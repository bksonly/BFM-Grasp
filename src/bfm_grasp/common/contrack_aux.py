"""Hand-only ConTrack reward terms, evaluated on privileged replay states.

Expert z encodes an eight-state window. Evaluate the same ConTrack functions
against each state and average; a repeated window shares one reward definition.
No privileged target or measured object state is added to actor observations.
"""
import torch

DT=1/60

def stability_terms(current, previous, dt=DT):
    def penalty(value,scale):return 1-torch.exp(-value.square().mean(-1,keepdim=True)/(scale*scale))
    dq=current['dq'];vel=current['object_velocity']
    ddq=(dq-previous['dq'])/dt
    acceleration=(vel-previous['object_velocity'])/dt
    return dict(qvel_finger=-penalty(dq,1),qvel_object_linear=-penalty(vel[:,:3],1),
                qvel_object_angular=-.1*penalty(vel[:,3:],1),
                qacc_finger=-10*penalty(ddq,50),qacc_object_linear=-100*penalty(acceleration[:,:3],50),
                qacc_object_angular=-10*penalty(acceleration[:,3:],50))

def tracking_terms(current, target):
    """Current (N,...), target (N,window,...); coordinates are environment-local."""
    qerr=(current['q'][:,None]-target['q']).square().mean(-1)
    perr=(current['object_position'][:,None]-target['object_position']).square().mean(-1)
    dot=(current['object_quaternion'][:,None]*target['object_quaternion']).sum(-1).abs().clamp(0,1)
    theta=2*torch.acos(dot)
    match=current['contact'][:,None].bool()&target['contact'].bool()
    distance=(current['contact_point'][:,None]-target['contact_point']).square().sum(-1)
    count=match.sum(-1)
    location=(torch.exp(-distance/.03**2)*match).sum(-1)/count.clamp(min=1)
    reduce=lambda value:((value*target['window_mask']).sum(1,keepdim=True)/target['window_mask'].sum(1,keepdim=True).clamp(min=1) if 'window_mask' in target else value.mean(1,keepdim=True))
    return dict(tracking_finger=10*reduce(torch.exp(-4*qerr)),
                tracking_object_position=500*reduce(torch.exp(-perr/.05**2)),
                tracking_object_rotation=500*reduce(torch.exp(-theta.square()/.5**2)),
                contact=100*reduce(match.float().mean(-1)),contact_position=100*reduce(location))

def auxiliary_reward(current, target, rp, expert_mask, dt=DT):
    terms=tracking_terms(current,target)
    reward=(rp+expert_mask.float()*sum(terms.values()))*dt
    assert reward.ndim==2 and reward.shape[1]==1
    return reward,terms
