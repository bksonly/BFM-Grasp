import sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3]/'third_party'/'BFM-Zero'))
import torch
from core import Reference,dimensions,make_agent
from scale_reference import ScaleExpertSampler
torch.set_num_threads(2);torch.manual_seed(42)
def ref(value):
    n=16;q=torch.zeros(n,44);corners=torch.full((n,24),value)
    targets=dict(q=q[:,:22],object_position=torch.zeros(n,3),object_quaternion=torch.tensor([1.,0,0,0]).expand(n,4).clone(),contact=torch.ones(n,10),contact_point=torch.full((n,10,3),value))
    return Reference(q,corners,targets)
canonical=[ref(float(i)) for i in range(10)];variants=[ref(float(i)) for i in range(270)]
dims=dimensions('binary');sampler=ScaleExpertSampler(canonical,variants,dims,'cpu')
classes=torch.arange(10).repeat_interleave(8)[:,None];vid=classes*27+torch.arange(80)[:,None]%27
batch=sampler.sample(80,classes,vid,starts=torch.zeros(10,dtype=torch.long))
assert torch.equal(batch['observation']['corners'][:,0],vid[:,0].float())
assert torch.equal(batch['canonical_next']['corners'][:,0],classes[:,0].float())
assert torch.equal(batch['targets']['contact_point'][:,0,0,0],vid[:,0].float())
permutation=torch.randperm(80);rematched=sampler.targets(vid.flatten(),batch['window_frames'][permutation]);assert torch.equal(rematched['contact_point'][:,0,0,0],vid[:,0].float())
agent,_=make_agent(device='cpu',batch=80);agent.reward_mode='conditional';agent.expert_batch=batch
z=agent.encode_expert(batch['next']['observation']);batch['next']['observation']['corners'].add_(999)
z2=agent.encode_expert(batch['next']['observation']);assert torch.equal(z,z2),'variant geometry leaked into canonical z'
obs={k:torch.randn(80,v)*.01 for k,v in dims.items()};obs['contact_binary']=torch.ones(80,10);obs['tactile']=torch.ones(80,10)
state=dict(q=torch.zeros(80,22),dq=torch.zeros(80,22),object_position=torch.zeros(80,3),object_quaternion=torch.tensor([1.,0,0,0]).expand(80,4).clone(),object_velocity=torch.zeros(80,6),contact=torch.ones(80,10),contact_point=torch.zeros(80,10,3))
transition=dict(observation=obs,action=torch.zeros(80,22),z=z,reference_context=torch.ones(80,1,dtype=torch.bool),reference_frame=torch.zeros(80,1,dtype=torch.long),next={'observation':obs,'terminated':torch.zeros(80,1,dtype=torch.bool)},object_id=classes,variant_id=vid,reward_state=state,rp=torch.zeros(80,1),aux_rewards={})
class Replay:
    def sample(self,n):assert n==80;return transition
metrics=agent.update({'train':Replay(),'expert_slicer':sampler},step=0)
assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
assert all(k in metrics for k in ['fb_loss','disc_loss','critic_loss','aux_critic_loss','actor_loss'])
print('PASS canonical z isolation, per-row scale matching, relabel target selection, full official loss update; metrics',len(metrics))
