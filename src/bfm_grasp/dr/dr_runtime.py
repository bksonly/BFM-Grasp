"""Opt-in BFM domain randomization. Never changes tracking targets or wrist commands."""
import json,math
from pathlib import Path
import torch
CONFIG={};ENABLED=False
DEFAULT=json.loads(Path(__file__).with_name('dr_defaults.json').read_text())
def configure(path):
    global CONFIG
    supplied=json.loads(Path(path).read_text()) if path else {}
    assert not set(supplied)-set(DEFAULT),'Unknown DR field'
    CONFIG={**DEFAULT,**supplied}
    for key in ('mass_multiplier','friction_multiplier','pd_stiffness_multiplier','pd_damping_multiplier'):
        lo,hi=CONFIG[key];assert 0<lo<=hi
    for key in ('position_xy_m','yaw_deg','q_noise_std','dq_noise_std','force_newton','torque_nm'):assert CONFIG[key]>=0
    for key in ('tactile_flip_probability','force_probability'):assert 0<=CONFIG[key]<=1
    assert isinstance(CONFIG['friction_buckets'],int) and CONFIG['friction_buckets']>=2
    return CONFIG

def uniform(n,bounds,device):
    lo,hi=bounds
    return torch.rand(n,device=device)*(hi-lo)+lo

def pose_state(state,vertices,table_z,origins,perturb=True):
    from isaaclab.utils.math import matrix_from_quat,quat_mul
    state=state.clone();n=len(state)
    if perturb:
        state[:,0,:2]+=(torch.rand(n,2,device=state.device)*2-1)*CONFIG['position_xy_m']
        yaw=(torch.rand(n,device=state.device)*2-1)*math.radians(CONFIG['yaw_deg'])
        rot=torch.zeros(n,4,device=state.device);rot[:,0]=torch.cos(yaw/2);rot[:,3]=torch.sin(yaw/2)
        state[:,0,3:7]=quat_mul(rot,state[:,0,3:7])
    # Maintain original positive clearance; raise only if scaled geometry penetrates table.
    R=matrix_from_quat(state[:,0,3:7]);bottom=(R[:,2]@vertices.T).min(-1).values+state[:,0,2]
    state[:,0,2]+=torch.clamp(table_z+origins[:,2]+.0001-bottom,min=0)
    return state

def randomize_dynamics(env,ids):
    if not ENABLED:return
    obj=env.scene['objects'];robot=env.scene['rby1a_sharpa'];n=len(ids)
    # PhysX buffers use CPU indices; one object per environment in this scene.
    index=ids.to('cpu');view=obj.root_physx_view
    if not hasattr(env,'dr_nominal'):
        env.dr_nominal=dict(mass=view.get_masses().clone(),inertia=view.get_inertias().clone(),material=view.get_material_properties().clone(),kp=robot.data.default_joint_stiffness.clone(),kd=robot.data.default_joint_damping.clone())
    base=env.dr_nominal
    if CONFIG['mass_multiplier']!=[1.,1.]:
        masses=view.get_masses();inertias=view.get_inertias();f=uniform(n,CONFIG['mass_multiplier'],masses.device)
        masses[index]=base['mass'][index]*f.reshape(-1,*([1]*(masses.ndim-1)))
        inertias[index]=base['inertia'][index]*f.reshape(-1,*([1]*(inertias.ndim-1)))
        view.set_masses(masses,index);view.set_inertias(inertias,index)
    if CONFIG['friction_multiplier']!=[1.,1.]:
        material=view.get_material_properties()
        # Reuse a bounded palette, rather than create continuously new materials.
        levels=torch.linspace(*CONFIG['friction_multiplier'],CONFIG['friction_buckets'],device=material.device)
        f=levels[torch.randint(len(levels),(n,),device=material.device)]
        material[index]=base['material'][index];material[index,:,:2]*=f[:,None,None]
        view.set_material_properties(material,index)
    for key,cfg,write in [('kp','pd_stiffness_multiplier',robot.write_joint_stiffness_to_sim),('kd','pd_damping_multiplier',robot.write_joint_damping_to_sim)]:
        if CONFIG[cfg]!=[1.,1.]:
            nominal=base[key][ids]
            lo,hi=CONFIG[cfg]
            multiplier=torch.rand_like(nominal)*(hi-lo)+lo
            write(nominal*multiplier,env_ids=ids)

def corrupt(obs):
    obs['actor_proprio']=obs['proprio'].clone()
    if not ENABLED:return obs
    for sl,key in [(slice(0,22),'q_noise_std'),(slice(22,44),'dq_noise_std')]:
        if CONFIG[key]:obs['actor_proprio'][:,sl]+=torch.randn_like(obs['actor_proprio'][:,sl])*CONFIG[key]
    if CONFIG['tactile_flip_probability']:
        flip=torch.rand_like(obs['tactile'])<CONFIG['tactile_flip_probability'];obs['tactile']=torch.where(flip,1-obs['tactile'],obs['tactile'])
    return obs

def force(env):
    if not ENABLED or CONFIG['force_probability']==0:return
    active=(torch.rand(env.num_envs,1,1,device=env.device)<CONFIG['force_probability']).float()
    f=(2*torch.rand(env.num_envs,1,3,device=env.device)-1)*CONFIG['force_newton']*active
    t=(2*torch.rand_like(f)-1)*CONFIG['torque_nm']*active
    env.scene['objects'].set_external_force_and_torque(f,t)


@torch.no_grad()
def refresh_reset(wrapped,env,bridge):
    """Synchronous reset: refresh four steps, validate two, restore exact sampled state.

    No resampling/rejection: all six steps are discarded, and a failed validation
    aborts. No rollout, reward, replay, normalizer or training step is generated.
    """
    global ENABLED
    assert bool((env.episode_length_buf==0).all()), 'DR reset requires synchronized episodes'
    saved=env.scene['objects'].data.object_state_w.clone()
    ids=torch.arange(env.num_envs,device=env.device)
    previous=ENABLED;ENABLED=False
    checks=[]
    def restore():
        wrapped.reset()
        env.scene['objects'].write_object_state_to_sim(saved)
        bridge.reset_history(ids)
    try:
        q=torch.stack([env.hand_motion[int(g)][0,6:] for g in env.group_id])
        action=(2*(q-bridge.term.lower)/(bridge.term.upper-bridge.term.lower)-1).clamp(-1,1)
        # Clear previous episode external forces during reset probes.
        zero=torch.zeros(env.num_envs,1,3,device=env.device)
        env.scene['objects'].set_external_force_and_torque(zero,zero)
        diagnostic_steps=int(__import__('os').environ.get('BFM_RESET_DIAGNOSTIC_STEPS','0'))
        for i in range(max(6,diagnostic_steps)):
            restore();obs=bridge.observe();bridge.before_step(obs,action);wrapped.step(action)
            speed=bridge.reward_state['object_velocity'][:,:3].norm(dim=-1)
            valid=torch.isfinite(speed)&(speed<=1.)
            checks.append(dict(stage='refresh' if i<4 else 'validation',max_speed_m_s=float(speed.max()),invalid_envs=int((~valid).sum()),invalid_ids=(~valid).nonzero().flatten().tolist(),invalid_groups=env.group_id[~valid].tolist()))
            if not diagnostic_steps and i>=4 and not bool(valid.all()):raise RuntimeError(f'DR reset validation failed: {checks}')
        if diagnostic_steps:raise RuntimeError(f'Diagnostic-only reset probes (never scored): {checks}')
        restore();bridge.captured=None
    finally:ENABLED=previous
    return checks
