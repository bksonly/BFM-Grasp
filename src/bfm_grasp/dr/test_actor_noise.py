import sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3]/'third_party'/'BFM-Zero'))
import torch
from types import SimpleNamespace as NS
import dr_runtime as dr
from core import dimensions,History,push_observation_histories,make_agent
from torch.utils._pytree import tree_map
torch.set_num_threads(2);torch.manual_seed(123)
dr.configure(None);dr.CONFIG.update(mass_multiplier=[1.,1.],friction_multiplier=[1.,1.],q_noise_std=.2,dq_noise_std=.3,tactile_flip_probability=1.,pd_stiffness_multiplier=[.9,1.1],pd_damping_multiplier=[.9,1.1]);dr.ENABLED=True
base={k:torch.zeros(8,v) for k,v in dimensions('binary').items()}
noisy=dr.corrupt(tree_map(torch.clone,base))
assert torch.equal(noisy['proprio'],base['proprio']) and not torch.equal(noisy['actor_proprio'],base['proprio'])
ah=History(8,'binary','cpu');ph=History(8,'binary','cpu')
for _ in range(6):push_observation_histories(ah,ph,noisy)
assert ph.value.count_nonzero()==0 and ah.value.count_nonzero()>0
noisy['history_actor']=ah.value.flatten(1);noisy['history_privileged']=ph.value.flatten(1)
agent,_=make_agent(device='cpu',batch=80)
model=agent._model
for net in ['_backward_map','_discriminator','_forward_map','_critic','_aux_critic']:
    layer=getattr(model,net).input_filter
    assert torch.equal(layer(base),layer(noisy)),net
assert not torch.equal(model._actor.input_filter(base),model._actor.input_filter(noisy))
# Every observation field is independently normalized: no actor statistics enter privileged filters.
model._obs_normalizer.train();model._obs_normalizer(noisy);model._obs_normalizer.eval()
x=model._obs_normalizer(base);y=model._obs_normalizer(noisy)
for net in ['_backward_map','_discriminator','_forward_map','_critic','_aux_critic']:
    layer=getattr(model,net).input_filter;assert torch.equal(layer(x),layer(y)),net
writes={}
robot=NS(write_joint_stiffness_to_sim=lambda x,env_ids:writes.update(kp=x),write_joint_damping_to_sim=lambda x,env_ids:writes.update(kd=x))
env=NS(scene={'objects':NS(root_physx_view=None),'rby1a_sharpa':robot},device='cpu',dr_nominal={'kp':torch.ones(4,22),'kd':torch.ones(4,22)})
dr.randomize_dynamics(env,torch.tensor([1,3]))
for x in writes.values():
    assert x.shape==(2,22) and ((x>=.9)&(x<=1.1)).all() and (x.std(-1)>0).all()
print('PASS per-joint PD; actor-only current/history noise; all five privileged filters isolated before/after normalization')
