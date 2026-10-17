import sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3]/'third_party'/'BFM-Zero'))
import torch
from torch.utils._pytree import tree_map,tree_leaves
from core import dimensions,Replay
from compact_replay import pack,unpack
torch.set_num_threads(2);n=128
def obs():
    x={k:torch.randn(n,v) for k,v in dimensions('binary').items()}
    x['tactile']=(torch.rand(n,10)>.5).float();x['contact_binary']=(torch.rand(n,10)>.5).float();x['privileged_tactile']=x['contact_binary'].clone()
    h=x['history_privileged'].reshape(n,4,76);h[:,:,44:54]=(torch.rand(n,4,10)>.5).float();h[:,:,54:]=x['history_actor'].reshape(n,4,76)[:,:,54:]
    return x
state={k:torch.randn(n,v) for k,v in dict(q=22,dq=22,object_position=3,object_quaternion=4,object_velocity=6,contact=10).items()};state['contact_point']=torch.randn(n,10,3)
t=dict(observation=obs(),action=torch.randn(n,22),z=torch.randn(n,256),reference_context=torch.ones(n,1,dtype=torch.bool),reference_frame=torch.zeros(n,1,dtype=torch.long),next={'observation':obs(),'terminated':torch.zeros(n,1,dtype=torch.bool)},object_id=torch.arange(n)[:,None],variant_id=torch.zeros(n,1,dtype=torch.long),reward_state=state,rp=torch.randn(n,1),aux_rewards={})
p=pack(t,True);u=unpack(p)
def equal(a,b,ids=None):
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k],ids) for k in a)
    return a.dtype==b.dtype and torch.equal(a if ids is None else a[ids],b)
assert equal(t,u)
r=Replay(256,'cpu');r.add(t);sample=r.sample(4096);ids=sample['object_id'].flatten()
assert equal(t,sample,ids)
r.add(t);r.add(t);sample=r.sample(4096);assert equal(t,sample,sample['object_id'].flatten())
before=sum(x.numel()*x.element_size() for x in tree_leaves(t))//n;after=sum(x.numel()*x.element_size() for x in tree_leaves(p))//n
print(dict(lossless=True,bytes_per_transition_before=before,bytes_per_transition_after=after,two_runs_full_replay_gib=after*1280000*10*2/1024**3))
