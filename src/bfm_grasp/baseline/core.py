"""BFM-Zero interface contract. No simulator or reference targets enter the actor."""
import itertools
import ast
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig, FBcprAuxModelArchiConfig
from humanoidverse.agents.nn_models import ActorArchiConfig, BackwardArchiConfig, ForwardArchiConfig, DiscriminatorArchiConfig, RewardNormalizerConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.normalizers import ObsNormalizerConfig, IdentityNormalizerConfig, BatchNormNormalizerConfig

TACTILE_DIMS={'none':0,'binary':10,'magnitude':10,'vector':30}
ACTOR_KEYS=['proprio','tactile','last_action','history_actor']
GOAL_KEYS=['proprio','corners','contact_binary']
FULL_KEYS=ACTOR_KEYS+['corners','object_velocity']

def dimensions(mode):
    n=TACTILE_DIMS[mode]
    return dict(proprio=44,tactile=n,last_action=22,history_actor=4*(66+n),corners=24,object_velocity=6,contact_binary=10)

def make_agent(mode='binary',device='cpu',width=2048,batch=1024,z_dim=256,goal_ratio=0):
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)
        device='cuda'
    dims=dimensions(mode)
    space=gym.spaces.Dict({k:gym.spaces.Box(-np.inf,np.inf,(v,),dtype=np.float32) for k,v in dims.items()})
    filt=lambda keys:DictInputFilterConfig(key=keys)
    assert width==2048 and z_dim==256 and goal_ratio in (0.,.2)
    # Import the actual upstream training configuration, not class defaults.
    source=Path(str(__import__('pathlib').Path(__file__).resolve().parents[3]/'third_party'/'BFM-Zero/humanoidverse/train.py')).read_text()
    calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='TrainConfig']
    agent_node=next(k.value for n in calls for k in n.keywords if k.arg=='agent')
    cfg=eval(compile(ast.Expression(agent_node),'<official BFM-Zero agent config>','eval'),globals())
    data=cfg.model_dump()
    data['model']['device']=device
    for name,keys in [('actor',ACTOR_KEYS),('b',GOAL_KEYS),('discriminator',GOAL_KEYS),('f',FULL_KEYS),('critic',FULL_KEYS),('aux_critic',FULL_KEYS)]:
        data['model']['archi'][name]['input_filter']=filt(keys).model_dump()
    data['model']['obs_normalizer']=ObsNormalizerConfig(normalizers={k:BatchNormNormalizerConfig(momentum=.01) for k in dims},allow_mismatching_keys=True).model_dump()
    data['train']['batch_size']=batch
    data['train']['train_goal_ratio']=float(goal_ratio)
    data['aux_rewards']=[];data['aux_rewards_scaling']={}
    if device=='cpu':data['compile']=False
    cfg=FBcprAuxAgentConfig.model_validate(data)
    from conditional_agent import ConditionalConTrackAgent
    return ConditionalConTrackAgent(obs_space=space,action_dim=22,cfg=cfg),dims

def box_corners(vertices):
    low,high=vertices.min(0).values,vertices.max(0).values
    bits=torch.tensor(list(itertools.product((0,1),repeat=3)),device=vertices.device)
    return torch.where(bits.bool(),high,low)

def local_corners(corners,object_pos,object_rot,wrist_pos,wrist_rot):
    world=torch.einsum('nij,kj->nki',object_rot,corners)+object_pos[:,None]
    return torch.einsum('nji,nkj->nki',wrist_rot,world-wrist_pos[:,None]).flatten(1)

class Tactile:
    """Ten PP/DP channels; binary policy input uses sensor contact flags directly."""
    def __init__(self,mode):self.mode=mode
    def reset(self,ids):pass
    def encode(self,force_local):
        assert force_local.shape[-2:]==(10,3)
        norm=force_local.norm(dim=-1)
        if self.mode=='none':return norm[:,:0]
        if self.mode=='binary':raise ValueError('Binary10 must use ConTrack contact-point validity, not a force threshold')
        if self.mode=='magnitude':return norm
        return force_local.flatten(1)

class History:
    def __init__(self,n,mode,device):
        self.value=torch.zeros(n,4,66+TACTILE_DIMS[mode],device=device)
    def push(self,obs):
        result=self.value.flatten(1).clone()
        frame=torch.cat([obs[k] for k in ('proprio','tactile','last_action')],-1)
        self.value=torch.cat([self.value[:,1:],frame[:,None]],1)
        return result
    def reset(self,ids):self.value[ids]=0

def absolute_targets(action,lower,upper):
    return (upper+lower)/2+(upper-lower)/2*action.clamp(-1,1)

class Reference:
    def __init__(self,proprio,corners,targets=None):
        assert len(proprio)==len(corners) and len(proprio)>=9
        self.proprio=proprio;self.corners=corners
        self.targets=targets
        if targets is None or 'contact' not in targets:
            raise ValueError('B/D require measured ten-body reference contacts')
        self.contact=targets['contact']
        assert self.contact.shape==(len(proprio),10)
        assert torch.all((self.contact==0)|(self.contact==1))
    def goal(self,frame):
        # Clamp only after the supplied expert ends. Measured successful experts
        # may include the complete lift and hold; synthetic targets are not data.
        idx=frame.clamp(0,len(self.proprio)-1)
        return {'proprio':self.proprio[idx],'corners':self.corners[idx],'contact_binary':self.contact[idx]}

class ExpertSampler:
    def __init__(self,references,dims,device,seq=8):
        self.refs=references;self.dims=dims;self.device=device;self.seq=seq
    def sample(self,batch,object_ids=None):
        assert batch%self.seq==0
        current=[];following=[];targets=[];groups=[]
        for block in range(batch//self.seq):
            group=int(object_ids[block*self.seq]) if object_ids is not None else torch.randint(len(self.refs),(1,)).item()
            if object_ids is not None:
                assert (object_ids[block*self.seq:(block+1)*self.seq]==group).all(), 'Balanced replay groups must align with expert sequence blocks'
            r=self.refs[group]
            start=torch.randint(len(r.proprio)-self.seq,(1,)).item()
            idx=torch.arange(start,start+self.seq,device=self.device)
            current.append(r.goal(idx));following.append(r.goal(idx+1))
            if r.targets is not None:
                targets.append({k:v[idx+1].unsqueeze(0).expand(self.seq,*v[idx+1].shape) for k,v in r.targets.items()})
            groups.append(torch.full((self.seq,1),group,device=self.device,dtype=torch.long))
        def merge(rows):
            # Unused expert channels are placeholders and MUST be filtered out of B/D.
            out={k:torch.zeros(batch,v,device=self.device) for k,v in self.dims.items()}
            out.update({k:torch.cat([r[k] for r in rows]) for k in GOAL_KEYS})
            return out
        result={'observation':merge(current),'next':{'observation':merge(following)},'object_id':torch.cat(groups)}
        if targets:result['targets']={k:torch.cat([t[k] for t in targets]) for k in targets[0]}
        return result

class Replay:
    def __init__(self,capacity,device):self.capacity=capacity;self.device=device;self.storage=None;self.cursor=0;self.size=0
    def add(self,tree):
        from torch.utils._pytree import tree_map,tree_leaves
        n=len(tree_leaves(tree)[0]);assert n<=self.capacity
        if self.storage is None:self.storage=tree_map(lambda x:torch.empty((self.capacity,*x.shape[1:]),dtype=x.dtype,device=self.device),tree)
        ids=(torch.arange(n,device=self.device)+self.cursor)%self.capacity
        tree_map(lambda dst,src:dst.index_copy_(0,ids,src.detach().to(self.device)),self.storage,tree)
        self.cursor=(self.cursor+n)%self.capacity;self.size=min(self.capacity,self.size+n)
    def sample(self,n):
        from torch.utils._pytree import tree_map
        assert self.size
        ids=torch.randint(self.size,(n,),device=self.device)
        return tree_map(lambda x:x[ids].clone(),self.storage)

class BalancedReplay:
    """Equal object sampling across live and retained physical rollouts."""
    def __init__(self,live,retained,block_size=1):
        self.live=live;self.retained=retained;self.block_size=block_size
    def sample(self,n):
        from torch.utils._pytree import tree_map
        sources=[self.live,*self.retained]
        assert n%self.block_size==0
        blocks=n//self.block_size
        extras=set(torch.randperm(len(sources))[:blocks%len(sources)].tolist()) if self.block_size>1 else set(range(blocks%len(sources)))
        sizes=[self.block_size*(blocks//len(sources)+(i in extras)) for i in range(len(sources))]
        rows=[source.sample(size) for source,size in zip(sources,sizes) if size]
        return tree_map(lambda *xs:torch.cat(xs),*rows)
