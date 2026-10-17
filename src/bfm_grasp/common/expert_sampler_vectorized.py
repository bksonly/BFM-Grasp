"""Vectorized expert slicing with the existing per-object eight-frame contract."""
import torch
from core import ExpertSampler,GOAL_KEYS

class VectorizedExpertSampler(ExpertSampler):
    def sample(self,batch,object_ids=None,starts=None):
        assert batch%self.seq==0
        blocks=batch//self.seq
        if object_ids is None:
            groups=torch.randint(len(self.refs),(blocks,),device=self.device)
        else:
            grouped=object_ids.reshape(blocks,self.seq)
            groups=grouped[:,0]
            assert (grouped==groups[:,None]).all(), 'Expert blocks must share one physical object'
        observation={k:torch.zeros(batch,n,device=self.device) for k,n in self.dims.items()}
        following={k:torch.zeros_like(v) for k,v in observation.items()}
        targets={}
        for group,ref in enumerate(self.refs):
            block_ids=(groups==group).nonzero().flatten()
            if not len(block_ids):continue
            initial=torch.randint(len(ref.proprio)-self.seq,(len(block_ids),),device=self.device) if starts is None else starts[block_ids]
            frames=initial[:,None]+torch.arange(self.seq,device=self.device)[None]
            rows=(block_ids[:,None]*self.seq+torch.arange(self.seq,device=self.device)[None]).flatten()
            for key,value in ref.goal(frames.flatten()).items():observation[key][rows]=value
            for key,value in ref.goal(frames.flatten()+1).items():following[key][rows]=value
            assert ref.targets is not None
            for key,value in ref.targets.items():
                if key not in targets:targets[key]=torch.empty((batch,self.seq,*value.shape[1:]),dtype=value.dtype,device=self.device)
                targets[key][rows]=value[frames+1].repeat_interleave(self.seq,dim=0)
        return dict(observation=observation,next={'observation':following},object_id=groups.repeat_interleave(self.seq)[:,None],targets=targets)
