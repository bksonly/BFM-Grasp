"""Canonical z, scale-matched discriminator positives and reward targets."""
import torch
from core import GOAL_KEYS
class ScaleExpertSampler:
    def __init__(self,refs,variants,dims,device,seq=8):
        self.refs=refs;self.dims=dims;self.device=device;self.seq=seq
        self.canonical={k:torch.stack([r.goal(torch.arange(len(r.proprio),device=device))[k] for r in refs]) for k in GOAL_KEYS}
        self.geometry={k:torch.stack([r.goal(torch.arange(len(r.proprio),device=device))[k] for r in variants]) for k in GOAL_KEYS}
        self.target_values={k:torch.stack([r.targets[k] for r in variants]) for k in variants[0].targets}
        self.length=len(refs[0].proprio)
    def targets(self,variants,frames):
        valid=frames<self.length;frames=frames.clamp(0,self.length-1)
        result={k:v[variants[:,None],frames] for k,v in self.target_values.items()}
        result['window_mask']=valid.float()
        return result
    def sample(self,batch,object_ids,variant_ids,starts=None):
        assert batch%8==0
        groups=object_ids.reshape(-1,8);assert (groups==groups[:,:1]).all()
        variants=variant_ids.flatten();assert torch.equal(variants//27,object_ids.flatten())
        starts=torch.randint(self.length-8,(batch//8,),device=self.device) if starts is None else starts
        frames=(starts[:,None]+torch.arange(8,device=self.device)).flatten()
        def obs(values,ids,idx):
            out={k:torch.zeros(batch,v,device=self.device) for k,v in self.dims.items()}
            out.update({k:v[ids,idx] for k,v in values.items()});return out
        windows=(starts[:,None]+1+torch.arange(8,device=self.device)).repeat_interleave(8,0)
        return dict(observation=obs(self.geometry,variants,frames),next={'observation':obs(self.geometry,variants,frames+1)},
                    canonical_next=obs(self.canonical,object_ids.flatten(),frames+1),window_frames=windows,
                    object_id=object_ids,variant_id=variant_ids,targets=self.targets(variants,windows))
