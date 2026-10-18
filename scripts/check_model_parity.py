"""Compare the same frozen checkpoint in historical and organized DR code."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--legacy-root');p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
if a.legacy_root:
 root=Path(a.legacy_root);sys.path[:0]=[str(root/'bfm_sharpa_dr'),str(root/'BFM-Zero')]
else:
 from _bootstrap import configure
 configure('dr')
import torch,numpy as np
from core import make_agent
from reference import load_reference
from motion_contract import tracking_context
torch.set_num_threads(4);torch.manual_seed(42)
agent,dims=make_agent(device='cpu',batch=32)
s=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
agent._model.load_state_dict(s['model']);agent._model.eval()
m=json.loads(Path(a.manifest).read_text());result={}
with torch.no_grad():
 for i,path in enumerate(m['canonical_references']):
  ref=load_reference(path,'cpu');z=tracking_context(agent._model,ref,8,dims)
  ids=torch.arange(0,len(z),5);obs={k:torch.randn(len(ids),v)*.01 for k,v in dims.items()}
  obs.update(ref.goal(ids));obs['actor_proprio']=obs['proprio'].clone();obs['tactile']=obs['contact_binary'].clone();obs['privileged_tactile']=obs['tactile'].clone()
  action=agent._model.actor(obs,z[ids],agent._model.cfg.actor_std).mean
  result[f'z_{i}']=z.numpy();result[f'action_{i}']=action.numpy()
np.savez_compressed(a.output,**result)
print(json.dumps({'output':a.output,'checkpoint_step':s['step'],'profile':'legacy' if a.legacy_root else 'organized','tensors':len(result)}))
