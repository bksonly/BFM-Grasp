"""Explicit baseline -> DR initialization; preserve networks, targets, six optimizers and normalizers."""
import sys,json,hashlib,argparse,importlib.util
from pathlib import Path
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3]/'third_party'/'BFM-Zero'))
import torch
from core import make_agent,dimensions
import dr_runtime
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--manifest',required=True,help='Prepared 27-scale cache manifest');a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(42)
source=Path(a.source);dest=Path(a.output);assert not dest.exists()
saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
assert saved['contract_version']=='official_bfmzero_hand_v3' and saved['reward_mode']=='conditional' and saved['goal_ratio']==0
spec=importlib.util.spec_from_file_location('baseline_core',str(Path(__file__).resolve().parents[1]/'baseline/core.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
assert saved['dimensions']==old.dimensions('binary')
baseline,_=old.make_agent(device='cpu',batch=80);baseline._model.load_state_dict(saved['model'],strict=True)
converted=saved['model'].copy()
for new,previous in [('actor_proprio','proprio'),('privileged_tactile','tactile'),('history_privileged','history_actor')]:
    prefix=f'_obs_normalizer._normalizers.{previous}.'
    for key,value in saved['model'].items():
        if key.startswith(prefix):converted[key.replace(prefix,f'_obs_normalizer._normalizers.{new}.',1)]=value.clone()
agent,dims=make_agent(device='cpu',batch=80);agent._model.load_state_dict(converted,strict=True)
for name,state in saved['optimizers'].items():getattr(agent,name).load_state_dict(state)
obs={k:torch.randn(8,v)*.03 for k,v in old.dimensions('binary').items()}
newobs={**obs,'actor_proprio':obs['proprio'].clone(),'privileged_tactile':obs['tactile'].clone(),'history_privileged':obs['history_actor'].clone()}
z=baseline._model.sample_z(8);action=torch.randn(8,22).clamp(-1,1)
errors={}
for name,extra in [('backward_map',()),('forward_map',(z,action)),('actor',(z,.2)),('critic',(z,action)),('discriminator',(z,)),('aux_critic',(z,action))]:
    x=getattr(baseline._model,name)(obs,*extra);y=getattr(agent._model,name)(newobs,*extra)
    if name=='actor':x=x.mean;y=y.mean
    errors[name]=float((x-y).abs().max());assert torch.equal(x,y),(name,errors[name])
# New channels add normalizer buffers only; all network/target parameters are unchanged.
assert all(torch.equal(v,converted[k]) for k,v in saved['model'].items())
manifest=Path(a.manifest)
metadata=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_step=saved['step'],source_updates=saved['total_updates'],output_errors=errors,optimizer_names=list(saved['optimizers']),replay='empty, recollect under DR',best_success='reset for new DR distribution',normalizer_mapping={'actor_proprio':'proprio','privileged_tactile':'tactile','history_privileged':'history_actor'})
saved.update(model=converted,dimensions=dims,best_success=None,dr_config=dr_runtime.configure(None),dr_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),migration=metadata)
dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');torch.save(saved,tmp);tmp.replace(dest);dest.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
