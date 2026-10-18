"""Plot actual future-eight-frame reference contexts from a frozen DR checkpoint."""
import argparse, hashlib, json, sys
from pathlib import Path
from _bootstrap import configure, ROOT
sys.path.insert(0, str(ROOT / '.cache/analysis-deps'))
configure('dr')
import numpy as np
import torch
from core import make_agent
from reference import load_reference
from motion_contract import tracking_context
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--manifest',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
torch.set_num_threads(4)
agent,dims=make_agent(device='cpu',batch=32)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
agent._model.load_state_dict(saved['model']);agent._model.eval()
manifest=json.loads(Path(a.manifest).read_text())
zs=[];labels=[];frames=[];names=[];references=[]
with torch.no_grad():
 for i,path in enumerate(manifest['canonical_references']):
  ref=load_reference(path,'cpu');z=tracking_context(agent._model,ref,8,dims).numpy()
  # Equal temporal sampling, all phases including approach and lift/hold.
  idx=np.arange(0,len(z),3)
  zs.append(z[idx]);frames.extend(idx.tolist());labels.extend([i]*len(idx))
  name=manifest['variants'][i*27]['name']
  names.append({'box08':'Box','pencil03':'Pencil','cylinder':'Cylindrical can','flat_can':'Flat can','bottle':'Bottle','mug':'Mug','tool':'Hammer','banana-A':'Banana','B09':'Bowl','P02':'Plate'}.get(name,name))
  references.append({'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()})
z=np.concatenate(zs);labels=np.asarray(labels)
y=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=42,max_iter=1500).fit_transform(z)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(10,6.2));colors=plt.get_cmap('tab10')
for i,name in enumerate(names):
 mask=labels==i;ax.scatter(y[mask,0],y[mask,1],s=15,alpha=.8,color=colors(i),label=name)
ax.set(xlabel='t-SNE 1',ylabel='t-SNE 2',title='Reference-conditioned grasp latents')
ax.spines[['top','right']].set_visible(False);ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',frameon=False)
fig.tight_layout();fig.savefig(out/'reference-latents.svg');fig.savefig(out/'reference-latents.png',dpi=180)
np.savez_compressed(out/'reference-latents.npz',z=z,embedding=y,labels=labels,frames=frames)
meta={'checkpoint':a.checkpoint,'checkpoint_step':saved['step'],'references':references,'names':names,'points':len(z),'window':8,'stride':3,'tsne':{'perplexity':30,'seed':42,'iterations':1500,'init':'pca','learning_rate':'auto'},'interpretation':'Each point is one time-window context from one of ten canonical trajectories, not an independent grasp trial. 2D separation alone does not establish generalization.'}
(out/'reference-latents.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
