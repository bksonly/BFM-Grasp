"""t-SNE of one mean frozen-B embedding per complete physical rollout."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/analysis-deps'))
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
NAMES=['Box','Pencil','Cylindrical can','Flat can','Bottle','Mug','Hammer','Banana','Bowl','Plate']
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
d=np.load(a.input);z=d['z'];labels=d['class_id'];success=d['success']
assert len(z)==len(labels)==len(success) and np.isfinite(z).all()
assert np.all(d['frames']==350), 'Expected complete initial-through-terminal states'
y=TSNE(n_components=2,perplexity=40,init='pca',learning_rate='auto',random_state=42,max_iter=1500).fit_transform(z)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(10.5,6.4));colors=plt.get_cmap('tab10')
for i,name in enumerate(NAMES):
 good=(labels==i)&success;bad=(labels==i)&~success
 ax.scatter(y[good,0],y[good,1],s=13,alpha=.65,color=colors(i),linewidths=0,label=name)
 ax.scatter(y[bad,0],y[bad,1],s=20,alpha=.9,color=colors(i),marker='x',linewidths=.7)
handles,labs=ax.get_legend_handles_labels();handles.append(Line2D([0],[0],marker='x',color='#444',linestyle='None',markersize=6));labs.append('Failed rollout')
ax.legend(handles,labs,bbox_to_anchor=(1.02,1),loc='upper left',frameon=False)
ax.set(xlabel='t-SNE 1',ylabel='t-SNE 2',title='Grasp rollout embeddings under domain randomization');ax.spines[['top','right']].set_visible(False);fig.tight_layout()
fig.savefig(out/'rollout-latents.svg');fig.savefig(out/'rollout-latents.png',dpi=180)
metadata={'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'rollouts':len(z),'successes':int(success.sum()),'objects':NAMES,'frames_per_rollout':350,'embedding':'project_z(mean_t B(actual privileged state_t)); includes initial and terminal pre-reset states','control_context':'unchanged canonical-reference future-eight-frame z; not the plotted trajectory embedding','tsne':{'perplexity':40,'seed':42,'iterations':1500,'init':'pca','learning_rate':'auto'},'points':[{'x':float(p[0]),'y':float(p[1]),'object':NAMES[int(c)],'variant_id':int(v),'success':bool(s)} for p,c,v,s in zip(y,labels,d['variant_id'],success)],'limitations':'B receives privileged geometry and contact information. Clustering is descriptive, not independent evidence of generalization. All rollouts, including failures, are included; no display jitter.'}
(out/'rollout-latents.json').write_text(json.dumps(metadata,indent=2));print(json.dumps({k:v for k,v in metadata.items() if k!='points'}))
