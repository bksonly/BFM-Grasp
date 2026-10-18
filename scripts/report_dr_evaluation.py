"""Summarize stopped DR runs and explicit independent evaluation artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--historical-root',required=True);p.add_argument('--evaluation-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
hist=Path(a.historical_root);ev=Path(a.evaluation_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
result={'training':{},'evaluation':{},'parity':{}}
fig,ax=plt.subplots(figsize=(9.5,4.8))
for group,label,color in [('resume','Transfer from ten-object policy','#2879b9'),('scratch','From scratch','#ed923b')]:
 path=hist/f'logs/bfm-dr-scale27-{group}-v1/train'
 e=EventAccumulator(str(path/'tensorboard'),size_guidance={'scalars':0});e.Reload();v=e.Scalars('Success/overall_macro_pct')
 result['training'][group]={'curve':[{'step':x.step,'percent':x.value} for x in v],'best':json.loads((path/'best.json').read_text())}
 ax.plot([x.step for x in v],[x.value for x in v],label=label,color=color,lw=2)
for folder in ['resume','resume-original','scratch-original','scratch-repeat','scale-1.1','scale-1.05-diagnostic','scale-1.15-refresh4','unseen-shapes','resume-rollout-embeddings','scratch-refresh4','scale-1.1-refresh4']:
 f=ev/folder/'evaluation.json'
 if not f.exists():continue
 d=json.loads(f.read_text());rows=[]
 for c in range(10):
  subset=[x for x in d['episodes'] if x.get('class_id',x['object_group']//27)==c]
  rows.append({'class_id':c,'successes':sum(x['success'] for x in subset),'count':len(subset)})
 result['evaluation'][folder]={'successes':d['successes'],'count':d['count'],'percent':100*d['successes']/d['count'],'per_class':rows}
a1=np.load(ev/'parity-organized.npz');a2=np.load(ev/'parity-legacy.npz')
result['parity']['frozen_model_max_absolute_error']=max(float(np.abs(a1[k]-a2[k]).max()) for k in a1.files)
f1=ev/'resume/evaluation.json';f2=ev/'resume-original/evaluation.json'
result['parity']['resume_all_episode_records_identical']=json.loads(f1.read_text())==json.loads(f2.read_text())
scratch=json.loads((ev/'scratch-repeat/evaluation.json').read_text())
for row in scratch['episodes']:row.pop('class_id',None)
result['parity']['scratch_all_episode_records_identical']=scratch==json.loads((ev/'scratch-original/evaluation.json').read_text())
result['limitations']=['One trained seed per condition.','Seen geometry evaluation includes all training DR; held-out scale and new mesh tests are separate.','The first organized scratch evaluation aborted before scoring due to one reset speed validation failure; this attempt is retained. No threshold was relaxed.']
ax.set(xlabel='Training steps',ylabel='Overall training success (%)',ylim=(0,100),title='Domain randomization: ten categories, 27 scales each');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'dr-training.svg');fig.savefig(out/'dr-training.png',dpi=180)
(out/'dr-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['parity']))
