"""Episode success measured on pre-reset states, independent of termination logic."""
from collections import defaultdict,deque
import torch

class EpisodeSuccessLog:
    def __init__(self,writer,groups,reference_mask,window=4096,group_names=None):
        self.group_names=group_names or {}
        self.group_ids=sorted(set(groups.cpu().tolist()))
        self.snapshot_counts={g:0 for g in self.group_ids}
        self.writer=writer;self.groups=groups;self.reference_mask=reference_mask.flatten()
        self.minimum=torch.full_like(groups,float('inf'),dtype=torch.float32)
        self.bottom=self.minimum.clone();self.frames=torch.zeros_like(groups)
        self.recent=defaultdict(lambda:deque(maxlen=window))
        self.counts=defaultdict(int);self.successes=defaultdict(int)

    def step(self,phase,lengths,rise,bottom,done,global_step):
        late=phase>=lengths-61
        self.minimum[late]=torch.minimum(self.minimum[late],rise[late])
        self.bottom[late]=torch.minimum(self.bottom[late],bottom[late])
        self.frames[late]+=1
        ids=done.nonzero().flatten()
        if not len(ids):return
        success=(self.minimum>=.04)&(self.bottom>.01)&(self.frames>=60)
        rows=torch.stack([self.groups[ids],self.reference_mask[ids].long(),success[ids].long()],1).cpu().tolist()
        touched=set()
        for group,is_ref,ok in rows:
            condition='reference_z' if is_ref else 'random_z'
            for key in ('all',f'object_{group}',f'all/{condition}',f'object_{group}/{condition}'):
                self.recent[key].append(ok);self.counts[key]+=1;self.successes[key]+=ok;touched.add(key)
        for key in sorted(touched):
            self.writer.add_scalar(f'train_success/{key}/rolling_rate',sum(self.recent[key])/len(self.recent[key]),global_step)
            self.writer.add_scalar(f'train_success/{key}/cumulative_rate',self.successes[key]/self.counts[key],global_step)
            self.writer.add_scalar(f'train_success/{key}/episodes',self.counts[key],global_step)
            self.writer.add_scalar(f'train_success/{key}/rolling_count',len(self.recent[key]),global_step)
        # Equal per-object windows prevent the global deque from favoring the
        # last physical group in a batch of simultaneously finished episodes.
        for condition,section in [('reference_z','Success'),('random_z','Success_random')]:
            rates=[];total_ok=0;total_n=0
            for group in self.group_ids:
                values=self.recent[f'object_{group}/{condition}']
                if not values:continue
                rate=sum(values)/len(values);rates.append(rate)
                total_ok+=sum(values);total_n+=len(values)
                name=self.group_names.get(group,f'object_{group}')
                self.writer.add_scalar(f'{section}/object_{group:02d}_{name}_pct',100*rate,global_step)
            if rates:
                self.writer.add_scalar(f'{section}/overall_macro_pct',100*sum(rates)/len(rates),global_step)
                self.writer.add_scalar(f'{section}/overall_micro_pct',100*total_ok/total_n,global_step)
                self.writer.add_scalar(f'{section}/worst_object_pct',100*min(rates),global_step)
                self.writer.add_scalar(f'{section}/objects_observed',len(rates),global_step)
        self.minimum[ids]=float('inf');self.bottom[ids]=float('inf');self.frames[ids]=0

    def reference_snapshot(self):
        """Compare only when every physical object has fresh reference-z episodes."""
        keys={g:f'object_{g}/reference_z' for g in self.group_ids}
        if not all(self.counts[k]>self.snapshot_counts[g] for g,k in keys.items()):return None
        rates={str(g):sum(self.recent[k])/len(self.recent[k]) for g,k in keys.items()}
        self.snapshot_counts={g:self.counts[k] for g,k in keys.items()}
        return dict(metric='reference_z_rolling_macro',macro_rate=sum(rates.values())/len(rates),
                    per_object_rate=rates,rolling_counts={str(g):len(self.recent[k]) for g,k in keys.items()},
                    episode_counts={str(g):self.counts[k] for g,k in keys.items()},
                    object_names={str(g):self.group_names.get(g,f'object_{g}') for g in self.group_ids})
