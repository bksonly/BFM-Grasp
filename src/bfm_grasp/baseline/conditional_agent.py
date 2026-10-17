"""Narrow extension of upstream FBcprAuxAgent for user-selected conditional rewards.

All upstream losses, optimizers and target updates are retained. Only latent
sampling bookkeeping and the auxiliary reward passed to the existing critic
are changed. Expert goals are paired with the same physical object class.
"""
import torch
from torch.utils._pytree import tree_map
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
from contrack_aux import auxiliary_reward

class FixedSample:
    def __init__(self,batch):self.batch=batch
    def sample(self,n):
        assert len(self.batch['object_id'])==n
        return self.batch

class ConditionalConTrackAgent(FBcprAuxAgent):
    def setup_compile(self):
        super().setup_compile()
        # Object matching/provenance uses variable index lists; compile neural
        # updates, not this Python bookkeeping.
        self.sample_mixed_z=type(self).sample_mixed_z.__get__(self)
        self.update_aux_critic=type(self).update_aux_critic.__get__(self)
        self._aux_update=super().update_aux_critic
        if self.cfg.compile:self._aux_update=torch.compile(self._aux_update,mode='reduce-overhead')

    def update(self,replay_buffer,step):
        train=tree_map(lambda x:x.to(self.device),replay_buffer['train'].sample(self.cfg.train.batch_size))
        self.reference_sampler=replay_buffer['expert_slicer']
        expert=self.reference_sampler.sample(self.cfg.train.batch_size,object_ids=train['object_id'])
        assert 'targets' in expert, 'Measured expert contact targets are required'
        self.reward_batch=train;self.expert_batch=expert
        # Upstream constructs an empty auxiliary reward before our override.
        # Bypass that placeholder, then normalize the real reward exactly once.
        normalizer=self._model._aux_reward_normalizer
        self._real_reward_normalizer=normalizer
        self._model._aux_reward_normalizer=torch.nn.Identity()
        try:
            return super().update({'train':FixedSample(train),'expert_slicer':FixedSample(expert)},step)
        finally:
            self._model._aux_reward_normalizer=normalizer

    @torch.no_grad()
    def sample_mixed_z(self,train_goal,expert_encodings,*args,**kwargs):
        batch=self.cfg.train.batch_size
        z=self._model.sample_z(batch,device=self.device)
        p=self.cfg.train.train_goal_ratio;e=self.cfg.train.expert_asm_ratio
        self.mixed_source=torch.multinomial(torch.tensor([p,e,1-p-e],device=self.device),batch,replacement=True)
        permutation=torch.randperm(batch,device=self.device)
        goals=self._model.project_z(self._model._backward_map(tree_map(lambda x:x[permutation],train_goal)))
        z=torch.where((self.mixed_source==0)[:,None],goals,z)
        # Permute expert windows and their reward targets together within each
        # physical object group. A pencil target is not assigned to a box scene.
        groups=self.reward_batch['object_id'].flatten()
        expert_groups=self.expert_batch['object_id'].flatten()
        self.expert_indices=torch.empty(batch,dtype=torch.long,device=self.device)
        for group in torch.unique(groups):
            ids=(groups==group).nonzero().flatten()
            source=(expert_groups==group).nonzero().flatten()
            assert len(ids)==len(source)
            self.expert_indices[ids]=source[torch.randperm(len(source),device=self.device)]
        z=torch.where((self.mixed_source==1)[:,None],expert_encodings[self.expert_indices],z)
        self.proposed_z=z.clone()
        return z

    def update_aux_critic(self,*args,**kwargs):
        target=tree_map(lambda x:x[self.expert_indices],self.expert_batch['targets'])
        expert=(self.mixed_source==1)[:,None]
        # The original 20% unrelabelled transitions keep their rollout intent.
        changed=(kwargs['z']==self.proposed_z).all(-1,keepdim=True)
        original_expert=self.reward_batch['reference_context'].bool()
        expert=torch.where(changed,expert,original_expert)
        target['window_mask']=torch.ones(target['q'].shape[:2],device=self.device)
        for group,ref in enumerate(self.reference_sampler.refs):
            ids=((~changed[:,0])&(self.reward_batch['object_id'][:,0]==group)).nonzero().flatten()
            if not len(ids):continue
            frame=self.reward_batch['reference_frame'][ids,0].long()
            indices=frame[:,None]+1+torch.arange(8,device=self.device)[None]
            valid=indices<len(ref.proprio)
            indices=indices.clamp(max=len(ref.proprio)-1)
            for key,value in ref.targets.items():target[key][ids]=value[indices]
            target['window_mask'][ids]=valid.float()
        if getattr(self,'reward_mode','conditional')=='rp-only':expert=torch.zeros_like(expert)
        reward,terms=auxiliary_reward(self.reward_batch['reward_state'],target,self.reward_batch['rp'],expert)
        before=int(self._real_reward_normalizer.counter.item())
        kwargs['aux_reward']=self._real_reward_normalizer(reward)
        assert int(self._real_reward_normalizer.counter.item())==before+1
        result=self._aux_update(*args,**kwargs)
        result['aux_reward/raw']=reward.mean()
        result['aux_reward/normalized']=kwargs['aux_reward'].mean()
        result['aux_reward/std']=self._real_reward_normalizer.S.sqrt().mean()
        result['z_source/relabelled_actual']=changed.float().mean()
        result.update({f'conditional_reward/{key}':value[expert].mean() if expert.any() else value.new_zeros(()) for key,value in terms.items()})
        result['conditional_reward/rp']=self.reward_batch['rp'].mean()
        result['conditional_reward/total']=reward.mean()
        for index,name in enumerate(('replay','expert','random')):
            result['z_source/'+name]=(self.mixed_source==index).float().mean()
        return result
