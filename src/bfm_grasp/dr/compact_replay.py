"""Lossless storage deduplication for binary actor/privileged observations; all continuous values stay float32."""
import torch
def pack(tree,validate=False):
    if 'observation' not in tree or 'history_privileged' not in tree['observation']:return tree
    out=dict(tree);out['next']=dict(tree['next'])
    for parent,key in [(out,'observation'),(out['next'],'observation')]:
        obs=dict(parent[key]);assert obs['tactile'].shape[-1]==10
        h=obs.pop('history_privileged').reshape(-1,4,76)
        if validate:
            assert torch.equal(obs['privileged_tactile'],obs['contact_binary'])
            assert torch.equal(h[:,:,54:],obs['history_actor'].reshape(-1,4,76)[:,:,54:])
            for v in [obs['tactile'],obs['contact_binary'],h[:,:,44:54]]:assert ((v==0)|(v==1)).all()
        obs.pop('privileged_tactile')
        obs['history_privileged_proprio']=h[:,:,:44].clone()
        obs['history_privileged_contact']=h[:,:,44:54].bool()
        obs['contact_binary']=obs['contact_binary'].bool();obs['tactile']=obs['tactile'].bool()
        parent[key]=obs
    return out
def unpack(tree):
    if 'observation' not in tree or 'history_privileged_proprio' not in tree['observation']:return tree
    out=dict(tree);out['next']=dict(tree['next'])
    for parent,key in [(out,'observation'),(out['next'],'observation')]:
        obs=dict(parent[key]);dtype=obs['proprio'].dtype
        obs['contact_binary']=obs['contact_binary'].to(dtype);obs['tactile']=obs['tactile'].to(dtype)
        obs['privileged_tactile']=obs['contact_binary'].clone()
        obs['history_privileged']=torch.cat([obs.pop('history_privileged_proprio'),obs.pop('history_privileged_contact').to(dtype),obs['history_actor'].reshape(-1,4,76)[:,:,54:]],-1).flatten(1)
        parent[key]=obs
    return out
