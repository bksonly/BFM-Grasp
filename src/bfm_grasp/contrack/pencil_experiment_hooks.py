"""Read-only collision checks and grasp metrics for the ConTrack A/B run."""
import os,json
from pathlib import Path
import torch
from pxr import Usd,UsdPhysics,PhysxSchema
from isaaclab.sim import get_current_stage
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.terminations import obj_pose_exceeded

def audit(env,env_ids):
    expected=os.environ.get('PENCIL_SELF_COLLISION','0')=='1'
    groups={'hand':[],'object':[],'table':[]};arts=[];filters=[]
    for prim in Usd.PrimRange(get_current_stage().GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        p=str(prim.GetPath())
        if not p.startswith('/World/envs/env_0/'):continue
        group='hand' if '/Rby1aSharpa' in p else 'object' if '/object_0' in p else 'table' if '/ReferenceTable' in p else None
        if group is None:continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):groups[group].append(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        if group=='hand' and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):arts.append(PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get())
        if group=='hand' and prim.HasAPI(UsdPhysics.FilteredPairsAPI):filters += [str(t) for t in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()]
    assert arts and all(v==expected for v in arts),(arts,expected)
    assert all(v and all(x is not False for x in v) for v in groups.values()),groups
    assert not any('/object_0' in t or '/ReferenceTable' in t for t in filters),filters
    assert env.scene['rby1a_sharpa'].num_joints==22
    result={'self_collision':expected,'external_collisions_enabled':True,'collider_counts':{k:len(v) for k,v in groups.items()},'num_joints':22}
    path=Path(env.cfg.log_dir);path.mkdir(parents=True,exist_ok=True)
    (path/'collision-audit.json').write_text(json.dumps(result,indent=2))
    print('COLLISION_AUDIT',result,flush=True)

def measured_pose_break(env):
    failed=obj_pose_exceeded(env)
    if not hasattr(env,'grasp_counts'):
        env.grasp_counts=torch.zeros(4,device=env.device)
        env.grasp_hold_min=torch.full((env.num_envs,),float('inf'),device=env.device)
    # Random-start episodes are reported separately from full first-frame attempts.
    first=env.frame_idx==env.episode_start_frame
    env.grasp_hold_min[first]=float('inf')
    height=env.scene['objects'].data.object_pos_w[:,0,2]-env.scene.env_origins[:,2]-env.ref_object_pos[0,0,2]
    hold=env.frame_idx>=env.ref_len-60
    env.grasp_hold_min[hold]=torch.minimum(env.grasp_hold_min[hold],height[hold])
    done=failed|(env.frame_idx>=env.ref_len-1)
    success=done&(~failed)&(env.frame_idx>=env.ref_len-1)&(env.grasp_hold_min>=.04)
    full=env.episode_start_frame==0
    env.grasp_counts += torch.stack([done.sum(),success.sum(),(done&full).sum(),(success&full).sum()])
    return failed
