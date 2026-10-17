"""Diagnostic scenes only; never used to claim grasp success."""
import os,torch,h5py
from pxr import Usd,UsdPhysics
import isaaclab.sim as sim

def spawn(prim_path,cfg,**kwargs):
    from grouped_scene import spawn_objects
    result=spawn_objects(prim_path,cfg,**kwargs)
    parents=sim.find_matching_prim_paths(prim_path.rsplit('/',1)[0])
    for parent in parents:
        prim=sim.get_current_stage().GetPrimAtPath(parent+'/object_0')
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
        if os.environ.get('CRAFTNET_ISOLATE_NO_OBJECT')=='1':
            prim.SetInstanceable(False)
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Set(False)
    return result

def configure(cfg):
    cfg.scene.objects.rigid_objects['object_0'].spawn.func=spawn
    if os.environ.get('CRAFTNET_ISOLATE_NO_TABLE')=='1':cfg.scene.reference_table=None

def set_object(env):
    if not hasattr(env,'isolation_object'):
        with h5py.File(env.cfg.reference_path) as f:
            g=f['object_tracks/0'];p=torch.tensor(g['translations'][:],device=env.device);q=torch.tensor(g['orientations_xyzw'][:],device=env.device)
        state=torch.zeros(len(p),13,device=env.device);state[:,:3]=p;state[:,3:7]=q[:,[3,0,1,2]];env.isolation_object=state
    idx=env.episode_length_buf.clamp(max=len(env.isolation_object)-1)
    state=env.isolation_object[idx,None].clone();state[:,:,:3]+=env.scene.env_origins[:,None]
    env.scene['objects'].write_object_state_to_sim(state)
