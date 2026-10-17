"""Deterministic per-environment assets and reference resets for one shared scene."""
import importlib.util,os,sys
from pathlib import Path
import h5py,numpy as np,torch
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.state import reset_reference_episode_first_frame

MODULES=[]
PATHS=[]
SCALE_CLASSES=0

def asset_index(index):
    return (index%SCALE_CLASSES)*27+(index//SCALE_CLASSES)%27 if SCALE_CLASSES else index%len(MODULES)

def spawn_objects(prim_path,cfg,translation=None,orientation=None,**kwargs):
    parents=sim.find_matching_prim_paths(prim_path.rsplit('/',1)[0])
    for parent in parents:
        index=int(parent.rsplit('_',1)[1]);m=MODULES[asset_index(index)]
        asset=m._rigid_objects['object_0']
        m.spawn_ref_usd_mesh(parent+'/object_0',asset.spawn,translation=asset.init_state.pos,orientation=asset.init_state.rot)
    return sim.get_current_stage().GetPrimAtPath(parents[0]+'/object_0')

def spawn_tables(prim_path,cfg,translation=None,orientation=None,**kwargs):
    parents=sim.find_matching_prim_paths(prim_path.rsplit('/',1)[0])
    for parent in parents:
        m=MODULES[asset_index(int(parent.rsplit('_',1)[1]))]
        position=(float(m._table_xy[0]),float(m._table_xy[1]),float(m._table_top)-.02)
        sim.spawn_cuboid(parent+'/ReferenceTable',cfg,translation=position)
    return sim.get_current_stage().GetPrimAtPath(parents[0]+'/ReferenceTable')

def configure(cfg,paths):
    global PATHS,SCALE_CLASSES
    PATHS=paths
    SCALE_CLASSES=len(paths)//27 if getattr(cfg,'scale_dr',False) else 0
    file=Path(str(__import__('pathlib').Path(__file__).resolve().parents[3]/'third_party'/'ConTrack/source/ConTrack/ConTrack/tasks/manager_based/mimic_rby1a_sharpa/rby1a_sharpa_env_cfg.py'))
    original=os.environ['CONTRACK_RBY1A_SHARPA_DATA']
    for i,path in enumerate(paths):
        os.environ['CONTRACK_RBY1A_SHARPA_DATA']=path
        spec=importlib.util.spec_from_file_location(f'bfm_asset_{i}',file)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);MODULES.append(module)
    os.environ['CONTRACK_RBY1A_SHARPA_DATA']=original
    assert all(np.allclose(m._base_t0,MODULES[0]._base_t0) for m in MODULES)
    cfg.scene.replicate_physics=False
    cfg.scene.objects.rigid_objects['object_0'].spawn.func=spawn_objects
    cfg.scene.reference_table.spawn.func=spawn_tables
    cfg.events.reset_to_demo.func=reset_grouped

def initialize(env):
    if hasattr(env,'group_qpos'):return
    env.group_id=torch.arange(env.num_envs,device=env.device)%len(PATHS)
    env.group_qpos=[];env.group_initial=[];env.group_vertices=[];env.group_table=[]
    for path in PATHS:
        with h5py.File(path) as f:
            q=f['qpos'][:,0];fps=float(np.asarray(f['fps']).max())
            g=f['object_tracks/0'];p=g['translations'][:];quat=g['orientations_xyzw'][:]
            state=np.zeros(13,np.float32);state[:3]=p[0];state[3:7]=quat[0][[3,0,1,2]]
            state[7:10]=(p[1]-p[0])*fps
            state[10:]=(Rotation.from_quat(quat[1])*Rotation.from_quat(quat[0]).inv()).as_rotvec()*fps
            tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=env.device)
            env.group_qpos.append(tensor(q));env.group_initial.append((tensor(q[0]),tensor((q[1]-q[0])*fps),tensor(state)))
            env.group_vertices.append(tensor(g['vertices'][:]));env.group_table.append(float(f.attrs['table_top_z']))
    env.group_lengths=torch.tensor([len(q) for q in env.group_qpos],device=env.device)[env.group_id]

def reset_grouped(env,env_ids):
    reset_reference_episode_first_frame(env,env_ids)
    initialize(env)
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    for group,(q,dq,state) in enumerate(env.group_initial):
        ids=env_ids[env.group_id[env_ids]==group]
        if not len(ids):continue
        name,joints,_=env.ref_hands[0]
        env.scene[name].write_joint_state_to_sim(q.expand(len(ids),-1),dq.expand(len(ids),-1),joint_ids=joints,env_ids=ids)
        values=state.expand(len(ids),1,-1).clone();values[:,:,0:3]+=env.scene.env_origins[ids,None]
        env.scene['objects'].write_object_state_to_sim(values,env_ids=ids)
