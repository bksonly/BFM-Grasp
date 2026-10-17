"""No arm or IK: externally prescribed root pose and 22 physical hand joints."""
import h5py,numpy as np,torch
from scipy.spatial.transform import Rotation
from isaaclab.managers import ActionTerm,EventTermCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from ConTrack.robots import rby1a_sharpa as calibration
from core import absolute_targets
from motion_contract import wrist_root_states
PATHS=[]
WRIST=['x_joint','y_joint','z_joint','x_rotation_joint','y_rotation_joint','z_rotation_joint']

def configure(cfg,paths):
    global PATHS
    PATHS=paths
    from grouped_scene import configure as configure_groups
    configure_groups(cfg,paths)
    # Source coordinates place the tabletop at z=0; do not overlap it with a ground plane.
    cfg.scene.ground=None
    robot=cfg.scene.rby1a_sharpa
    # Disable only contacts within this articulation; external colliders remain enabled.
    robot.spawn.articulation_props.enabled_self_collisions=False
    import os
    if os.environ.get('CRAFTNET_FREE_WRIST')=='1':robot.spawn.articulation_props.fix_root_link=False
    if os.environ.get('CRAFTNET_DIAG_OBJECT_HULL')=='1':
        from grouped_scene import MODULES
        from pathlib import Path
        from pxr import Usd,UsdPhysics
        for module in MODULES:
            spawn=module._rigid_objects['object_0'].spawn
            source=Path(spawn.usd_path);target=source.with_name(source.stem+'-audit-convex-hull.usd')
            stage=Usd.Stage.Open(str(source));stage.GetRootLayer().Export(str(target))
            stage=Usd.Stage.Open(str(target))
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set('convexHull')
            stage.GetRootLayer().Save();spawn.usd_path=str(target)
    robot.spawn.usd_path=str(__import__('pathlib').Path(__file__).resolve().parents[3]/'assets/sharpa/rby1a_sharpa_tactile.usd')
    robot.init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0),joint_pos={'.*':0.})
    finger=robot.actuators['sharpa'].copy()
    finger.joint_names_expr=calibration.SHARPA_RIGHT_JOINT_NAMES
    for key in ('effort_limit_sim','stiffness','damping','armature','friction','velocity_limit_sim'):
        values=getattr(finger,key)
        if isinstance(values,dict):setattr(finger,key,{k:v for k,v in values.items() if k.startswith('right_')})
    robot.actuators={'sharpa':finger}
    scale=float(os.environ.get('CRAFTNET_DIAG_GAIN_SCALE','1'))
    if scale!=1:
        finger.stiffness={k:v*scale for k,v in finger.stiffness.items()}
        finger.damping={k:v*scale**.5 for k,v in finger.damping.items()}
    if os.environ.get('CRAFTNET_DIAG_NO_FRICTION')=='1':finger.friction=0.
    if os.environ.get('CRAFTNET_CONTACT_ISOLATION')=='1':
        from contact_isolation import configure as configure_isolation
        configure_isolation(cfg)
    for name in list(vars(cfg.events)):
        if not name.startswith('_'):setattr(cfg.events,name,None)
    cfg.events.reset_to_demo=EventTermCfg(func=reset,mode='reset')

class HandOnlyAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);self.robot=env.scene['rby1a_sharpa']
        self.fingers=self.robot.find_joints(calibration.SHARPA_RIGHT_JOINT_NAMES,preserve_order=True)[0]
        assert len(self.robot.joint_names)==22 and not any('arm_' in n for n in self.robot.joint_names)
        self.ids=[0]*7+self.fingers
        limits=self.robot.data.soft_joint_pos_limits[0,self.fingers];self.lower=limits[:,0];self.upper=limits[:,1]
        self._raw=torch.zeros(env.num_envs,22,device=env.device);self.target=torch.zeros_like(self._raw)
        self.root_target=torch.zeros(env.num_envs,13,device=env.device)
        initialize(env)
    @property
    def action_dim(self):return 22
    @property
    def raw_actions(self):return self._raw
    @property
    def processed_actions(self):return self.target
    def process_actions(self,actions):
        self._raw=actions.clamp(-1,1)
        for group,q in enumerate(self._env.hand_motion):
            ids=self._env.group_id==group
            frames=self._env.episode_length_buf[ids].clamp(max=len(q)-1)
            self.root_target[ids]=self._env.hand_roots[group][frames]
            self.root_target[ids,:3]+=self._env.scene.env_origins[ids]
        self._env.hand_wrist_twist.copy_(self.root_target[:,7:])
        self.target[:]=absolute_targets(self._raw,self.lower,self.upper)
    def apply_actions(self):
        import os
        if os.environ.get('CRAFTNET_CONTACT_ISOLATION')=='1':
            from contact_isolation import set_object
            set_object(self._env)
        self.robot.write_root_state_to_sim(self.root_target)
        self.robot.set_joint_position_target(self.target,joint_ids=self.fingers)
    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self._raw[ids]=0;self.target[ids]=0

def initialize(env):
    if hasattr(env,'hand_motion'):return
    index=torch.arange(env.num_envs,device=env.device)
    classes=len(PATHS)//27
    env.group_id=(index%classes)*27+(index//classes)%27 if getattr(env.cfg,'scale_dr',False) else index%len(PATHS)
    env.class_id=env.group_id//27 if getattr(env.cfg,'scale_dr',False) else env.group_id
    env.variant_id=env.group_id.clone()
    env.hand_wrist_twist=torch.zeros((env.num_envs,6),device=env.device)
    env.group_vertices=[];env.group_table=[];env.hand_motion=[];env.hand_objects=[];env.hand_roots=[]
    tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=env.device)
    for path in PATHS:
        with h5py.File(path) as f:
            q=f['qpos'][:,0];p=f['wrist_position'][:];quat=f['wrist_quaternion_xyzw'][:]
            angles=np.unwrap(Rotation.from_quat(quat).as_euler('XYZ'),axis=0)
            env.hand_motion.append(tensor(np.concatenate([p,angles,q[:,7:]],axis=1)))
            fps=float(np.asarray(f['fps']).reshape(-1)[0])
            env.hand_roots.append(tensor(wrist_root_states(p,quat,fps)))
            g=f['object_tracks/0'];state=np.zeros(13);state[:3]=g['translations'][0];state[3:7]=g['orientations_xyzw'][0][[3,0,1,2]]
            env.hand_objects.append(tensor(state));env.group_vertices.append(tensor(g['vertices'][:]));env.group_table.append(float(f.attrs['table_top_z']))
    env.group_lengths=torch.tensor([len(q) for q in env.hand_motion],device=env.device)[env.group_id]
    env.ref_len=len(env.hand_motion[0]);env.cfg.reference_path=PATHS[0]
    with h5py.File(PATHS[0]) as f:env.ref_qpos=tensor(f['qpos'][:])

def reset(env,env_ids):
    initialize(env)
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    robot=env.scene['rby1a_sharpa'];joint_ids=robot.find_joints(calibration.SHARPA_RIGHT_JOINT_NAMES,preserve_order=True)[0]
    for group,q in enumerate(env.hand_motion):
        ids=env_ids[env.group_id[env_ids]==group]
        if not len(ids):continue
        root=env.hand_roots[group][0].expand(len(ids),-1).clone();root[:,:3]+=env.scene.env_origins[ids]
        env.hand_wrist_twist[ids]=root[:,7:]
        robot.write_root_state_to_sim(root,env_ids=ids)
        robot.write_joint_state_to_sim(q[0,6:].expand(len(ids),-1),torch.zeros(len(ids),22,device=env.device),joint_ids=joint_ids,env_ids=ids)
        state=env.hand_objects[group].expand(len(ids),1,-1).clone();state[:,:,:3]+=env.scene.env_origins[ids,None]
        if getattr(env.cfg,'scale_dr',False):
            import dr_runtime
            state=dr_runtime.pose_state(state,env.group_vertices[group],env.group_table[group],env.scene.env_origins[ids],perturb=dr_runtime.ENABLED)
        env.scene['objects'].write_object_state_to_sim(state,env_ids=ids)
    if getattr(env.cfg,'scale_dr',False):
        import dr_runtime
        dr_runtime.randomize_dynamics(env,env_ids)

