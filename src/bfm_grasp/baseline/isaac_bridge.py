"""Separate BFM action/observation bridge; baseline ConTrack files are unchanged."""
import torch
from isaaclab.managers import ActionTerm,ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa.mdp.state import prepare_reference_state
from core import absolute_targets,box_corners,local_corners,History,Tactile

@configclass
class HandActionCfg(ActionTermCfg):
    class_type:type=None
    reference_path:str=''

class HandAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        env.cfg.reference_path=cfg.reference_path;prepare_reference_state(env,None)
        assert len(env.ref_hands)==1 and env.num_objects==1
        name,ids,_=env.ref_hands[0];self.robot=env.scene[name];self.ids=ids
        limits=self.robot.data.soft_joint_pos_limits[0,ids[7:]]
        self.lower=limits[:,0];self.upper=limits[:,1]
        self._raw=torch.zeros(env.num_envs,22,device=env.device)
        self.target=torch.zeros(env.num_envs,29,device=env.device)
    @property
    def action_dim(self):return 22
    @property
    def raw_actions(self):return self._raw
    @property
    def processed_actions(self):return self.target[:,7:]
    def process_actions(self,actions):
        self._raw=actions.clamp(-1,1)
        idx=self._env.episode_length_buf.clamp(max=self._env.ref_len-1)
        self.target[:,:7]=self._env.ref_qpos[idx,0,:7]
        if hasattr(self._env,'group_id'):
            for group,q in enumerate(self._env.group_qpos):
                mask=self._env.group_id==group
                self.target[mask,:7]=q[self._env.episode_length_buf[mask].clamp(max=len(q)-1),:7]
        self.target[:,7:]=absolute_targets(self._raw,self.lower,self.upper)
    def apply_actions(self):
        self.robot.set_joint_position_target(self.target,joint_ids=self.ids)
        e=self._env
        if e.unused_joint_ids:self.robot.set_joint_position_target(e.unused_default_qpos[None].expand(e.num_envs,-1),joint_ids=e.unused_joint_ids)
    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self._raw[ids]=0;self.target[ids]=0

HandActionCfg.class_type=HandAction

class Bridge:
    def __init__(self,env,vertices,mode='none'):
        self.env=env;self.mode=mode
        self.term=env.action_manager.get_term('hand')
        self.robot=self.term.robot;self.ids=self.term.ids[7:]
        self.merged_wrist='right_hand_C_MC' not in self.robot.body_names
        self.wrist=self.robot.body_names.index('link_right_arm_6' if self.merged_wrist else 'right_hand_C_MC')
        self.corners=box_corners(vertices)
        self.vertices=vertices
        import h5py
        with h5py.File(env.cfg.reference_path) as f:self.table_z=float(f.attrs['table_top_z'])
        self.history=History(env.num_envs,mode,env.device);self.tactile=Tactile(mode)
        from ten_body_contacts import TenBodyContacts
        self.force_provider=TenBodyContacts(env,self.robot)
        self.last=torch.zeros(env.num_envs,22,device=env.device)
        self.captured=None
    def observe(self):
        e=self.env;r=self.robot.data;o=e.scene['objects'].data
        wp=r.body_pos_w[:,self.wrist];wr=matrix_from_quat(r.body_quat_w[:,self.wrist])
        angular=r.body_ang_vel_w[:,self.wrist]
        wrist_velocity=r.body_lin_vel_w[:,self.wrist]
        if hasattr(e,'hand_wrist_twist'):
            # PhysX reports zero twist for the prescribed fixed root. Use the
            # finite difference of its external trajectory for relative states.
            wrist_velocity=e.hand_wrist_twist[:,:3]
            angular=e.hand_wrist_twist[:,3:]
        if self.merged_wrist:
            # URDF right_sharpa_mount is merged into the last arm rigid body.
            offset=wr@torch.tensor([0.,0.,-.1261],device=e.device)
            wp=wp+offset;wrist_velocity=wrist_velocity+torch.cross(angular,offset,dim=-1)
            wr=wr@torch.diag(torch.tensor([-1.,1.,-1.],device=e.device))
        op=o.object_pos_w[:,0];orr=matrix_from_quat(o.object_quat_w[:,0])
        delta=op-wp
        lin=o.object_lin_vel_w[:,0]-wrist_velocity-torch.cross(angular,delta,dim=-1)
        vel=torch.cat([torch.einsum('nji,nj->ni',wr,lin),torch.einsum('nji,nj->ni',wr,o.object_ang_vel_w[:,0]-angular)],-1)
        corners=local_corners(self.corners,op,orr,wp,wr)
        if hasattr(e,'group_id'):
            for group,vertices in enumerate(e.group_vertices):
                mask=e.group_id==group
                corners[mask]=local_corners(box_corners(vertices),op[mask],orr[mask],wp[mask],wr[mask])
        return dict(proprio=torch.cat([r.joint_pos[:,self.ids],r.joint_vel[:,self.ids]],-1).clone(),
            corners=corners,object_velocity=vel,contact_binary=self.force_provider.binary().clone(),
            tactile=(self.force_provider.binary() if self.mode=='binary' else self.tactile.encode(self.force_provider.read())) if self.mode!='none' else torch.zeros(e.num_envs,0,device=e.device),last_action=self.last.clone(),history_actor=self.history.value.flatten(1).clone())
    def before_step(self,obs,action):
        o=self.env.scene['objects'].data
        self.previous_velocity=dict(dq=self.robot.data.joint_vel[:,self.ids].clone(),
                                    object_velocity=torch.cat([o.object_lin_vel_w[:,0],o.object_ang_vel_w[:,0]],-1).clone())
        self.history.push(obs);self.last=action.detach().clone()
    def capture(self):
        # Called by the termination manager after physics, BEFORE auto-reset.
        self.captured=self.observe()
        o=self.env.scene['objects'].data
        position=self.force_provider.points_world()
        contact=torch.isfinite(position).all(-1)
        relative=torch.nan_to_num(position-o.object_pos_w[:,0,None],nan=0.)
        object_rotation=matrix_from_quat(o.object_quat_w[:,0])
        local=torch.einsum('nji,nkj->nki',object_rotation,relative)
        self.reward_state=dict(q=self.robot.data.joint_pos[:,self.ids].clone(),dq=self.robot.data.joint_vel[:,self.ids].clone(),
                              object_position=(o.object_pos_w[:,0]-self.env.scene.env_origins).clone(),
                              object_quaternion=o.object_quat_w[:,0].clone(),
                              object_velocity=torch.cat([o.object_lin_vel_w[:,0],o.object_ang_vel_w[:,0]],-1).clone(),
                              contact=contact.float(),contact_point=local)
        from contrack_aux import stability_terms
        self.rp=sum(stability_terms(self.reward_state,self.previous_velocity).values())
        self.height=o.object_pos_w[:,0,2].clone()
        rotation=matrix_from_quat(o.object_quat_w[:,0])
        self.bottom=(rotation[:,2]@self.vertices.T).min(-1).values+self.height-self.table_z
        if hasattr(self.env,'group_id'):
            for group,vertices in enumerate(self.env.group_vertices):
                mask=self.env.group_id==group
                self.bottom[mask]=(rotation[mask,2]@vertices.T).min(-1).values+self.height[mask]-self.env.group_table[group]-self.env.scene.env_origins[mask,2]
            return self.env.episode_length_buf>=self.env.group_lengths-1
        return self.env.episode_length_buf>=self.env.ref_len-1
    def reset_history(self,ids):
        self.history.reset(ids);self.tactile.reset(ids);self.last[ids]=0

def capture_terminal(env):
    if hasattr(env,'bfm_bridge'):return env.bfm_bridge.capture()
    return env.episode_length_buf>=env.ref_len-1

def unused_observation(env):
    return env.scene['rby1a_sharpa'].data.joint_pos[:,:1]
