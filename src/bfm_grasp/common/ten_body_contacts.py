"""Original ConTrack ten PP/DP bodies, filtered against the manipulated object."""
import torch
from isaaclab.utils.math import matrix_from_quat
from contact_reference import LINKS
KEYS=tuple('contact_right_'+f+'_'+part for f in ('index','middle','ring','pinky','thumb') for part in ('link1','link2'))

def configure_tactile(cfg):
    for key in KEYS:
        sensor=getattr(cfg.scene,key)
        assert sensor.track_contact_points and len(sensor.filter_prim_paths_expr)==1
        sensor.track_friction_forces=True

class TenBodyContacts:
    def __init__(self,env,robot):
        self.robot=robot;self.sensors=[];self.ids=[]
        self.peak=torch.zeros(10,device=env.device)
        self.active_counts=torch.zeros(10,device=env.device,dtype=torch.long)
        assert not any('elastomer' in name for name in robot.body_names), 'Independent elastomer bodies forbidden'
        for link,key in zip(LINKS,KEYS):
            name='right_'+link
            sensor=env.scene[key]
            assert sensor.body_names==[name],(key,sensor.body_names)
            assert len(sensor.cfg.filter_prim_paths_expr)==1
            self.sensors.append(sensor);self.ids.append(robot.body_names.index(name))
    def binary(self):
        self.read()
        flags=torch.stack([torch.isfinite(s.data.contact_pos_w[:,0,0]).all(-1) for s in self.sensors],dim=1)
        self.active_counts+=flags.sum(0)
        return flags.float()
    def points_world(self):
        return torch.stack([s.data.contact_pos_w[:,0,0] for s in self.sensors],dim=1)
    def read(self):
        world=torch.stack([s.data.force_matrix_w[:,0,0]+s.data.friction_forces_w[:,0,0] for s in self.sensors],dim=1)
        assert torch.isfinite(world).all(), 'Non-finite object-filtered contact force'
        self.peak=torch.maximum(self.peak,world.norm(dim=-1).max(dim=0).values)
        rotation=matrix_from_quat(self.robot.data.body_quat_w[:,self.ids])
        return torch.einsum('nfji,nfj->nfi',rotation,world)
