"""Unfiltered contact forces on dedicated physical elastomer bodies."""
import torch
import os
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat
FINGERS=('thumb','index','middle','ring','pinky')

def body_name(finger):
    return 'right_'+finger+('_DP' if os.environ.get('CRAFTNET_MERGED_TACTILE')=='1' else '_elastomer')

def configure_tactile(cfg):
    from pathlib import Path
    path=Path(__file__).resolve().parents[3]/'assets/sharpa/rby1a_sharpa_tactile.usd'
    if not path.exists():raise FileNotFoundError('Build the dedicated elastomer USD before tactile training')
    cfg.scene.rby1a_sharpa.spawn.usd_path=str(path)
    for finger in FINGERS:
        setattr(cfg.scene,'tactile_'+finger,ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Rby1aSharpa/'+body_name(finger),
            update_period=0.0,filter_prim_paths_expr=[],history_length=0))

class ElastomerForces:
    def __init__(self,env,robot):
        self.robot=robot;self.sensors=[];self.ids=[]
        self.peak=torch.zeros(5,device=env.device)
        for finger in FINGERS:
            name=body_name(finger)
            if name not in robot.body_names:raise RuntimeError(f'Tactile rigid body missing: {name}')
            sensor=env.scene['tactile_'+finger]
            assert sensor.body_names==[name],sensor.body_names
            assert not sensor.cfg.filter_prim_paths_expr
            self.sensors.append(sensor);self.ids.append(robot.body_names.index(name))
    def read(self):
        world=torch.stack([s.data.net_forces_w[:,0] for s in self.sensors],dim=1)
        if not torch.isfinite(world).all():raise RuntimeError('Non-finite elastomer force: invalidate this physics rollout')
        self.peak=torch.maximum(self.peak,world.norm(dim=-1).max(dim=0).values)
        rotation=matrix_from_quat(self.robot.data.body_quat_w[:,self.ids])
        return torch.einsum('nfji,nfj->nfi',rotation,world)

def configure_probe(cfg):
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    cfg.scene.tactile_probe=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/TactileProbe',
        spawn=sim.SphereCfg(radius=.008,rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            collision_props=sim.CollisionPropertiesCfg(),mass_props=sim.MassPropertiesCfg(mass=.01)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,5)))

def place_probe(env,provider,step):
    if not hasattr(provider,'probe_centers'):
        import trimesh
        from pathlib import Path
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        asset=Path(provider.robot.cfg.spawn.usd_path).with_suffix('.urdf')
        urdf=ET.parse(asset).getroot();centers=[]
        for finger in FINGERS:
            collision=urdf.find(f"link[@name='right_{finger}_elastomer']/collision")
            mesh=collision.find('geometry/mesh');origin=collision.find('origin')
            center=trimesh.load(mesh.attrib['filename'],process=False).bounds.mean(0)
            center*=__import__('numpy').fromstring(mesh.get('scale','1 1 1'),sep=' ')
            if origin is not None:
                center=Rotation.from_euler('xyz',__import__('numpy').fromstring(origin.get('rpy','0 0 0'),sep=' ')).apply(center)
                center+=__import__('numpy').fromstring(origin.get('xyz','0 0 0'),sep=' ')
            centers.append(center)
        provider.probe_centers=torch.tensor(__import__('numpy').array(centers),dtype=torch.float32,device=env.device)
    probe=env.scene['tactile_probe'];state=probe.data.default_root_state.clone()
    state[:,:3]=env.scene.env_origins+torch.tensor([0,0,5.],device=env.device)
    if step%32>=16:
        finger=(step//32)%5;body=provider.ids[finger]
        rot=matrix_from_quat(provider.robot.data.body_quat_w[:,body])
        state[:,:3]=provider.robot.data.body_pos_w[:,body]+rot@provider.probe_centers[finger]
    probe.write_root_state_to_sim(state)
