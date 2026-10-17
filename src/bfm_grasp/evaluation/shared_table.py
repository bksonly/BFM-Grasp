"""Ten independent policy instances rendered together over one physical table."""
import json
from pathlib import Path
import h5py,numpy as np,torch
OFFSETS=None
NAMES=['box08','pencil03','cylinder','flat_can','bottle','mug','tool','banana-A','B09','P02']
def configure(cfg,paths):
    global OFFSETS
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim
    assert len(paths)==10 and cfg.scene.num_envs==10
    offsets=[]
    for i,path in enumerate(paths):
        with h5py.File(path) as f:
            center=f['object_tracks/0/translations'][0];top=float(f.attrs['table_top_z'])
        target=np.array([(i%5-2)*.6,(.5-i//5)*.65,0.])
        offsets.append([target[0]-center[0],target[1]-center[1],-top])
    OFFSETS=np.array(offsets,dtype=np.float32)
    cfg.scene.reference_table=AssetBaseCfg(prim_path='/World/SharedTable',collision_group=-1,
        spawn=sim.CuboidCfg(size=(3.4,1.6,.04),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.35,.35,.35))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.02)))
    cfg.viewer.origin_type='world';cfg.viewer.lookat=(0.,0.,.08);cfg.viewer.eye=(0.,-3.5,1.5);cfg.viewer.resolution=(1600,900)
def place(env,out):
    assert OFFSETS is not None
    env.scene.env_origins[:]=torch.as_tensor(OFFSETS,device=env.device)
    Path(out).mkdir(parents=True,exist_ok=True)
    (Path(out)/'table-layout.json').write_text(json.dumps(dict(layout='two rows of five; one shared physical tabletop',names=NAMES,offsets=OFFSETS.tolist(),table_size_m=[3.4,1.6,.04],cross_object_collision='environment filtering retained',evaluation='one rollout per object in this overview; independent batch evaluation supplies success rates'),indent=2))
