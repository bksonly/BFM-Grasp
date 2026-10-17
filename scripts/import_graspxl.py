"""Convert a trusted GraspXL Sharpa .npy + object mesh to our hand-only reference protocol."""
import argparse,hashlib,json,subprocess,sys,tempfile,os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from pathlib import Path
import h5py,numpy as np,trimesh
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/bfm_grasp/common"))
from table_frame import write_table_frame
p=argparse.ArgumentParser()
p.add_argument("--source",type=Path,required=True,help="Trusted GraspXL dictionary .npy (loads pickle)")
p.add_argument("--mesh",type=Path,required=True)
p.add_argument("--object-id",required=True)
p.add_argument("--source-fps",type=float,required=True,help="Explicit timing assumption; legacy experiments used 30")
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():p.error("Output already exists")
if a.source_fps<=0:p.error("FPS must be positive")
raw=np.load(a.source,allow_pickle=True).item();hand=raw["right_hand"];obj=raw[a.object_id]
angles=np.asarray(hand["pose"])[:,6:];assert angles.shape[1]==22 and len(angles)>=2
names=json.loads((ROOT/"assets/sharpa/joint_order.json").read_text());assert len(names)==22
# This input schema is already retargeted to Sharpa; MANO angles are not accepted.
import xml.etree.ElementTree as ET
joints={j.get("name"):j for j in ET.parse(ROOT/"third_party/ConTrack/assets/urdf/sharpa_right.urdf").getroot().findall("joint")}
for i,name in enumerate(names):
    limit=joints[name].find("limit")
    assert np.all(angles[:,i]>=float(limit.get("lower"))-1e-4) and np.all(angles[:,i]<=float(limit.get("upper"))+1e-4),name
mesh=trimesh.load(a.mesh,force="mesh",process=False)
old=np.arange(len(angles))/a.source_fps;new=np.linspace(0,old[-1],round(old[-1]*60)+1)
def linear(x):
    x=np.asarray(x);return np.stack([np.interp(new,old,x[:,i]) for i in range(x.shape[1])],axis=-1)
def quaternion(x):return Slerp(old,Rotation.from_rotvec(x))(new).as_quat()
q=linear(angles);wp=linear(hand["trans"]);wq=quaternion(hand["rot"]);op=linear(obj["trans"]);oq=quaternion(obj["rot"])
# Keep the official original z=0 tabletop; only center the scene in XY.
offset=np.r_[op[:,0:2].mean(0),0.];wp-=offset;op-=offset
# Same 0.5s close hold, 1s smooth 5cm lift, 1s hold, final padding convention.
h=np.r_[np.zeros(30),.05*(3*np.linspace(0,1,60)**2-2*np.linspace(0,1,60)**3),np.full(60,.05),.05]
def extend(x,lift=False):
    tail=np.repeat(x[-1:],len(h),axis=0)
    if lift:tail[:,2]+=h
    return np.concatenate([x,tail])
q=extend(q);wp=extend(wp,True);op=extend(op,True);wq=extend(wq);oq=extend(oq);n=len(q)
a.output.parent.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix="bfm-reference-") as temp:
    source=Path(temp)/"motion.h5"
    with h5py.File(source,"w") as f:
        f["qpos"]=np.concatenate([np.zeros((n,7)),q],axis=-1)[:,None,:].astype("f4")
        f["fps"]=np.array([60],dtype="f4");f["is_rhand"]=np.array([1],dtype="u1");f["urdf_name"]=np.array([b"rby1a_sharpa"]);f["base_translation"]=np.zeros((1,3),dtype="f4")
        f["wrist_position"]=wp.astype("f4");f["wrist_quaternion_xyzw"]=wq.astype("f4")
        for k,v in {"vertices":mesh.vertices,"faces":mesh.faces,"translations":op,"orientations_xyzw":oq}.items():f["object_tracks/0/"+k]=v
        f.create_group("contacts")
        f.attrs.update(reference_kind="graspxl_geometric_task_extension_v1",hand_only=True,source_fps_assumption=a.source_fps,terminal_padding_frames=1,source_file_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),source_object_id=a.object_id,table_center_xy=op[0,:2],lift_test="Prescribed wrist and object TARGET lift 5cm; physical object stays dynamic")
        write_table_frame(f,-offset)
    subprocess.run([sys.executable,str(ROOT/"scripts/generate_contacts.py"),"--source",str(source),"--output",str(Path(temp)/"complete.h5")],check=True)
    import shutil
    shutil.copy2(Path(temp)/"complete.h5",a.output)
    shutil.copy2(Path(temp)/"complete.contacts.json",a.output.with_suffix(".contacts.json"))
print("Converted reference and geometric contacts:",a.output)
print("Inspect kinematic replay and physics before using this new reference as training data.")
