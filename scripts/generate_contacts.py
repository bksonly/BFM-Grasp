"""Generate explicit geometric ten-body targets, including fixed-child elastomer geometry."""
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import numpy as np
import pinocchio as pin
import trimesh
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/bfm_grasp/common"))
from contact_reference import LINKS, load_direct_contacts

p=argparse.ArgumentParser()
p.add_argument('--source',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
urdf=root/'assets/sharpa/rby1a_sharpa_tactile.urdf'
xml=ET.parse(urdf).getroot()
model=pin.buildModelFromUrdf(str(urdf));data=model.createData()
links={l.get('name'):l for l in xml.findall('link')}
fixed={}
for joint in xml.findall('joint'):
    if joint.get('type')=='fixed':
        fixed.setdefault(joint.find('parent').get('link'),[]).append(joint.find('child').get('link'))
def descendants(name):
    return [name]+[n for c in fixed.get(name,[]) for n in descendants(c)]
names=json.loads((root/'assets/sharpa/joint_order.json').read_text())
assert len(names)==22 and all(model.existJointName(n) for n in names)
segments=[];manifest={}
for link in LINKS:
    children=descendants('right_'+link)
    manifest[link]=children
    if link.endswith('DP'):
        assert 'right_'+link[:-2]+'elastomer' in children
    geometry=[]
    for child in children:
        for c in links[child].findall('collision'):
            mesh=c.find('geometry/mesh')
            assert mesh is not None, child
            vertices=trimesh.load(urdf.parent/mesh.get('filename'),force='mesh',process=False).vertices.copy()
            vertices*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            # Include face centroids as well as all vertices; no truncated vertex subset.
            loaded=trimesh.load(urdf.parent/mesh.get('filename'),force='mesh',process=False)
            centers=loaded.triangles_center*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            vertices=np.concatenate([vertices,centers])
            o=c.find('origin')
            if o is not None:
                vertices=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).apply(vertices)+np.fromstring(o.get('xyz','0 0 0'),sep=' ')
            geometry.append((model.getFrameId(child),vertices))
    segments.append(geometry)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
assert not out.exists(), 'Do not overwrite a reference already used by a run'
shutil.copy2(a.source,out)
with h5py.File(out,'r+') as f:
    object_mesh=trimesh.Trimesh(f['object_tracks/0/vertices'][:],f['object_tracks/0/faces'][:],process=False)
    qref=f['qpos'][:,0,7:];wp=f['wrist_position'][:]
    wr=Rotation.from_quat(f['wrist_quaternion_xyzw'][:]).as_matrix()
    op=f['object_tracks/0/translations'][:]
    rot=Rotation.from_quat(f['object_tracks/0/orientations_xyzw'][:]).as_matrix()
    flags=np.zeros((len(qref),1,10),dtype=np.uint8)
    points=np.zeros((len(qref),1,10,3),dtype=np.float32)
    distance=np.empty((len(qref),1,10),dtype=np.float32)
    for t in range(len(qref)):
        q=pin.neutral(model)
        for name,value in zip(names,qref[t]):q[model.joints[model.getJointId(name)].idx_q]=value
        pin.framesForwardKinematics(model,data,q)
        for i,geometry in enumerate(segments):
            local=[]
            for frame,v in geometry:
                tf=data.oMf[frame]
                world=(v@tf.rotation.T+tf.translation)@wr[t].T+wp[t]
                local.append((world-op[t])@rot[t])
            local=np.concatenate(local)
            # AABB distance is a lower bound: discarded points cannot satisfy
            # the 5 mm surface threshold. This changes cost, not contact labels.
            bounds=object_mesh.bounds
            bound=np.linalg.norm(np.maximum(np.maximum(bounds[0]-local,local-bounds[1]),0),axis=1)
            candidate=bound<=.005
            distance[t,0,i]=bound.min()
            if not candidate.any():continue
            nearest,dist,_=trimesh.proximity.closest_point(object_mesh,local[candidate])
            close=dist<=.005
            flags[t,0,i]=close.any()
            # Approximate a patch centroid, not a single closest vertex.
            if close.any():points[t,0,i]=nearest[close].mean(0)
        if t%50==0:print('frame',t,'/',len(qref),flush=True)
    del f['contacts']
    f['contacts/link_names']=np.asarray(LINKS,dtype='S')
    f['contacts/0/is_contact']=flags
    f['contacts/0/points']=points
    f['contacts/0/min_aabb_distance_lower_bound_m']=distance
    f.attrs['contact_source']='geometric approximation: all collision mesh vertices and triangle centroids, fixed descendants included, unsigned distance <=5mm; NOT measured contact/force'
    f.attrs['contact_schema']='direct_pp_dp_10_v1'
    f.attrs['contact_frame']='object_local'
    f.attrs['contact_threshold_m']=.005
    loaded,lp=load_direct_contacts(f,len(qref),1,1)
    assert np.array_equal(loaded[:,:,0],flags) and np.allclose(lp[:,:,0],points)
    assert np.isfinite(points).all() and np.isfinite(distance).all()
    on=points[flags.astype(bool)]
    surface_distance=np.array([])
    if len(on):_,surface_distance,_=trimesh.proximity.closest_point(object_mesh,on)
    summary=dict(links=manifest,frames=len(qref),contact_fraction=flags.mean(axis=(0,1)).tolist(),
                 initial=flags[0,0].tolist(),final=flags[-1,0].tolist(),
                 max_centroid_surface_distance_m=float(surface_distance.max()) if len(surface_distance) else None)
summary['file_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
out.with_suffix('.contacts.json').write_text(json.dumps(summary,indent=2))
print('COMPLETE',str(out),flush=True)
