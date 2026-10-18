"""Freeze same-category held-out meshes by geometry, before running any policy."""
import argparse,json,hashlib,shutil
from pathlib import Path
import h5py,numpy as np,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p=argparse.ArgumentParser();p.add_argument('--training-manifest',required=True);p.add_argument('--candidates',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',required=True);p.add_argument('--mesh-choices');a=p.parse_args()
m=json.loads(Path(a.training_manifest).read_text());candidates=json.loads(Path(a.candidates).read_text());selected=json.loads(Path(a.selection).read_text());exclude={x['uid'] for x in selected}|{'560b013c4aed4cc093de013c517b696d','098db6acfad0452285ce77781932fd24'}
choices=json.loads(Path(a.mesh_choices).read_text()) if a.mesh_choices else {}
groups={'cylinder':'cylinder','bottle':'bottle','mug':'mug','banana-A':'curved','B09':'bowl'}
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);records=[];renders=[]
for c,name in enumerate(m['class_names']):
 original=next(x for x in m['variants'] if x['class_id']==c and x['scale']==[1.,1.,1.])
 dest=out/(name+'.h5');shutil.copy2(original['motion'],dest)
 record={'class_id':c,'name':name,'motion':str(dest),'held_out_shape':False}
 if name in groups:
  with h5py.File(dest) as f:old=f['object_tracks/0/vertices'][:];oldfaces=f['object_tracks/0/faces'][:]
  oldextent=np.ptp(old,axis=0);ranked=[]
  category=next((x['category'] for x in selected if x['group']==groups[name]),None)
  for row in candidates:
   if row['group']!=groups[name] or row['uid'] in exclude:continue
   if name in choices and row['uid']!=choices[name]:continue
   if name=='tool' and row['category']!=category:continue
   mesh=trimesh.load(Path(row['folder'])/'object.obj',force='mesh',process=False)
   extent=mesh.extents;score=float(np.linalg.norm(extent/extent.max()-oldextent/oldextent.max()))
   ranked.append((score,row['uid'],row,mesh))
  assert ranked,name
  score,uid,row,mesh=sorted(ranked,key=lambda x:(x[0],x[1]))[0]
  factor=oldextent.max()/mesh.extents.max()
  vertices=(mesh.vertices-mesh.bounds.mean(0))*factor+(old.min(0)+old.max(0))/2
  with h5py.File(dest,'r+') as f:
   for key,value in [('vertices',vertices),('faces',mesh.faces)]:
    del f['object_tracks/0/'+key];f['object_tracks/0/'+key]=value
   f.attrs['evaluation_only']=True;f.attrs['evaluation_shape_uid']=uid
  record.update(held_out_shape=True,uid=uid,category=row['category'],mesh_path=str(Path(row['folder'])/'object.obj'),uniform_size_normalization=factor,selection_score=score,selection_rule='Visual geometry review for complete same-category mesh, then closest normalized extents; training IDs excluded; no policy outcome used')
  renders.append((name,old,oldfaces,vertices,mesh.faces))
 record['motion_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();records.append(record)
(out/'manifest.json').write_text(json.dumps({'variants_per_class':1,'variants':records,'split':'held_out_shape_with_separate_control_classes','canonical_reference_policy':'unchanged training class reference'},indent=2))
fig=plt.figure(figsize=(13,5.5))
for col,(name,old,of,new,nf) in enumerate(renders):
 for row,(v,f) in enumerate([(old,of),(new,nf)]):
  ax=fig.add_subplot(2,len(renders),row*len(renders)+col+1,projection='3d');ax.add_collection3d(Poly3DCollection(v[f],facecolor='#6aa9b8',edgecolor='none'))
  center=(old.max(0)+old.min(0))/2;radius=np.ptp(old,axis=0).max()*.65
  ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius));ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(elev=20,azim=-55);ax.set_title(name+(' / training' if row==0 else ' / unseen'),fontsize=10)
fig.tight_layout();fig.savefig(out/'shape-comparison.png',dpi=180)
print(json.dumps(records,indent=2))
