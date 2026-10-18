"""Create held-out physical scale overlays; never alter canonical reference z."""
import argparse,hashlib,json,shutil
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--training-manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
m=json.loads(Path(a.training_manifest).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
for scale in [1.1,1.05,1.15]:
 records=[];folder=out/str(scale);folder.mkdir(exist_ok=True)
 for c,name in enumerate(m['class_names']):
  original=next(x for x in m['variants'] if x['class_id']==c and x['scale']==[1.,1.,1.])
  dest=folder/(name+'.h5')
  if not dest.exists():
   shutil.copy2(original['motion'],dest)
   with h5py.File(dest,'r+') as f:
    f['object_tracks/0/vertices'][:]*=scale
    f.attrs['evaluation_geometry_scale']=scale
    f.attrs['evaluation_only']=True
  with h5py.File(dest) as f,h5py.File(original['motion']) as g:
   assert np.allclose(f['object_tracks/0/vertices'][:],g['object_tracks/0/vertices'][:]*scale)
   assert np.array_equal(f['qpos'][:],g['qpos'][:])
  records.append({'class_id':c,'name':name,'scale':[scale]*3,'motion':str(dest),'motion_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
 manifest={'variants_per_class':1,'variants':records,'split':'seen' if scale==1.1 else 'interpolation' if scale==1.05 else 'extrapolation','canonical_reference_policy':'unchanged training class reference'}
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(folder/'manifest.json')
