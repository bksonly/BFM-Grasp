"""GraspXL tabletop is the original z=0 plane, never an object mesh bound."""
import numpy as np

def write_table_frame(file,raw_to_world_translation,source='GraspXL official tabletop viewer: original z=0'):
    translation=np.asarray(raw_to_world_translation,dtype=np.float64)
    if translation.shape!=(3,) or not np.isfinite(translation).all():raise ValueError('Invalid scene translation')
    file.attrs['table_frame_version']=1
    file.attrs['table_source_plane_z']=0.0
    file.attrs['table_raw_to_world_translation']=translation
    file.attrs['table_raw_to_world_rotation']=np.eye(3)
    file.attrs['table_top_z']=float(translation[2])
    file.attrs['table_height_source']=source
    validate_table_frame(file)

def validate_table_frame(file):
    if 'table_top_z' not in file.attrs:return
    version=file.attrs.get('table_frame_version',0)
    if version not in (1,2):
        raise ValueError(f'{file.filename}: unverified tabletop frame; migrate before screening/training')
    t=np.asarray(file.attrs['table_raw_to_world_translation'])
    r=np.asarray(file.attrs['table_raw_to_world_rotation'])
    if t.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(t).all() or not np.isfinite(r).all():raise ValueError('Invalid table transform')
    if not np.allclose(r,np.eye(3),atol=1e-9):raise ValueError('Rotated tabletop requires explicit plane handling')
    source=float(file.attrs['table_source_plane_z'])
    correction=0.
    if version==2:
        correction=float(file.attrs['table_geometry_correction_z'])
        if not np.isfinite(correction) or correction>0 or not file.attrs.get('table_geometry_correction_reason'):
            raise ValueError('Invalid or undocumented geometry calibration')
    if source!=0 or not np.isclose(float(file.attrs['table_top_z']),t[2]+correction,atol=1e-8,rtol=0):
        raise ValueError(f'{file.filename}: tabletop does not match transformed original z=0')
