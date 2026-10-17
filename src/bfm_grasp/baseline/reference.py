import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import numpy as np
import torch
import pinocchio as pin
from scipy.spatial.transform import Rotation
from core import Reference,box_corners,local_corners

def load_reference(path,device,root=Path(__file__).resolve().parents[3],source='measured'):
    """Load source reference or a complete measured successful rollout."""
    with h5py.File(path) as f:
        if source == 'graspxl-geometric':
            if f.attrs.get('reference_kind') != 'graspxl_geometric_task_extension_v1':
                raise ValueError('Expected validated GraspXL geometric reference cache')
            if f.attrs.get('contact_schema') != 'direct_pp_dp_10_v1' or 'geometric' not in f.attrs.get('contact_source',''):
                raise ValueError('Missing geometric ten-body contact provenance')
        elif source != 'measured':
            raise ValueError(source)
        if source == 'measured' and 'lift_test' in f.attrs and f.attrs.get('reference_kind','')!='successful_physics_rollout':
            raise ValueError('Synthetic lift targets are not expert states; use a complete measured successful rollout')
        q=f['qpos'][:,0];pos=f['object_tracks/0/translations'][:]
        rot=Rotation.from_quat(f['object_tracks/0/orientations_xyzw'][:]).as_matrix()
        vertices=f['object_tracks/0/vertices'][:];fps=float(np.asarray(f['fps']).reshape(-1)[0])
        if 'wrist_position' in f:
            tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
            wp=f['wrist_position'][:];wr=Rotation.from_quat(f['wrist_quaternion_xyzw'][:]).as_matrix()
            corners=local_corners(box_corners(tensor(vertices)),tensor(pos),tensor(rot),tensor(wp),tensor(wr))
            dq=f['qvel'][:,0,7:] if 'qvel' in f else np.gradient(q[:,7:],1/fps,axis=0)
            targets=None
            if 'contacts/link_names' in f:
                from contact_reference import load_direct_contacts
                flags,points=load_direct_contacts(f,len(q),1,1)
                xyzw=f['object_tracks/0/orientations_xyzw'][:]
                targets=dict(q=tensor(q[:,7:]),object_position=tensor(pos),
                             object_quaternion=tensor(xyzw[:,[3,0,1,2]]),
                             contact=tensor(flags[:,0,0]),contact_point=tensor(points[:,0,0]))
            return Reference(tensor(np.concatenate([q[:,7:],dq],-1)),corners,targets)
    xml=ET.parse(Path(__file__).resolve().parents[3]/'third_party/ConTrack/assets/urdf/rby1a_sharpa_kinematics.urdf').getroot()
    for j in xml.findall('joint'):
        limit=j.find('limit')
        if limit is not None:
            for field in ('effort','velocity'):limit.attrib.setdefault(field,'0')
    model=pin.buildModelFromXML(ET.tostring(xml,encoding='unicode'));data=model.createData();full=pin.neutral(model)
    ids=[model.joints[model.getJointId(f'right_arm_{i}')].idx_q for i in range(7)]
    frame=model.getFrameId('right_hand_C_MC');wp=[];wr=[]
    for angles in q:
        full[ids]=angles[:7];pin.framesForwardKinematics(model,data,full)
        wp.append(data.oMf[frame].translation.copy());wr.append(data.oMf[frame].rotation.copy())
    tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
    corners=local_corners(box_corners(tensor(vertices)),tensor(pos),tensor(rot),tensor(wp),tensor(wr))
    dq=np.gradient(q[:,7:],1/fps,axis=0)
    return Reference(tensor(np.concatenate([q[:,7:],dq],-1)),corners)
