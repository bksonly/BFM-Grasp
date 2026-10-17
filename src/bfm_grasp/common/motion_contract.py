"""Testable wrist and context contracts, independent of Isaac Sim."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def wrist_root_states(position, quaternion_xyzw, fps):
    """Return pose and world-frame finite-difference linear/angular velocity."""
    position=np.asarray(position,dtype=np.float64)
    quaternion_xyzw=np.asarray(quaternion_xyzw,dtype=np.float64)
    assert position.shape==(len(quaternion_xyzw),3) and len(position)>=2
    assert fps>0 and np.isfinite(position).all() and np.isfinite(quaternion_xyzw).all()
    rotation=Rotation.from_quat(quaternion_xyzw)
    linear=np.gradient(position,1/float(fps),axis=0)
    # R_next R_prev^-1 yields a world-frame increment, not Euler angle rates.
    intervals=(rotation[1:]*rotation[:-1].inv()).as_rotvec()*fps
    angular=np.empty_like(position)
    angular[0]=intervals[0];angular[-1]=intervals[-1]
    angular[1:-1]=.5*(intervals[:-1]+intervals[1:])
    return np.concatenate([position,rotation.as_quat()[:,[3,0,1,2]],linear,angular],axis=-1).astype(np.float32)


def grouped_reference_mask(group_ids, fraction=.5):
    """Sample independently inside each physical object group, without parity bias."""
    assert 0<=fraction<=1
    mask=torch.zeros_like(group_ids,dtype=torch.bool)
    for group in torch.unique(group_ids):
        ids=(group_ids==group).nonzero().flatten()
        n=int(len(ids)*fraction)
        selected=ids[torch.randperm(len(ids),device=ids.device)[:n]]
        mask[selected]=True
    return mask[:,None]


def tracking_context(model,reference,window,dims):
    """Equivalent to upstream tracking_inference on original next observations.

    The shortened final window does not duplicate the final observation.
    After the complete reference, the external controller holds the final z.
    """
    n=len(reference.proprio)
    goals={k:torch.zeros(n-1,v,device=reference.proprio.device) for k,v in dims.items()}
    goals.update(reference.goal(torch.arange(1,n,device=reference.proprio.device)))
    encoding=model.backward_map(goals)
    # A bounded offset sum avoids subtracting long cumulative sums.
    total=torch.zeros_like(encoding);count=torch.zeros((n-1,1),device=encoding.device)
    for offset in range(min(window,n-1)):
        total[:n-1-offset]+=encoding[offset:]
        count[:n-1-offset]+=1
    return model.project_z(total/count)
