import json,h5py,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

class HeightAudit:
    def __init__(self,env,path,out):
        self.env=env;self.out=Path(out);self.rows=[]
        with h5py.File(path) as f:
            g=f['object_tracks/0'];self.p=g['translations'][:];self.q=g['orientations_xyzw'][:];self.v=g['vertices'][:]
            self.table=float(f.attrs['table_top_z'])
            self.wp=f['wrist_position'][:];self.joints=f['qpos'][:,0,7:]
        self.capture('after_reset',0)
    def capture(self,label,frame):
        e=self.env;o=e.scene['objects'].data
        p=(o.object_pos_w[0,0]-e.scene.env_origins[0]).cpu().numpy();q=o.object_quat_w[0,0].cpu().numpy()[[1,2,3,0]]
        i=min(frame,len(self.p)-1)
        bottom=lambda p,q:float((Rotation.from_quat(q).apply(self.v)+p)[:,2].min())
        self.rows.append(dict(label=label,frame=i,reference_position=self.p[i].tolist(),live_position=p.tolist(),
            position_error=(p-self.p[i]).tolist(),reference_bottom=bottom(self.p[i],self.q[i]),live_bottom=bottom(p,q),table_z=self.table))
        term=e.action_manager.get_term('hand');robot=term.robot
        wrist=(robot.data.root_pos_w[0]-e.scene.env_origins[0]).cpu().numpy()
        fingers=robot.data.joint_pos[0,term.fingers].cpu().numpy()
        self.rows[-1].update(wrist_reference=self.wp[i].tolist(),wrist_live=wrist.tolist(),wrist_error=(wrist-self.wp[i]).tolist(),
                            finger_error=(fingers-self.joints[i]).tolist(),finger_max_error=float(np.abs(fingers-self.joints[i]).max()),
                            is_fixed_base=robot.is_fixed_base)
        self.rows[-1].update(joint_names=[robot.joint_names[j] for j in term.fingers],
            reference_q=self.joints[i].tolist(),actual_q=fingers.tolist(),pd_target=term.target[0].cpu().tolist())
        (self.out/'height-audit.json').write_text(json.dumps(self.rows,indent=2))
