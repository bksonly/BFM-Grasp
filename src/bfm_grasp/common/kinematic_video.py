import json,h5py,numpy as np,torch,imageio.v2 as imageio
from PIL import Image,ImageDraw

def record(env,path,out):
    with h5py.File(path) as f:
        q=f['qpos'][:,0,7:];wp=f['wrist_position'][:];wq=f['wrist_quaternion_xyzw'][:]
        op=f['object_tracks/0/translations'][:];oq=f['object_tracks/0/orientations_xyzw'][:];fps=float(f['fps'][0])
    term=env.action_manager.get_term('hand');robot=term.robot
    target=out/'videos/reference-kinematic.mp4';target.parent.mkdir(exist_ok=True)
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=env.device)
    error=0.
    env.render(recompute=True)
    for _ in range(10):env.sim.render()
    with imageio.get_writer(str(target),fps=fps,codec='libx264',quality=8) as writer:
        for frame in range(len(q)+60):
            i=min(frame,len(q)-1)
            hand=np.r_[wp[i],wq[i,[3,0,1,2]],np.zeros(6)][None]
            obj=np.r_[op[i],oq[i,[3,0,1,2]],np.zeros(6)][None,None]
            robot.write_root_state_to_sim(tensor(hand))
            robot.write_joint_state_to_sim(tensor(q[i:i+1]),torch.zeros(1,22,device=env.device),joint_ids=term.fingers)
            env.scene['objects'].write_object_state_to_sim(tensor(obj))
            env.sim.forward();env.scene.update(0.)
            env.sim.render()
            image=env.render(recompute=True)
            if frame==len(q)-1:
                imageio.imwrite(out/'final.png',image)
                assert image[80:].std()>10,'Blank kinematic rendering'
            image=Image.fromarray(image);draw=ImageDraw.Draw(image)
            draw.rectangle((8,8,620,38),fill='black')
            draw.text((16,16),f'KINEMATIC ONLY | reference frame {i}/{len(q)-1} | no added lift',fill='white')
            writer.append_data(np.asarray(image))
            error=max(error,float((robot.data.joint_pos[0,term.fingers]-tensor(q[i])).abs().max()))
    result={'frames':len(q),'fps':fps,'physics_steps':0,'max_joint_error_rad':error,'object_follows_reference':True,'video':str(target)}
    (out/'kinematic.json').write_text(json.dumps(result,indent=2));(out/'result.json').write_text(json.dumps(result,indent=2))
