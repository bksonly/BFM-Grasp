import json,torch
from types import SimpleNamespace as NS
from pathlib import Path
import dr_runtime as dr
torch.manual_seed(7);torch.set_num_threads(2)
assert dr.configure(None)==dr.configure(str(Path(__file__).with_name('dr_defaults.json')))
dr.ENABLED=True
class View:
    def __init__(self):
        self.mass=torch.ones(64,1);self.inertia=torch.ones(64,9);self.material=torch.ones(64,1,3);self.material[:,:,2]=0
    def get_masses(self):return self.mass.clone()
    def get_inertias(self):return self.inertia.clone()
    def get_material_properties(self):return self.material.clone()
    def set_masses(self,x,ids):self.mass[ids]=x[ids]
    def set_inertias(self,x,ids):self.inertia[ids]=x[ids]
    def set_material_properties(self,x,ids):self.material[ids]=x[ids]
view=View();writes={}
robot=NS(data=NS(default_joint_stiffness=torch.ones(64,22),default_joint_damping=torch.ones(64,22)),write_joint_stiffness_to_sim=lambda x,env_ids:writes.update(kp=x),write_joint_damping_to_sim=lambda x,env_ids:writes.update(kd=x))
env=NS(scene={'objects':NS(root_physx_view=view),'rby1a_sharpa':robot},device='cpu')
values=[]
for _ in range(100):
    dr.randomize_dynamics(env,torch.arange(64))
    assert ((view.mass>=.8)&(view.mass<=1.2)).all()
    assert torch.equal(view.inertia,view.mass.expand(-1,9))
    assert (view.material[:,:,2]==0).all()
    values.append(view.material[:,:,0].flatten().clone())
    for x in writes.values():assert ((x>=.8)&(x<=1.2)).all() and (x.std(-1)>0).all()
assert torch.cat(values).unique().numel()==64
print('PASS defaults synchronized; mass/inertia non-compounding; per-joint PD ranges; only 64 friction levels across 100 resets')
