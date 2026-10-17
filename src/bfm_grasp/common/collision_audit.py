import json
from pxr import Usd,UsdPhysics,PhysxSchema
import isaaclab.sim as sim

def check(out):
    stage=sim.get_current_stage();groups={'hand':[],'object':[],'table':[]};articulations=[];filters=[]
    for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        path=str(prim.GetPath())
        if not path.startswith('/World/envs/env_0/'):continue
        group='hand' if '/Rby1aSharpa' in path else 'object' if '/object_0' in path else 'table' if '/ReferenceTable' in path else None
        if group is None:continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):groups[group].append({'path':path,'enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()})
        if group=='hand' and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):articulations.append({'path':path,'self_collisions':PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get()})
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            targets=[str(x) for x in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()]
            filters.append({'path':path,'targets':targets})
    assert articulations and all(x['self_collisions'] is False for x in articulations)
    assert all(rows and all(x['enabled'] is not False for x in rows) for rows in groups.values())
    assert not any('/object_0' in t or '/ReferenceTable' in t for x in filters if '/Rby1aSharpa' in x['path'] for t in x['targets'])
    report={'articulations':articulations,'colliders':groups,'filtered_pairs':filters,'external_collision_geometry_enabled':True}
    (out/'collision-audit.json').write_text(json.dumps(report,indent=2))
