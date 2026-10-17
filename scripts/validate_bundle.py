"""CPU-only check of the shipped references, geometry and import layout."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/"data/references/manifest.json").read_text())["objects"]
assert len(rows)==10 and len({x["name"] for x in rows})==10
for row in rows:
    assert row["motion"]==row["raw"]["path"]
    for kind in ["raw","contrack"]:
        item=row[kind];path=ROOT/item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
        with h5py.File(path) as f:
            n=len(f["qpos"]);assert n==350 and f["qpos"].shape==(n,1,29)
            assert f["wrist_position"].shape==(n,3)
            assert f["contacts/0/is_contact"].shape==(n,1,10)
            assert np.isin(f["contacts/0/is_contact"][:],[0,1]).all()
            assert f.attrs["contact_frame"]=="object_local"
            assert float(f["fps"][0])==60
            assert f.attrs["reference_kind"]==("successful_physics_rollout" if kind=="contrack" else "graspxl_geometric_task_extension_v1")
for p in (ROOT/"assets/sharpa").glob("*.urdf"):
    for mesh in ET.parse(p).getroot().iter("mesh"):
        assert (p.parent/mesh.get("filename")).is_file()
for item in json.loads((ROOT/"assets/manifest.json").read_text())["files"]:
    assert hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest()==item["sha256"]
assert not (ROOT/".gitmodules").exists()
print("PASS: 20 references, hashes, protocol, shared wrist source, hand mesh paths; no submodules")
