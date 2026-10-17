"""Launch a bundled ten-object BFM experiment; no workstation-specific paths."""
import argparse, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("mode",choices=["train","smoke","eval"])
p.add_argument("--references",choices=["raw","contrack"],default="contrack")
p.add_argument("--manifest",type=Path,default=ROOT/"data/references/manifest.json")
p.add_argument("--device",default="cuda:0")
p.add_argument("--num-envs",type=int)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--checkpoint",type=Path)
p.add_argument("--dry-run",action="store_true")
a,extra=p.parse_known_args()
if a.mode=="eval" and not a.checkpoint:p.error("eval requires --checkpoint; trained weights are not bundled")
if a.output.exists():p.error("Choose a new output directory; do not overwrite a run")
rows=json.loads(a.manifest.read_text())["objects"]
def resolve(path):
    path=Path(path)
    return str(path if path.is_absolute() else ROOT/path)
entry="evaluate_table.py" if a.mode=="eval" else "train_bfm_baseline.py"
cmd=[sys.executable,str(ROOT/"scripts"/entry),"--hand-only","--headless","--device",a.device,
     "--reference-source","measured" if a.references=="contrack" else "graspxl-geometric",
     "--reference",resolve(rows[0][a.references]["path"]),"--motion",resolve(rows[0]["motion"]),
     "--output",str(a.output.resolve()),"--tactile","binary","--z-window","8","--goal-ratio","0",
     "--reward-mode","conditional","--startup-warmup-episodes","1","--seed","42"]
for row in rows[1:]:cmd += ["--expert",resolve(row[a.references]["path"]),"--group-motion",resolve(row["motion"])]
if a.mode=="train":cmd += ["--num-envs",str(a.num_envs or 4096),"--network-width","2048","--batch","4096","--steps","24576","--warmup","3","--update-every","1","--updates-per-step","16","--replay-capacity","1280000","--replay-device","cpu","--save-every","2048","--save-seconds","1800","--tb-interval","32"]
elif a.mode=="smoke":cmd += ["--num-envs",str(a.num_envs or len(rows)),"--network-width","2048","--collect-only","--batch","80","--steps","8","--warmup","3","--update-every","1","--replay-capacity","256"]
else:cmd += ["--num-envs",str(a.num_envs or len(rows)),"--network-width","2048","--batch","80","--steps","349","--evaluate","--video","--resume",str(a.checkpoint.resolve())]
cmd += extra
print(json.dumps({"command":cmd,"smoke_is_not_research_configuration":a.mode=="smoke"},indent=2),flush=True)
if not a.dry_run:
    env=os.environ.copy();env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONUNBUFFERED="1")
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if not (a.output/"result.json").is_file():
        raise RuntimeError("Simulator exited without result.json; inspect the simulator error output")
