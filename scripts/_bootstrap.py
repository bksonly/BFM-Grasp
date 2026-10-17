"""Select an experiment profile and its shared implementation, without loading a simulator."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "bfm_grasp"
def configure(profile):
    if profile not in {"baseline", "dr", "evaluation"}:
        raise ValueError(profile)
    paths = [SOURCE / profile]
    if profile == "evaluation":
        paths.append(SOURCE / "baseline")
    paths += [SOURCE / "common", ROOT / "third_party/BFM-Zero",
              ROOT / "third_party/ConTrack/source/ConTrack"]
    sys.path[:0] = [str(path) for path in paths]
    return SOURCE / profile
def launch(profile):
    runpy.run_path(str(configure(profile) / "train.py"), run_name="__main__")
