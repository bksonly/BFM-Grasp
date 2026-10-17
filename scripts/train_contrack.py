"""Run the actual hand-only ConTrack PPO profile used to produce the experts."""
import importlib.abc,importlib.util,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/"src/bfm_grasp/contrack"
PREFIX="ConTrack.tasks.manager_based.mimic_rby1a_sharpa"
OVERRIDES={PREFIX+".rby1a_sharpa_env_cfg":"environment.py",**{PREFIX+".mdp."+x:x+".py" for x in ["actions","state","rewards"]}}
class ProfileModules(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in OVERRIDES:
            return importlib.util.spec_from_file_location(fullname,PROFILE/OVERRIDES[fullname])
sys.meta_path.insert(0,ProfileModules())
sys.path[:0]=[str(PROFILE),str(ROOT/"src/bfm_grasp/common")]
runpy.run_path(str(PROFILE/"train.py"),run_name="__main__")
