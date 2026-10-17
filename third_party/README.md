# Bundled algorithm and environment source

These are ordinary files tracked by this private repository. No external BFM-Zero or ConTrack checkout is required to obtain their algorithm source.

- BFM-Zero: all upstream tracked source/configuration/build/license files; excludes locomotion data/robot assets (`humanoidverse/data/`) and demonstration media (`model/`), which are not used by our Sharpa task.
- ConTrack: upstream source packages, PPO runner and environment configurations, with only Sharpa left/right robot geometry retained. Unused RBY1A body/arm, xArm and XHand meshes and their visual URDFs are excluded. A geometry-free RBY1A–Sharpa kinematic URDF is retained only for legacy trajectory wrist FK. Unrelated upstream demo trajectories under data/ are also excluded; our 20 required references live at the repository root. Other upstream robot tasks are not supported by this asset subset. The three current environment modifications are included directly. Do not apply `patches/ConTrack.patch` again.
- Each original LICENSE and copyright notice is retained. `source-manifest.json` records upstream commits and SHA-256 hashes of the imported files. `dependencies.lock.json` records upstream repository URLs.

## Runtime boundary

Isaac Sim/IsaacLab, PyTorch and Python packages remain installed runtime dependencies. BFM and ConTrack are ordinary tracked source files, with no nested Git repositories, gitlinks, or Git LFS pointers.

Twenty reference H5 files and the generated hand USD are bundled in data/references and assets/sharpa. See the root README for tested runtime versions and commands. The 270-variant DR cache and trained policy weights are not included.

source-manifest.json records the imported upstream source hashes and exclusions. The hand-only PPO overrides are explicit in src/bfm_grasp/contrack; source changes in the bundled ConTrack environment locate the in-repository table-frame validator. The original licenses continue to apply.
