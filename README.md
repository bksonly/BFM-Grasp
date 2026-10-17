# BFM-Grasp( )

**A General-Purpose Dexterous Grasping Function for Code-as-Policies**

Private research repository. One policy controls 22 Sharpa finger joints using proprioception, 10 binary tactile channels, history and a reference-derived latent. Wrist motion is external. Training uses privileged information; the actor does not receive live object state or images.

## What is included

- **Static source files**, not Git submodules: official BFM networks/losses and ConTrack PPO/environment source, with original licenses and provenance.
- **20 reference trajectories**: ten GraspXL-derived references with offline geometric contacts and the same ten ConTrack-corrected, measured-contact references. The GraspXL-derived files include resampling and the close/lift/hold extension; they are not untouched dataset `.npy` files.
- The full hand-only Sharpa USD with merged elastomer geometry, plus Sharpa meshes and URDFs. Object meshes are embedded in the reference H5 files.
- Training, smoke-check and table-evaluation entry points; private review website.

No trained checkpoints, replay buffers or simulator binaries are bundled. Evaluation requires your trained checkpoint. DR's 270-variant cache is not bundled; the provided ten-object commands run the fixed-geometry baseline.

## Setup

Use Linux with an NVIDIA GPU and an **installed, licensed Isaac Sim / IsaacLab environment**. The tested workstation uses Python 3.11, Isaac Sim 5.1.0, IsaacLab commit `42e61645c96bac08135566634785cdc87728d5ab`, and PyTorch 2.7.0+cu128. Install IsaacLab and its `isaaclab_tasks` package into the active simulator Python environment first. This repository does not install or accept simulator licenses on your behalf.

```bash
# In the simulator Python environment, from the repository root:
python -m pip install -r requirements.txt
python scripts/validate_bundle.py
python scripts/run.py smoke --references contrack --device cuda:0 --output runs/smoke-contrack
```

The smoke check uses ten environments and the official network architecture. It runs startup physics warmup, a complete reset, and short rollout collection; it does not validate learning performance. First use may compile kernels. Choose a fresh output directory for each run. The two upstream libraries are loaded from this checkout; do not install their full locomotion environments or clone their repositories separately.

## Train the two ten-object baselines

```bash
python scripts/run.py train --references contrack --device cuda:0 --output runs/contrack10
python scripts/run.py train --references raw --device cuda:1 --output runs/raw10
```

Each command starts one experiment. Run only on a GPU with sufficient free memory. Defaults preserve the research setting: 4,096 environments, batch 4,096, 16 updates per sampling step, 24,576 steps, official 2,048-wide networks, conditional auxiliary reward, 8-frame latent windows, and no replay-derived targets. Each run has CPU replay capacity of 1.28 million transitions per object; this is a large-memory workstation configuration. `--dry-run` prints the command without launching it. Do not interpret a smaller smoke check as the full experiment.

Both groups use **the same raw-reference wrist trajectory** from `data/references/manifest.json`; only the learning reference changes. Warmup remains enabled. Checkpoints and training logs are written to the selected output directory. See [algorithm description](docs/算法描述.md) for the exact losses and success metric.

```bash
# A single shared-table video, using your trained baseline checkpoint:
python scripts/run.py eval --references contrack --checkpoint runs/contrack10/best.ckpt --device cuda:0 --output runs/table-eval
```

The shared-table video is a demonstration, not a statistical success-rate evaluation. Checkpoint architecture and observation contract must match the selected profile.

## Use other GraspXL trajectories

Use the **Sharpa tabletop** trajectories already retargeted to this hand, together with the matching object mesh. MANO trajectories and trajectories for another hand are not interchangeable. The converter loads GraspXL dictionary `.npy` files with pickle: use trusted dataset files only.

```bash
python scripts/import_graspxl.py --source /path/to/sharpa_0.npy --mesh /path/to/OBJECT_ID.obj --object-id OBJECT_ID --source-fps 30 --output data/custom/object.h5
```

Set `--source-fps` explicitly: 30 Hz is the assumption used in our original experiments, not a verified universal dataset frame rate. The converter validates the 22-joint order/limits, resamples to 60 Hz, preserves the original z=0 tabletop, appends 0.5 s close hold + 1 s smooth 5 cm lift + 1 s hold, and computes ten PP/DP geometric contact targets including fixed elastomer geometry. These targets are geometric approximations, not measured forces or simulated successful grasps. Inspect kinematic replay and physical behavior before trusting a new object. Conversion does not guarantee a successful grasp.

Create a manifest using the same structure as `data/references/manifest.json`. For a raw-only custom set, each object needs `name`, `raw: {"path": "data/custom/object.h5", "sha256": "..."}`, and `motion: "data/custom/object.h5"`; paths are relative to the repository root or absolute. Then use `scripts/run.py train --references raw --manifest data/custom/manifest.json --output runs/custom`. You can first preview a converted reference through `scripts/train_bfm_baseline.py --hand-only --headless --device cuda:0 --reference-source graspxl-geometric --reference data/custom/object.h5 --motion data/custom/object.h5 --output runs/custom-preview --num-envs 1 --kinematic-video`.

For residual PPO correction, the hand-only ConTrack profile is available:

```bash
python scripts/train_contrack.py --task Isaac-Rby1a-Sharpa-Mimic-Manager-v0 --data data/custom/object.h5 --experiment_name custom-sharpa --run_name seed42 --num_envs 16384 --max_iterations 800 --save_interval 100 --headless --logger tensorboard --video_num_envs 0 --seed 42 --device cuda:0
```

This preserves the original PPO optimizer and task reward with hand-only actions/reset and startup warmup. Exporting a newly trained successful rollout into a measured-contact expert remains a separate step; do not relabel geometric contacts as measured contacts. The shipped 20 files already contain the required cached data.

## Layout

```text
assets/sharpa/            # Portable hand USD, URDF and joint order
data/references/        # raw/ + contrack/: ten files each, shared manifest
src/bfm_grasp/
  common/                # Shared rewards, contacts, reset checks, logging
  baseline/              # Fixed-object BFM
  dr/                    # DR observations, geometry, dynamics and replay
  evaluation/            # Shared-table evaluation
  contrack/              # Actual hand-only PPO task overrides
scripts/                 # Run, validate, import and train entry points
third_party/             # Ordinary vendored BFM-Zero and ConTrack files
site/                    # Private review website
```

See [code guide](docs/CODE_GUIDE.md) and [reference protocol](docs/算法描述.md). `dependencies.lock.json` records upstream versions; original LICENSE files remain in `third_party/`. Dataset and asset rights still apply; keeping this repository private does not grant redistribution rights.

## Publication

Keep this repository and webpage private until reviewed. Preview locally with `python -m http.server 8765 --bind 127.0.0.1 --directory site`. GitHub Pages is not enabled. The current single-commit history is intended to avoid downloading removed robot assets; local pre-cleanup history is backed up separately on the workstation.
