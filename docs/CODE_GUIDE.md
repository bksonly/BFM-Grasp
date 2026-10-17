# Code guide

## Two training stages

1. `scripts/train_contrack.py` selects `src/bfm_grasp/contrack/`, the actual hand-only PPO profile. A process-local import selector supplies its actions/state/rewards/environment under the original task module names; upstream PPO and runners remain in the bundled library. PPO lives in `third_party/ConTrack/rsl_rl_contrack/algorithms/ppo.py`; its runner is `runners/on_policy_runner.py`. The base Sharpa environment is `third_party/ConTrack/source/ConTrack/ConTrack/tasks/manager_based/mimic_rby1a_sharpa/`, including `rby1a_sharpa_env_cfg.py`, `mdp/` and `agents/`.
2. `scripts/train_bfm.py` selects the DR profile. Official BFM loss/update code lives in `third_party/BFM-Zero/humanoidverse/agents/{fb,fb_cpr,fb_cpr_aux}/`; neural architectures live in `nn_models.py`. Our `core.py` assembles task inputs/configuration and replay, and `conditional_agent.py` connects the task auxiliary reward.

## Shared code and profiles

`common/` contains one copy of modules that were byte-identical across all three experimental directories: auxiliary rewards, contact processing, expert sampling, warmup, success reporting and audits. This eliminates 32 duplicate file copies, including modules shared by baseline and evaluation.

`baseline/` retains fixed-object observation and replay behavior. `dr/` retains its distinct actor-noise isolation, privileged observations, packed replay, scale references and dynamics randomization. These are intentional algorithm-interface differences, not accidental duplicate copies. They remain explicit rather than being hidden in a collection of conditionals.

`evaluation/` contains only its training/evaluation driver and shared-table scene overrides; it reuses baseline modules and common modules.

`scripts/_bootstrap.py` sets a deterministic import order: selected profile, baseline fallback for evaluation, common modules, then bundled upstream packages. Run one profile per process. Launch through scripts/, not individual internal modules. The internal modules retain their historical simple imports for a low-risk structural refactor; future package conversion can replace those with explicit package imports.

## Reading order

Read `docs/算法描述.md`, then the selected profile's `train.py`, `core.py`, `isaac_bridge.py` and `grouped_scene.py`. Read `common/contrack_aux.py` for auxiliary reward and upstream `fb_cpr_aux/agent.py` for official optimization. DR settings and their existing CPU regression tests are alongside `dr_runtime.py` in `dr/`.

## Upstream size and scope

BFM-Zero occupies about 2.4 MiB: all tracked source/configuration files, excluding unused locomotion robot/data assets and demo media. ConTrack source plus Sharpa assets occupies about 15 MiB after removing unrelated upstream demo trajectories. Only Sharpa left/right meshes and visual URDFs are retained; unrelated RBY1A body/arm, xArm and XHand geometry is removed. A geometry-free kinematic URDF preserves legacy wrist FK. These sizes describe current files, not historical Git objects. Git history and runtime-generated files from the original upstream checkouts are not copied. Source hashes and exclusions are recorded in `third_party/source-manifest.json`.

The production training checkout is unchanged. The shipped bundle has passed CPU data checks, a ten-object BFM startup/reset/collection smoke check, and a one-iteration hand-only PPO run. These are integration checks, not convergence or performance validation. See README for required simulator installation, included data, and commands.

## Reproducibility bundle

`data/references/manifest.json` selects 20 ordinary H5 files and pins their SHA-256 hashes. Raw and corrected learning references use the same raw wrist motion. `assets/sharpa/` supplies the complete generated hand USD and portable URDF. `scripts/run.py` resolves every bundled path relative to the repository root. Historical workstation-only orchestration scripts were removed from the clean distribution.
