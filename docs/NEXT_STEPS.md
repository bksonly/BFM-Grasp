# Next stage: parity validation and banana-to-basket demo

## Execution gate

The user will inspect DR training curves on the morning after 2026-10-17 and explicitly instruct stopping. No scheduled stop or polling is requested. Continue the existing DR jobs until then. All new development takes place in BFM-Grasp and is versioned with Git.

After stopping: save final state, evaluate the agreed checkpoints, and add measured DR results and videos to the private site.

## GPU A: validate the code organization

Use identical checkpoints, reference hashes, assets, configurations and seeds in the historical and organized code. First compare observation assembly, normalized inputs, latent z and actor outputs on identical recorded tensors. Then compare physics resets, object trajectories and per-object success over repeated episodes. Report numerical tolerances and simulator nondeterminism. A short smoke test is not a substitute for this parity experiment. Exercise both baseline and DR contracts; restore/cache DR geometry explicitly rather than silently reverting to fixed geometry.

## GPU B: banana-to-basket demo

Scene: a furnished room, table, open basket, existing banana-A mesh, and the full Sharpa hand. Wrist motion is controlled externally. Keep BFM running for finger stabilization during grasp, lift and transport. Release by a documented rule-based return toward the initial finger pose. No attachment constraint, teleportation or hidden force should substitute for grasping.

Candidate resources:
- Official Code as Policies: https://github.com/google-research/google-research/tree/master/code_as_policies
- Isaac Sim Simple Room asset: /Isaac/Environments/Simple_Room/simple_room.usd (verify against the installed 5.1 asset set before use).
- Poly Haven basket candidates: https://polyhaven.com/models (Wicker Basket 01/02; select an open variant and verify geometry/license before bundling).

Use CaP's separation of language-program generation, skill interfaces, and reusable prompts. Avoid replacing the existing IsaacLab simulator with a second simulator just to use the original examples.

Planner interface proposal: observe, move_wrist, grasp(reference_id), release, and inspect_result. Exact signatures, frames, units and constraints must be fixed before freezing the reusable prompt.

## No-API mode

A session-controlled bridge can pause simulation, write a camera observation/request, wait for a structured response generated in this conversation, execute it, and produce the next observation. This avoids the need for a model API key for the initial demo. It requires the active assistant session; it is not a standalone unattended API service. Keep this interface replaceable by a real model endpoint later.

Visual planning needs metric grounding: calibrated camera intrinsics/extrinsics and depth, or a clearly defined tabletop coordinate/grid interface. A single uncalibrated RGB image does not guarantee centimeter-accurate wrist placement. Separate visual estimates from simulator ground truth used only for evaluation. Record exactly what information the planner receives; actor remains visual-blind.

Freeze after a successful run: prompt, tool schema, reference catalog, camera configuration, exact observation images, generated plan, execution trace, seed/configuration and outcome. Others should be able to reuse the prompt with another vision-language model; replay of one saved plan must be labeled separately from fresh planning.

Acceptance: banana clears the table, stays grasped during transport, is released into the basket, and remains inside after settling. Include failed attempts in quantitative reporting; distinguish a demonstration from generalization evaluation.
