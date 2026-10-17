# Development workflow

- This checkout is the primary development repository. Use Git commits for subsequent code and documentation changes; do not duplicate whole experiment source directories for versioning.
- `/home/zzy/CraftNet` is the historical experiment workspace. Existing DR jobs continue there until the user reviews curves and explicitly requests stopping them. Do not migrate, stop, restart, or edit those jobs automatically.
- Before GPU work, inspect existing processes and available memory. Never launch duplicate jobs.
- After the user's stop instruction: preserve final checkpoints, evaluate DR, and update the private website with measured results. Then use one available GPU for old-versus-organized-code parity validation and the other for the banana-to-basket demo.
- Repository and website remain private. Keep large generated outputs and checkpoints out of Git; commit manifests, configurations, scripts and concise results.
- Do not label a conversational/offline LLM demonstration as an autonomous API integration. Preserve prompts, image inputs, tool outputs and generated plans for reproducibility.
