# Champions RL

An offline battle bot for **Pokémon Champions, Regulation M-C, Battle Stadium Singles**. It plays complete battles programmatically: six-Pokémon preview, select three and a lead, moves, switches, forced replacements, and Mega Evolution. No account, battle server, OCR, tablet, or controller is involved.

The method is **knowledge-guided reinforcement learning with imitation pretraining**. Game facts become structured inputs and pretrained embeddings; demonstrations teach an initial policy; masked PPO improves the policy through battles against a mixture of opponents and frozen versions of itself.

**A working training system is not a verified strong player.** Read the actual evaluation reports under `runtime_data/`. The program only marks the requested 60% target as passed after a separate test benchmark passes the checks below. There is no measured claim of beating Jaxcalibur.

## Use the installed environment

From this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m champions_rl status
.\.venv\Scripts\python.exe -m champions_rl play --checkpoint runtime_data\bc_v2.pt --opponent balanced
```

`play` writes a complete public Showdown protocol log and a JSON result to `runtime_data/play/`. A neural policy receives only its own player-visible state. An opponent's actual moveset, items, selected trio, and pending action are not inputs.

The first installed experiments are `runtime_data/run_001/` and `runtime_data/run_002/`. `latest.pt` is an inference checkpoint; `trainer_state.pt` also includes the optimizer and PyTorch RNG state. Training logs contain **training** win rates, which are not benchmark results.

## Reproduce installation

Prerequisites: Windows, Node.js 22.18+ and `uv` on PATH. Run:

```powershell
.\tools\setup.ps1
```

The script creates this project's `.venv`, installs the pinned Python packages including CUDA 12.8 PyTorch, and installs/builds the pinned simulator. The installed configuration was checked with Python 3.12 and an RTX 4070 12 GB. PyTorch includes the required CUDA runtime; a separate system CUDA toolkit is not needed. The simulator and feature extraction run on the CPU; the network trains on the GPU.

All battles and training run offline after installation. The copied reference policy is self-contained in `opponents/autoplay_reference/`; the original `Poke-Champ-AutoPlay` checkout is not needed at runtime.

## Full training process

1. **Prepare facts and legal sets.** Already generated and included. Rebuild after deliberately changing the pinned format or simulator:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl prepare
   ```

   `data/knowledge.json` contains the restricted species, legal learnsets, abilities, items, types, natures, and move properties. `data/sets.json` contains 2,145 simulator-validated example sets covering all 269 selectable species/forms in this snapshot. There are 366 forms including battle-only transformations, representing 231 National Dex numbers, and 511 learnable/forced moves. These are all **eligible facts**, not an exhaustive enumeration of every possible legal competitive set.

   Set generation uses September 2026 usage marginals plus legal fallback choices. Combinations are synthetic examples, not claimed to be authentic joint human sets. Team manifests include meta-weighted, uniform, offensive, bulky, and weather variants. Train/validation/test contain 128/64/128 disjoint team compositions; individual species and sets can overlap.

2. **Collect demonstrations from several policies.** Each completed game is a durable shard. Reuse `runtime_data/demos_v2` from the installed run or create a new directory:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl collect --teacher balanced --games 1000 --seed 101 --output runtime_data\my_demos
   .\.venv\Scripts\python.exe -m champions_rl collect --teacher setup --games 1000 --seed 201 --output runtime_data\my_demos
   .\.venv\Scripts\python.exe -m champions_rl collect --teacher defensive --games 500 --seed 301 --output runtime_data\my_demos
   .\.venv\Scripts\python.exe -m champions_rl collect --teacher reference --games 100 --seed 401 --output runtime_data\my_demos
   ```

   The reference teacher uses the original scoring, opponent predictions, 32 seeded simulations, and prepared damage/speed preview cache. Its demonstrations use its saved six on both sides. Other teachers use diverse teams. Reference collection is much slower than the simple opponents; degraded fallback decisions are rejected, not silently counted as the original bot.

3. **Load domain knowledge and imitate.** The first phase learns species→types and move→properties in the embedding tables. Behavior cloning then learns legal actions from the demonstrations, weighting the expensive reference examples more heavily:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl pretrain --data runtime_data\my_demos --output runtime_data\my_bc.pt --steps 5000
   ```

4. **Improve through battles.** The transformer has policy, value, and opponent-action heads. PPO uses legal-action masking, undiscounted match rewards, GAE, and entropy regularization. The default opponent mix contains random, damage, balanced, setup, defensive, stochastic, and frozen neural opponents:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl train --initial runtime_data\my_bc.pt --steps 1000000 --envs 32 --output runtime_data\my_run
   ```

   `--steps` counts learning-player decisions, not games; the final rollout can round it up. By default 80% of non-reference teams are freshly generated legal compositions and 20% come from the training manifest; exact held-out compositions are excluded. `--fixed-teams` reproduces the first experiments, which used only the fixed manifest. `--reference-fraction 0.05` adds the full original bot to the online mix, at a substantial throughput cost. The default is zero; its knowledge is initially supplied through demonstrations. Frozen opponents occupy `--snapshot-fraction 0.3` of the mix after snapshots become available.

   `--heuristic-weight 2` optionally adds a fixed tactical logit prior and learns a residual policy around it. Both rollout action probabilities and PPO updates include the same prior, and checkpoints record its strength. This is an experimental hybrid variant, not a proven improvement over the pure neural policy.

   To stop cleanly, create the configured stop file. To resume, remove it and load the trainer state:

   ```powershell
   New-Item -ItemType File runtime_data\STOP
   Remove-Item -LiteralPath runtime_data\STOP
   .\.venv\Scripts\python.exe -m champions_rl train --resume runtime_data\my_run\trainer_state.pt --steps 1000000 --envs 32 --output runtime_data\my_run
   ```

   New trainer states preserve weights, optimizer, RNG streams, and the frozen opponent league, then start fresh battles. In-progress games are not restored, so this is not a bit-for-bit continuation. Early experiment checkpoints predate league/RNG persistence and restart those components from the supplied seed. `--steps` on resume means **additional** decisions. A STOP file applies to all trainers using that path; use distinct `--stop-file` paths for separate runs.

5. **Select using validation only.** Evaluate immutable numbered checkpoints; compare stochastic and deterministic play on validation before choosing deployment settings:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl evaluate --checkpoint runtime_data\my_run\latest.pt --games 100 --split validation --output runtime_data\my_validation
   ```

   Add `--deterministic` to use the highest-probability legal action. Each report records the setting, checkpoint hash, source hash, teams, seeds, and policy provenance. `evaluated_model.pt` freezes the exact evaluated weights. Do not use test results to repeatedly tune the model.

6. **Run the final frozen benchmark.** Once validation is good enough:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl evaluate --checkpoint runtime_data\my_validation\evaluated_model.pt --games 200 --workers 4 --split test --output runtime_data\final_test
   ```

   The default suite includes the original reference and five other styles. Against the original, both players use its saved six, so a counter-team does not manufacture an advantage. Seats alternate and matchup seeds are paired. Other styles use the held-out team manifest. Original-roster results generalize to that roster and matchup distribution; they do not prove superiority on every possible team.

   Parallel evaluation assigns disjoint game IDs to independent processes and sums game counts. Each pair holds the teams and engine seed fixed while swapping the policies, so the learner controls each team once. Checkpoint/source/data hashes must agree across workers. Public logs, per-game reference trial audits, and a snapshot of the evaluated implementation are retained alongside reports. Four evaluation processes and the reference bot's four probe workers per process suit a many-core CPU; use `--workers 1` on a smaller machine. This does not reduce the reference policy's 32 trials.

   Passing requires at least 200 reference games, at least 60% wins **and a 95% Wilson lower bound of at least 60%**, at least four other styles with 200 games and 55% wins each, no adapter errors or reference fallbacks, and at most 1% truncations. A 60/100 result does not pass this stricter uncertainty check. Errors remain in the denominator; time limits are truncations, not invented wins. The final report explicitly records whether each condition passed.

## Knowledge, features, and rewards

`data/feature_schema.json` lists every input. Entity features include species/type, known own stats, estimated enemy stats with knowledge flags, HP, status, boosts, PP, revealed moves/items/abilities, field conditions, and recent public move history. Candidate features include approximate damage, KO margin, accuracy, priority, recovery, setup, hazards, switches, Mega options, and preview coverage. Contextual Pokémon embeddings connect each candidate to its actor and selected team.

The native simulator resolves battle mechanics. Approximate damage features are decision hints, not a substitute for the simulator and not exact predictions of a hidden opponent set. The full legacy catalog is retained only inside the isolated reference adapter for compatibility; the learned model's vocabulary is the restricted Champions export.

Rewards are +1 for winning, -1 for losing, and 0 for a draw, plus `0.15 × (next_potential - current_potential)`. Potential measures visible HP and remaining Pokémon advantage and is zero at true terminal states. With gamma=1 the shaping terms telescope, preserving the completed-match objective instead of rewarding damage farming or excessive switching. Time caps bootstrap the value estimate and stop GAE from crossing into a reset game.

## Programmatic use and checks

```python
from champions_rl.environment import ChampionsEnv
import numpy as np

with ChampionsEnv(opponent="balanced") as env:
    observation, info = env.reset(seed=42)
    while True:
        action = int(np.flatnonzero(info["action_mask"])[0])  # replace with your policy
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover restricted legality and every generated set/team, hidden-information separation, legal masks, deterministic replay, trapping retries, Illusion, revival, Gym operation, reward telescoping, truncation bootstrapping, checkpoints, and the acceptance gate.

## Scope and provenance

- The engine is Pokémon Showdown's Champions mod at `c0845f33d381402dfcb18ce324c20a94f18af075`, format `gen9championsbssregmc`. See the [pinned implementation](https://github.com/smogon/pokemon-showdown/tree/c0845f33d381402dfcb18ce324c20a94f18af075/data/mods/champions) and [official M-C announcement](https://champions-news.pokemon-home.com/en/page/816.html). Do not substitute generic Gen 9 OU or Scarlet/Violet Battle Stadium rules.
- Champions uses its stat-point system (66 total, 32 per stat), level 50, six choose three, and one Mega. Tera, Z-Moves, and Dynamax are not enabled. Showdown stores Champions stat points in the `evs` field; these are not ordinary 252-EV spreads.
- Offline turn caps are implemented; cartridge decision clocks and total time limits are not simulated. Simulator accuracy depends on the pinned mod. Rare public-state effects and the legacy adapter remain areas for further differential testing.
- `opponents/autoplay_reference/SOURCE_MANIFEST.json` identifies the original files and snapshot hashes. Imports and runtime paths were namespaced; only pure state helpers were extracted from the original perception layer. No OCR/device code is used. `tools/import_reference.py` refreshes that snapshot only when explicitly run with a source directory.
- [Jaxcalibur](https://jaxcalibur.github.io/) informed the entity transformer, history, opponent-action auxiliary task, league training, and evaluation approach. Its article describes a different format and much larger training scale. This project uses a CPU Showdown simulator with GPU learning, not its JAX simulator, search implementation, code, or weights. Beating it has not been demonstrated.
