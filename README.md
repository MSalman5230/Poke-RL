# Champions RL

An offline battle bot for **Pokémon Champions, Regulation M-C, Battle Stadium Singles**. It plays complete battles programmatically: six-Pokémon preview, select three and a lead, moves, switches, forced replacements, and Mega Evolution. No account, battle server, OCR, tablet, or controller is involved.

The method is **knowledge-guided reinforcement learning with imitation pretraining**. Game facts become structured inputs and pretrained embeddings; demonstrations teach an initial policy; masked PPO improves the policy through battles against a mixture of opponents and frozen versions of itself.

**A working training system is not a verified strong player.** Read the actual evaluation reports under `runtime_data/`. The program only marks the requested 60% target as passed after a separate test benchmark passes the checks below. There is no measured claim of beating Jaxcalibur.

## Use the installed environment

From this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m champions_rl status
.\.venv\Scripts\python.exe -m champions_rl play --checkpoint runtime_data\bc_v6_final.pt --opponent balanced
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

   `data/knowledge.json` contains the restricted species, legal learnsets, abilities, items, types, natures, and move properties. `data/sets.json` contains 2,189 simulator-validated example sets covering all 269 selectable species/forms in this snapshot. There are 366 forms including battle-only transformations, representing 231 National Dex numbers, and 511 learnable/forced moves. These are all **eligible facts**, not an exhaustive enumeration of every possible legal competitive set.

   The library includes 50 authored complete strategy templates. Mixed sampling selects coherent rain, sun, sand, snow, balance, or offense teams 65% of the time, with broader coverage otherwise. Synthetic fallback sets have obvious item/move/investment contradictions repaired; they are not claimed to be tournament sets. Team identities ignore roster and move-slot permutations. Train/validation/test contain 128/64/128 disjoint compositions; `validation_competitive` adds 64 curated-mixture compositions. Existing benchmark manifests survive `prepare`. Individual species and sets can overlap.

2. **Collect demonstrations from several policies.** Each completed game is a durable shard. The current feature schema is v6. Its fingerprint covers the knowledge, set priors, encoder, public-state parser, and native damage implementation. Older encoded vectors are incompatible. Use `runtime_data/demos_v6_final` or collect fresh data:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl collect --teacher balanced --games 1000 --seed 101 --output runtime_data\my_demos
   .\.venv\Scripts\python.exe -m champions_rl collect --teacher setup --games 1000 --seed 201 --output runtime_data\my_demos
   .\.venv\Scripts\python.exe -m champions_rl collect --teacher defensive --games 500 --seed 301 --output runtime_data\my_demos
   .\.venv\Scripts\python.exe -m champions_rl collect --teacher reference --games 100 --seed 401 --output runtime_data\my_demos
   ```

   The reference teacher uses the original scoring, opponent predictions, 32 seeded simulations, and prepared damage/speed preview cache. Its demonstrations use its saved six on both sides. Other teachers use diverse teams. Reference collection is much slower than the simple opponents; degraded fallback decisions are rejected, not silently counted as the original bot.

3. **Load domain knowledge and imitate.** The first phase learns species→types and move→properties in the embedding tables. Behavior cloning then learns legal actions from the demonstrations, giving reference examples 50% of the sampling mass when both expert and scripted data exist. Whole games are held out. Step zero is a candidate, and checkpoint selection uses the same 50/50 reference-versus-other weighting in validation. Per-teacher losses and agreement are reported. A better imitation score still requires paired battle validation before promotion:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl pretrain --data runtime_data\my_demos --output runtime_data\my_bc.pt --steps 5000
   ```

4. **Improve through battles.** The 574,786-parameter transformer (two layers, width 128, four attention heads) has policy, public-value, and semantic opponent-move/switch heads. A separate 428,841-parameter critic sees both actual teams only during training. Its weights and private inputs are excluded from deployed policy checkpoints. PPO uses legal-action masking, undiscounted match rewards, GAE, and entropy regularization. Use the parallel trainer:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl train-fast --initial runtime_data\my_bc.pt --steps 2000000 --workers 16 --envs 256 --horizon 16 --rollout-decisions 32768 --batch-size 512 --learning-rate 0.00003 --output runtime_data\my_run
   ```

   The central learner batches neural decisions from parallel CPU battle workers. It freezes policy weights while accumulating at least 32,768 decisions per update (configurable with `--rollout-decisions`; the final requested batch may be smaller). Defaults are 10% current-policy self-play, 10% snapshots, and 80% scripted opponents. Sampling shifts toward difficult scripted styles while retaining coverage. The league keeps the imitation anchor, difficult older opponents, and four recent snapshots rather than only the latest 12. These difficulty estimates come from training matches, not an independent strength ranking.

   Current-policy self-play learns from **both players' decisions**, with separate observations, rewards, and advantage calculations for each seat. Waiting phases contribute value targets and rewards but no policy loss. `--steps` counts actual learning-player decisions, not games or waiting phases; the final rollout can round it up. A self-play win rate is not a strength metric and is not included in opponent win-rate summaries.

   Actor and critic have independent optimizers. Actor KL protection uses up to 4,096 representative positions; stopping the actor does not stop the critic from learning from every rollout record. Defaults are 512-position minibatches, four epochs, actor learning rate 3e-5 and critic learning rate 3e-4. Logs include unique sample coverage, completed epochs, and the triggering KL. Completed games supply terminal-result targets to the value heads; unfinished games retain bootstrap targets. GAE still supplies policy advantages.

   The fast script pool contains balanced, hyperoffense, setup_sweeper, attrition, pivot, and a small random fraction. Near-duplicate legacy styles remain available for old benchmarks. Repeat `--league-checkpoint PATH` to seed the league with compatible frozen policies. Every ten updates, a separate validation process runs 256 paired games per style and against the frozen starting policy, plus 256 saved-roster games. Numbered `critic_*.pt` files match numbered actor checkpoints. Queued evaluations survive trainer restarts and preserve interrupted reports.

   An optional separate `--reference-workers 1 --reference-weight 0.1` process plays the full original 32-trial bot using its saved roster. Its model is refreshed between complete games; fast workers never wait for these games. Completed games enter a bounded replay buffer and train a small additional [self-imitation learning loss (Oh et al., 2018)](https://proceedings.mlr.press/v80/oh18b.html), reinforcing the learner’s own actions whose returns exceed its value estimate. This is not expert teaching; it defaults off in the new training recipe. These delayed episodes are **off-policy** and never enter the fresh PPO ratio calculation. Truncated reference games do not supply Monte Carlo targets. `--reference-weight 0.1` controls this loss; `--reference-workers 0` disables new reference games. The number of reference games depends on their throughput, rather than being a fixed fraction of fast games. Their training wins are not an independent benchmark.

   In the fast pool, 15% of battles use the familiar saved roster against other styles or neural opponents. For the rest, 80% of team selections are freshly generated legal compositions and 20% come from the training manifest; exact held-out compositions and three designated team cores are excluded. `validation_cores` contains 72 teams around those cores. This exclusion applies prospectively: transferred older weights may have encountered similar cores previously. Species, moves, items, abilities, natures, and stat-point spreads vary through the legal set library described above. The earlier serial `train` command remains available for old experiments.

   `--heuristic-weight 2` optionally adds a fixed tactical logit prior and learns a residual policy around it. Both rollout action probabilities and PPO updates include the same prior, and checkpoints record its strength. This is an experimental hybrid variant, not a proven improvement over the pure neural policy.

   To stop cleanly, create the configured stop file. To resume, remove it and load the trainer state:

   ```powershell
   New-Item -ItemType File runtime_data\STOP
   Remove-Item -LiteralPath runtime_data\STOP
   .\.venv\Scripts\python.exe -m champions_rl train-fast --resume runtime_data\my_run\trainer_state.pt --steps 1000000 --workers 16 --envs 128 --output runtime_data\my_run
   ```

   New trainer states preserve weights, optimizer, RNG streams, and the frozen opponent league, then start fresh battles. Keep the output directory to reload its durable reference replay shards. In-progress games are not restored, so this is not a bit-for-bit continuation. Changing the worker count starts new worker RNG streams. Early experiment checkpoints predate league/RNG persistence and restart those components from the supplied seed. `--steps` on resume means **additional** decisions. A STOP file applies to all trainers using that path; use distinct `--stop-file` paths for separate runs. New models preserve public move order, actor identity, PP use, status progress, and effect ages. Feature-v2 weights remain archived and require their frozen implementation. New checkpoints and demonstration shards fingerprint the knowledge and set-prior snapshots. Writes flush data before atomic replacement to reduce loss after a restart.

5. **Select using validation only.** Evaluate immutable numbered checkpoints; compare stochastic and deterministic play on validation before choosing deployment settings:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl evaluate --checkpoint runtime_data\my_run\latest.pt --opponents balanced,hyperoffense,setup_sweeper,attrition,pivot --games 256 --workers 4 --deterministic --split validation --output runtime_data\my_validation
   ```

   Add `--deterministic` to use the highest-probability legal action. Each report records the setting, checkpoint hash, source hash, teams, seeds, and policy provenance. `evaluated_model.pt` freezes the exact evaluated weights. Do not use test results to repeatedly tune the model.

   Run at least 128 reference games separately. Repeat script evaluations with `--roster saved` to isolate skill on the deployment roster, and with `--split validation_cores` to test the excluded team cores. `--frozen-opponent frozen_NAME=PATH` adds a frozen model named in `--opponents`. Compare completed identical schedules with `tools/compare_evaluations.py --baseline FOLDER --candidate FOLDER --output FILE`; its bootstrap keeps both seats and all opponents within each seed block.

6. **Run the final frozen benchmark.** Once validation is good enough:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl evaluate --checkpoint runtime_data\my_validation\evaluated_model.pt --opponents reference,balanced,hyperoffense,setup_sweeper,attrition,pivot --games 256 --workers 4 --deterministic --split test --output runtime_data\final_test
   ```

   This suite includes the original reference and five distinct scripted styles. Against the original, both players use its saved six, so a counter-team does not manufacture an advantage. Seats alternate and matchup seeds are paired. Other styles use the held-out team manifest. Original-roster results generalize to that roster and matchup distribution; they do not prove superiority on every possible team.

   Parallel evaluation assigns disjoint game IDs to independent processes and sums game counts. Each pair holds the teams and engine seed fixed while swapping the policies, so the learner controls each team once. Checkpoint/source/data hashes must agree across workers. Public logs, per-game reference trial audits, and a snapshot of the evaluated implementation are retained alongside reports. Four evaluation processes and the reference bot's four probe workers per process suit a many-core CPU; use `--workers 1` on a smaller machine. This does not reduce the reference policy's 32 trials.

   Passing requires at least 200 reference games, at least 60% wins **and a 95% Wilson lower bound of at least 60%**, at least four other styles with 200 games and 55% wins each, no adapter errors or reference fallbacks, and at most 1% truncations. A 60/100 result does not pass this stricter uncertainty check. Errors remain in the denominator; time limits are truncations, not invented wins. The final report explicitly records whether each condition passed.

## Expert corrections and optional search

The resumable collector saves each completed game independently, including compressed raw public observations, and records failed strict-reference attempts separately. Repeating the same command skips completed shards. Full reference decisions retain all 32 simulations; unsupported varied-team preview states are recorded as errors and never become labels. `--teams train` varies the reference expert's own team as well as its opponent; `--teams cross` keeps the expert's own team fixed. `python -m champions_rl.reencode` rebuilds features from recorded public states and recomputes scripted labels after mechanics changes. It preserves reference expert labels and the outcomes of the original behavior trajectories.

```powershell
.\.venv\Scripts\python.exe -m champions_rl.expert_data --output runtime_data\my_demos --games 2048 --teacher mixture --teams train --workers 6 --seed 91001
.\.venv\Scripts\python.exe -m champions_rl.expert_data --output runtime_data\my_demos --games 128 --teacher reference --teams saved --workers 4 --seed 51001
.\.venv\Scripts\python.exe -m champions_rl.expert_data --output runtime_data\my_demos --games 96 --teacher reference --teams saved --workers 4 --seed 81001 --student runtime_data\my_bc.pt --beta 0.25
.\.venv\Scripts\python.exe -m champions_rl pretrain --data runtime_data\my_demos --initial runtime_data\my_bc.pt --output runtime_data\my_dagger.pt
```

The third command follows [DAgger’s dataset aggregation approach](https://proceedings.mlr.press/v15/ross11a.html): the expert labels the learner's current position; the learner normally acts, with a 25% expert-action mixture. Stored labels are the expert's preferred action, even when the learner chooses differently. `--opponents reference` visits positions against the full original opponent too. Metadata records disagreements and the student checkpoint hash.

For a specific observed failure, the collector also accepts a script name such as `--opponents balanced` and `--student-deterministic`. Aggregate separate data folders with `pretrain --data ORIGINAL_FOLDER CORRECTION_FOLDER`. Optional `--correction-mass 0.25` assigns 25% of training and validation weight to teacher labels on learner-visited states; the remaining mass is divided equally between original reference and other demonstrations. It defaults to zero, retaining the earlier teacher mixture. This is a candidate experiment: correcting one roster can reduce performance elsewhere, so evaluate both before promotion.

`play` defaults to deterministic actions; `--no-deterministic` samples the policy. Evaluation requires an explicit `--deterministic` for that mode. Compare both on identical seeds and teams, including `--split validation_competitive`.

```powershell
.\.venv\Scripts\python.exe -m champions_rl.value_audit --checkpoint runtime_data\my_run\latest.pt --critic-state runtime_data\my_run\trainer_state.pt --output runtime_data\value_audit --games 256
.\.venv\Scripts\python.exe -m champions_rl evaluate --checkpoint runtime_data\my_run\latest.pt --critic-state runtime_data\my_run\trainer_state.pt --opponents balanced --games 256 --workers 4 --deterministic --search --output runtime_data\search_validation
```

Search remains off by default. It considers the top three options, 16 sampled hidden worlds, two semantic-head replies per world, and one simulated turn. The private critic scores each hypothetical complete state; it never receives the opponent's actual hidden state. Candidate comparisons use common random seeds and a paired confidence margin across complete sampled worlds, with a correction for multiple candidates. Uncertain comparisons keep the policy's choice. This confidence check does not establish a win-rate improvement.

Mechanics hydration is adapted from the frozen reference into `simulator/hypothetical.cjs`; the original snapshot is unchanged. Unknown durations and hidden trios remain hypotheses; unsupported states fall back to the policy with recorded diagnostics. Use matching actor and critic snapshots, first audit held-out terminal-result prediction by turn, then compare search against the plain policy on the same battle schedule. Search teaching is blocked without a certificate for the exact actor/critic hashes showing a positive paired improvement over at least 256 games. No such certificate has been established. The earlier value audit showed weak early-game predictions, so additional search distillation is deferred.

The recovered paired baseline is in `benchmarks/2026-10-29-scripted-round-robin.json`: 1,792 games using the saved-roster mirror, no invalid final games. The reference ranked first, then defensive. This measures that matchup distribution and seed sensitivity, not a universal luck ceiling. Recovery preserved invalid attempts and reran the same interrupted seeds.

## Knowledge, features, and rewards

`data/feature_schema.json` lists every input. Entity features include species/type, known own stats, estimated enemy stats with knowledge flags, HP, status, boosts, PP, revealed moves/items/abilities, field conditions, and recent public move history. Candidate features include approximate damage, KO margin, accuracy, priority, recovery, setup, hazards, switches, Mega options, and preview coverage. Contextual Pokémon embeddings connect each candidate to its actor and selected team.

Unknown opponents use compatible complete-set hypotheses weighted by usage, conditioned on revealed moves/items/abilities. Revealing one status move retains plausible unrevealed attacks; unknown stats use invested spreads and natures. Native tactical queries enumerate damage rolls for hypothetical sets and expose KO, survival-risk, and effective-order hints. Ordinary random critical hits are omitted, multi-hit rolls are correlated approximations, and survival/order features are not complete joint-turn probabilities. Public ages and unknown effect durations remain estimates. The native simulator resolves the actual battle. The full legacy catalog is retained only inside the isolated reference adapter for compatibility; the learned model's vocabulary is the restricted Champions export.

The native calculator restores public hit counters, cumulative faint counts, boosts, and supported effects. Tests compare actual simulator move resolution for Disguise, Rage Fist, Substitute, survival items/abilities, and conditional moves. Unsupported estimates carry a reliability flag. Compatible move queries share prepared matchup state; mutable state is restored between rolls. Raw distributions are cached without HP only where mechanics permit, and Python recomputes HP-dependent KO/survival summaries. HP-sensitive moves, abilities/items, multi-hit effects, Mega forms, and revival history retain their required dependencies. A switch reliability flag remains false because entry effects are not a full switch simulation.

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

Tests cover restricted legality and every generated set/team, hidden-information separation, legal masks, deterministic replay, trapping retries, Illusion, revival, Gym operation, reward telescoping, truncation bootstrapping, dual-seat parallel trajectories, batched inference, replay-loss gradients, checkpoint/resume, process cleanup, and the acceptance gate.

## Review-cycle measurements

The corrected native-feature pipeline achieved 358, 410, and 441 decisions/s with 12, 16, and 24 workers respectively in short cold-cache probes using 256 battles and two PPO epochs. The fastest measured whole-process rate was about 407 decisions/s. These probes preceded the final cumulative-faint cache fix; concurrent evaluation and the four-epoch full run have different costs. The requested 1,000 decisions/s target has **not** been demonstrated. Changes to both features and worker configuration prevent attributing the improvement to caching alone.

The rebuilt v6 demonstration set contains 13,337 decisions from 1,175 valid games, including 1,545 reference decisions. Reference own-team variation produced 55 valid games and nine strict preview failures; failed games were excluded. Re-encoding corrected 36 scripted labels. Balanced held-out imitation loss improved from 2.462 at step zero to 1.453 at step 201, which was saved rather than the later overfit checkpoint. These losses measure teacher agreement, not battle strength.

The long run in `runtime_data/ppo_v6_full` requests two million decisions with a separate private critic and frozen paired validation every ten updates. Use its metrics and completed evaluation reports for current progress; an incomplete run is not evidence of achieving the playing-strength target. The original reference and old checkpoints remain independent yardsticks. Search is still experimental and no search labels were added in this review cycle.

At the frozen 355,544-decision checkpoint, PPO won 254/256 saved-roster games against balanced, compared with 29/256 for its starting imitation policy. Across the five common scripts on varied validation teams, it improved by 3.2 percentage points (paired bootstrap interval +0.8 to +5.7 points). That varied-team score remains close to the older transferred anchor. These are exploratory validation results from repeated checkpoint selection, not a final certification. The initial v6 imitation candidate won 63/128 strict reference games with no invalid games; the older anchor won 55/128 after five CPU-starved games were retried on their original seeds, with the failed attempts preserved. The stronger reference acceptance gate has not been established by these results.

The 178,392-decision critic audit also exposed a generalization gap: private-value explained variance was +0.115 on new battles using training-manifest teams and -0.237 on validation-manifest teams. These audits contain 256 games each and use the same frozen actor/critic pair. The private critic's low optimization loss is insufficient evidence that search can rank unseen continuations reliably. Detailed results and limitations are recorded in `benchmarks/2026-10-29-v6-review.json`.

## Historical feature-v2 speed measurements

These archived results predate native tactical features and the private critic; they are not current throughput claims. Measured on 2026-10-29 with an i9-13900K and RTX 4070, using the same 512,833-parameter starting checkpoint. Each entry is the median of two sequential runs with different seeds; each run requests 20,000 decisions for 64 battles or 40,000 for 128 battles (rounded up to a complete rollout):

| Trainer | CPU workers | Fast battles | Background reference workers | Learning decisions/s |
| --- | ---: | ---: | ---: | ---: |
| Serial | 1 | 64 | 0 | 357 |
| Parallel | 1 | 64 | 0 | 477 |
| Parallel | 2 | 64 | 0 | 777 |
| Parallel | 4 | 64 | 0 | 1,097 |
| Parallel | 8 | 64 | 0 | 1,467 |
| Parallel | 8 | 128 | 1 | 1,675 |
| Parallel | 12 | 128 | 1 | 1,868 |
| Parallel | 16 | 128 | 1 | 2,025 |

Eight workers with 64 battles provide **4.1×** the serial training throughput, or **3.6×** when process startup and shutdown are included. The 16-worker / 128-battle configuration is the fastest tested, including a concurrent full reference opponent. These are short runs; throughput changes with battle lengths, retained snapshots, PPO early stopping, and other machine activity. Full results and provenance are saved in [64-battle measurements](benchmarks/2026-10-29-workers64.json) and [128-battle measurements with reference games](benchmarks/2026-10-29-workers128-reference.json).

A [longer integration run](benchmarks/2026-10-29-sustained-reference.json) resumed from 160,070 fast-training decisions and added 204,211 in 96.14 seconds (about 2,124 decisions/s). It successfully retained 12 league snapshots after creating 18, and its two phases completed 17 full reference games with 161 audited 32-trial decisions, no fallbacks, and no errors. That legacy checkpoint remains archived at `runtime_data/parallel_reference_smoke/latest.pt`; continuing it requires its frozen feature-v2 implementation.

Speed is the improvement established here. A separate deterministic validation of that checkpoint ran 200 held-out games per style: damage 48.5%, balanced 49%, setup 51.5%, defensive 50%, stochastic 56%, with zero errors or truncations. The report is retained in `runtime_data/validation_parallel_fast/`. This does not establish the 60% original-bot target or superiority to Jaxcalibur. Reference training-game wins are not a substitute for an independent frozen evaluation.

Run one benchmark suite at a time, with other training/evaluation jobs stopped:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_training.py --initial runtime_data\bc_v6_final.pt --output runtime_data\speed_check --steps 32768 --workers 12,16,24 --envs 256 --repeats 2
```

Each run starts a fresh process from the same weights. `report.json` records source/checkpoint hashes, seeds, hardware, training throughput, whole-process time, inference batch sizes, and rollout/optimization time. The output folder must be new so previous measurements cannot be mixed into a run. Add `--reference-workers 1` to measure concurrent original-bot work. Use an existing compatible checkpoint for `--initial` if the local experiment checkpoint is unavailable.

Parallel self-play supplies two learner trajectories; the serial trainer supplies one. Their default opponent mixtures differ, so this compares training systems, not identical battle trajectories or playing strength. `--workers` controls CPU processes; `--envs` is the total number of concurrent fast battles across those processes. Increasing workers indefinitely can reduce performance through smaller inference batches and contention. GPU learning remains centralized, and no CUDA toolkit reinstall is needed to change these settings.

## Scope and provenance

- The engine is Pokémon Showdown's Champions mod at `c0845f33d381402dfcb18ce324c20a94f18af075`, format `gen9championsbssregmc`. See the [pinned implementation](https://github.com/smogon/pokemon-showdown/tree/c0845f33d381402dfcb18ce324c20a94f18af075/data/mods/champions) and [official M-C announcement](https://champions-news.pokemon-home.com/en/page/816.html). Do not substitute generic Gen 9 OU or Scarlet/Violet Battle Stadium rules.
- Champions uses its stat-point system (66 total, 32 per stat), level 50, six choose three, and one Mega. Tera, Z-Moves, and Dynamax are not enabled. Showdown stores Champions stat points in the `evs` field; these are not ordinary 252-EV spreads.
- Offline turn caps are implemented; cartridge decision clocks and total time limits are not simulated. Simulator accuracy depends on the pinned mod. Rare public-state effects and the legacy adapter remain areas for further differential testing.
- `opponents/autoplay_reference/SOURCE_MANIFEST.json` identifies the original files and snapshot hashes. Imports and runtime paths were namespaced; only pure state helpers were extracted from the original perception layer. No OCR/device code is used. `tools/import_reference.py` refreshes that snapshot only when explicitly run with a source directory.
- [Jaxcalibur](https://jaxcalibur.github.io/) informed the entity transformer, history, opponent-action auxiliary task, league training, and evaluation approach. Its article describes a different format and much larger training scale. This project uses a CPU Showdown simulator with GPU learning, not its JAX simulator, search implementation, code, or weights. Beating it has not been demonstrated.
