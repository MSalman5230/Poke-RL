# Champions RL

An offline battle bot for **Pokémon Champions, Regulation M-C, Battle Stadium Singles**. It plays complete battles programmatically: six-Pokémon preview, select three and a lead, moves, switches, forced replacements, and Mega Evolution. No account, battle server, OCR, tablet, or controller is involved.

The method is **knowledge-guided reinforcement learning with imitation pretraining**. Game facts become structured inputs and pretrained embeddings; demonstrations teach an initial policy; masked PPO improves the policy through battles against a mixture of opponents and frozen versions of itself.

The frozen two-million-decision policy won **255/256 independent test games against the original bot on its saved roster**. The full acceptance gate still failed: four scripted styles on varied teams remained below 55%. Generalization needs more work, and no comparison against Jaxcalibur has been measured. See the results and limits below.

## Use the installed environment

From this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m champions_rl status
.\.venv\Scripts\python.exe -m champions_rl play --checkpoint runtime_data\candidates\v6_2m_review\policy.pt --legacy-runtime runtime_data\baselines\v6_3cc5468 --opponent balanced
```

`play` writes a complete public Showdown protocol log and a JSON result to `runtime_data/play/`. A neural policy receives only its own player-visible state. An opponent's actual moveset, items, selected trio, and pending action are not inputs.

The source now uses **feature version 8**. The tested v6 policy requires the hash-verified frozen runtime shown above; v7 weights require `--legacy-runtime runtime_data/baselines/v7_e3e2d8a`. Neither version is silently loaded into a different encoder. `runtime_data/v8_pilot_warm_start.pt` is a compatible migration for experiments, not a promoted policy. Its new disguise-order weights start at zero and need training.

**Training is paused.** The saved user instruction in `runtime_data/PAUSED_BY_USER.json` requires explicit permission to resume training; a general continuation does not suffice. `runtime_data/STOP` remains in place. The longer v7 run stopped at 1,259,684 cumulative decisions, update 36. Its checkpoint and optimizer state are preserved; do not resume that state into the v8 encoder.

The completed run is `runtime_data/ppo_v6_full/`. `latest.pt` is an inference checkpoint; `trainer_state.pt` also includes both optimizers, RNG streams, and the frozen opponent league. The candidate folder preserves the tested policy and its result separately. Training logs contain **training** win rates, which are not benchmark results.

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

   The library includes 50 authored complete strategy templates. The v7 `diverse` generator selects usage-teammate compositions 60% of the time, the earlier mixed generator 20%, broad species coverage 10%, and usage-ranked teams 10%. It keeps complete sets intact, checks item/species constraints, and accounts for weather dependencies and team coverage. These are synthetic plausible teams, not recovered tournament rosters. Team identities ignore roster and move-slot permutations. Train/validation/test contain 128/64/128 disjoint compositions; `validation_competitive` adds 64 curated-mixture compositions. The separate `test_v7` manifest reserves 128 fresh teams for a future frozen candidate. Existing benchmark manifests survive `prepare`. Individual species and sets can overlap; held-out three-Pokemon cores are excluded from new training compositions.

2. **Collect demonstrations from several policies.** Each completed game is a durable shard. The current feature schema is v8. Its fingerprint covers the knowledge, set priors, encoder, public-state parser, and native damage implementation. Older encoded vectors are incompatible. Use the rebuilt `runtime_data/demos_v8` or collect fresh data:

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

4. **Improve through battles.** The v8 transformer has **629,949 parameters** (two layers, width 128, four attention heads), with policy, public-value, and semantic opponent-move/switch heads. A separate **435,369-parameter critic** sees both actual teams only during training. Its weights and private inputs are excluded from deployed policy checkpoints. PPO uses legal-action masking, undiscounted match rewards, GAE, and entropy regularization. Use the parallel trainer:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl train-fast --initial runtime_data\my_bc.pt --steps 2000000 --workers 16 --envs 256 --horizon 16 --rollout-decisions 32768 --batch-size 512 --learning-rate 0.00003 --output runtime_data\my_run
   ```

   The central learner batches neural decisions from parallel CPU battle workers. It freezes policy weights while accumulating at least 32,768 decisions per update (configurable with `--rollout-decisions`; the final requested batch may be smaller). Defaults are 10% current-policy self-play, 10% snapshots, and 80% scripted opponents. Sampling shifts toward difficult scripted styles while retaining coverage. The league keeps the imitation anchor, difficult older opponents, and four recent snapshots rather than only the latest 12. These difficulty estimates come from training matches, not an independent strength ranking.

   The current recipe uses the public value head for PPO advantages by default. The private critic trains separately at learning rate `1e-4` for two passes, using bootstrapped targets. Logs include both heads' pre-update explained variance on completed-game returns. Private training loss alone does not justify switching the baseline. Training-only identity dropout masks 25% of species/move/item/ability identities consistently within each observation; PPO records retain the exact masked input used to sample the action and old probability. Inference uses all available identities.

   Current-policy self-play learns from **both players' decisions**, with separate observations, rewards, and advantage calculations for each seat. Waiting phases contribute value targets and rewards but no policy loss. `--steps` counts actual learning-player decisions, not games or waiting phases; the final rollout can round it up. A self-play win rate is not a strength metric and is not included in opponent win-rate summaries.

   Actor and critic have independent optimizers. Actor KL protection uses up to 4,096 representative positions; stopping the actor does not stop the critic from learning from every rollout record. Defaults are 512-position minibatches, four epochs, actor learning rate 3e-5 and critic learning rate 3e-4. Logs include unique sample coverage, completed epochs, and the triggering KL. Completed games supply terminal-result targets to the value heads; unfinished games retain bootstrap targets. GAE still supplies policy advantages.

   The fast script pool contains balanced, hyperoffense, setup_sweeper, attrition, pivot, and a small random fraction. Near-duplicate legacy styles remain available for old benchmarks. Repeat `--league-checkpoint PATH` to seed the league with compatible frozen policies. Every ten updates, a separate validation process runs 256 paired games per style and against the frozen starting policy, plus 256 saved-roster games. Numbered `critic_*.pt` files match numbered actor checkpoints. Queued evaluations survive trainer restarts and preserve interrupted reports.

   An optional separate `--reference-workers 1 --reference-weight 0.1` process plays the full original 32-trial bot using its saved roster. Its model is refreshed between complete games; fast workers never wait for these games. Completed games enter a bounded replay buffer and train a small additional [self-imitation learning loss (Oh et al., 2018)](https://proceedings.mlr.press/v80/oh18b.html), reinforcing the learner’s own actions whose returns exceed its value estimate. This is not expert teaching; it defaults off in the new training recipe. These delayed episodes are **off-policy** and never enter the fresh PPO ratio calculation. Truncated reference games do not supply Monte Carlo targets. `--reference-weight 0.1` controls this loss; `--reference-workers 0` disables new reference games. The number of reference games depends on their throughput, rather than being a fixed fraction of fast games. Their training wins are not an independent benchmark.

   In the fast pool, 15% of battles use the familiar saved roster against other styles or neural opponents. For the rest, 80% of team selections are freshly generated legal compositions and 20% come from the training manifest; exact held-out compositions and three designated team cores are excluded. `validation_cores` contains 72 teams around those cores. This exclusion applies prospectively: transferred older weights may have encountered similar cores previously. Species, moves, items, abilities, natures, and stat-point spreads vary through the legal set library described above. The earlier serial `train` command remains available for old experiments.

   `--heuristic-weight 2` optionally adds a fixed tactical logit prior and learns a residual policy around it. Both rollout action probabilities and PPO updates include the same prior, and checkpoints record its strength. This is an experimental hybrid variant, not a proven improvement over the pure neural policy.

   To stop cleanly, create the configured stop file. After explicit permission to resume, remove it and load a feature-compatible trainer state. The current v7 pause must not be cleared implicitly:

   ```powershell
   New-Item -ItemType File runtime_data\STOP
   Remove-Item -LiteralPath runtime_data\STOP
   .\.venv\Scripts\python.exe -m champions_rl train-fast --resume runtime_data\my_run\trainer_state.pt --steps 1000000 --workers 16 --envs 128 --output runtime_data\my_run
   ```

   New trainer states preserve weights, optimizer, RNG streams, and the frozen opponent league, then start fresh battles. Keep the output directory to reload its durable reference replay shards. In-progress games are not restored, so this is not a bit-for-bit continuation. Changing the worker count starts new worker RNG streams. Early experiment checkpoints predate league/RNG persistence and restart those components from the supplied seed. `--steps` on resume means **additional** decisions. A STOP file applies to all trainers using that path; use distinct `--stop-file` paths for separate runs. New models preserve public move order, actor identity, PP use, status progress, and effect ages. Feature-v2 weights remain archived and require their frozen implementation. New checkpoints and demonstration shards fingerprint the knowledge and set-prior snapshots. Writes flush data before atomic replacement to reduce loss after a restart.

   On Windows, parallel training and battle evaluation hold a [temporary system power request](https://learn.microsoft.com/en-us/windows/desktop/api/winbase/nf-winbase-setthreadexecutionstate) while running, then restore the calling thread's state. This prevents automatic idle sleep without changing the power plan or keeping the display on; explicit user sleep still works. New training phases report both wall time and system-awake time, since sleep can invalidate reference deadlines and distort throughput. Earlier phases did not measure awake time separately.

5. **Select using validation only.** Evaluate immutable numbered checkpoints; compare stochastic and deterministic play on validation before choosing deployment settings:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl evaluate --checkpoint runtime_data\my_run\latest.pt --opponents balanced,hyperoffense,setup_sweeper,attrition,pivot --games 256 --workers 4 --deterministic --split validation --output runtime_data\my_validation
   ```

   Add `--deterministic` to use the highest-probability legal action. Each report records the setting, checkpoint hash, source hash, teams, seeds, and policy provenance. `evaluated_model.pt` freezes the exact evaluated weights. Do not use test results to repeatedly tune the model.

   Run at least 128 reference games separately. Repeat script evaluations with `--roster saved` to isolate skill on the deployment roster, and with `--split validation_cores` to test the excluded team cores. `--frozen-opponent frozen_NAME=PATH` adds a frozen model named in `--opponents`. Compare completed identical schedules with `tools/compare_evaluations.py --baseline FOLDER --candidate FOLDER --output FILE`; its bootstrap keeps both seats and all opponents within each seed block.

6. **Run the final frozen benchmark.** Once validation is good enough:

   ```powershell
   .\.venv\Scripts\python.exe -m champions_rl evaluate --checkpoint runtime_data\my_validation\evaluated_model.pt --opponents reference,balanced,hyperoffense,setup_sweeper,attrition,pivot --games 256 --workers 4 --deterministic --split test --seed 190731 --output runtime_data\final_test
   ```

   This suite includes the original reference and five distinct scripted styles. Against the original, both players use its saved six, so a counter-team does not manufacture an advantage. Seats alternate and matchup seeds are paired. Other styles use the held-out team manifest. Original-roster results generalize to that roster and matchup distribution; they do not prove superiority on every possible team.

   Parallel evaluation assigns disjoint game IDs to independent processes and sums game counts. Each pair holds the teams and engine seed fixed while swapping the policies, so the learner controls each team once. Checkpoint/source/data hashes must agree across workers. Public logs, per-game reference trial audits, and a snapshot of the evaluated implementation are retained alongside reports. Four evaluation processes and the reference bot's four probe workers per process suit a many-core CPU; use `--workers 1` on a smaller machine. This does not reduce the reference policy's 32 trials.

   Passing requires at least 200 reference games, at least 60% wins **and a 95% Wilson lower bound of at least 60%**, at least four other styles with 200 games and 55% wins each, no adapter errors or reference fallbacks, and at most 1% truncations. A 60/100 result does not pass this stricter uncertainty check. Errors remain in the denominator; time limits are truncations, not invented wins. The final report explicitly records whether each condition passed.

Test evaluation defaults to seed 190731; validation defaults to 92341. The reference mirror uses the same saved teams in both splits, so a new split name alone does not make those battles independent. When supplying custom seeds, reserve non-overlapping seed ranges for validation and test, and freeze the candidate before inspecting test results.

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

The saved-roster balanced-only correction experiment demonstrated that risk: 25% correction mass raised wins against balanced from 29/256 to 102/256, but reduced reference wins from 63/128 to 21/128. Its paired reference difference was -32.8 points (95% interval -41.4 to -24.2). That checkpoint was rejected and never fed into the main PPO run. The interrupted reference evaluation was recovered on its original schedule, retaining invalid attempts; it is validation evidence, not a clean test certification.

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

### V8 regression fixes and the next decision

The public parser now tracks uncertainty about the active opponent's identity separately from previously revealed abilities. A real Corviknight's earlier Pressure reveal no longer rules out a later Zoroark disguise. The simulator regression switches Corviknight out, brings in a disguised Zoroark, then verifies that the legal return to real Corviknight remains representable by the opponent-action head. Masked auxiliary targets raise an error; PPO logs policy, public-value, private-value, and opponent losses separately.

Preview features include the potential Illusion disguise target, and preview search keeps reserve orders that produce different disguises. Simulator tests verify `[Zoroark, Corviknight, Tyranitar]` versus `[Zoroark, Tyranitar, Corviknight]`. The migration copies compatible v7 weights explicitly, initializes the new inputs neutrally, and transfers no optimizer. Earlier demonstrations are rebuilt from their public observations; their original files remain intact. The small correction corpus holds out every fifth correction game in chronological order, with an explicit whole-game partition; the original datasets retain their historical hash split.

The agreed promotion targets are **at least 60% of decisive pairs won on both broad teams and held-out cores, at least 60% head-to-head against the strongest available frozen policy, and no saved-team regression**, with uncertainty reported. A decisive pair is a 2-0 sweep or a 0-2 loss. Reports retain 1-1 splits, draws, errors, and overall win rate; a split does not establish that team luck caused the result. The historical `acceptance` field still describes the older v6 gate and is not sufficient for v8 promotion.

Evaluation supports a different frozen implementation for each neural opponent:

```powershell
.\.venv\Scripts\python.exe -m champions_rl evaluate --checkpoint runtime_data\v8_pilot_warm_start.pt --opponents frozen_v6 --frozen-opponent frozen_v6=runtime_data/candidates/v6_2m_review/policy.pt --frozen-opponent-runtime frozen_v6=runtime_data/baselines/v6_3cc5468 --games 256 --workers 2 --deterministic --output runtime_data\v8_vs_v6
```

The saved-team reference regression is still under investigation. Frozen v6, v7 imitation, the v7 neural pilot, and the paused later policy all choose Gholdengo / Volcarona / Garchomp in the same order in that mirror preview. A changed preview choice does not explain the drop from 126/128 to 110/128. The correction collection therefore queries frozen v6 on positions from fresh losses by the v7 pilot against the strict reference bot. Its demonstrated teaching advantage applies to this saved-roster matchup. These are public-state teacher preferences, not counterfactual proofs that each alternative wins; value targets retain the actual learner outcomes.

The bounded collection completed 128 games, including 19 learner losses, with no invalid games in the resumed collection. It reviewed 160 positions and retained 15 teacher disagreements from 11 games: nine training decisions and six decisions in two held-out games. Ten disagreements were Garchomp choosing Earthquake where v6 preferred Outrage against Volcarona. This is a narrow retention dataset, not broad tactical coverage or 160 verified mistakes. An earlier reference timeout was rejected, and its 26 completed games were preserved during recovery.

`tools/correction_experiment.py` prepares a fixed-budget comparison without training by default. The review decisions, corpus limits, test results, and agreed targets are recorded in [the v8 protocol](benchmarks/2026-10-30-v8-correction-protocol.json). The control retains the rebuilt original demonstrations; the treatment adds the new disagreement labels at 5% sampling mass. Both use the same v8 start, 400 imitation updates, seed, learning rate, batch size, and 25% identity dropout. Evaluation uses their final weights, avoiding different checkpoint-selection rules. The primary schedule contains 2,048 direct games (1,024 seat-swapped pairs); secondary schedules check broad teams, held-out cores, saved-roster mirrors, saved versus varied teams, the original reference, and the best completed v7 broad-validation checkpoint. Frozen v6 is reevaluated on the saved-team validation schedules as the retention anchor. That checkpoint is a practical comparison opponent, not a proven universal champion. No automatic promotion or prospective `test_v7` evaluation occurs.

Terminal search expansion, evidence-conditioned beliefs, human replay imitation, an independently trained exploiter, public-value representation gradients, dropout ablation, a larger model, and a 10M-decision run remain separate experiments. The current correction comparison changes none of them. Search has not earned teaching status, and approximate replay masks must not be represented as exact legality.

### V7 representation and generalization experiments

V7 encodes each move with its own PP, reveal probability, and mechanical attributes, then pools the set independently of arbitrary slot order. Conversion retains its meaningful first-slot flag. Opponent switch predictions score each destination Pokemon with battle context and public eligibility masks. The actor and critic now receive public hit counts, cumulative faints, last move, and Stockpile/Perish state; the critic additionally receives the simulator's private state during training.

The fixed 208-position representation audit found zero changed legal choices under equivalent move reorderings, versus 29 for the frozen v6 model. Maximum remapped probability error was below `6e-7`; opponent switch scores followed permuted destinations. Rage Fist and cumulative faint changes now alter critic inputs. These tests establish representation correctness, not stronger play by themselves.

The v7 dropout imitation candidate scored 734/1,280 on broad validation and 144/256 on held-out cores, versus v6's 717/1,280 and 140/256. Paired differences were +1.3 points (95% interval -2.6 to +5.3) and +1.6 (-5.5 to +8.6). The matched no-dropout candidate scored 721/1,280 and 144/256. Neither the migration nor dropout has yet shown a statistically established battle improvement. `benchmarks/2026-10-30-v7-experiment-protocol.json` records the two 262,144-decision mixture pilots and promotion criteria before their results. The prospective `test_v7` schedule, seed 290731, remains unused.

Both pilots completed eight updates: 262,322 decisions for the 75% scripted / 10% self-play / 15% historical control, and 262,244 for the 50% / 25% / 25% alternative. The same frozen schedules gave:

| Validation schedule | Frozen v6 | V7 control | V7 neural-heavy mixture |
| --- | ---: | ---: | ---: |
| Five scripts, broad teams | 717/1,280 | 697/1,280 | 746/1,280 |
| Balanced, held-out cores | 140/256 | 147/256 | 141/256 |
| Balanced, saved mirror | 254/256 | 254/256 | 253/256 |
| Balanced, saved team versus varied teams | 220/256 | 220/256 | 234/256 |
| Frozen v7 imitation opponent, broad teams | Not measured | 113/256 | 131/256 |

All games in this table completed without errors, fallbacks, or truncations. Relative to the control, the neural-heavy mixture gained 3.8 points on broad teams (paired 95% interval +0.3 to +7.3) and 7.0 against the frozen imitation policy (+1.6 to +12.5), while cores changed by -2.3 points (-8.2 to +3.5). These are exploratory, single-training-seed results; the runs shared hardware and are not an isolated equal-compute comparison. Neither candidate established gains over v6 on both primary schedules, so neither was promoted. That continuation was paused by the user at 1,259,684 cumulative decisions (update 36); its original plan remains in `benchmarks/2026-10-30-v7-stage2-protocol.json`. It is not running, and v8 development does not resume it.

The neural-heavy pilot subsequently won **110/128 reference validation games (85.9%)**, with a Wilson interval of 78.9% to 90.9% and no errors, fallbacks, or truncations. Every audited original-bot decision used 32 trials. This exceeds 60% on the validation schedule but trails v6's historical 126/128 result; it does not pass the full independent-test gate or establish preserved reference strength. V6 remains the default frozen baseline. Details are in `benchmarks/2026-10-30-v7-neural-reference-validation.json`.

On 256 held-out games, the neural-heavy pilot's public/private value explained variance was +0.181/+0.228; sampled play gave +0.172/+0.223 on a separate 256-game trajectory audit. The private head's paired advantage intervals still included zero in both modes. PPO therefore keeps public advantages, and search remains experimental. `tools/audit_value_calibration.py` resamples entire seed blocks when comparing heads. `tools/summarize_validation_families.py` reports descriptive weaknesses by opponent and learner/enemy team family without treating repeated turns or seeds as independent evidence.

The terminal preview-search experiment samples 16 public-compatible worlds and runs all top-three distinct trio/lead options to completion using the frozen policy on both sides. It uses no value head; a failed/capped continuation cancels the comparison. A paired margin is required to override the base choice. On the initial 32-game core trial it won 16 games, exactly matching the plain policy (paired difference interval -9.4 to +9.4 points), despite five overrides and 1,536 continuation games. It stays off, with no search labels added to training. The separate simulator-verified immediate-endgame diagnostic was solved in all three fixtures; this small result is not a battle-strength benchmark.

`python -m champions_rl.replays --input replay.json --output runtime_data/replays` imports offline `gen9championsbssregmc` protocol logs through the existing public-state parser. Ordinary spectator replays yield public evidence and observed action records, but lack full own sets and trustworthy legal masks for policy imitation. A SHA-bound `--companion` can supply independently captured decision frames and explicit complete teams. Only verified move-phase frames become imitation samples; unseen enemy choices remain unlabeled, and observed replies are targets only. Nothing is added to training automatically. Parser/importer fingerprints prevent reuse of stale imports. Beyond simulator-generated regression tests, an initial three-public-replay probe imported 45 observed-action records with zero parser errors, zero complete teams, and zero imitation samples. Its sources and hashes are in `benchmarks/2026-10-30-v7-human-replay-probe.json`; these records have not been used for training.

### Completed V6 run

The run in `runtime_data/ppo_v6_full` completed **2,001,783 policy decisions in 57 PPO updates**. Every update used 100% of the actor and critic samples for four epochs. The representative KL guard remained enabled; no update triggered it. Frozen paired validation ran every ten updates, and the final checkpoint received the same suite. These are observed outcomes, not an isolated causal ablation of each fix.

| Validation schedule | Starting imitation policy | Final PPO policy |
| --- | ---: | ---: |
| Original reference, saved roster | 63/128 (49.2%) | 126/128 (98.4%) |
| Balanced script, saved roster | 29/256 (11.3%) | 254/256 (99.2%) |
| Five scripts, broad team mix | 646/1,280 (50.5%) | 717/1,280 (56.0%) |
| Five scripts, competitive teams | 648/1,280 (50.6%) | 928/1,280 (72.5%) |
| Balanced script, excluded team cores | 110/256 (43.0%) | 140/256 (54.7%) |

The final evaluations above had no errors, fallbacks, or truncations. Paired bootstrap differences were +49.2 points against the reference, +5.5 on broad teams, +21.9 on competitive teams, and +11.7 on excluded cores. The corresponding 95% intervals were +43.0 to +55.5, +1.6 to +9.6, +17.2 to +26.7, and +4.3 to +19.1. Blocks retain both seats and all opponents. Repeated validation selected the candidate; it is not an independent test claim. Reference performance is specifically the saved-roster mirror, and broad-team generalization remains much weaker.

The final actor was frozen at `runtime_data/candidates/v6_2m_review/policy.pt` before independent testing. Its selection and new-seed test plan were committed in `9c30b11` and recorded in `benchmarks/2026-10-30-v6-candidate-selection.json`. At 1.41M decisions, deterministic play beat sampled play by 4.2 points on the competitive schedule (interval +1.2 to +7.3), so deterministic play was retained.

The independent test used seed 190731, disjoint from the validation schedule, with the frozen weights and action mode:

| Opponent | Wins / games | Win rate |
| --- | ---: | ---: |
| Original reference, saved roster | 255/256 | 99.6% |
| Balanced, varied teams | 129/256 | 50.4% |
| Hyperoffense, varied teams | 139/256 | 54.3% |
| Setup sweeper, varied teams | 139/256 | 54.3% |
| Attrition, varied teams | 152/256 | 59.4% |
| Pivot, varied teams | 133/256 | 52.0% |

The reference's 95% Wilson lower bound is 97.8%, passing the reference target. **The full acceptance gate failed** because four styles did not reach 55%. All 1,536 scheduled games completed with zero errors or fallbacks; one attrition game reached the turn cap and remains a draw in the denominator. All 2,484 audited reference decisions used 32 trials. The candidate remains marked `reference_target_met_generalization_gate_failed`; it was not promoted as a generally strong agent. The complete result and hashes are in `benchmarks/2026-10-30-v6-independent-test.json`. No policy changes were made after viewing this test; later model selection must use validation and reserve a fresh independent test.

Search remains disabled. The final 256-game value audit gave public/private explained variance of +0.127/-0.274 on unseen teams. Whole-game cross-fitted shrinkage reduced the private critic's overconfidence but left explained variance at -0.021; no transform was deployed. Audits default to deterministic play; `--no-deterministic` measures the sampled policy used in training. A separate sampled audit also failed to justify search. Low training loss does not establish reliable ranking of alternative continuations, and no search-distillation labels were added.

The rebuilt v6 demonstrations contain 13,337 decisions from 1,175 valid games, including 1,545 reference decisions. Varied expert own teams supplied 55 valid games; nine strict preview failures were excluded. Re-encoding corrected 36 scripted labels. Balanced held-out imitation loss improved from 2.462 at step zero to 1.453 at the selected step 201. Teacher agreement alone was never treated as battle promotion; the failed correction-overweighting experiment is described above.

The final training phase averaged 339 decisions per awake second with 16 CPU workers and four PPO epochs. Short earlier cold-cache probes measured 358/410/441 decisions/s with 12/16/24 workers and two epochs (about 407/s including full process time at the fastest setting). Those probes preceded the final faint-counter fix and ran without concurrent validation; they are not a controlled caching ablation. The **1,000 decisions/s target remains unmet**. Battle collection and feature preparation still dominate learning time.

Windows idle sleep interrupted two earlier reference decisions and inflated one rollout's wall time by about 45 minutes. Original attempts and event evidence remain in `benchmarks/2026-10-30-sleep-interruption.json`. Recovery reruns only invalid or missing scheduled games, preserving valid losses. Training/evaluation now hold temporary power requests; older wall timing is not relabeled as measured awake time. Full experiment provenance and limits are in `benchmarks/2026-10-29-v6-review.json`.

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
.\.venv\Scripts\python.exe tools\benchmark_training.py --initial runtime_data\v8_pilot_warm_start.pt --output runtime_data\speed_check --steps 32768 --workers 12,16,24 --envs 256 --repeats 2
```

Each run starts a fresh process from the same weights. `report.json` records source/checkpoint hashes, seeds, hardware, training throughput, whole-process time, inference batch sizes, and rollout/optimization time. The output folder must be new so previous measurements cannot be mixed into a run. Add `--reference-workers 1` to measure concurrent original-bot work. Use an existing compatible checkpoint for `--initial` if the local experiment checkpoint is unavailable.

Parallel self-play supplies two learner trajectories; the serial trainer supplies one. Their default opponent mixtures differ, so this compares training systems, not identical battle trajectories or playing strength. `--workers` controls CPU processes; `--envs` is the total number of concurrent fast battles across those processes. Increasing workers indefinitely can reduce performance through smaller inference batches and contention. GPU learning remains centralized, and no CUDA toolkit reinstall is needed to change these settings.

## Scope and provenance

- The engine is Pokémon Showdown's Champions mod at `c0845f33d381402dfcb18ce324c20a94f18af075`, format `gen9championsbssregmc`. See the [pinned implementation](https://github.com/smogon/pokemon-showdown/tree/c0845f33d381402dfcb18ce324c20a94f18af075/data/mods/champions) and [official M-C announcement](https://champions-news.pokemon-home.com/en/page/816.html). Do not substitute generic Gen 9 OU or Scarlet/Violet Battle Stadium rules.
- Champions uses its stat-point system (66 total, 32 per stat), level 50, six choose three, and one Mega. Tera, Z-Moves, and Dynamax are not enabled. Showdown stores Champions stat points in the `evs` field; these are not ordinary 252-EV spreads.
- Offline turn caps are implemented; cartridge decision clocks and total time limits are not simulated. Simulator accuracy depends on the pinned mod. Rare public-state effects and the legacy adapter remain areas for further differential testing.
- `opponents/autoplay_reference/SOURCE_MANIFEST.json` identifies the original files and snapshot hashes. Imports and runtime paths were namespaced; only pure state helpers were extracted from the original perception layer. No OCR/device code is used. `tools/import_reference.py` refreshes that snapshot only when explicitly run with a source directory.
- [Jaxcalibur](https://jaxcalibur.github.io/) informed the entity transformer, history, opponent-action auxiliary task, league training, and evaluation approach. Its article describes a different format and much larger training scale. This project uses a CPU Showdown simulator with GPU learning, not its JAX simulator, search implementation, code, or weights. Beating it has not been demonstrated.
