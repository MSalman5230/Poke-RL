# Opponent-action prediction rules

This section predicts one opponent move for a one-turn exchange. It is separate from estimating the opponent's build and from scoring our possible responses.

All rules below are implemented. This file describes current behavior; Markdown is documentation, and changing it alone does not change the bot.

## Current rules

| ID | Rule |
| --- | --- |
| OA-01 | Use the active opponent's scoped predicted build, observations and regenerated Showdown legality. For transformed Ditto, use its current copied moves/form/ability and retained native item/HP; copied moves never extend the native build. Apply BE-21 mandatory Mega filtering before any probe: one legal version per move. Predict no voluntary opponent switch. |
| OA-02 | For ordinary enemy options, probe each supplied legal choice in isolation with all 32 recorded seeds. At a normal move prompt, Python supplies the first established legal Mega move context, if any, as defender_mega_action. Each trial applies that evolution before the enemy attack using native Showdown stats, typing, ability activation and simultaneous-Mega ordering. The reference move is never executed; no turn or residuals advance. No supplied defender evolution preserves the current form, including already evolved members. Replacement prompts, spent Mega slots and absent established Mega moves never infer another evolution. If evolution opens a replacement prompt, fail the isolated probe under the existing BE-23 scored recovery instead of inventing a replacement or executing the interrupted attack. Measure actual enemy-move-inflicted HP damage, including called attacks, excluding our recoil, setup costs, ability damage and residuals. Node validates and executes supplied choices; Python owns their selection. Context-dependent moves use OA-10. |
| OA-03 | For each contextual enemy option, take the highest complete reference-action mean under OA-10. Then select the enemy option with the highest mean immediate damage and keep that one reply fixed across all our candidate actions. Never maximize independently per seed or choose a different enemy reply for each of our candidates. |
| OA-04 | If no option has positive expected damage, choose the most-used legal move. Move weights use the predicted battle form before the species, cutoff 1500 before 0, and the first available usage share for each move. |
| OA-05 | Exact damage or usage ties retain simulator move order after mandatory Mega filtering. If every fallback usage weight is missing, keep the first legal move. |
| OA-06 | A normal move prompt with no legal opponent reply causes an explicit decision failure and BE-23 scored recovery. At a replacement prompt, an absent reply is allowed when the phase does not require one. |
| OA-07 | After a normal faint replacement, choose a fresh reply against the replacement for its next exchange, using the same damage/usage rules. For a mid-turn replacement, an observed pending opponent move takes precedence; use the predicted reply for an unobserved pending move. Resolve only actions still pending in that turn. |
| OA-08 | Save probed options, damage, samples, ordinary defender_mega_action (or null), contextual reference choices and per-seed hits/replacement choices, reply, usage weights and seeds. The saved root, seeds and defender reference reproduce the ordinary probe's evolution assumption. Continuations preserve their chosen replies and inputs. Predictions and hypothetical outcomes never update observations. |
| OA-09 | Recompute replies from exact post-setup/entry/survivor/pivot states through OA-01 to OA-10, including OA-02's legal upcoming defender evolution. Each reply remains fixed across that state's alternatives, with no voluntary enemy switch. Preserve spent-action handling, applicable trial positions and exact continuation PRNG. Cache only mechanically identical probe inputs with the same seeds, move contexts (including the derived defender Mega reference), weights and replacement matrices; probe clones reset their PRNG, actual continuations never do. Reuse complete contextual exchanges and paused pivot prompts in the initial evaluation only when root, seed, own choice, fixed reply and replacement ranks match exactly. |
| OA-10 | Use genuine full-exchange contexts for Sucker Punch, Upper Hand, Counter, Mirror Coat, Metal Burst, Comeuppance, Focus Punch, Shell Trap, Beak Blast, Payback, Avalanche, Revenge, Assurance, Me First, Copycat and Mirror Move. Contexts are established legal staying-in moves including Status choices, after mandatory Mega filtering. Run every context across the same 32 seeds, resolve pending replacements through the shared Python TS-09 resolver, and attribute damage to the original enemy actor. Choose the highest complete context mean, never per-seed best contexts. A phase with no legal move context records an inapplicability reason and zero contextual damage; incomplete required work fails the decision. No invented incoming hit, queue action or guaranteed success. |

## Where these rules live

- [Reply selection and policy identity](../../bot/decision/opponent_action.py)
- [Usage weights, reply selection and pending-move precedence](../../bot/decision/opponent_action.py)
- [Seeds and fixed-reply orchestration](../../bot/decision/greedy.py)
- [Replacement continuation planning](../../bot/decision/rollouts.py)
- [Shared mechanics batches and pending-prompt resolution](../../bot/decision/experiments.py)
- [Mechanics-only damage probes and supplied action execution](../../bot/simulator/node/worker.cjs)
- [Reply, mechanics, replacement and replay regressions](../../tests/test_showdown.py)

`choose` and `pending_reply` are the only implementations of reply-choice rules. Normal turns, replacement continuations and replay call these Python functions. Node reports probe measurements and executes supplied actions; it never selects a reply. The pinned Showdown dependency stays unchanged.

## Maintaining this file

When a reply rule is added, changed or removed, update its implementation and this file in the same change. Keep existing IDs stable, allocate a new ID for a new rule, and record the date, affected IDs and reason below. Describe implemented behavior here; label proposed ideas explicitly until they are implemented. Update relevant regression coverage when behavior changes.

## Change history

| Date | Rules | Change |
| --- | --- | --- |
| 2026-10-26 | OA-01 to OA-08 | Documented the existing fixed immediate-damage reply baseline; no policy changes. |
| 2026-10-26 | OA-03 to OA-08 | Removed Node reply-choice copies. All phases and replay reuse Python reply/precedence functions; diagnostics also retain replacement probe options and planning outcomes. Prediction behavior is unchanged. |
| 2026-10-26 | OA-07 to OA-09 | Reused reply prediction on actual per-trial replacement entries and post-setup states, with exact-input probe caching. No new opponent switching or inference policy. |
| 2026-10-26 | OA-09 | Reused per-entry reply prediction for voluntary switches' next decisions. Preserve the original reply during switching, then recompute against each resulting state without adding voluntary enemy switches. |
| 2026-10-26 | OA-09 | Reused reply prediction for staying-in/sacrifice continuations, including conditional survivor/replacement groups within the same 32 trials. Keep each next-turn reply fixed across that state's alternatives and do not replay an already-spent attack. |
| 2026-10-26 | OA-09 | Reused fresh next-turn replies across every fixed pivot receiver and recognized live pivot replacement. Keep the original reply during each first exchange, preserve spent-action handling, and add no voluntary opponent switch prediction. |
| 2026-10-26 | OA-01 to OA-03, OA-05, OA-08 to OA-10 | Added forced Mega choice and real conditional-move probes. Python retains all reply/context/replacement policy; ordinary probes keep their fast path. |
| 2026-10-26 | OA-02, OA-08, OA-09 | Ordinary enemy probes now account for our legal upcoming Mega form through an explicit Python-supplied defender reference. Native mechanics apply both sides' evolutions before the isolated attack, with the reference saved for replay. Fixes first-turn base-form defense/type/ability probes without executing our move, adding full-turn contexts, changing the fixed-reply policy or updating observations. |
| 2026-10-27 | OA-01, OA-02, OA-06 | Opposing transformed Ditto now supplies its copied legal attack options to the same Python reply predictor, including emergency scoring. |
| 2026-10-28 | OA-07 | Rule unchanged; its observations changed. Live captions are no longer read: the View Log is read at fight prompts and a replacement screen has none, so at our own pivot's replacement prompt whether the opponent already acted this turn is unobserved and OA-07 uses the predicted reply (its existing unobserved case). Battle Info there still supplies the opponent's identity, HP, stages and effects. |
