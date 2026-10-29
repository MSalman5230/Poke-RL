'use strict';
// The only mutable battle objects live here. No simulated result is an observation.
const readline = require('readline');
const fs = require('fs');
const crypto = require('crypto');
const os = require('os');
const {Worker, isMainThread, parentPort} = require('worker_threads');
const {Battle, Dex, PRNG} = require('pokemon-showdown');
const previewProfiles = require('./preview.cjs');
const REVISION = 'c0845f33d381402dfcb18ce324c20a94f18af075';
const FORMAT = 'gen9championsbssregmc';
const dex = Dex.mod('champions');
const id = x => Dex.toID(x);
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const identity = {revision: REVISION, format: FORMAT, node: process.version,
  worker_sha256: sha(__filename), lock_sha256: sha(__dirname + '/package-lock.json'),
  preview_sha256: sha(__dirname + '/preview.cjs'),
  battle_sha256: sha(require.resolve('pokemon-showdown/dist/sim/battle')),
  champions_sha256: sha(require.resolve('pokemon-showdown/dist/data/mods/champions/scripts'))};
const requestedWorkers = Number(process.env.PCA_PROBE_WORKERS ?? 4);
const probeWorkerCount = isMainThread ? Math.min(4, os.availableParallelism(),
  Number.isSafeInteger(requestedWorkers) ? Math.max(0, requestedWorkers) : 4) : 0;
identity.probe_workers = probeWorkerCount;
const lock = JSON.parse(fs.readFileSync(__dirname + '/package-lock.json', 'utf8'));
if (!lock.packages['node_modules/pokemon-showdown'].resolved.endsWith(REVISION)) {
  throw new Error('Showdown lockfile revision mismatch');
}
function check(deadline) { if (Date.now() >= deadline) throw new Error('deadline'); }
function wireRefs(value) {
  // The pinned build emits [DataMove:id], but its deserializer recognizes [Move:id].
  // Normalize that mechanics reference at our serialization boundary, without editing Showdown.
  if (typeof value === 'string') return value.replace(/^\[DataMove:([a-z0-9]+)\]$/, '[Move:$1]');
  if (Array.isArray(value)) return value.map(wireRefs);
  if (value && typeof value === 'object') {
    for (const key of Object.keys(value)) value[key] = wireRefs(value[key]);
  }
  return value;
}
function serialize(b) { return wireRefs(b.toJSON()); }
function clone(root, seed) {
  // fromJSON retains some arrays (including log); copying the serialized root is essential.
  const b = Battle.fromJSON(wireRefs(structuredClone(root)));
  if (seed) b.prng = new PRNG(seed);
  b.send = () => {};
  return b;
}
function state(b, key, target, source, duration) {
  const result = {id: key, target, source, sourceSlot: source?.getSlot?.(), effectOrder: b.effectOrder++};
  if (duration > 0) result.duration = duration;
  return result;
}
function find(b, side, name) {
  return b.sides[side === 'me' ? 0 : 1].pokemon.find(p => p.m.identity === name);
}
function hydrate(input) {
  if (input.format !== FORMAT) throw new Error('Unsupported format');
  const b = new Battle({formatid: FORMAT, seed: [1, 2, 3, 4]});
  const assumptions = [];
  // Suppress initial entry callbacks. Observed boosts/fields/items will be hydrated below.
  input.sides.forEach((s, i) => b.setPlayer(i ? 'p2' : 'p1', {name: i ? 'Them' : 'Us',
    team: s.pokemon.map(m => ({...m.set, ability: 'No Ability', item: ''}))}));
  b.choose('p1', 'team ' + input.sides[0].pokemon.map((_, i) => i + 1).join(','));
  b.choose('p2', 'team ' + input.sides[1].pokemon.map((_, i) => i + 1).join(','));
  if (b.requestState !== 'move') throw new Error('Could not initialize battle');
  b.turn = input.turn;
  b.queue.clear();
  b.faintQueue = [];
  b.log = [];
  b.inputLog = [];
  b.sentLogPos = 0;
  b.lastMove = null;
  b.field.weather = '';
  b.field.weatherState = {id: '', effectOrder: 0};
  b.field.terrain = '';
  b.field.terrainState = {id: '', effectOrder: 0};
  b.field.pseudoWeather = {};
  input.sides.forEach((s, i) => {
    const side = b.sides[i];
    side.sideConditions = {};
    side.slotConditions = [{}];
    side.totalFainted = Math.max(s.total_fainted || 0, s.pokemon.filter(p => !p.hp).length);
    side.faintedLastTurn = side.pokemon.find(p => p.name === s.fainted_last_turn) || null;
    side.faintedThisTurn = side.pokemon.find(p => p.name === s.fainted_this_turn) || null;
    side.pokemon.forEach((p, j) => {
      const m = s.pokemon[j];
      p.set = {...p.set, ...m.set, moves: [...m.set.moves]};
      side.team[j] = p.set;
      p.m.identity = m.identity;
      p.hp = m.hp; p.maxhp = m.maxhp; p.baseMaxhp = m.maxhp;
      p.fainted = !m.hp; p.faintQueued = false;
      p.storedStats = Object.fromEntries(Object.entries(m.stats).filter(([k]) => k !== 'hp'));
      p.baseStoredStats = {...p.storedStats};
      p.boosts = {atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0};
      p.types = [...m.types];
      p.baseTypes = [...dex.species.get(m.set.species).types];
      p.ability = id(m.ability); p.baseAbility = id(m.base_ability);
      p.abilityState = state(b, p.ability, p, p);
      if (m.type_ability_used) p.abilityState[id(m.type_ability_used)] = true;
      p.item = id(m.item); p.itemState = state(b, p.item, p, p);
      p.lastItem = ''; p.usedItemThisTurn = false; p.itemKnockedOff = !p.item && !!m.set.item;
      p.volatiles = {};
      p.status = m.status;
      p.statusState = state(b, m.status, p, p);
      const progress = m.progress || {};
      if (p.status === 'slp') {
        p.statusState.time = Math.max(1, 3 - (progress.sleepElapsed || 0));
        p.statusState.startTime = 3;
        if (progress.sleepSource !== 'rest') assumptions.push({fact: 'sleep duration', species: m.identity,
          elapsed: progress.sleepElapsed || 0, remaining: p.statusState.time, assumption: 'conditional three-check duration'});
      }
      if (p.status === 'tox') p.statusState.stage = progress.toxicStage || 0;
      p.moveSlots.forEach((slot, k) => {
        const move = m.moves[k];
        if (!move || slot.id !== move.id) throw new Error('Hydrated move slot identity mismatch');
        if (move.pp < 0 || move.pp > slot.maxpp) throw new Error('Hydrated PP outside Champions maximum');
        slot.pp = move.pp;
        slot.disabled = false; slot.disabledSource = ''; slot.used = move.pp < slot.maxpp;
      });
      p.baseMoveSlots = p.moveSlots;
      p.switchFlag = false; p.forceSwitchFlag = false;
      p.lastMove = null; p.lastMoveUsed = null; p.lastMoveEncore = null;
      p.activeTurns = j ? 0 : (s.fresh ? 1 : 2);
      p.activeMoveActions = j || s.fresh ? 0 : 1;
      p.newlySwitched = false; p.beingCalledBack = false;
      p.isStarted = !j; p.isActive = !j;
      p.trapped = false; p.maybeTrapped = false; p.maybeDisabled = false;
      p.hurtThisTurn = null; p.lastDamage = 0; p.attackedBy = [];
      const hits = m.times_attacked ?? 0;
      if (!Number.isSafeInteger(hits) || hits < 0) throw new Error('Invalid received-hit count');
      p.timesAttacked = hits;
      if (m.transformed) {
        const copy = m.transformed;
        if (!p.setSpecies(dex.species.get(copy.form), undefined, true)) throw new Error('Invalid copied form');
        p.transformed = true;
        p.storedStats = {...copy.stats};
        p.types = [...copy.types];
        p.apparentType = copy.types.join('/');
        p.ability = id(copy.ability);
        p.abilityState = state(b, p.ability, p, p);
        p.timesAttacked = hits;
        p.moveSlots = copy.moves.map(mv => {
          const move = dex.moves.get(mv.name);
          const maxpp = Math.min(5, move.pp);
          const pp = Math.max(0, mv.pp - (copy.pp_used?.[mv.name] || 0));
          if (!move.exists || pp > maxpp) throw new Error('Invalid transformed move');
          return {move: move.name, id: move.id, pp, maxpp, target: move.target,
            disabled: false, disabledSource: '', used: pp < maxpp, virtual: true};
        });
        for (const [key, data] of Object.entries(copy.copied_effects || {})) {
          p.volatiles[key] = {...state(b, key, p, p), ...data, target: p, source: p};
        }
      }
      p.canMegaEvo = s.mega_used || m.mega_evolved ? null : b.actions.canMegaEvo(p);
      p.canMegaEvoX = null; p.canMegaEvoY = null;
      if (m.hero_ready && p.species.baseSpecies === 'Palafin') p.m.hero_ready = true;
    });
    side.pokemonLeft = side.pokemon.filter(p => p.hp > 0).length;
    const p = side.active[0];
    if (p.ability === 'supremeoverlord') p.abilityState.fallen = Math.min(side.totalFainted, 5);
    p.boosts = {...p.boosts, ...s.boosts};
    p.lastMove = s.last_move ? dex.moves.get(s.last_move) : null;
    p.lastMoveUsed = p.lastMove;
    p.lastMoveEncore = p.lastMove;
    const v = s.observation;
    for (const effect of s.volatiles) {
      let key = effect.id;
      if (['typechange', 'stall', 'lightscreen', 'reflect'].includes(key)) continue;
      if (key === 'slowstart') {
        p.abilityState.counter = effect.remaining ?? Math.max(0, 5 - (effect.elapsed || 0));
        continue;
      }
      const foeSource = ['leechseed', 'partiallytrapped', 'taunt', 'encore', 'disable', 'trapped', 'confusion'].includes(key);
      const sourceSide = effect.source_side || effect.sourceSide || (foeSource ? (i ? 'me' : 'opp') : (i ? 'opp' : 'me'));
      const source = find(b, sourceSide, effect.source) || b.sides[sourceSide === 'me' ? 0 : 1].active[0];
      const conditionKey = key.startsWith('perish') ? 'perishsong' : key.replace(/^(protosynthesis|quarkdrive)(atk|def|spa|spd|spe)$/, '$1');
      const cond = dex.conditions.get(conditionKey);
      if (!cond.exists) throw new Error('Unmapped observed volatile: ' + key);
      let remaining = effect.remaining;
      if (remaining === undefined && cond.duration) remaining = Math.max(1, cond.duration - (effect.elapsed || 0));
      const value = state(b, key, p, source, remaining);
      value.sourceEffect = dex.moves.get(effect.move || key);
      if (key === 'substitute') {
        const owner = find(b, i ? 'opp' : 'me', effect.source) || p;
        value.hp = v.substitute_hp ?? Math.floor(owner.maxhp / 4);
        if (v.substitute_hp === undefined) assumptions.push({fact: 'substitute HP', species: p.m.identity,
          source: owner.m.identity, assumed: value.hp});
      }
      if (key === 'confusion') {
        value.time = Math.max(1, 4 - (effect.elapsed || 0));
        assumptions.push({fact: 'confusion remaining', assumed: value.time, elapsed: effect.elapsed || 0});
      }
      if (key === 'partiallytrapped') {
        value.duration = remaining ?? Math.max(1, 6 - (effect.elapsed || 0));
        value.boundDivisor = source.item === 'bindingband' ? 6 : 8;
        value.sourceEffect = dex.moves.get(effect.move || 'Infestation');
        assumptions.push({fact: 'partial trap duration', assumed: value.duration, elapsed: effect.elapsed || 0});
      }
      if (key === 'leechseed') value.leechSeeder = source.getSlot();
      if (key === 'encore' || key === 'disable' || key === 'choicelock' || key === 'lockedmove') {
        value.move = id(effect.move || s.last_move);
        if (!value.move) throw new Error(key + ' has no observed move');
      }
      if (key === 'lockedmove') {
        value.trueDuration = Math.max(1, 3 - (effect.elapsed || 0));
        value.duration = Math.min(2, value.trueDuration);
        assumptions.push({fact: 'rampage duration', assumed: 3, completed: effect.elapsed || 0});
      }
      if (key.startsWith('perish')) { value.time = Number(key.slice(-1)); key = 'perishsong'; value.id = key; }
      if (/^(protosynthesis|quarkdrive)(atk|def|spa|spd|spe)$/.test(key)) {
        const match = key.match(/^(protosynthesis|quarkdrive)(.+)$/);
        key = match[1]; value.id = key; value.bestStat = match[2]; value.fromBooster = false;
      }
      if (key === 'slowstart') value.duration = remaining ?? 5;
      p.volatiles[key] = value;
    }
    if (v.charging && v.charging.phase !== 'accelerated') {
      const move = id(v.charging.move);
      p.volatiles.twoturnmove = {...state(b, 'twoturnmove', p, p, 2), move};
      p.volatiles[move] = {...state(b, move, p, p, 2), targetLoc: 1};
    }
    if (v.protect_streak || v.protect_turn === input.turn - 1) {
      p.volatiles.stall = {...state(b, 'stall', p, p, 2), counter: 3 ** (v.protect_streak || 1)};
    }
    if (p.ability === 'truant' && v.last_action_kind === 'move') p.volatiles.truant = state(b, 'truant', p, p);
    for (const [key, count] of Object.entries(s.conditions)) {
      const cond = dex.conditions.get(key);
      if (!cond.exists) throw new Error('Unmapped side condition: ' + key);
      side.sideConditions[key] = {...state(b, key, side, p, cond.duration ? count : undefined),
        ...(key === 'spikes' || key === 'toxicspikes' ? {layers: count} : {})};
    }
  });
  const field = input.field;
  const setter = b.p1.active[0];
  if (field.weather) {
    b.field.weather = field.weather;
    b.field.weatherState = state(b, field.weather, b.field, setter, field.weather_turns);
  }
  if (field.terrain) {
    b.field.terrain = field.terrain;
    b.field.terrainState = state(b, field.terrain, b.field, setter, field.terrain_turns);
  }
  if (field.trickroom) b.field.pseudoWeather.trickroom = state(b, 'trickroom', b.field, setter, field.trickroom);
  for (const [key, value] of Object.entries(field.extra)) {
    if (!dex.conditions.get(key).exists) throw new Error('Unmapped field effect: ' + key);
    b.field.pseudoWeather[key] = state(b, key, b.field, setter, value.remaining);
  }
  for (const [side, wish] of Object.entries(input.wishes)) {
    const source = find(b, side, wish.source);
    if (!source && wish.amount === undefined) throw new Error('Wish source HP unavailable');
    const s = b.sides[side === 'me' ? 0 : 1];
    s.slotConditions[0].wish = {...state(b, 'wish', s, source || s.active[0]), isSlotCondition: true,
      hp: wish.amount ?? source.maxhp / 2, startingTurn: b.getOverflowedTurnCount() + wish.turns - 2};
  }
  for (const [targetSide, future] of Object.entries(input.future_moves)) {
    const source = find(b, future.sourceSide, future.source);
    if (!source) throw new Error('Delayed attack source unavailable');
    const side = b.sides[targetSide === 'me' ? 0 : 1];
    const m = dex.moves.get(future.move);
    side.slotConditions[0].futuremove = {...state(b, 'futuremove', side, source), isSlotCondition: true,
      targetSlot: side.active[0].getSlot(), endingTurn: b.getOverflowedTurnCount() + future.turns - 1, move: m.id,
      moveData: {id: m.id, name: m.name, accuracy: m.accuracy, basePower: m.basePower, category: m.category,
        priority: 0, flags: {futuremove: 1}, effectType: 'Move', type: m.type, ignoreImmunity: false}};
  }
  b.updateSpeed();
  for (const p of b.getAllActive()) {
    b.runEvent('DisableMove', p);
    b.runEvent('TrapPokemon', p);
    b.runEvent('MaybeTrapPokemon', p);
  }
  if (input.kind === 'send') {
    for (let i = 0; i < 2; i++) {
      const p = b.sides[i].active[0], s = input.sides[i];
      if (!i || !p.hp || s.replacement.kind === 'pivot') p.switchFlag = s.observation.pending_pass ? id(s.observation.pending_pass) : true;
    }
    b.midTurn = true;
    b.makeRequest('switch');
  } else {
    b.midTurn = false;
    b.makeRequest('move');
  }
  return {battle: b, assumptions};
}
function legal(b, sideIndex, movesOnly = false) {
  const side = b.sides[sideIndex], req = side.activeRequest;
  if (!req || req.wait) return [];
  const list = [];
  if (req.active) for (const m of req.active[0].moves) {
    if (m.disabled) continue;
    const base = {kind: 'move', name: m.id, choice: 'move ' + m.id, mega: false};
    list.push(base);
    if (req.active[0].canMegaEvo) list.push({...base, choice: base.choice + ' mega', mega: true});
  }
  if (!movesOnly && (req.forceSwitch?.[0] || req.active && !req.active[0].trapped)) {
    side.pokemon.forEach((p, i) => {
      if (p.hp && !side.active.includes(p)) list.push({kind: 'switch', identity: p.m.identity, choice: 'switch ' + (i + 1)});
    });
  }
  return list;
}
function summary(b) {
  return {turn: b.turn, request: b.requestState, ended: b.ended, winner: b.winner || null,
    field: {weather: b.field.weather, weather_turns: b.field.weatherState.duration ?? null,
      terrain: b.field.terrain, terrain_turns: b.field.terrainState.duration ?? null,
      pseudoWeather: Object.fromEntries(Object.entries(b.field.pseudoWeather).map(([k, v]) => [k, v.duration ?? null]))},
    sides: b.sides.map(s => ({left: s.pokemonLeft, active: s.active[0]?.m.identity,
      pokemon: s.pokemon.map(p => ({identity: p.m.identity, species: p.species.name, hp: p.hp, maxhp: p.maxhp,
        status: p.status, item: p.item, ability: p.ability, boosts: {...p.boosts},
        pp: p.moveSlots.map(m => ({id: m.id, pp: m.pp, disabled: m.disabled})), types: [...p.types]}))}))};
}
function choose(b, side, choice) {
  if (!b.choose(side, choice)) throw new Error('Rejected simulator choice ' + side + ': ' + choice);
}
// Observe resolved mechanics. These wrappers call each original exactly once and never roll RNG.
function measurements(b, previous, isolated = false) {
  const trace = structuredClone(previous || {queues: [], moves: [], damage: [], calculations: []});
  trace.inflicted ||= [];
  let actionIndex = -1, calculation = null, statCalculation = null;
  const wrap = (object, key, callback) => {
    const original = object[key];
    Object.defineProperty(object, key, {configurable: true, value: callback(original)});
  };
  const actionInfo = action => ({actor: action.pokemon?.m.identity, side: action.pokemon?.side.n,
    move: action.move?.id, priority: action.priority || 0, speed: action.speed || 0,
    fractionalPriority: action.fractionalPriority || 0});
  wrap(b.queue, 'sort', original => function(...args) {
    const result = original.apply(this, args);
    const moves = this.list.filter(a => a.choice === 'move').map(actionInfo);
    if (moves.length) trace.queues.push(moves);
    return result;
  });
  wrap(b, 'runAction', original => function(action) {
    actionIndex = -1;
    if (action.choice === 'move') {
      actionIndex = trace.moves.length;
      trace.moves.push({...actionInfo(action), can_act: !!action.pokemon.hp});
    }
    return original.call(this, action);
  });
  wrap(b.actions, 'getDamage', original => function(source, target, move, ...args) {
    const m = typeof move === 'string' ? b.dex.moves.get(move) : move;
    const saved = calculation;
    calculation = m && typeof m === 'object' ? {actor: source.m.identity, target: target.m.identity,
      move: m.id, owner: (m.overrideOffensivePokemon === 'target' ? target : source).m.identity,
      stat: m.overrideOffensiveStat || (m.category === 'Physical' ? 'atk' : 'spa'), action_index: actionIndex} : null;
    try { return original.call(this, source, target, move, ...args); }
    finally { calculation = saved; }
  });
  wrap(b, 'runEvent', original => function(event, ...args) {
    const result = original.call(this, event, ...args);
    // Observe the stage after Unaware and other ModifyBoost handlers have run.
    if (event === 'ModifyBoost' && statCalculation && result && typeof result === 'object') {
      trace.calculations.push({...statCalculation, stage: result[statCalculation.stat] ?? 0});
    }
    return result;
  });
  for (const side of b.sides) for (const p of side.pokemon) {
    wrap(p, 'calculateStat', original => function(stat, boost, ...args) {
      const saved = statCalculation;
      statCalculation = calculation && calculation.owner === this.m.identity && calculation.stat === stat ? calculation : null;
      try { return original.call(this, stat, boost, ...args); }
      finally { statCalculation = saved; }
    });
    wrap(p, 'damage', original => function(amount, source, effect) {
      const hp = this.hp;
      const result = original.call(this, amount, source, effect);
      const current = trace.moves[actionIndex];
      // Attribution for probes includes called moves, but excludes recoil, abilities and residuals.
      if (source && source.side !== this.side && effect?.effectType === 'Move' && hp > this.hp &&
          (isolated || current?.actor === source.m.identity)) {
        trace.inflicted.push({actor: source.m.identity, side: source.side.n, target: this.m.identity,
          target_side: this.side.n, move: effect.id, amount: hp - this.hp});
      }
      if (source && current?.actor === source.m.identity && current.move === effect?.id &&
          effect?.effectType === 'Move' && b.activeMove?.id === effect.id && hp > this.hp) {
        const used = [...trace.calculations].reverse().find(c => c.actor === source.m.identity &&
          c.target === this.m.identity && c.move === effect.id && c.action_index === actionIndex);
        trace.damage.push({actor: source.m.identity, side: source.side.n, target: this.m.identity,
          target_side: this.side.n, move: effect.id, amount: hp - this.hp, knockout: !this.hp,
          action_index: actionIndex, offensive: used || null});
      }
      return result;
    });
  }
  return trace;
}
let probeWorkers;
function getProbeWorkers() {
  if (probeWorkers) return probeWorkers;
  probeWorkers = Array.from({length: probeWorkerCount}, () => {
    const worker = new Worker(__filename), pending = new Map();
    let sequence = 0, failure = null, resolveReady, rejectReady;
    const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
    const fail = error => {
      failure = error;
      rejectReady(error);
      for (const job of pending.values()) job.reject(error);
      pending.clear();
    };
    worker.on('error', fail);
    worker.on('exit', code => fail(new Error('Probe worker exited: ' + code)));
    worker.on('message', message => {
      if (message.ready) {
        const keys = ['revision', 'format', 'worker_sha256', 'lock_sha256', 'battle_sha256', 'champions_sha256'];
        if (keys.some(key => message.identity?.[key] !== identity[key])) fail(new Error('Probe worker identity mismatch'));
        else resolveReady();
        return;
      }
      const job = pending.get(message.id);
      if (!job) { fail(new Error('Stale probe worker response')); return; }
      pending.delete(message.id);
      if (message.error) job.reject(new Error(message.error));
      else job.resolve(message.rows);
    });
    return {ready, async run(root, seeds, options, deadline, defenderMegaAction) {
      await ready;
      if (failure) throw failure;
      return new Promise((resolve, reject) => {
        const id = ++sequence;
        pending.set(id, {resolve, reject});
        worker.postMessage({id, root, seeds, options, deadline, defender_mega_action: defenderMegaAction});
      });
    }};
  });
  return probeWorkers;
}
function probeOptions(probeRoot, seeds, options, deadline, defenderMegaAction) {
  return options.map(({index, action}) => {
    let damage = 0;
    for (const seed of seeds) {
      check(deadline);
      const q = clone(probeRoot, seed), attacker = q.p2.active[0], target = q.p1.active[0];
      try {
        const trace = measurements(q, null, true);
        if (defenderMegaAction) {
          // Only the supplied evolutions run here: no reference move, turn or residuals.
          // Showdown owns Speed order/ties and ability activation for both sides' Megas.
          q.updateSpeed();
          q.queue.clear();
          q.queue.addChoice({choice: 'megaEvo', pokemon: target});
          if (action.mega) q.queue.addChoice({choice: 'megaEvo', pokemon: attacker});
          q.queue.sort();
          while (q.queue.peek()) {
            check(deadline);
            const evolution = q.queue.shift();
            if (evolution.choice !== 'megaEvo') throw new Error('Unexpected Mega probe preparation action');
            q.runAction(evolution);
            if (q.requestState === 'switch') throw new Error('Mega probe requires a replacement choice');
          }
        } else if (action.mega) {
          q.actions.runMegaEvo(attacker);
        }
        q.actions.runMove(action.name, attacker, 1);
        damage += trace.inflicted.filter(h => h.side === 1 && h.target_side === 0 && h.actor === attacker.m.identity)
          .reduce((sum, h) => sum + h.amount, 0);
      } finally {
        q.destroy();
      }
    }
    return {index, value: {...action, damage: damage / seeds.length}};
  });
}
async function probe(serialized, seeds, deadline, supplied, defenderMegaAction) {
  const b = clone(serialized);
  let probeRoot, options;
  try {
    if (defenderMegaAction && (b.requestState !== 'move' || !b.p1.active[0].hp ||
        defenderMegaAction.kind !== 'move' || defenderMegaAction.mega !== true ||
        !legal(b, 0, true).some(a => a.mega && a.choice === defenderMegaAction.choice && a.name === defenderMegaAction.name))) {
      throw new Error('Illegal defender Mega probe action');
    }
    // A faint replacement may leave the opponent waiting. Probe the live foe's moves in a move request.
    const foe = b.p2.active[0];
    if (!foe.hp) return [];
    b.p1.active[0].switchFlag = false; foe.switchFlag = false;
    b.makeRequest('move');
    const available = legal(b, 1, true);
    options = supplied === undefined ? available : supplied;
    if (options.some(a => !available.some(x => x.choice === a.choice && x.name === a.name && x.mega === a.mega))) {
      throw new Error('Illegal supplied probe choice');
    }
    probeRoot = serialize(b);
  } finally {
    b.destroy();
  }
  const tasks = options.map((action, index) => ({index, action}));
  const workers = getProbeWorkers();
  if (workers.length <= 1 || tasks.length <= 1) return probeOptions(probeRoot, seeds, tasks, deadline, defenderMegaAction).map(row => row.value);
  const chunks = Array.from({length: Math.min(workers.length, tasks.length)}, () => []);
  tasks.forEach((task, i) => chunks[i % chunks.length].push(task));
  const rows = (await Promise.all(chunks.map((chunk, i) => workers[i].run(probeRoot, seeds, chunk, deadline, defenderMegaAction)))).flat();
  rows.sort((a, b) => a.index - b.index);
  if (rows.length !== tasks.length || rows.some((row, i) => row.index !== i ||
      row.value?.name !== options[i].name || row.value?.mega !== options[i].mega || !Number.isFinite(row.value?.damage))) {
    throw new Error('Incomplete or malformed probe worker response');
  }
  return rows.map(row => row.value);
}
// Policy stays in Python. Each case stops at the next choice request.
function step(root, task) {
  const b = clone(task.root || root, task.seed);
  try {
    const trace = task.measure === false ? null : measurements(b, task.trace);
    if (task.kind === 'turn') {
      if (!b.ended) {
        if (b.requestState !== 'move') throw new Error('Turn rollout requires a move prompt');
        choose(b, 'p1', task.action.choice);
        if (!task.reply) throw new Error('No predicted opponent action');
        choose(b, 'p2', task.reply.choice);
      }
    } else if (task.kind === 'send') {
      if (task.residual) {
        if (b.p2.active[0].hp && task.pending_action) {
          b.queue.addChoice({choice: 'move', pokemon: b.p2.active[0],
            moveid: task.pending_action.name, mega: !!task.pending_action.mega, targetLoc: 1});
        }
        b.queue.addChoice({choice: 'residual'});
      }
    } else if (task.kind === 'resume') {
      choose(b, task.side ? 'p2' : 'p1', task.choice);
    } else {
      throw new Error('Unknown mechanics step ' + task.kind);
    }
    const pending = [];
    if (!b.ended && b.requestState === 'switch') {
      for (let i = 0; i < 2; i++) {
        const side = b.sides[i];
        if (side.activeRequest?.forceSwitch?.[0] && !side.isChoiceDone()) {
          pending.push({side: i, opponent: b.sides[1 - i].active[0]?.m.identity,
            options: legal(b, i).filter(a => a.kind === 'switch')});
        }
      }
      if (!pending.length) throw new Error('Unresolved replacement request');
    }
    const result = {id: task.id, after: summary(b), pending, measurements: trace,
      protocol: b.log.filter(line => !line.startsWith('|t:|'))};
    if (task.retain_root || pending.length) result.root = serialize(b);
    if (task.retain_root) result.legal = [legal(b, 0), legal(b, 1)];
    return result;
  } finally {
    b.destroy();
  }
}
function steps(request, deadline) {
  const rows = [], errors = [];
  for (const task of request.cases) {
    try {
      check(deadline);
      rows.push(step(request.root, task));
    } catch (e) {
      errors.push(String(e.message));
      break;
    }
  }
  return {rows, errors, deadline_passed: Date.now() >= deadline};
}
async function dispatch(request) {
  const deadline = request.deadline_ms;
  if (!Number.isFinite(deadline) || deadline <= 0) throw new Error('A finite positive deadline_ms is required');
  switch (request.op) {
  case 'ping':
    if (Dex.formats.get(FORMAT).mod !== 'champions') throw new Error('Champions format unavailable');
    await Promise.all(getProbeWorkers().map(worker => worker.ready));
    return {identity};
  case 'prepare': {
    const {battle, assumptions} = hydrate(request.input);
    const result = {root: serialize(battle), assumptions, legal: [legal(battle, 0), legal(battle, 1)], summary: summary(battle)};
    battle.destroy();
    return result;
  }
  case 'probe_options': {
    const b = clone(request.root), canExchange = b.requestState === 'move';
    const alive = b.p2.active[0].hp > 0;
    b.p1.active[0].switchFlag = false; b.p2.active[0].switchFlag = false;
    if (alive) b.makeRequest('move');
    const result = {legal: [b.p1.active[0].hp ? legal(b, 0, true) : [], alive ? legal(b, 1, true) : []],
      can_exchange: canExchange, summary: summary(b)};
    b.destroy();
    return result;
  }
  case 'probe': return {rows: await probe(request.root, request.seeds, deadline, request.options, request.defender_mega_action)};
  case 'step': return steps(request, deadline);
  case 'preview_profiles': return previewProfiles(request, deadline, {Battle, PRNG, dex, FORMAT, check, clone});
  case 'inspect': {
    const b = clone(request.root);
    const result = {summary: summary(b), legal: [legal(b, 0), legal(b, 1)]};
    b.destroy();
    return result;
  }
  case 'stats': {
    const b = new Battle({formatid: FORMAT});
    const result = {stats: request.members.map(set => {
      const s = dex.species.get(set.species);
      return Object.fromEntries(['hp', 'atk', 'def', 'spa', 'spd', 'spe'].map(k => [k,
        s.id === 'shedinja' && k === 'hp' ? 1 : b.dex.data.Scripts.statModify.call(b, s.baseStats, set, k)]));
    })};
    b.destroy();
    return result;
  }
  default: throw new Error('Unknown operation ' + request.op);
  }
}
if (!isMainThread) {
  if (Dex.formats.get(FORMAT).mod !== 'champions') throw new Error('Champions format unavailable');
  parentPort.on('message', request => {
    try {
      parentPort.postMessage({id: request.id, rows: probeOptions(request.root, request.seeds, request.options,
        request.deadline, request.defender_mega_action)});
    } catch (e) {
      parentPort.postMessage({id: request.id, error: String(e.stack || e)});
    }
  });
  parentPort.postMessage({ready: true, identity});
} else readline.createInterface({input: process.stdin, crlfDelay: Infinity}).on('line', async line => {
  let request;
  try {
    request = JSON.parse(line);
    const result = await dispatch(request);
    process.stdout.write(JSON.stringify({id: request.id, result}) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({id: request?.id, error: String(e.stack || e)}) + '\n');
  }
});
