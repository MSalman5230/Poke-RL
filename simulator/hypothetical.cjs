'use strict';
// Mechanics hydration adapted verbatim from the user's frozen battle bot.
// Source: opponents/autoplay_reference/bot/simulator/node/worker.cjs (SOURCE_MANIFEST.json).
// This isolated copy accepts explicitly hypothetical worlds, never a live game ID.
const {Battle,Dex,PRNG}=require('pokemon-showdown');
const FORMAT='gen9championsbssregmc';
const dex=Dex.mod('champions');
const id=Dex.toID;
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
module.exports={hydrate,serialize,clone};
