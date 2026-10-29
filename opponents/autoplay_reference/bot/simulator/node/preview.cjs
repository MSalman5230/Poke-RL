'use strict';
// Raw, isolated attack measurements for the Python preview policy. No ranking.
module.exports = function previewProfiles(request, deadline, api) {
  const {Battle, PRNG, dex, FORMAT, check, clone} = api;
  const rows = [], errors = [];
  function median(b) {
    const random = b.prng.random.bind(b.prng);
    Object.defineProperty(b.prng, 'random', {configurable: true, value(from, to) {
      random(from, to);
      if (from === undefined) return .5;
      return to ? Math.floor((to - from) / 2) + from : Math.floor(from / 2);
    }});
    Object.defineProperty(b, 'randomizer', {configurable: true, value(damage) {
      this.random(16);
      return this.trunc(this.trunc(damage * 92.5) / 100);
    }});
  }
  function start(task) {
    const b = new Battle({formatid: FORMAT, seed: task.seed, send: () => {}});
    try {
      median(b);
      for (let i = 0; i < 2; i++) {
        const input = task.members[i];
        const set = structuredClone(input.set);
        if (task.copy_context === 'active' && task.copy_sides.includes(i)) set.ability = 'No Ability';
        if (input.form?.endsWith('-Hero')) set.species = input.form;
        b.setPlayer(i ? 'p2' : 'p1', {name: i ? 'Them' : 'Us', team: [set]});
      }
      if (!b.choose('p1', 'team 1') || !b.choose('p2', 'team 1') || b.requestState !== 'move') {
        throw new Error('Preview root did not reach a move prompt');
      }
      b.queue.clear();
      for (let i = 0; i < 2; i++) {
        const p = b.sides[i].active[0], input = task.members[i];
        p.m.identity = input.identity;
        if (input.form?.includes('-Mega') && p.species.name !== input.form) {
          if (p.canMegaEvo !== input.form) throw new Error('Incompatible preview Mega form');
          b.queue.addChoice({choice: 'megaEvo', pokemon: p});
        }
      }
      b.queue.sort();
      while (b.queue.peek()) {
        check(deadline);
        const action = b.queue.shift();
        if (action.choice !== 'megaEvo') throw new Error('Unexpected preview entry choice');
        b.runAction(action);
      }
      for (const side of task.copy_sides || []) {
        const p = b.sides[side].active[0];
        p.baseAbility = dex.toID(task.members[side].set.ability);
        if (task.copy_context === 'active' || !p.transformed && p.baseAbility !== 'imposter') {
          p.ability = p.baseAbility;
          p.transformInto(b.sides[1-side].active[0]);
        }
      }
      b.updateSpeed();
      return b;
    } catch (error) {
      b.destroy();
      throw error;
    }
  }
  function attack(root, side, option, reference) {
    const b = clone(root), actor = b.sides[side].active[0], target = b.sides[1 - side].active[0];
    try {
      median(b);
      let accuracy = 1;
      let healing = 0, healed = 0;
      const heal = actor.heal;
      Object.defineProperty(actor, 'heal', {configurable: true, value(amount, ...args) {
        const before = this.hp;
        const result = heal.call(this, amount, ...args);
        healed += this.hp - before;
        return result;
      }});
      const runEvent = b.runEvent;
      Object.defineProperty(b, 'runEvent', {configurable: true, value(event, ...args) {
        const result = runEvent.call(this, event, ...args);
        if (event === 'TryHeal' && args[0] === actor && typeof result === 'number' && result > 0) healing += result;
        return result;
      }});
      const hitAccuracy = b.actions.hitStepAccuracy;
      Object.defineProperty(b.actions, 'hitStepAccuracy', {configurable: true, value(...args) {
        const chance = b.randomChance;
        b.randomChance = function(n, d) {
          chance.call(this, n, d);
          accuracy = Math.max(0, Math.min(1, n / d));
          return n > 0; // Conditional-on-hit damage; Python accounts for accuracy.
        };
        try { return hitAccuracy.apply(this, args); }
        finally { b.randomChance = chance; }
      }});
      function queueReference() {
        b.queue.clear();
        if (reference) b.queue.addChoice({choice: 'move', pokemon: target, moveid: reference, targetLoc: 1});
      }
      function use() {
        check(deadline);
        queueReference();
        const move = dex.getActiveMove(option.name);
        const scheduled = b.queue.resolveAction({choice: 'move', pokemon: actor, moveid: option.name, targetLoc: 1});
        const action = scheduled.find(a => a.choice === 'move');
        const beforeMoves = [...b.queue.list, ...scheduled].filter(a => a.choice === 'beforeTurnMove' || a.choice === 'priorityChargeMove');
        b.speedSort(beforeMoves);
        for (const before of beforeMoves) b.runAction(before);
        const earlier = b.queue.list.find(a => a.choice === 'move' && a.pokemon === target);
        if (option.prior_reference && reference && target.hp && earlier && b.comparePriority(earlier, action) <= 0) {
          b.actions.runMove(reference, target, 1);
        }
        const hp = target.hp, ownHP = actor.hp;
        healing = 0; healed = 0;
        let actions = 0;
        const recharging = !!actor.volatiles.mustrecharge;
        if (actor.hp && target.hp) {
          do {
            check(deadline);
            const charging = actor.volatiles.twoturnmove;
            const chargeMove = charging?.move;
            const wasCharging = chargeMove && !!actor.volatiles[chargeMove];
            b.actions.runMove(option.name, actor, 1);
            actions++;
            // Isolated measurements do not run turn-end expiration. The strike
            // removes its move marker even on immunity/miss; end only that
            // completed lock, preserving boosts, items and recharge state.
            if (wasCharging && actor.volatiles.twoturnmove === charging && !actor.volatiles[chargeMove]) {
              actor.removeVolatile('twoturnmove');
            }
          } while (actor.hp && target.hp && (actor.volatiles.twoturnmove || recharging && actions === 1) && actions < 2);
          if (actor.hp && target.hp && actor.volatiles.twoturnmove) {
            throw new Error('Incomplete preview charging attack');
          }
        }
        return {damage: hp - target.hp, source_loss: ownHP - actor.hp + healed - healing,
          healing_potential: healing, healing_applied: healed,
          actions: Math.max(1, actions), accuracy, priority: action?.priority ?? move.priority,
          fractional_priority: action?.fractionalPriority || 0, speed: action?.speed ?? actor.getActionSpeed(),
          target_speed: target.getActionSpeed(), target_hp: hp, source_hp: ownHP,
          source_fainted: !actor.hp, target_fainted: !target.hp,
          flinch: !!target.volatiles.flinch, target_item: target.item, target_species: target.species.name};
      }
      const first = use();
      let repeat = first;
      if (actor.hp && target.hp) {
        // Preserve used items, broken Disguise, self drops and changed abilities.
        // Damaged HP disables Sturdy/Multiscale on subsequent hits.
        target.hp = Math.max(1, target.maxhp - 1);
        actor.hp = actor.maxhp;
        actor.switchFlag = false;
        target.switchFlag = false;
        b.turn++;
        repeat = use();
      }
      return {name: option.name, reference: reference || null, first, repeat};
    } finally {
      b.destroy();
    }
  }
  for (const task of request.tasks) {
    let b;
    try {
      check(deadline);
      b = start(task);
      const root = b.toJSON();
      const sides = b.sides.map(s => ({species: s.active[0].species.name, hp: s.active[0].hp,
        maxhp: s.active[0].maxhp, speed: s.active[0].getActionSpeed(), item: s.active[0].item,
        ability: s.active[0].ability, boosts: {...s.active[0].boosts}}));
      const copy_failures = Object.fromEntries((task.copy_sides || []).filter(i => !b.sides[i].active[0].transformed)
        .map(i => [String(i), 'native Transform rejected copy']));
      const profiles = task.options.map((options, side) => options.filter(option =>
        !copy_failures[String(side)] && b.sides[side].active[0].moveSlots.some(m => m.id === option.name)).flatMap(option =>
        (option.references?.length ? option.references : [null]).map(reference => attack(root, side, option, reference))));
      rows.push({id: task.id, sides, profiles, ...(task.copy_sides ? {copy_failures} : {})});
    } catch (error) {
      errors.push({id: task.id, error: String(error.message)});
    } finally {
      b?.destroy();
    }
  }
  return {rows, errors};
};
