// Export reader metadata; battle mechanics stay in Showdown.
'use strict';
const fs = require('fs');
const {Dex} = require('pokemon-showdown');
const {MovesText} = require('pokemon-showdown/dist/data/text/moves');
const REVISION = 'c0845f33d381402dfcb18ce324c20a94f18af075';
const dex = Dex.mod('champions');
const species = {}, moves = {}, items = {}, abilities = {}, natures = {}, types = {};
for (const s of dex.species.all()) {
  if (!s.exists || ['CAP', 'Custom'].includes(s.isNonstandard)) continue;
  let learnset = [];
  try { learnset = [...new Set(dex.species.getFullLearnset(s.id).flatMap(x => Object.keys(x.learnset)))]; } catch {}
  species[s.id] = {id: s.id, name: s.name, types: s.types, base: s.baseStats,
    abilities: s.abilities, possibleAbilities: [...new Set(Object.values(s.abilities))],
    baseSpecies: s.baseSpecies, weight: s.weightkg, num: s.num, otherFormes: s.otherFormes || [],
    requiredItem: s.requiredItem || null, requiredItems: s.requiredItems || [],
    battleOnly: s.battleOnly || null, learnset};
}
for (const m of dex.moves.all()) {
  if (!m.exists) continue;
  moves[m.id] = JSON.parse(JSON.stringify({...m, bp: m.basePower,
    max_pp: m.noPPBoosts ? m.pp : (m.pp / 5 + 1) * 4,
    accuracy: m.accuracy === true ? 100 : m.accuracy, alwaysHits: m.accuracy === true,
    selfBoosts: m.self?.boosts || (m.target === 'self' ? m.boosts : null),
    targetBoosts: m.target !== 'self' ? m.boosts : null,
    description: MovesText[m.id]?.shortDesc || null,
    secondaryEffects: m.secondaries || (m.secondary ? [m.secondary] : []),
    conditionalHandlers: Object.keys(m).filter(k => typeof m[k] === 'function'),
    secondaries: !!(m.secondary || m.secondaries?.length)}));
}
for (const i of dex.items.all()) if (i.exists) items[i.id] = {name: i.name, megaStone: i.megaStone, megaEvolves: i.megaEvolves};
for (const a of dex.abilities.all()) if (a.exists) abilities[a.id] = {name: a.name};
for (const n of dex.natures.all()) natures[n.name] = {plus: n.plus || null, minus: n.minus || null};
for (const t of dex.types.all()) types[t.name] = t.damageTaken;
fs.writeFileSync(process.argv[2], JSON.stringify({schema_version: 2,
  source: {repository: 'smogon/pokemon-showdown', revision: REVISION, format: 'gen9championsbssregmc', mod: 'champions'},
  species, moves, items, abilities, natures, types}) + '\n');
