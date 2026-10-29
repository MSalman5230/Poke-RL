'use strict';
// Mechanics and legality are taken from the same pinned engine used for battles.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {Dex, TeamValidator} = require('pokemon-showdown');
const {MovesText} = require('pokemon-showdown/dist/data/text/moves');
const {AbilitiesText} = require('pokemon-showdown/dist/data/text/abilities');
const {ItemsText} = require('pokemon-showdown/dist/data/text/items');
const FORMAT = 'gen9championsbssregmc';
const dex = Dex.mod('champions');
const validator = new TeamValidator(FORMAT);
const species = {}, moves = {}, abilities = {}, items = {}, types = {}, natures = {};
const standard = x => x.exists && !x.isNonstandard;
for (const s of dex.species.all()) {
  if (!standard(s) || s.isCosmeticForme) continue;
  if (validator.checkSpecies({species:s.name, ability:s.abilities['0']}, s, s, {})) continue;
  let learnset = [];
  if (!s.battleOnly) {
    const candidates = [...new Set(dex.species.getFullLearnset(s.id).flatMap(x => Object.keys(x.learnset)))];
    learnset = candidates.filter(id => standard(dex.moves.get(id)) &&
      !validator.checkCanLearn(id, s, undefined, {species:s.name, level:50, ability:s.abilities['0']})).sort();
  }
  species[s.id] = {id:s.id, name:s.name, num:s.num, baseSpecies:s.baseSpecies, types:s.types,
    baseStats:s.baseStats, abilities:s.abilities, weightkg:s.weightkg, battleOnly:s.battleOnly || null,
    isMega:!!s.isMega, requiredItem:s.requiredItem || null, requiredItems:s.requiredItems || [],
    learnset, cosmetic:!!s.isCosmeticForme};
}
const usableMoves = new Set(Object.values(species).flatMap(s => s.learnset));
// Struggle is a legal forced action even though it cannot be selected in a set.
usableMoves.add('struggle');
for (const id of [...usableMoves].sort()) {
  const m = dex.moves.get(id);
  moves[id] = {id, name:m.name, type:m.type, category:m.category, basePower:m.basePower,
    accuracy:m.accuracy === true ? 100 : m.accuracy, alwaysHits:m.accuracy === true,
    pp:m.noPPBoosts ? m.pp : (m.pp / 5 + 1) * 4, priority:m.priority, target:m.target,
    flags:m.flags, boosts:m.boosts || null, self:m.self || null, heal:m.heal || null,
    drain:m.drain || null, recoil:m.recoil || null, status:m.status || null,
    volatileStatus:m.volatileStatus || null, sideCondition:m.sideCondition || null,
    weather:m.weather || null, terrain:m.terrain || null,
    secondary:m.secondary || null, secondaries:m.secondaries || [],
    selfSwitch:m.selfSwitch || false, forceSwitch:!!m.forceSwitch,
    multihit:m.multihit || null, critRatio:m.critRatio || 1,
    description:MovesText[id]?.shortDesc || '',
    conditionalHandlers:Object.keys(m).filter(k => typeof m[k] === 'function')};
}
for (const a of dex.abilities.all()) if (standard(a) && Object.values(species).some(s=>Object.values(s.abilities).includes(a.name))) {
  abilities[a.id] = {id:a.id,name:a.name,description:AbilitiesText[a.id]?.shortDesc || ''};
}
for (const i of dex.items.all()) if (standard(i)) {
  // Mega Stones whose recipients are outside this regulation are omitted.
  if (i.megaStone && !Object.values(typeof i.megaStone === 'object' ? i.megaStone : {form:i.megaStone}).some(n=>species[Dex.toID(n)])) continue;
  items[i.id] = {id:i.id,name:i.name,megaStone:i.megaStone || null,megaEvolves:i.megaEvolves || null,
    isBerry:!!i.isBerry,description:ItemsText[i.id]?.shortDesc || ''};
}
for (const t of dex.types.all()) if (t.name !== 'Stellar') types[t.name] = t.damageTaken;
for (const n of dex.natures.all()) natures[n.name] = {plus:n.plus || null,minus:n.minus || null};
const output = {schema:1, format:FORMAT, mod:'champions', generation:9,
  revision:'c0845f33d381402dfcb18ce324c20a94f18af075',
  rules:{teamSize:6,pickedTeamSize:3,level:50,maxStatPoints:66,maxPerStat:32,megaPerBattle:1,
    duplicateItems:false,duplicateSpecies:false,terastallization:false,zMoves:false,dynamax:false,
    clockSimulation:false},
  sources:['https://github.com/smogon/pokemon-showdown/tree/c0845f33d381402dfcb18ce324c20a94f18af075/data/mods/champions',
    'https://champions-news.pokemon-home.com/en/page/816.html'],
  species,moves,abilities,items,types,natures};
const out = process.argv[2] || path.join(__dirname,'../data/knowledge.json');
fs.mkdirSync(path.dirname(out),{recursive:true});
fs.writeFileSync(out,JSON.stringify(output)+'\n');
console.log(JSON.stringify({path:out,species:Object.keys(species).length,
  selectable:Object.values(species).filter(s=>!s.battleOnly).length,
  nationalNumbers:new Set(Object.values(species).filter(s=>!s.battleOnly).map(s=>s.num)).size,
  moves:Object.keys(moves).length,items:Object.keys(items).length,
  sha256:crypto.createHash('sha256').update(fs.readFileSync(out)).digest('hex')}));
