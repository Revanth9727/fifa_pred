let liveState = { matches: [], standings: {}, champion: null, status: "loading" };
let selectedTeam = "";
let simResults = [];
let simPollTimer = null;
let autoRefreshInFlight = false;

const teamColors = {
  Argentina: ["#75aadb", "#ffffff"], Brazil: ["#009c3b", "#ffdf00"],
  England: ["#ffffff", "#cf142b"], France: ["#0055a4", "#ef4135"],
  Germany: ["#111111", "#ffce00"], Spain: ["#c60b1e", "#ffc400"],
  Portugal: ["#006600", "#ff0000"], Netherlands: ["#ff7f00", "#21468b"],
  Mexico: ["#006847", "#ce1126"], "United States": ["#3c3b6e", "#b22234"],
  Canada: ["#ff0000", "#ffffff"], Japan: ["#ffffff", "#bc002d"],
  Morocco: ["#c1272d", "#006233"], Senegal: ["#00853f", "#fdef42"],
  Ghana: ["#006b3f", "#fcd116"], Uruguay: ["#0038a8", "#fcd116"],
  Colombia: ["#fcd116", "#003893"], Switzerland: ["#d52b1e", "#ffffff"],
  Belgium: ["#111111", "#fae042"], Croatia: ["#f00000", "#171796"],
  Australia: ["#004b3a", "#ffcd00"], "South Korea": ["#c60c30", "#003478"],
  "Saudi Arabia": ["#006c35", "#ffffff"], "C\u00f4te d'Ivoire": ["#f77f00", "#009e60"],
  "South Africa": ["#007a4d", "#ffb612"], Czechia: ["#d7141a", "#11457e"],
  "Bosnia and Herzegovina": ["#002395", "#ffcd00"], Qatar: ["#8d1b3d", "#ffffff"],
  Haiti: ["#00209f", "#d21034"], Scotland: ["#003380", "#ffffff"],
  Paraguay: ["#d52b1e", "#ffffff"], "T\u00fcrkiye": ["#e30a17", "#ffffff"],
  "Cura\u00e7ao": ["#003da5", "#f9e814"], Ecuador: ["#ffd100", "#003893"],
  Sweden: ["#fecc02", "#006aa7"], Tunisia: ["#e70013", "#ffffff"],
  Egypt: ["#ce1126", "#ffffff"], Iran: ["#239f40", "#da0000"],
  "New Zealand": ["#000000", "#ffffff"], "Cabo Verde": ["#003893", "#cf2027"],
  Iraq: ["#007a3d", "#ce1126"], Norway: ["#ef2b2d", "#ffffff"],
  Algeria: ["#006233", "#ffffff"], Austria: ["#ed2939", "#ffffff"],
  Jordan: ["#007a3d", "#ce1126"], "DR Congo": ["#007fff", "#ce1126"],
  Uzbekistan: ["#1eb53a", "#009fcc"], Panama: ["#da121a", "#003580"],
};

const fmtPct = (value) => value == null ? "-" : `${(value * 100).toFixed(1)}%`;

function colors(team) {
  return teamColors[team] || ["#003f5c", "#ffa600"];
}

function teamPill(team) {
  const [a, b] = colors(team);
  return `<span class="teamPill" style="--team-a:${a};--team-b:${b}">${team || "TBD"}</span>`;
}

function lockedCount() {
  return liveState.matches.filter((match) => match.locked).length;
}

function renderScoreboard() {
  const champion = liveState.champion;
  document.getElementById("scoreboard").innerHTML = [
    tile("Tournament State", liveState.status.replaceAll("_", " ")),
    tile("Played", `${lockedCount()} / 104`),
    tile("Next", nextMatchLabel()),
    tile("Champion", champion ? champion : "Hidden"),
  ].join("");
  document.getElementById("statusLine").textContent = champion
    ? `${champion} won this simulated World Cup. Reset to play another path.`
    : "Click each match card to inspect it, or press Simulate on the card to reveal its score.";
  document.getElementById("championBanner").innerHTML = champion
    ? `Champion: ${teamPill(champion)}`
    : "Champion hidden";
}

function tile(label, value) {
  return `<div class="scoreTile"><span>${label}</span><strong>${value}</strong></div>`;
}

function nextMatchLabel() {
  const next = liveState.matches.find((match) => match.available && !match.locked);
  return next ? `#${next.match_no} ${next.home} vs ${next.away}` : "Complete";
}

function groupMatches() {
  const q = document.getElementById("searchBox").value.trim().toLowerCase();
  return liveState.matches.filter((match) => {
    if (match.stage !== "Group") return false;
    const text = `${match.home} ${match.away} ${match.group}`.toLowerCase();
    return (!selectedTeam || match.home === selectedTeam || match.away === selectedTeam) && (!q || text.includes(q));
  });
}

function matchScore(match) {
  if (!match.locked) return "vs";
  return `${match.home_score} - ${match.away_score}`;
}

function resultNote(match) {
  if (!match.locked) return "";
  if (match.result_note) return match.result_note;
  if (match.went_to_shootout && match.winner) {
    return `${match.winner} win ${match.penalties_home}-${match.penalties_away} on penalties`;
  }
  if (match.went_to_et) return "After extra time";
  return match.winner ? `${match.winner} win` : "Draw";
}

function matchCard(match) {
  const [ha, hb] = colors(match.home);
  const [aa, ab] = colors(match.away);
  const cls = match.locked ? "played" : match.available ? "ready" : "locked";
  return `<article class="matchCard ${cls}" data-match-id="${match.match_id}">
    <div class="cardStripe" style="background:linear-gradient(90deg, ${ha}, ${hb}, ${ab}, ${aa})"></div>
    <div class="matchMeta"><span>#${match.match_no}</span><span>${match.locked ? "Final" : match.available ? "Ready" : "Locked"}</span></div>
    <div class="teamsLine">${teamPill(match.home)}<strong>${matchScore(match)}</strong>${teamPill(match.away)}</div>
    ${match.locked ? `<div class="resultNote">${resultNote(match)}</div>` : ""}
    <div class="probLine">
      <span>${match.home} ${fmtPct(match.home_win)}</span>
      <span>Draw ${fmtPct(match.draw)}</span>
      <span>${match.away} ${fmtPct(match.away_win)}</span>
    </div>
    <button class="simulateMatch" data-match-id="${match.match_id}" ${match.available && !match.locked ? "" : "disabled"}>
      ${match.locked ? "Played" : "Simulate"}
    </button>
  </article>`;
}

function standingsTable(group) {
  const rows = (liveState.standings[group] || []).map((row, idx) => `
    <tr class="${idx < 2 ? "advance" : idx === 2 ? "third" : ""}">
      <td>${idx + 1}</td><td>${row.team}</td><td>${row.played}</td><td>${row.points}</td><td>${row.gd}</td>
    </tr>
  `).join("");
  return `<table class="standings"><thead><tr><th></th><th>Team</th><th>P</th><th>Pts</th><th>GD</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderGroups() {
  const byGroup = {};
  for (const match of groupMatches()) {
    byGroup[match.group] = byGroup[match.group] || [];
    byGroup[match.group].push(match);
  }
  document.getElementById("groupCards").innerHTML = Object.keys(byGroup).sort().map((group) => `
    <section class="groupCard">
      <div class="groupHeader"><h3>Group ${group}</h3><span>${byGroup[group].filter((m) => m.locked).length}/6 played</span></div>
      ${standingsTable(group)}
      <div class="matchStack">${byGroup[group].map(matchCard).join("")}</div>
    </section>
  `).join("");
  wireMatchCards();
}

function knockoutMatches() {
  return liveState.matches.filter((match) => match.stage !== "Group");
}

// ---------------------------------------------------------------------------
// Bracket slot helpers
// ---------------------------------------------------------------------------
const ROUND_BASE = { 'Round of 32': 72, 'Round of 16': 88, 'Quarter-final': 96, 'Semi-final': 100 };

function parseSlotPart(part) {
  part = part.trim();
  const m = part.match(/Winner (Round of 32|Round of 16|Quarter-final|Semi-final) (\d+)/);
  if (m) return 'W' + (ROUND_BASE[m[1]] + parseInt(m[2]));
  return part;
}

function parseMatchSlots(slotStr) {
  if (!slotStr) return [null, null];
  if (slotStr === 'Semi-final losers') return ['L101', 'L102'];
  if (slotStr === 'Semi-final winners') return ['W101', 'W102'];
  const parts = slotStr.split(' vs ');
  if (parts.length !== 2) return [null, null];
  return [parseSlotPart(parts[0]), parseSlotPart(parts[1])];
}

// Pool of groups each variable 3rd-place slot can draw from (from tournament.yaml)
const THIRD_PLACE_POOLS = {
  A: ['C','E','F','H','I'], B: ['E','F','G','I','J'],
  D: ['B','E','F','I','J'], E: ['A','B','C','D','F'],
  G: ['A','E','H','I','J'], I: ['C','D','F','G','H'],
  K: ['D','E','I','J','L'], L: ['E','H','I','J','K'],
};

function computeDoneGroups(matches) {
  const counts = {};
  for (const m of matches) {
    if (m.stage !== 'Group') continue;
    if (!counts[m.group]) counts[m.group] = { total: 0, locked: 0 };
    counts[m.group].total++;
    if (m.locked) counts[m.group].locked++;
  }
  const done = new Set();
  for (const [g, v] of Object.entries(counts)) {
    if (v.total > 0 && v.total === v.locked) done.add(g);
  }
  return done;
}

function confirmedForSlot(slot, doneGroups, byNo) {
  if (!slot || slot.startsWith('3~')) return false;
  if (slot.startsWith('W')) {
    const n = parseInt(slot.slice(1));
    const m = byNo && byNo[n];
    return !!(m && m.winner);
  }
  if (slot.startsWith('L')) {
    const n = parseInt(slot.slice(1));
    const m = byNo && byNo[n];
    return !!(m && m.winner);
  }
  return doneGroups.has(slot.slice(1));
}

function likelyForSlot(slot, byNo, stds) {
  if (!slot) return null;
  if (slot.startsWith('W')) {
    const n = parseInt(slot.slice(1));
    const m = byNo && byNo[n];
    if (!m) return null;
    if (m.winner) return m.winner;
    if (m.home_win != null) return m.home_win >= 0.5 ? m.home : m.away;
    return null;
  }
  if (slot.startsWith('L')) {
    const n = parseInt(slot.slice(1));
    const m = byNo && byNo[n];
    if (!m || !m.winner) return null;
    return m.winner === m.home ? m.away : m.home;
  }
  // Variable 3rd-place slots: assignment depends on Annex C combinatorics,
  // only resolved once all groups finish — show nothing to avoid misleading duplicates.
  if (slot.startsWith('3~')) return null;
  const pos = parseInt(slot[0]) - 1;
  const grp = slot.slice(1);
  if (!stds || !stds[grp] || stds[grp].length <= pos) return null;
  return stds[grp][pos].team || null;
}

const R32_SLOTS = [
  ['2A','2B'],['1C','2F'],['1F','2C'],['2E','2I'],
  ['1H','2J'],['2K','2L'],['2D','2G'],['1J','2H'],
  ['1A','3~A'],['1B','3~B'],['1D','3~D'],['1E','3~E'],
  ['1G','3~G'],['1I','3~I'],['1K','3~K'],['1L','3~L'],
];
const R16_SLOTS = [
  ['W73','W75'],['W84','W86'],['W78','W77'],['W83','W85'],
  ['W74','W76'],['W81','W88'],['W80','W79'],['W82','W87'],
];
const QF_SLOTS  = [['W89','W90'],['W91','W92'],['W93','W94'],['W95','W96']];
const SF_SLOTS  = [['W97','W98'],['W99','W100']];
const FIN_SLOT  = ['W101','W102'];

function bMatchCard(match, hSlot, aSlot, byNo, stds, doneGroups) {
  const dg = doneGroups || new Set();

  function slotNode(slot, team) {
    if (team) return { dot: colors(team)[0], node: `<span class="bTeamName">${team}</span>` };
    const l = likelyForSlot(slot, byNo, stds);
    if (!l) return { dot: '#ddd', node: '' };
    if (confirmedForSlot(slot, dg, byNo)) {
      return { dot: colors(l)[0], node: `<span class="bTeamName bConfirmed">${l}</span>` };
    }
    return { dot: '#8899aa', node: `<span class="bLikely">~ ${l}</span>` };
  }

  if (!match) {
    const h = slotNode(hSlot, null);
    const a = slotNode(aSlot, null);
    const cls = (h.node || a.node) ? 'bCardPending' : 'bCardPending bCardEmpty';
    return `<div class="bCard ${cls}">
      <div class="bTeamRow"><span class="bDot" style="background:${h.dot}"></span>${h.node}</div>
      <div class="bTeamRow"><span class="bDot" style="background:${a.dot}"></span>${a.node}</div>
      <div class="bInfoRow"></div>
    </div>`;
  }
  const cls = match.locked ? "bCardPlayed" : match.available ? "bCardReady" : "bCardLocked";
  const hw = match.locked && match.winner === match.home;
  const aw = match.locked && match.winner === match.away;
  let infoRow = "";
  if (match.available && !match.locked) {
    infoRow = `<button class="bSimBtn simulateMatch" data-match-id="${match.match_id}">▶ Sim</button>`;
  } else if (match.locked && match.went_to_shootout) {
    infoRow = `<span class="bInfo">Pens: ${match.penalties_home}–${match.penalties_away}</span>`;
  } else if (match.locked && match.went_to_et) {
    infoRow = `<span class="bInfo">AET</span>`;
  }
  const h = slotNode(hSlot, match.home);
  const a = slotNode(aSlot, match.away);
  return `<div class="bCard ${cls}" data-match-id="${match.match_id}">
    <div class="bTeamRow ${hw ? "bWinner" : ""}">
      <span class="bDot" style="background:${h.dot}"></span>
      ${h.node}
      <span class="bScr">${match.locked ? match.home_score : (match.home_win != null && match.home ? fmtPct(match.home_win) : "")}</span>
    </div>
    <div class="bTeamRow ${aw ? "bWinner" : ""}">
      <span class="bDot" style="background:${a.dot}"></span>
      ${a.node}
      <span class="bScr">${match.locked ? match.away_score : ""}</span>
    </div>
    <div class="bInfoRow">${infoRow}</div>
  </div>`;
}

function renderBracket() {
  const board = document.getElementById("bracketBoard");
  const byId = {}, byNo = {};
  for (const m of liveState.matches) {
    byId[m.match_id] = m;
    byNo[m.match_no]  = m;
  }
  const stds = liveState.standings || {};
  const doneGroups = computeDoneGroups(liveState.matches);

  function card(matchId, slotDef) {
    const m = byId[matchId] || null;
    const [hs, as] = m ? parseMatchSlots(m.slot || '') : slotDef;
    return bMatchCard(m, hs, as, byNo, stds, doneGroups);
  }

  const r32 = (i) => card(`R32${i+1}`, R32_SLOTS[i]);
  const r16 = (i) => card(`R16${i+1}`, R16_SLOTS[i]);
  const qf  = (i) => card(`QF${i+1}`,  QF_SLOTS[i]);
  const sf  = (i) => card(`SF${i+1}`,  SF_SLOTS[i]);
  const finM = byId['F1'] || null;
  const [fhs, fas] = finM ? parseMatchSlots(finM.slot || '') : FIN_SLOT;

  const pair   = (a, b) => `<div class="bPair">${a}${b}</div>`;
  const single = (m)    => `<div class="bSingle">${m}</div>`;

  board.innerHTML = `
    <div class="bTree">
      <div class="bTreeLabels">
        <div>Round of 32</div><div>Round of 16</div><div>Quarter-finals</div><div>Semi-finals</div>
        <div>⚽ Final</div>
        <div>Semi-finals</div><div>Quarter-finals</div><div>Round of 16</div><div>Round of 32</div>
      </div>
      <div class="bTreeBody">
        <div class="bRound bRoundLeft">
          ${pair(r32(0),r32(2))}${pair(r32(11),r32(13))}${pair(r32(5),r32(4))}${pair(r32(10),r32(12))}
        </div>
        <div class="bRound bRoundLeft">
          ${pair(r16(0),r16(1))}${pair(r16(2),r16(3))}
        </div>
        <div class="bRound bRoundLeft">
          ${pair(qf(0),qf(1))}
        </div>
        <div class="bRound bRoundLeft">
          ${single(sf(0))}
        </div>
        <div class="bCenter">
          ${bMatchCard(finM, fhs, fas, byNo, stds, doneGroups)}
          ${liveState.champion
            ? `<div class="bChampion">🏆 ${liveState.champion}</div>`
            : `<div class="bChampionPlaceholder">Champion TBD</div>`}
        </div>
        <div class="bRound bRoundRight">
          ${single(sf(1))}
        </div>
        <div class="bRound bRoundRight">
          ${pair(qf(2),qf(3))}
        </div>
        <div class="bRound bRoundRight">
          ${pair(r16(4),r16(5))}${pair(r16(6),r16(7))}
        </div>
        <div class="bRound bRoundRight">
          ${pair(r32(1),r32(3))}${pair(r32(8),r32(15))}${pair(r32(7),r32(6))}${pair(r32(9),r32(14))}
        </div>
      </div>
    </div>
  `;
  wireMatchCards();

  const koPending = liveState.matches.filter((m) => m.stage !== "Group" && m.available && !m.locked);
  const simKnockBtn = document.getElementById("simKnockoutBtn");
  if (simKnockBtn) {
    simKnockBtn.style.display = koPending.length ? "" : "none";
    if (koPending.length) simKnockBtn.textContent = `Simulate ${koPending[0].stage}`;
  }
}

function renderResults() {
  document.getElementById("resultRows").innerHTML = liveState.matches
    .slice()
    .sort((a, b) => a.match_no - b.match_no)
    .map((match) => `
    <tr><td>${match.match_no}</td><td>${match.stage === "Group" ? `Group ${match.group}` : match.stage}</td><td>${match.home || "TBD"}</td><td>${match.locked ? `${match.home_score}-${match.away_score}` : "-"}</td><td>${match.away || "TBD"}</td><td>${match.locked ? "Final" : "Scheduled"}</td></tr>
  `).join("");
}

function renderAll() {
  renderScoreboard();
  renderGroups();
  renderBracket();
  renderResults();
}

function wireMatchCards() {
  document.querySelectorAll(".matchCard, .bCard").forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.target.closest("button")) return;
      const match = liveState.matches.find((m) => m.match_id === card.dataset.matchId);
      if (match) openMatchDrawer(match);
    });
  });
  document.querySelectorAll(".simulateMatch").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      await simulateMatch(button.dataset.matchId);
    });
  });
}

function openMatchDrawer(match) {
  const pens = match.went_to_shootout
    ? `${match.penalties_home} - ${match.penalties_away}`
    : "No";
  openDrawer(`${match.home} vs ${match.away}`, `
    <div class="drawerGrid">
      <div class="drawerMetric"><span>Stage</span><strong>${match.stage}</strong></div>
      <div class="drawerMetric"><span>Status</span><strong>${match.locked ? "Final" : match.available ? "Ready" : "Locked"}</strong></div>
      <div class="drawerMetric"><span>Score</span><strong>${match.locked ? `${match.home_score}-${match.away_score}` : "Hidden"}</strong></div>
      <div class="drawerMetric"><span>Winner</span><strong>${match.winner || (match.locked ? "Draw" : "Hidden")}</strong></div>
      <div class="drawerMetric"><span>${match.home} win</span><strong>${fmtPct(match.home_win)}</strong></div>
      <div class="drawerMetric"><span>Draw</span><strong>${fmtPct(match.draw)}</strong></div>
      <div class="drawerMetric"><span>${match.away} win</span><strong>${fmtPct(match.away_win)}</strong></div>
      <div class="drawerMetric"><span>Extra time</span><strong>${match.went_to_et ? "Yes" : "No"}</strong></div>
      <div class="drawerMetric"><span>Penalties</span><strong>${pens}</strong></div>
      ${match.locked ? `<div class="drawerMetric drawerNote"><span>Decision</span><strong>${resultNote(match)}</strong></div>` : ""}
    </div>
  `);
}

function openDrawer(title, body) {
  document.getElementById("drawerTitle").textContent = title;
  document.getElementById("drawerBody").innerHTML = body;
  document.getElementById("detailDrawer").classList.add("open");
}

function closeDrawer() {
  document.getElementById("detailDrawer").classList.remove("open");
}

async function loadLiveState() {
  const res = await fetch("/api/live-tournament");
  liveState = await res.json();
  renderAll();
}

async function simulateMatch(matchId) {
  setBusy(true);
  try {
    const res = await fetch(`/api/live-tournament/matches/${matchId}/simulate`, { method: "POST" });
    if (res.ok) {
      liveState = await res.json();
      renderAll();
    }
  } catch (err) {
    console.error("simulateMatch failed:", err);
  } finally {
    setBusy(false);
  }
}

async function simulateKnockoutRound() {
  const pending = liveState.matches.filter((m) => m.stage !== "Group" && m.available && !m.locked);
  if (!pending.length) return;
  setBusy(true);
  try {
    for (const match of pending) {
      const res = await fetch(`/api/live-tournament/matches/${match.match_id}/simulate`, { method: "POST" });
      if (!res.ok) break;
      liveState = await res.json();
      renderAll();
    }
  } catch (err) {
    console.error("simulateKnockoutRound failed:", err);
  } finally {
    setBusy(false);
  }
}

async function simulateAllGroupStage() {
  const pending = liveState.matches.filter((m) => m.stage === "Group" && !m.locked);
  if (!pending.length) return;
  setBusy(true);
  try {
    for (const match of pending) {
      const res = await fetch(`/api/live-tournament/matches/${match.match_id}/simulate`, { method: "POST" });
      if (!res.ok) break;
      liveState = await res.json();
      renderAll();
    }
  } catch (err) {
    console.error("simulateAllGroupStage failed:", err);
  } finally {
    setBusy(false);
  }
}

async function resetTournament() {
  setBusy(true);
  try {
    const res = await fetch("/api/live-tournament/reset", { method: "POST" });
    liveState = await res.json();
    renderAll();
  } catch (err) {
    console.error("resetTournament failed:", err);
  } finally {
    setBusy(false);
  }
}

let metricsData = null;

async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) return;
    metricsData = await res.json();
  } catch (_) {
    return;
  }
  renderMetrics();
}

function metricCard(label, value) {
  return `<div class="metricCard"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderMetrics() {
  if (!metricsData) return;
  const { simulation: sim, model_quality: mq } = metricsData;

  document.getElementById("simQualityCards").innerHTML = sim.available ? [
    metricCard("Simulations", sim.n_runs.toLocaleString()),
    metricCard("Mean Champion %", fmtPct(sim.mean_p_champion)),
    metricCard("Std Dev", fmtPct(sim.std_p_champion)),
    metricCard("Mean Std Error", fmtPct(sim.mean_se)),
    metricCard("Mean 95% CI Width", fmtPct(sim.mean_ci_width)),
  ].join("") : `<p class="simEmpty">Run a simulation first.</p>`;

  document.getElementById("convergenceChart").innerHTML = sim.available
    ? renderConvergenceSVG(sim)
    : `<div class="chartPlaceholder">Run a simulation first.</div>`;

  document.getElementById("seTableBody").innerHTML = (sim.available ? sim.teams : []).map((t) => `
    <tr>
      <td>${teamPill(t.team)}</td>
      <td>${fmtPct(t.p_champion)}</td>
      <td>${fmtPct(t.se)}</td>
      <td>${fmtPct(t.ci_lower)} – ${fmtPct(t.ci_upper)}</td>
    </tr>
  `).join("");

  const mqSection = document.getElementById("modelQualitySection");
  if (mq.available) {
    mqSection.style.display = "block";
    document.getElementById("modelQualityCards").innerHTML = [
      metricCard("Brier Score", mq.brier.toFixed(6)),
      metricCard("Log Loss", mq.log_loss.toFixed(4)),
    ].join("");
    if (mq.captured_at) {
      document.getElementById("metricsTimestamp").textContent = new Date(mq.captured_at).toLocaleString();
    }
    document.getElementById("calibrationChart").innerHTML = renderCalibrationSVG(mq.reliability_bins);
  } else {
    mqSection.style.display = "none";
  }

  renderSliceMetrics(metricsData.slices || []);
}

function renderSliceMetrics(slices) {
  const section = document.getElementById("sliceMetricsSection");
  const tbody = document.getElementById("sliceTableBody");
  if (!slices || !slices.length) {
    section.style.display = "none";
    return;
  }
  section.style.display = "block";
  tbody.innerHTML = slices.map((s) => {
    const biasColor = s.large_bias ? "color:var(--wine);font-weight:850" : "color:var(--grass)";
    const biasSign = s.W_bias > 0 ? "+" : "";
    const gBias = s.goals_bias != null ? s.goals_bias : 0;
    const gSign = gBias > 0 ? "+" : "";
    return `<tr>
      <td>${s.bucket}</td>
      <td>${s.n}</td>
      <td>${fmtPct(s.pred_W)}</td>
      <td>${fmtPct(s.actual_W)}</td>
      <td style="${biasColor}">${biasSign}${(s.W_bias * 100).toFixed(1)}pp</td>
      <td>${fmtPct(s.pred_D)}</td>
      <td>${fmtPct(s.actual_D)}</td>
      <td>${gSign}${gBias.toFixed(3)}</td>
      <td>${s.brier.toFixed(4)}</td>
      <td>${s.log_loss.toFixed(3)}</td>
      <td style="${biasColor}">${s.large_bias ? "⚠ Bias" : "✓ OK"}</td>
    </tr>`;
  }).join("");
}

function renderConvergenceSVG(sim) {
  const top5 = sim.teams.slice(0, 5);
  if (!top5.length) return `<div class="chartPlaceholder">No team data.</div>`;
  const W = 560, H = 240;
  const pad = { top: 18, right: 24, bottom: 46, left: 58 };
  const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;
  const ns = [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000];
  const logMin = Math.log(ns[0]), logMax = Math.log(ns[ns.length - 1]);
  const xS = (n) => pad.left + (Math.log(n) - logMin) / (logMax - logMin) * cW;
  const maxSE = Math.max(...top5.map((t) => Math.sqrt(t.p_champion * (1 - t.p_champion) / ns[0])), 1e-6);
  const yS = (se) => pad.top + cH - (se / maxSE) * cH;
  const colors = ["#00a7a5", "#9b174c", "#f4c430", "#0e7a4f", "#3c7dd1"];

  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const se = f * maxSE, y = yS(se);
    return `<line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${(pad.left + cW).toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(7,27,37,0.07)" stroke-width="1"/>
            <text x="${(pad.left - 6).toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-size="10" fill="#647780">${(se * 100).toFixed(2)}%</text>`;
  }).join("");

  const xTicks = [1000, 5000, 10000, 25000, 50000, 100000, 200000].map((n) => {
    const x = xS(n), label = `${n / 1000}k`;
    return `<line x1="${x.toFixed(1)}" y1="${(pad.top + cH).toFixed(1)}" x2="${x.toFixed(1)}" y2="${(pad.top + cH + 5).toFixed(1)}" stroke="#647780" stroke-width="1"/>
            <text x="${x.toFixed(1)}" y="${(pad.top + cH + 16).toFixed(1)}" text-anchor="middle" font-size="10" fill="#647780">${label}</text>`;
  }).join("");

  const curN = Math.min(Math.max(sim.n_runs, ns[0]), ns[ns.length - 1]);
  const cx = xS(curN);
  const curLine = `<line x1="${cx.toFixed(1)}" y1="${pad.top}" x2="${cx.toFixed(1)}" y2="${(pad.top + cH).toFixed(1)}" stroke="#f4c430" stroke-width="1.5" stroke-dasharray="4,3"/>
                   <text x="${(cx + 4).toFixed(1)}" y="${(pad.top + 13).toFixed(1)}" font-size="10" fill="#f4c430" font-weight="bold">n=${sim.n_runs.toLocaleString()}</text>`;

  const lines = top5.map((t, i) => {
    const d = ns.map((n) => {
      const se = Math.sqrt(t.p_champion * (1 - t.p_champion) / n);
      return `${xS(n).toFixed(1)},${yS(se).toFixed(1)}`;
    }).join("L");
    return `<path d="M${d}" fill="none" stroke="${colors[i]}" stroke-width="2"/>`;
  }).join("");

  const legend = top5.map((t, i) =>
    `<rect x="${(pad.left + i * 106).toFixed(1)}" y="${(H - 10).toFixed(1)}" width="8" height="8" fill="${colors[i]}" rx="2"/>
     <text x="${(pad.left + i * 106 + 11).toFixed(1)}" y="${(H - 2).toFixed(1)}" font-size="10" fill="#647780">${t.team.slice(0, 13)}</text>`
  ).join("");

  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block">
    <rect x="${pad.left}" y="${pad.top}" width="${cW}" height="${cH}" fill="rgba(7,27,37,0.02)" rx="4"/>
    ${grid}${xTicks}
    <text x="${(pad.left + cW / 2).toFixed(1)}" y="${(pad.top + cH + 34).toFixed(1)}" text-anchor="middle" font-size="11" fill="#647780">Simulations (n)</text>
    <text x="14" y="${(pad.top + cH / 2).toFixed(1)}" text-anchor="middle" font-size="11" fill="#647780" transform="rotate(-90,14,${(pad.top + cH / 2).toFixed(1)})">Std Error</text>
    ${lines}${curLine}${legend}
  </svg>`;
}

function renderCalibrationSVG(bins) {
  if (!bins || !bins.length) return `<div class="chartPlaceholder">No calibration data yet.</div>`;
  const W = 320, H = 290;
  const pad = { top: 18, right: 18, bottom: 48, left: 52 };
  const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;
  const xS = (v) => pad.left + v * cW;
  const yS = (v) => pad.top + cH - v * cH;

  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) => {
    const y = yS(v), x = xS(v);
    return `<line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${(pad.left + cW).toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(7,27,37,0.07)" stroke-width="1"/>
            <text x="${(pad.left - 6).toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-size="10" fill="#647780">${(v * 100).toFixed(0)}%</text>
            <line x1="${x.toFixed(1)}" y1="${pad.top}" x2="${x.toFixed(1)}" y2="${(pad.top + cH).toFixed(1)}" stroke="rgba(7,27,37,0.07)" stroke-width="1"/>
            <text x="${x.toFixed(1)}" y="${(pad.top + cH + 16).toFixed(1)}" text-anchor="middle" font-size="10" fill="#647780">${(v * 100).toFixed(0)}%</text>`;
  }).join("");

  const diagonal = `<line x1="${xS(0)}" y1="${yS(0)}" x2="${xS(1).toFixed(1)}" y2="${yS(1).toFixed(1)}" stroke="rgba(7,27,37,0.22)" stroke-width="1.5" stroke-dasharray="6,4"/>`;

  const maxN = Math.max(...bins.map((b) => b.n));
  const dots = bins.map((b) => {
    const cx = xS(b.mean_pred).toFixed(1), cy = yS(b.observed_rate).toFixed(1);
    const r = (3 + (b.n / maxN) * 9).toFixed(1);
    return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="rgba(0,167,165,0.65)" stroke="#00a7a5" stroke-width="1.5">
              <title>n=${b.n} · pred ${(b.mean_pred * 100).toFixed(1)}% · actual ${(b.observed_rate * 100).toFixed(1)}%</title>
            </circle>`;
  }).join("");

  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:360px;height:auto;display:block">
    <rect x="${pad.left}" y="${pad.top}" width="${cW}" height="${cH}" fill="rgba(7,27,37,0.02)" rx="4"/>
    ${grid}${diagonal}${dots}
    <text x="${(pad.left + cW / 2).toFixed(1)}" y="${(pad.top + cH + 36).toFixed(1)}" text-anchor="middle" font-size="11" fill="#647780">Predicted Probability</text>
    <text x="14" y="${(pad.top + cH / 2).toFixed(1)}" text-anchor="middle" font-size="11" fill="#647780" transform="rotate(-90,14,${(pad.top + cH / 2).toFixed(1)})">Observed Rate</text>
  </svg>`;
}

async function runMonteCarloSimulation() {
  const nRuns = parseInt(document.getElementById("nRunsInput").value, 10) || 25000;
  const statusEl = document.getElementById("simJobState");
  setBusy(true);
  statusEl.textContent = "Starting…";
  const res = await fetch(`/api/simulate?n_runs=${nRuns}`, { method: "POST" });
  if (!res.ok) {
    statusEl.textContent = "Failed to start";
    setBusy(false);
    return;
  }
  const job = await res.json();
  if (!job.accepted) {
    statusEl.textContent = "Job already running";
    setBusy(false);
    return;
  }
  statusEl.textContent = `Running ${nRuns.toLocaleString()} simulations…`;
  pollSimJob();
}

function pollSimJob() {
  if (simPollTimer) clearInterval(simPollTimer);
  const statusEl = document.getElementById("simJobState");
  simPollTimer = setInterval(async () => {
    const res = await fetch("/api/results/job");
    if (!res.ok) return;
    const job = await res.json();
    if (job.status === "complete") {
      clearInterval(simPollTimer);
      simPollTimer = null;
      statusEl.textContent = "Complete";
      await Promise.all([loadSimResults(), loadMetrics()]);
      setBusy(false);
    } else if (job.status === "failed") {
      clearInterval(simPollTimer);
      simPollTimer = null;
      statusEl.textContent = `Failed: ${job.error || "unknown error"}`;
      setBusy(false);
    }
  }, 1000);
}

async function loadSimResults() {
  const res = await fetch("/api/dashboard");
  if (!res.ok) return;
  const data = await res.json();
  simResults = data.probabilities || [];
  renderMonteCarlo();
}

function probBar(value, maxVal) {
  if (value == null) return "-";
  const pct = (value * 100).toFixed(1);
  const w = maxVal > 0 ? ((value / maxVal) * 100).toFixed(1) : "0";
  return `<span class="champBar" style="--w:${w}%">${pct}%</span>`;
}

function renderMonteCarlo() {
  const tbody = document.getElementById("simResultRows");
  if (!tbody) return;
  if (!simResults.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="simEmpty">No results yet — run a simulation above.</td></tr>`;
    return;
  }
  const maxChamp = Math.max(...simResults.map((r) => r.p_champion || 0));
  tbody.innerHTML = simResults.map((row) => `
    <tr>
      <td>${teamPill(row.team)}</td>
      <td class="champCell">${probBar(row.p_champion, maxChamp)}</td>
      <td>${fmtPct(row.p_reach_final)}</td>
      <td>${fmtPct(row.p_reach_sf)}</td>
      <td>${fmtPct(row.p_reach_qf)}</td>
      <td>${fmtPct(row.p_reach_r16)}</td>
      <td>${fmtPct(row.p_reach_r32)}</td>
    </tr>
  `).join("");
}

async function loadUpcoming() {
  try {
    const res = await fetch("/api/upcoming");
    if (!res.ok) return;
    const data = await res.json();
    renderUpcoming(data.upcoming || [], data.error);
  } catch (_) {}
}

function renderUpcoming(matches, error) {
  const section = document.getElementById("upcomingSection");
  const grid = document.getElementById("upcomingCards");
  if (error && !matches.length) {
    section.style.display = "block";
    grid.innerHTML = `<p class="simEmpty" style="color:var(--wine)">${error}</p>`;
    return;
  }
  if (!matches.length) {
    section.style.display = "none";
    return;
  }
  section.style.display = "block";
  grid.innerHTML = matches.map((m) => {
    const dt = m.date ? new Date(m.date) : null;
    const timeStr = dt ? dt.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
    const liveBadge = m.live ? `<span class="liveBadge">&#9679; LIVE ${m.clock || ""}</span>` : "";
    const scoreStr = m.live ? `${m.live_home ?? ""} – ${m.live_away ?? ""}` : "";
    return `<div class="upcomingCard ${m.live ? "upcomingLive" : ""}">` +
      `<div class="upcomingTeams">${teamPill(m.home)} <span class="upcomingVs">${scoreStr || "vs"}</span> ${teamPill(m.away)}</div>` +
      `<div class="upcomingMeta">${timeStr} ${liveBadge}</div>` +
      `</div>`;
  }).join("");
}

async function refreshResults() {
  const statusEl = document.getElementById("jobState");
  statusEl.textContent = "Fetching live results…";
  setBusy(true);
  const res = await fetch("/api/results/refresh", { method: "POST" });
  if (!res.ok) {
    statusEl.textContent = "Failed to start refresh";
    setBusy(false);
    return;
  }
  const job = await res.json();
  if (!job.accepted) {
    statusEl.textContent = "A job is already running";
    setBusy(false);
    return;
  }
  pollRefreshJob(false);
}

function pollRefreshJob(silent = false) {
  if (simPollTimer) clearInterval(simPollTimer);
  const statusEl = document.getElementById("jobState");
  simPollTimer = setInterval(async () => {
    const res = await fetch("/api/results/job");
    if (!res.ok) return;
    const job = await res.json();
    if (!silent) statusEl.textContent = job.message || job.status;
    if (job.status === "complete") {
      clearInterval(simPollTimer);
      simPollTimer = null;
      if (!silent) statusEl.textContent = job.message || "Results updated";
      await Promise.all([loadLiveState(), loadUpcoming(), loadSimResults(), loadMetrics()]);
      autoRefreshInFlight = false;
      if (!silent) setBusy(false);
    } else if (job.status === "failed") {
      clearInterval(simPollTimer);
      simPollTimer = null;
      if (!silent) statusEl.textContent = `Refresh failed: ${job.error || "unknown error"}`;
      autoRefreshInFlight = false;
      if (!silent) setBusy(false);
    }
  }, 1500);
}

async function autoRefreshResults() {
  if (autoRefreshInFlight) return;
  try {
    const jobRes = await fetch("/api/results/job");
    if (jobRes.ok) {
      const job = await jobRes.json();
      if (job.status === "running") return;
    }
    autoRefreshInFlight = true;
    const res = await fetch("/api/results/refresh", { method: "POST" });
    if (!res.ok) {
      autoRefreshInFlight = false;
      return;
    }
    const job = await res.json();
    if (job.accepted) {
      pollRefreshJob(true);
    } else {
      autoRefreshInFlight = false;
    }
  } catch (_) {
    autoRefreshInFlight = false;
  }
}

async function startRetrain() {
  const statusEl = document.getElementById("retrainStatus");
  statusEl.style.display = "block";
  statusEl.textContent = "Starting retrain…";
  setBusy(true);
  const res = await fetch("/api/retrain", { method: "POST" });
  if (!res.ok) {
    statusEl.textContent = "Failed to start retrain.";
    setBusy(false);
    return;
  }
  const job = await res.json();
  if (!job.accepted) {
    statusEl.textContent = "A job is already running.";
    setBusy(false);
    return;
  }
  statusEl.textContent = "Retraining model…";
  pollRetrainJob();
}

function pollRetrainJob() {
  if (simPollTimer) clearInterval(simPollTimer);
  const statusEl = document.getElementById("retrainStatus");
  simPollTimer = setInterval(async () => {
    const res = await fetch("/api/results/job");
    if (!res.ok) return;
    const job = await res.json();
    statusEl.textContent = job.message || job.status;
    if (job.status === "complete") {
      clearInterval(simPollTimer);
      simPollTimer = null;
      await Promise.all([loadSimResults(), loadMetrics()]);
      setBusy(false);
    } else if (job.status === "failed") {
      clearInterval(simPollTimer);
      simPollTimer = null;
      statusEl.textContent = `Retrain failed: ${job.error || "unknown error"}`;
      setBusy(false);
    }
  }, 1500);
}

function setBusy(value) {
  const excluded = new Set(["drawerClose", "chatBubble", "chatClose", "chatSend"]);
  document.querySelectorAll("button").forEach((button) => {
    if (!excluded.has(button.id)) button.disabled = value && !button.classList.contains("simulateMatch");
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((el) => el.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((el) => el.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "metrics") loadMetrics();
  });
});

document.getElementById("searchBox").addEventListener("input", renderGroups);
document.getElementById("resetBtn").addEventListener("click", resetTournament);
document.getElementById("simGroupBtn").addEventListener("click", simulateAllGroupStage);
document.getElementById("simKnockoutBtn").addEventListener("click", simulateKnockoutRound);
document.getElementById("refreshBtn").addEventListener("click", refreshResults);
document.getElementById("retrainBtn").addEventListener("click", startRetrain);
document.getElementById("runSimBtn").addEventListener("click", runMonteCarloSimulation);
document.getElementById("drawerClose").addEventListener("click", closeDrawer);

loadLiveState();
loadSimResults();
loadMetrics();
loadUpcoming();
setInterval(loadUpcoming, 90000);
setInterval(autoRefreshResults, 120000);

// ---- Chat ----
let chatHistory = [];

function toggleChat() {
  const panel = document.getElementById("chatPanel");
  const bubble = document.getElementById("chatBubble");
  const willOpen = !panel.classList.contains("chatPanelOpen");
  panel.classList.toggle("chatPanelOpen", willOpen);
  panel.setAttribute("aria-hidden", willOpen ? "false" : "true");
  bubble.setAttribute("aria-expanded", willOpen ? "true" : "false");
  if (willOpen) {
    document.getElementById("chatInput").focus();
  }
}

function appendChatMsg(role, text) {
  const box = document.getElementById("chatMessages");
  const isUser = role === "user";
  const div = document.createElement("div");
  div.className = `chatMsg ${isUser ? "chatMsgUser" : "chatMsgAssistant"}`;
  const bubble = document.createElement("div");
  bubble.className = "chatMsgBubble";
  bubble.textContent = text;
  div.appendChild(bubble);
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function appendTypingIndicator() {
  const box = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.id = "chatTyping";
  div.className = "chatMsg chatMsgAssistant";
  div.innerHTML = `<div class="chatTypingBubble">
    <span></span>
    <span></span>
    <span></span>
  </div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("chatTyping");
  if (el) el.remove();
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  input.disabled = true;
  document.getElementById("chatSend").disabled = true;

  appendChatMsg("user", message);
  chatHistory.push({ role: "user", content: message });
  appendTypingIndicator();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: chatHistory.slice(-20) }),
    });
    const data = await res.json();
    removeTypingIndicator();
    const reply = data.reply || "Sorry, I couldn't get a response.";
    appendChatMsg("assistant", reply);
    chatHistory.push({ role: "assistant", content: reply });
  } catch (_) {
    removeTypingIndicator();
    appendChatMsg("assistant", "⚠️ Network error — please try again.");
  } finally {
    input.disabled = false;
    document.getElementById("chatSend").disabled = false;
    input.focus();
  }
}

document.getElementById("chatBubble").addEventListener("click", toggleChat);
document.getElementById("chatClose").addEventListener("click", toggleChat);
document.getElementById("chatSend").addEventListener("click", sendChat);
document.getElementById("chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
