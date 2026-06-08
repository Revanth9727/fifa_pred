# data_sources.md

Where each input comes from, what it gives you, and how often to refresh. **Verify current access terms and URLs before relying on any source** — availability and licensing change. Downloads land in `data/raw/`; only `data/ingest.py` fetches them.

## Prediction inputs

| Source | Provides | Where | Refresh |
|---|---|---|---|
| International results (1872–present) | The training backbone: scores, date, teams, tournament/friendly, venue, neutral flag (~45k matches) | Kaggle dataset "International football results from 1872 to present" (Mart Jürisoo) | Once, then per match-day during the tournament |
| Elo ratings | Primary team-strength signal | eloratings.net (World Football Elo Ratings) — downloadable tables | Weekly / after each match |
| FIFA ranking / SPI | Secondary strength signals | fifa.com rankings; SPI from public sports-analytics sources | Per ranking release |
| Squad market values | Squad-quality proxy (rank signal, not level) | Transfermarkt — **ToS-sensitive; check scraping/usage terms or use a licensed export** | At squad announcement (~early June) |
| Club form / xG | Player and team form: goals, assists, xG/xA, minutes | FBref (sports-reference); StatsBomb open data (GitHub) | At squad announcement |
| Fixtures + venues | Schedule, venues, rest/travel computation | Official FIFA 2026 fixture list | Static; results filled live |

## LLM extraction inputs (role 1 — structure only)

| Source | Provides | Where | Refresh |
|---|---|---|---|
| Team news / injury reports | Raw text for availability + lineup-certainty extraction | News aggregators (e.g. GDELT, a news API), official club/national-team channels, reputable beat reporters | Daily near and during the tournament |
| Official lineups | Confirmed XI (triggers supersession) | Official match-day team-sheet releases (~1hr pre-kickoff) | Per match, at release |

**Leakage rule (ai_rules.md #4):** extraction runs on *raw* information only — injuries, team news, lineup announcements. Never market commentary, tipster content, or odds-derived punditry. Tag each item with a `source_type` (`official` / `beat_report` / `rumor`) at ingestion so the adjustment layer and eval can weight and ablate by trust tier.

## Benchmark only — never a feature (eval/ only)

| Source | Provides | Where | Refresh |
|---|---|---|---|
| Outright / futures odds | Pre-tournament champion + finalist market | Bookmaker futures markets / an odds API | Now, then periodically |
| Per-match closing lines | Single-game market for match-model calibration | Closing odds (e.g. Pinnacle) via an odds API; historical match odds from Football-Data.co.uk | Per match, at close |

These load only inside `eval/calibration.py`. `tests/test_no_leakage.py` enforces that no prediction module can reach them.

## Notes

- **Entity resolution (role 2):** names and club strings differ across Transfermarkt, FBref, Elo, and the official squad list. `data/entity_resolve.py` reconciles them at build time; validate the joins before trusting downstream features.
- **Mismatch coverage:** lopsided internationals are rare in the results data, so squad/strength signals are least reliable exactly where the group stage needs them. This is a known limitation, not a missing source.
- **De-vig before scoring:** futures carry a larger overround than match lines — normalize (proportional or Shin's) so probabilities sum to 1 before any comparison.
