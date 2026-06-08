# worldcup-predictor

A calibrated probability model for the 2026 FIFA World Cup. ML at the core, the Claude API at the boundaries.

A goals-based match model feeds a Monte Carlo simulator that plays the 48-team tournament 50,000 times and reports each team's odds of reaching every stage. The Claude API reads raw text and emits *structure* — it never produces a number. The success bar is matching the calibration of the de-vigged closing line, not "calling the winner."

## What's where

- `CLAUDE.md` — orientation + the five invariants (read first)
- `docs/` — `arch.md`, `ai_rules.md`, `folder_struct.md`, `setup.md`, `windsurf-prompts.md`, `data_sources.md`, `data_exp.md`
- `config/` — `tournament.yaml`, `teams.yaml` (the real final draw), `settings.yaml`
- `src/wcpredict/` — the package
- `tests/` — correctness-critical tests live here (sampler, bracket, adjust)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...        # or put it in .env
```

## Run

```bash
wcpredict build-data    # ingest + join -> match table + features
wcpredict fit           # train the match model on structured history
wcpredict simulate      # 50k tournaments -> stage/finalist/champion probabilities
wcpredict evaluate      # de-vig the market + score calibration (with vs without the LLM adjustment)
wcpredict update --results <match-day>   # fix results, apply confirmed lineups, re-simulate
```

## The non-negotiables

1. The Claude API emits structure, never magnitude — all numbers come from calibrated code.
2. LLM-extracted signal is an inference-time adjustment, never a training column.
3. Market/odds data lives only in `eval/`. Never a feature.
4. The simulator samples; it never averages.
5. Entity resolution and explanation stay out of the training and eval loops.

See `docs/arch.md` for the full design and `docs/ai_rules.md` for the enforced rules.

## Status note

`config/tournament.yaml` leaves the round-of-32 slot pairings and best-third allocation table empty on purpose — those come from FIFA's official 2026 bracket and must be filled before `simulate` runs. Everything else is wired.
