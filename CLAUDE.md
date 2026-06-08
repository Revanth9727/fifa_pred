# CLAUDE.md

Read this first. It orients you; the details live in `docs/`.

## What this is

A system that predicts the 2026 FIFA World Cup (48 teams, US/Canada/Mexico, June–July 2026). The output is a **calibrated probability distribution** over outcomes — per-team odds of reaching each stage, the final, and the title — not a single pick.

It is **ML at the core, LLM at the boundaries.** A statistical match model plus a Monte Carlo tournament simulator own all the numbers. An LLM reads unstructured text and emits *structure* that feeds, explains, or cleans those numbers — it never produces a number itself.

## The success criterion

Not "did we call the champion." The bar is **matching the calibration of the de-vigged closing line.** Worse-calibrated than the close → the model is wrong. Matching it → the model is good. Consistently beating it → suspect a bug or leakage before celebrating.

## Invariants (do not violate — see `docs/ai_rules.md` for the full list)

1. **The LLM emits structure, never magnitude.** It returns typed categorical objects (status, role, source). All numbers come from a calibrated function or small auxiliary model.
2. **LLM-extracted signal is an inference-time adjustment, never a training column.** The match model trains only on structured historical data.
3. **Market data lives only in `eval/`.** It is the benchmark, never a feature. Anything that lets odds reach the model is a bug.
4. **The simulator samples, it never averages.** Resolving a match by comparing expected goals is the canonical failure.
5. **Roles 2 (entity resolution) and 3 (explanation) stay out of the training and eval loops entirely.**

## The shape, in one paragraph

Data → features → match model (goals-based: bivariate Poisson + Dixon–Coles) → a pre-match **adjustment layer** that applies LLM-extracted late information → the sampler (the seam) → Monte Carlo simulator over the 48-team bracket → outputs. A separate `eval/` module de-vigs the market and scores calibration. The adjustment layer has two channels: **availability** (shifts mean λ) and **lineup certainty** (controls the dispersion of a λ-mixture the sampler draws over). See `docs/arch.md`.

## Ethos

Calibrated, not clever. Decisive defaults over configurable everything. Every signal must earn its place through an A/B against the close — if the LLM adjustment doesn't improve calibration, cut it. Concentrate testing on the two correctness-critical files: `sim/sampler.py` and `sim/bracket.py`, plus the adjustment layer.

## Where to look

- `docs/arch.md` — full architecture and the seam contract
- `docs/ai_rules.md` — hard rules and conventions you must follow
- `docs/folder_struct.md` — the file layout and per-file responsibilities
- `docs/setup.md` — environment and how to run each stage
- `docs/windsurf-prompts.md` — the ordered build prompts
- `docs/data_sources.md` — where each data source comes from
- `docs/data_exp.md` — schemas and field meanings
