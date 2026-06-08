#!/usr/bin/env python3
"""
Pull and freeze the 2026 World Cup squad snapshot from Wikipedia.

Output:
  data/raw/squads_2026_<cutoff>.csv
  outputs/squad_unmatched_<cutoff>.csv

The resolver is deterministic/offline by default. If ANTHROPIC_API_KEY is set,
team-name fallback can use data/entity_resolve.py, but player IDs are resolved
against the local Transfermarkt player table to keep prediction numeric inputs
outside the LLM boundary.
"""
from __future__ import annotations

import argparse
import difflib
from io import StringIO
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpredict.data.entity_resolve import load_canonical_teams, make_resolver

WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"


TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Curaçao": "Curaçao",
    "Curacao": "Curaçao",
    "Czech Republic": "Czechia",
    "DR Congo": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Türkiye": "Türkiye",
    "Turkey": "Türkiye",
    "United States": "United States",
    "USA": "United States",
}

PLAYER_ALIASES = {
    ("Scotland", "Andy Robertson"): "Andrew Robertson",
    ("Uruguay", "José Giménez"): "José María Giménez",
    ("Argentina", "Nicolás González"): "Nico González",
}


def _settings() -> dict[str, Any]:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _cutoff(settings: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    return str(settings.get("snapshot", {}).get("cutoff"))


def _norm(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace(".", " ").replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean(value: Any) -> str:
    text = re.sub(r"\[[^\]]+\]", "", str(value))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_players() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "players.csv"
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, compression="gzip")


def _parse_team_tables() -> pd.DataFrame:
    response = requests.get(
        WIKI_URL,
        headers={"User-Agent": "wcpredict-squad-freeze/0.1 (research; contact local)"},
        timeout=30,
    )
    response.raise_for_status()
    rows: list[dict[str, Any]] = []
    canonical = load_canonical_teams(ROOT / "config" / "teams.yaml")
    resolve_team, _ = make_resolver(canonical, TEAM_ALIASES)
    canonical_keys = {_norm(name): name for name in canonical}
    soup = BeautifulSoup(response.text, "html.parser")

    current_team: str | None = None
    for node in soup.find_all(["h2", "h3", "h4", "table"]):
        if node.name in {"h2", "h3", "h4"}:
            heading = _clean(node.get_text(" "))
            heading = heading.replace("[edit]", "").strip()
            resolved_heading = resolve_team(heading)
            if _norm(resolved_heading) in canonical_keys:
                current_team = resolved_heading
            continue
        if node.name != "table" or current_team is None:
            continue

        try:
            parsed = pd.read_html(StringIO(str(node)))
        except ValueError:
            continue
        if not parsed:
            continue
        table = parsed[0]
        cols = [_clean(col[-1] if isinstance(col, tuple) else col) for col in table.columns]
        df = table.copy()
        df.columns = cols
        lowered = {c.lower().strip().rstrip("."): c for c in cols}

        # Wikipedia squad tables usually include these columns. A heading row
        # immediately before the table supplies current_team.
        name_col = next((lowered[c] for c in lowered if c in {"player", "name"}), None)
        pos_col = next((lowered[c] for c in lowered if c in {"pos", "position"}), None)
        club_col = next((lowered[c] for c in lowered if c == "club"), None)
        age_col = next((lowered[c] for c in lowered if c == "age"), None)
        dob_col = next((lowered[c] for c in lowered if "date of birth" in c), None)
        caps_col = next((lowered[c] for c in lowered if c == "caps"), None)
        if not name_col or not pos_col:
            continue

        for _, row in df.iterrows():
            raw_name = _clean(row.get(name_col, ""))
            if not raw_name or raw_name.lower() in {"player", "nan"}:
                continue
            age_value = row.get(age_col, pd.NA) if age_col else pd.NA
            if pd.isna(age_value) and dob_col:
                match = re.search(r"aged\s+(\d+)", str(row.get(dob_col, "")))
                age_value = int(match.group(1)) if match else pd.NA
            rows.append({
                "team": current_team,
                "name": re.sub(r"\s+\(captain\)", "", raw_name, flags=re.I).strip(),
                "position": _clean(row.get(pos_col, "")),
                "club": _clean(row.get(club_col, "")) if club_col else "",
                "age": pd.to_numeric(age_value, errors="coerce"),
                "caps": pd.to_numeric(row.get(caps_col, pd.NA), errors="coerce") if caps_col else pd.NA,
            })

    out = pd.DataFrame(rows).drop_duplicates(subset=["team", "name"])
    return out


def _resolve_players(squads: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = players.copy()
    p["_name_key"] = p["name"].map(_norm)
    p["_country_key"] = p["country_of_citizenship"].fillna("").map(_norm)
    p["_value"] = pd.to_numeric(p.get("market_value_in_eur"), errors="coerce").fillna(0.0)

    by_name = p.sort_values("_value", ascending=False).groupby("_name_key").head(8)
    name_keys = sorted(by_name["_name_key"].dropna().unique())

    resolved_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    for _, row in squads.iterrows():
        lookup_name = PLAYER_ALIASES.get((str(row["team"]), str(row["name"])), row["name"])
        key = _norm(lookup_name)
        team_key = _norm(row["team"])
        candidates = p[p["_name_key"] == key]
        match_method = "exact"
        if candidates.empty:
            close = difflib.get_close_matches(key, name_keys, n=1, cutoff=0.91)
            if close:
                candidates = by_name[by_name["_name_key"] == close[0]]
                match_method = "fuzzy"
        if not candidates.empty:
            same_country = candidates[candidates["_country_key"] == team_key]
            chosen_pool = same_country if not same_country.empty else candidates
            chosen = chosen_pool.sort_values("_value", ascending=False).iloc[0]
            out = row.to_dict()
            out.update({
                "player_id": int(chosen["player_id"]),
                "tm_name": chosen["name"],
                "tm_country": chosen.get("country_of_citizenship", ""),
                "market_value_in_eur": float(chosen["_value"]),
                "resolve_method": match_method,
            })
            resolved_rows.append(out)
        else:
            unmatched_rows.append(row.to_dict())

    resolved = pd.DataFrame(resolved_rows)
    unmatched = pd.DataFrame(unmatched_rows)
    if not unmatched.empty:
        # Best high-value candidate by fuzzy name, used only for manual review.
        flags = []
        for _, row in unmatched.iterrows():
            key = _norm(row["name"])
            close = difflib.get_close_matches(key, name_keys, n=3, cutoff=0.80)
            cand = by_name[by_name["_name_key"].isin(close)].sort_values("_value", ascending=False)
            if cand.empty:
                flags.append({"candidate_player_id": pd.NA, "candidate_name": "", "candidate_market_value": 0.0, "high_value_flag": False})
            else:
                top = cand.iloc[0]
                value = float(top["_value"])
                flags.append({
                    "candidate_player_id": int(top["player_id"]),
                    "candidate_name": top["name"],
                    "candidate_market_value": value,
                    "high_value_flag": value >= 20_000_000,
                })
        unmatched = pd.concat([unmatched.reset_index(drop=True), pd.DataFrame(flags)], axis=1)
    return resolved, unmatched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args(argv)

    settings = _settings()
    cutoff = _cutoff(settings, args.cutoff)
    squads = _parse_team_tables()
    if squads.empty:
        raise RuntimeError("No squad tables parsed from Wikipedia")
    players = _read_players()
    resolved, unmatched = _resolve_players(squads, players)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"squads_2026_{cutoff}.csv"
    resolved.insert(0, "cutoff", cutoff)
    resolved.to_csv(out_path, index=False)

    report_dir = ROOT / "outputs"
    report_dir.mkdir(parents=True, exist_ok=True)
    unmatched_path = report_dir / f"squad_unmatched_{cutoff}.csv"
    unmatched.to_csv(unmatched_path, index=False)

    print(f"Wrote squad snapshot: {out_path} ({len(resolved):,} resolved rows)")
    print(f"Wrote unmatched report: {unmatched_path} ({len(unmatched):,} unmatched rows)")
    if not unmatched.empty and "high_value_flag" in unmatched.columns:
        print(f"High-value unmatched candidates: {int(unmatched['high_value_flag'].sum())}")
    print("Teams parsed:", squads["team"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
