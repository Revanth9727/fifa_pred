from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

from wcpredict.cli import (
    _counts_to_probabilities,
    _fit_with_decay,
    _load_squad_team_features,
    _mask_features,
    _parse_results,
    _paths,
    _ablation_ladder,
    _select_recency_decay,
    SPLIT_DATE,
)
from wcpredict.eval.calibration import evaluate_by_slice
from wcpredict.model.calibration import fit_wdl_calibration
from wcpredict.sim.simulator import STAGES, TournamentSimulator, _build_context
from wcpredict.sim.bracket import build_r32
from wcpredict.sim.sampler import sample_match


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
WC_DATE_RANGE = "20260611-20260719"  # Group stage open – Final

# ESPN display names → internal team names used in groups config
_ESPN_ALIASES: dict[str, str] = {
    "USA": "United States",
    "Ivory Coast": "C\u00f4te d'Ivoire",
    "Cote d'Ivoire": "C\u00f4te d'Ivoire",
    "Czech Republic": "Czechia",
    "Cape Verde": "Cabo Verde",
    "Turkey": "T\u00fcrkiye",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Curacao": "Cura\u00e7ao",
}


@dataclass
class JobState:
    status: str = "idle"
    message: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


class PredictionService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or project_root()
        self.settings = load_yaml(self.root / "config" / "settings.yaml")
        self.teams_cfg = load_yaml(self.root / "config" / "teams.yaml")
        self.groups: dict[str, list[str]] = {
            str(group): list(teams)
            for group, teams in self.teams_cfg["groups"].items()
        }
        self.raw_dir = self.root / self.settings["paths"]["raw"]
        self.outputs_dir = self.root / self.settings["paths"]["outputs"]
        self.processed_dir = self.root / self.settings["paths"]["processed"]
        self.results_path = self.raw_dir / "worldcup_2026_results.csv"
        self.live_state_path = self.processed_dir / "live_tournament_state.json"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.job = JobState()
        self._future: Future | None = None
        self._espn_cache: dict[str, Any] = {}
        self._espn_cache_ts: float = 0.0
        self._chat_context_cache: str = ""
        self._chat_context_ts: float = 0.0

    @property
    def model_path(self) -> Path:
        return self.outputs_dir / "match_model.pkl"

    @property
    def probabilities_path(self) -> Path:
        return self.outputs_dir / "probabilities.csv"

    @property
    def counts_path(self) -> Path:
        return self.outputs_dir / "simulation_counts.json"

    def health(self) -> dict[str, Any]:
        return {
            "model": self.model_path.exists(),
            "probabilities": self.probabilities_path.exists(),
            "results": self.results_path.exists(),
            "groups": len(self.groups),
            "teams": sum(len(teams) for teams in self.groups.values()),
            "job": self.job.__dict__,
        }

    def _rng(self) -> np.random.Generator:
        seed = int(time.time() * 1000) % (2**32 - 1)
        return np.random.default_rng(seed)

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    # Home-crowd Elo boost applied to the three host nations.
    # ~80 points corresponds to documented WC host-nation win-rate uplift (~60%).
    _HOST_ELO_BUMP = 80.0

    def _load_strengths(self) -> dict[str, float]:
        path = self.raw_dir / "elo_ratings_wc2026.csv"
        if not path.exists():
            return {}
        elo = pd.read_csv(path)
        if "snapshot_date" in elo.columns and "date" not in elo.columns:
            elo = elo.rename(columns={"snapshot_date": "date"})
        team_col = "team" if "team" in elo.columns else "country"
        rating_col = "elo" if "elo" in elo.columns else "rating"
        if team_col not in elo.columns or rating_col not in elo.columns:
            return {}
        if "date" in elo.columns:
            elo["date"] = pd.to_datetime(elo["date"], errors="coerce")
            elo = elo.sort_values("date")
        wc_teams = {team for group in self.groups.values() for team in group}
        latest = elo.dropna(subset=[team_col, rating_col]).groupby(team_col).tail(1)
        strengths = {
            str(row[team_col]): float(row[rating_col])
            for _, row in latest.iterrows()
            if str(row[team_col]) in wc_teams
        }
        host_nations = {
            t["name"] for t in self.teams_cfg.get("teams", [])
            if t.get("host", False)
        }
        for team in host_nations:
            if team in strengths:
                strengths[team] += self._HOST_ELO_BUMP
        return strengths

    def _active_groups(self) -> set[str]:
        path = self.outputs_dir / "active_feature_groups.json"
        if path.exists():
            return set(json.loads(path.read_text()).get("active_groups", []))
        return {"squad_depth", "star_power", "position_groups", "recent_form", "travel_rest", "gk"}

    def _team_features(self) -> dict[str, dict[str, float]]:
        team_features_df = _load_squad_team_features(self.settings)
        if team_features_df is None:
            return {}
        active = self._active_groups()
        masked = team_features_df.copy()
        if "n_squad" in masked.columns:
            low_squad = masked["n_squad"] < 5
            masked.loc[low_squad, "star_concentration"] = np.nan
            masked.loc[low_squad, "star_power_rank"] = np.nan
        if "squad_depth" not in active:
            masked["squad_quality_rank"] = 0.5
            masked["squad_depth_rank"] = 0.5
        if "star_power" not in active:
            masked["star_power_rank"] = 0.5
            masked["star_concentration"] = 0.5
        if "position_groups" not in active:
            masked["attack_value_rank"] = 0.5
            masked["midfield_value_rank"] = 0.5
            masked["defense_value_rank"] = 0.5
        if "recent_form" not in active:
            masked["club_minutes_rank"] = 0.5
            masked["club_form_rank"] = 0.5
            masked["national_form"] = 0.5
        if "gk" not in active:
            masked["gk_rating"] = 0.5
        return masked.set_index("team").to_dict(orient="index")

    def group_fixtures(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        match_no = 1
        for group, teams in self.groups.items():
            fixtures = [
                (teams[0], teams[1]), (teams[0], teams[2]), (teams[0], teams[3]),
                (teams[1], teams[2]), (teams[1], teams[3]), (teams[2], teams[3]),
            ]
            for idx, (home, away) in enumerate(fixtures, start=1):
                rows.append({
                    "match_id": f"G{group}{idx}",
                    "match_no": match_no,
                    "stage": "Group",
                    "group": group,
                    "slot": f"Group {group} match {idx}",
                    "home": home,
                    "away": away,
                    "status": "scheduled",
                })
                match_no += 1
        return rows

    def _initial_live_state(self) -> dict[str, Any]:
        matches = []
        for row in self.group_fixtures():
            matches.append({
                **row,
                "home_score": None,
                "away_score": None,
                "winner": None,
                "went_to_et": False,
                "went_to_shootout": False,
                "penalties_home": None,
                "penalties_away": None,
                "locked": False,
                "available": True,
            })
        return {
            "version": 1,
            "status": "group_stage",
            "champion": None,
            "matches": matches,
            "standings": self._standings_from_matches(matches),
        }

    def load_live_state(self) -> dict[str, Any]:
        if not self.live_state_path.exists():
            return self.reset_live_state()
        return json.loads(self.live_state_path.read_text())

    def save_live_state(self, state: dict[str, Any]) -> dict[str, Any]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.live_state_path.write_text(json.dumps(state, indent=2) + "\n")
        return state

    def reset_live_state(self) -> dict[str, Any]:
        return self.save_live_state(self._initial_live_state())

    def _standings_from_matches(self, matches: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        standings: dict[str, dict[str, dict[str, Any]]] = {}
        for group, teams in self.groups.items():
            standings[group] = {
                team: {"team": team, "played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0}
                for team in teams
            }
        for match in matches:
            if match.get("stage") != "Group" or not match.get("locked"):
                continue
            group = match["group"]
            home = match["home"]
            away = match["away"]
            gh = int(match["home_score"])
            ga = int(match["away_score"])
            for team, gf, ga_ in [(home, gh, ga), (away, ga, gh)]:
                row = standings[group][team]
                row["played"] += 1
                row["gf"] += gf
                row["ga"] += ga_
                row["gd"] = row["gf"] - row["ga"]
            if gh > ga:
                standings[group][home]["points"] += 3
            elif ga > gh:
                standings[group][away]["points"] += 3
            else:
                standings[group][home]["points"] += 1
                standings[group][away]["points"] += 1
        return {
            group: sorted(
                rows.values(),
                key=lambda row: (-row["points"], -row["gd"], -row["gf"], row["team"]),
            )
            for group, rows in standings.items()
        }

    def _group_stage_complete(self, matches: list[dict[str, Any]]) -> bool:
        group_matches = [match for match in matches if match["stage"] == "Group"]
        return bool(group_matches) and all(match.get("locked") for match in group_matches)

    def _knockout_round_complete(self, matches: list[dict[str, Any]], stage: str) -> bool:
        stage_matches = [match for match in matches if match["stage"] == stage]
        return bool(stage_matches) and all(match.get("locked") for match in stage_matches)

    def _rank_thirds(self, standings: dict[str, list[dict[str, Any]]]) -> set[str]:
        thirds = [(group, rows[2]) for group, rows in standings.items()]
        ranked = sorted(
            thirds,
            key=lambda item: (
                -item[1]["points"],
                -item[1]["gd"],
                -item[1]["gf"],
                item[1]["team"],
            ),
        )
        return {group for group, _ in ranked[:8]}

    def _append_round(
        self,
        state: dict[str, Any],
        *,
        stage: str,
        prefix: str,
        start_no: int,
        matchups: list[tuple[str, str, str]],
    ) -> None:
        existing = {match["match_id"] for match in state["matches"]}
        for idx, (home, away, slot) in enumerate(matchups, start=1):
            match_id = f"{prefix}{idx}"
            if match_id in existing:
                continue
            state["matches"].append({
                "match_id": match_id,
                "match_no": start_no + idx - 1,
                "stage": stage,
                "group": "",
                "slot": slot,
                "home": home,
                "away": away,
                "status": "ready",
                "home_score": None,
                "away_score": None,
                "winner": None,
                "went_to_et": False,
                "went_to_shootout": False,
                "penalties_home": None,
                "penalties_away": None,
                "locked": False,
                "available": True,
            })

    def _advance_live_state(self, state: dict[str, Any]) -> dict[str, Any]:
        matches = state["matches"]
        standings = self._standings_from_matches(matches)
        state["standings"] = standings
        if self._group_stage_complete(matches):
            group_standings = {
                group: [row["team"] for row in rows]
                for group, rows in standings.items()
            }
            thirds = self._rank_thirds(standings)
            r32 = build_r32(group_standings, thirds)
            self._append_round(
                state,
                stage="Round of 32",
                prefix="R32",
                start_no=73,
                matchups=[(m.home, m.away, m.label) for m in r32],
            )
            state["status"] = "round_of_32"

        round_plan = [
            ("Round of 32", "Round of 16", "R16", 89),
            ("Round of 16", "Quarter-final", "QF", 97),
            ("Quarter-final", "Semi-final", "SF", 101),
        ]
        for current_stage, next_stage, prefix, start_no in round_plan:
            if self._knockout_round_complete(matches, current_stage):
                winners = [
                    match["winner"]
                    for match in sorted(matches, key=lambda row: row["match_no"])
                    if match["stage"] == current_stage
                ]
                if current_stage == "Round of 32":
                    _R16_PAIRS = [(0,2),(11,13),(5,4),(10,12),(1,3),(8,15),(7,6),(9,14)]
                    matchups = [
                        (winners[i], winners[j], f"Winner Round of 32 {i+1} vs Winner Round of 32 {j+1}")
                        for i, j in _R16_PAIRS
                    ]
                else:
                    matchups = [
                        (winners[i], winners[i + 1], f"Winner {current_stage} {i + 1} vs Winner {current_stage} {i + 2}")
                        for i in range(0, len(winners), 2)
                    ]
                self._append_round(
                    state,
                    stage=next_stage,
                    prefix=prefix,
                    start_no=start_no,
                    matchups=matchups,
                )
                state["status"] = next_stage.lower().replace(" ", "_").replace("-", "_")

        if self._knockout_round_complete(matches, "Semi-final"):
            semis = [match for match in sorted(matches, key=lambda row: row["match_no"]) if match["stage"] == "Semi-final"]
            winners = [match["winner"] for match in semis]
            losers = [match["away"] if match["winner"] == match["home"] else match["home"] for match in semis]
            self._append_round(
                state,
                stage="Third-place playoff",
                prefix="TP",
                start_no=103,
                matchups=[(losers[0], losers[1], "Semi-final losers")],
            )
            self._append_round(
                state,
                stage="Final",
                prefix="F",
                start_no=104,
                matchups=[(winners[0], winners[1], "Semi-final winners")],
            )
            state["status"] = "finals_ready"

        final = next((match for match in matches if match["stage"] == "Final" and match.get("locked")), None)
        if final:
            state["champion"] = final["winner"]
            state["status"] = "complete"
        return state

    def simulate_live_match(self, match_id: str) -> dict[str, Any]:
        state = self.load_live_state()
        match = next((item for item in state["matches"] if item["match_id"] == match_id), None)
        if match is None:
            raise ValueError(f"Unknown match_id: {match_id}")
        if match.get("locked"):
            return state
        if not match.get("available", False):
            raise ValueError(f"Match is not available yet: {match_id}")

        model = self._load_model()
        strengths = self._load_strengths()
        team_features = self._team_features()
        prob = self._match_probability_row(
            model=model,
            strengths=strengths,
            team_features=team_features,
            home=match["home"],
            away=match["away"],
            max_goals=int(self.settings.get("match_model", {}).get("max_goals", 10)),
        )
        dist = model.predict(_build_context(
            match["home"],
            match["away"],
            neutral=True,
            settings=self.settings,
            model=model,
            elo_diff=strengths.get(match["home"], 1600.0) - strengths.get(match["away"], 1600.0),
            team_features=team_features,
        ))
        result = sample_match(
            dist,
            self._rng(),
            self.settings,
            et_allowed=match["stage"] != "Group",
            strength_diff=(strengths.get(match["home"], 1600.0) - strengths.get(match["away"], 1600.0)) / 400.0,
        )
        home_total = result.goals_home + result.et_goals_home
        away_total = result.goals_away + result.et_goals_away
        winner = None
        if result.winner == "home":
            winner = match["home"]
        elif result.winner == "away":
            winner = match["away"]
        result_note = None
        if result.went_to_shootout and winner is not None:
            result_note = (
                f"{winner} win {result.penalties_home}-{result.penalties_away} on penalties"
            )
        elif result.went_to_et:
            result_note = "After extra time"
        match.update({
            "home_score": home_total,
            "away_score": away_total,
            "winner": winner,
            "went_to_et": result.went_to_et,
            "went_to_shootout": result.went_to_shootout,
            "penalties_home": result.penalties_home,
            "penalties_away": result.penalties_away,
            "result_note": result_note,
            "locked": True,
            "status": "final",
            **prob,
        })
        return self.save_live_state(self._advance_live_state(state))

    def _match_probability_row(
        self,
        *,
        model: Any,
        strengths: dict[str, float],
        team_features: dict[str, dict[str, float]],
        home: str,
        away: str,
        max_goals: int,
    ) -> dict[str, float]:
        ctx = _build_context(
            home,
            away,
            neutral=True,
            settings=self.settings,
            model=model,
            elo_diff=strengths.get(home, 1600.0) - strengths.get(away, 1600.0),
            team_features=team_features,
        )
        dist = model.predict(ctx)
        win, draw, loss = model.derive_wdl(dist, max_goals=max_goals)
        return {
            "home_win": win,
            "draw": draw,
            "away_win": loss,
            "lambda_home": dist.lambda_a,
            "lambda_away": dist.lambda_b,
        }

    def _local_results(self) -> dict[tuple[str, str], tuple[int, int]]:
        if not self.results_path.exists():
            return {}
        return _parse_results(self.results_path)

    def _result_for(self, home: str, away: str) -> tuple[int, int] | None:
        results = self._local_results()
        if (home, away) in results:
            return results[(home, away)]
        if (away, home) in results:
            ga, gh = results[(away, home)]
            return gh, ga
        return None

    def match_predictions(self) -> list[dict[str, Any]]:
        state = self.load_live_state()
        return sorted(state.get("matches", []), key=lambda row: int(row.get("match_no", 0)))

    def probabilities(self) -> list[dict[str, Any]]:
        if not self.probabilities_path.exists():
            return []
        df = pd.read_csv(self.probabilities_path)
        return df.replace({np.nan: None}).to_dict(orient="records")

    def dashboard(self) -> dict[str, Any]:
        matches = self.match_predictions()
        probabilities = self.probabilities()
        completed = sum(1 for row in matches if row.get("locked"))
        group_completed = sum(1 for row in matches if row.get("stage") == "Group" and row.get("locked"))
        return {
            "health": self.health(),
            "matches": matches,
            "probabilities": probabilities,
            "summary": {
                "matches_total": len(matches),
                "group_matches": 72,
                "completed_matches": completed,
                "completed_group_matches": group_completed,
                "actual_results_synced": self.results_path.exists(),
                "last_updated": time.time(),
            },
        }

    def _norm_team(self, name: str) -> str:
        """Normalise an ESPN display name to the internal team name."""
        return _ESPN_ALIASES.get(name, name)

    def _fetch_espn_data_cached(self) -> dict[str, Any]:
        """Return ESPN data, refreshing at most once every 90 seconds."""
        if time.time() - self._espn_cache_ts < 90 and self._espn_cache:
            return self._espn_cache
        data = self._fetch_espn_data()
        self._espn_cache = data
        self._espn_cache_ts = time.time()
        return data

    def _fetch_espn_data(self) -> dict[str, Any]:
        """
        Fetch all WC match data from ESPN's public scoreboard API.

        Returns
        -------
        dict with keys:
          "completed" : pd.DataFrame  — finished matches (team_home/away, goals_home/away, source)
          "upcoming"  : list[dict]    — future/live matches (home, away, date, status, live_score)
          "error"     : str | None    — human-readable error if the call failed
        """
        headers = {"User-Agent": "wcpredict/0.1 (research)", "Accept": "application/json"}
        params = {"dates": WC_DATE_RANGE, "limit": "300"}
        empty_df = pd.DataFrame(columns=["team_home", "team_away", "goals_home", "goals_away", "source"])

        try:
            resp = requests.get(ESPN_SCOREBOARD, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return {"completed": empty_df, "upcoming": [], "error": str(exc)}

        completed: list[dict] = []
        upcoming: list[dict] = []

        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            status = comp.get("status", {})
            status_type = status.get("type", {})
            is_complete = status_type.get("completed", False)
            state = status_type.get("state", "pre")  # pre / in / post

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            home = self._norm_team(home_c.get("team", {}).get("displayName", ""))
            away = self._norm_team(away_c.get("team", {}).get("displayName", ""))

            if is_complete:
                try:
                    completed.append({
                        "team_home": home,
                        "team_away": away,
                        "goals_home": int(home_c.get("score", 0)),
                        "goals_away": int(away_c.get("score", 0)),
                        "home_winner": bool(home_c.get("winner", False)),
                        "away_winner": bool(away_c.get("winner", False)),
                        "source": "espn",
                    })
                except (ValueError, TypeError):
                    pass
            else:
                row: dict[str, Any] = {
                    "home": home,
                    "away": away,
                    "date": event.get("date", ""),
                    "status": status_type.get("description", "Scheduled"),
                    "live": state == "in",
                }
                if state == "in":
                    try:
                        row["live_home"] = int(home_c.get("score", 0))
                        row["live_away"] = int(away_c.get("score", 0))
                        row["clock"] = status.get("displayClock", "")
                    except (ValueError, TypeError):
                        pass
                upcoming.append(row)

        completed_df = (
            pd.DataFrame(completed)
            .drop_duplicates(subset=["team_home", "team_away"], keep="last")
            .reset_index(drop=True)
            if completed else empty_df
        )
        return {"completed": completed_df, "upcoming": upcoming, "error": None}

    def fetch_results_from_espn(self) -> pd.DataFrame:
        """Return a DataFrame of completed WC matches from ESPN."""
        return self._fetch_espn_data()["completed"]

    def get_upcoming_matches(self) -> dict[str, Any]:
        """Return upcoming / live WC matches for the UI (real teams only, live first)."""
        data = self._fetch_espn_data_cached()
        known = {team for group in self.groups.values() for team in group}
        real = [
            m for m in data["upcoming"]
            if m["home"] in known or m["away"] in known
        ]
        # live matches first, then chronological
        real.sort(key=lambda m: (0 if m.get("live") else 1, m.get("date", "")))
        return {
            "upcoming": real[:10],
            "error": data["error"],
        }

    def save_results(self, results: pd.DataFrame) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        cols = ["team_home", "team_away", "goals_home", "goals_away", "source"]
        if results.empty:
            pd.DataFrame(columns=cols).to_csv(self.results_path, index=False)
        else:
            results[cols].to_csv(self.results_path, index=False)
        return self.results_path

    def run_simulation(self, n_runs: int | None = None, seed: int | None = None) -> dict[str, Any]:
        model = self._load_model()
        strengths = self._load_strengths()
        results = self._local_results()
        team_features = self._team_features()
        runs = int(n_runs or self.settings.get("simulation", {}).get("n_runs", 25000))
        actual_seed = int(seed if seed is not None else self.settings.get("simulation", {}).get("random_seed", 42))
        sim = TournamentSimulator(
            self.groups,
            model,
            self.settings,
            strengths=strengths or None,
            realized_results=results,
            team_features=team_features,
        )
        counts = sim.run(n_runs=runs, seed=actual_seed)
        probs = _counts_to_probabilities(counts, n_runs=runs, groups=self.groups)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        probs.to_csv(self.probabilities_path, index=False)
        self.counts_path.write_text(json.dumps(counts, indent=2, sort_keys=True))
        return {
            "runs": runs,
            "seed": actual_seed,
            "completed_results": len(results),
            "probabilities": str(self.probabilities_path),
            "counts": str(self.counts_path),
        }

    def _sync_wikipedia_results_to_live_state(self, fetched: pd.DataFrame) -> int:
        """
        Lock group-stage matches in live_tournament_state.json using scores
        from the Wikipedia fetch.  Returns the number of newly locked matches.

        Only group-stage matches are synced — knockout fixtures aren't in the
        fetched table until the bracket is set.  Matches already locked are
        left untouched.
        """
        if fetched.empty:
            return 0

        results: dict[tuple[str, str], tuple[int, int]] = {}
        for _, row in fetched.iterrows():
            h = str(row.get("team_home", ""))
            a = str(row.get("team_away", ""))
            gh = row.get("goals_home")
            ga = row.get("goals_away")
            if h and a and gh is not None and ga is not None:
                results[(h, a)] = (int(gh), int(ga))

        state = self.load_live_state()
        newly_locked = 0

        model = None
        strengths: dict[str, float] = {}
        team_features: dict[str, dict[str, float]] = {}
        try:
            if self.model_path.exists():
                model = self._load_model()
                strengths = self._load_strengths()
                team_features = self._team_features()
        except Exception:
            pass

        max_goals = int(self.settings.get("match_model", {}).get("max_goals", 10))

        # Build a lookup that also carries winner flags for penalty resolution
        winner_flags: dict[tuple[str, str], tuple[bool, bool]] = {}
        for _, row in fetched.iterrows():
            h = str(row.get("team_home", ""))
            a = str(row.get("team_away", ""))
            if h and a:
                winner_flags[(h, a)] = (
                    bool(row.get("home_winner", False)),
                    bool(row.get("away_winner", False)),
                )

        for match in state["matches"]:
            if match.get("locked"):
                continue
            home, away = match["home"], match["away"]
            score = results.get((home, away)) or results.get((away, home))
            if score is None:
                continue
            is_reversed = (home, away) not in results
            gh, ga = (score[1], score[0]) if is_reversed else score

            # Determine winner; for knockout draws (ET/pens) use ESPN winner flag
            is_group = match.get("stage") == "Group"
            went_to_et = False
            went_to_shootout = False
            winner: str | None = None
            if gh > ga:
                winner = home
            elif ga > gh:
                winner = away
            else:
                # Draw — only valid in group stage; in knockout use ESPN winner flag
                if not is_group:
                    hw, aw = winner_flags.get((home, away), winner_flags.get((away, home), (False, False)))
                    if is_reversed:
                        hw, aw = aw, hw
                    if hw:
                        winner = home
                        went_to_et = True
                        went_to_shootout = True
                    elif aw:
                        winner = away
                        went_to_et = True
                        went_to_shootout = True
                    else:
                        continue  # can't resolve yet

            prob: dict[str, Any] = {}
            if model is not None and is_group:
                try:
                    prob = self._match_probability_row(
                        model=model,
                        strengths=strengths,
                        team_features=team_features,
                        home=home,
                        away=away,
                        max_goals=max_goals,
                    )
                except Exception:
                    pass

            match.update({
                "home_score": gh,
                "away_score": ga,
                "winner": winner,
                "went_to_et": went_to_et,
                "went_to_shootout": went_to_shootout,
                "penalties_home": None,
                "penalties_away": None,
                "result_note": None,
                "locked": True,
                "status": "final",
                **prob,
            })
            newly_locked += 1

        if newly_locked:
            self.save_live_state(self._advance_live_state(state))

        return newly_locked

    def refresh_results_only(self) -> dict[str, Any]:
        """
        Fast refresh: fetch from ESPN → save CSV → sync live state.
        Does NOT run Monte Carlo (use run_simulation() / start_simulation_job() for that).
        Completes in a few seconds.
        """
        self.job.message = "Fetching results from ESPN…"
        data = self._fetch_espn_data()
        if data["error"]:
            raise RuntimeError(f"ESPN fetch failed: {data['error']}")
        fetched = data["completed"]
        n_fetched = len(fetched)
        self.save_results(fetched)
        self.job.message = f"Fetched {n_fetched} result(s) — syncing live state…"
        synced = self._sync_wikipedia_results_to_live_state(fetched)
        msg = f"Fetched {n_fetched} result(s), {synced} newly locked"
        self.job.message = msg
        return {
            "fetched_results": n_fetched,
            "synced_to_live_state": synced,
            "message": msg,
        }

    def refresh_and_simulate(self, n_runs: int | None = None, seed: int | None = None) -> dict[str, Any]:
        out = self.refresh_results_only()
        self.job.message = f"{out['message']} — running simulation…"
        sim = self.run_simulation(n_runs=n_runs, seed=seed)
        return {**out, **sim}

    def start_refresh_job(self, n_runs: int | None = None, seed: int | None = None) -> dict[str, Any]:
        """
        Fetch ESPN results and sync the live state. If new real results were
        locked and web.auto_retrain_after_result_refresh is enabled, retrain
        the model and rerun probabilities in the same background job.
        """
        if self._future is not None and not self._future.done():
            return {"accepted": False, "job": self.job.__dict__}

        self.job = JobState(status="running", message="Fetching live results…", started_at=time.time())

        def task() -> dict[str, Any]:
            try:
                out = self.refresh_results_only()
                auto_retrain = bool(
                    self.settings.get("web", {}).get("auto_retrain_after_result_refresh", True)
                )
                if auto_retrain and int(out.get("synced_to_live_state", 0)) > 0:
                    self.job.message = "New real results synced — retraining model…"
                    retrain_out = self.retrain_with_live_results(n_runs=n_runs, seed=seed)
                    out = {**out, "retrained": True, **retrain_out}
                else:
                    out = {**out, "retrained": False}
                self.job.status = "complete"
                self.job.message = (
                    f"{out['message']} — model retrained"
                    if out.get("retrained")
                    else out["message"]
                )
                self.job.finished_at = time.time()
                self.job.error = None
                return out
            except Exception as exc:
                self.job.status = "failed"
                self.job.message = "Refresh failed"
                self.job.finished_at = time.time()
                self.job.error = str(exc)
                raise

        self._future = self.executor.submit(task)
        return {"accepted": True, "job": self.job.__dict__}

    def start_simulation_job(self, n_runs: int | None = None, seed: int | None = None) -> dict[str, Any]:
        if self._future is not None and not self._future.done():
            return {"accepted": False, "job": self.job.__dict__}

        self.job = JobState(status="running", message="Running tournament simulation", started_at=time.time())

        def task() -> dict[str, Any]:
            try:
                out = self.run_simulation(n_runs=n_runs, seed=seed)
                self.job.status = "complete"
                self.job.message = "Simulation complete"
                self.job.finished_at = time.time()
                self.job.error = None
                return out
            except Exception as exc:  # pragma: no cover - surfaced via status endpoint
                self.job.status = "failed"
                self.job.message = "Simulation failed"
                self.job.finished_at = time.time()
                self.job.error = str(exc)
                raise

        self._future = self.executor.submit(task)
        return {"accepted": True, "job": self.job.__dict__}

    def _simulation_stats(self, probs: list[dict]) -> dict[str, Any]:
        if not probs:
            return {"available": False}
        n_runs = int(self.settings.get("simulation", {}).get("n_runs", 25000))
        if self.counts_path.exists():
            counts = json.loads(self.counts_path.read_text())
            estimates = []
            for row in probs:
                p = row.get("p_champion") or 0
                tc = counts.get(row["team"], {}).get("champion", 0)
                if p > 0 and tc > 0:
                    estimates.append(round(tc / p))
            if estimates:
                n_runs = int(np.median(estimates))
        teams_stats = []
        for row in probs:
            p = float(row.get("p_champion") or 0)
            se = float((p * (1 - p) / n_runs) ** 0.5) if n_runs > 0 else 0.0
            teams_stats.append({
                "team": row["team"],
                "p_champion": p,
                "se": se,
                "ci_lower": float(max(0.0, p - 1.96 * se)),
                "ci_upper": float(min(1.0, p + 1.96 * se)),
            })
        champion_probs = [t["p_champion"] for t in teams_stats]
        ses = [t["se"] for t in teams_stats]
        ci_widths = [t["ci_upper"] - t["ci_lower"] for t in teams_stats]
        return {
            "available": True,
            "n_runs": n_runs,
            "mean_p_champion": float(np.mean(champion_probs)),
            "std_p_champion": float(np.std(champion_probs)),
            "mean_se": float(np.mean(ses)),
            "mean_ci_width": float(np.mean(ci_widths)),
            "teams": teams_stats,
        }

    def _latest_eval_metrics(self) -> dict[str, Any] | None:
        metrics_dir = self.outputs_dir / "metrics"
        if not metrics_dir.exists():
            return None
        files = sorted(metrics_dir.glob("metrics_*.json"))
        if not files:
            return None
        with open(files[-1]) as f:
            return json.load(f)

    def metrics(self) -> dict[str, Any]:
        probs = self.probabilities()
        sim = self._simulation_stats(probs)
        eval_data = self._latest_eval_metrics()
        if eval_data:
            mq: dict[str, Any] = {
                "available": True,
                "brier": eval_data.get("brier"),
                "log_loss": eval_data.get("log_loss"),
                "captured_at": eval_data.get("captured_at"),
                "reliability_bins": eval_data.get("reliability_bins", []),
            }
        else:
            mq = {"available": False}
        slices = self._slice_metrics()
        return {"simulation": sim, "model_quality": mq, "slices": slices}

    def _slice_metrics(self) -> list[dict[str, Any]]:
        """Compute per-Elo-bucket WDL bias for the validation set."""
        try:
            paths = _paths(self.settings)
            match_table_path = paths["match_table"]
            features_path = paths["features"]
            if not match_table_path.exists() or not features_path.exists() or not self.model_path.exists():
                return []
            table = pd.read_parquet(match_table_path)
            features = pd.read_parquet(features_path)
            table["date"] = pd.to_datetime(table["date"])
            scored = table[["goals_home", "goals_away"]].notna().all(axis=1)
            val_mask = (table["date"] >= SPLIT_DATE) & scored
            val_table = table.loc[val_mask].reset_index(drop=True)
            val_feats = features.loc[val_mask].reset_index(drop=True)
            if val_table.empty:
                return []
            model = self._load_model()
            return evaluate_by_slice(val_table, val_feats, model)
        except Exception:
            return []

    def _build_2026_feature_rows(self, rows_2026: pd.DataFrame) -> pd.DataFrame:
        """
        Build a features DataFrame for completed 2026 WC matches, aligned
        row-for-row with rows_2026.  Uses squad snapshot features; sets
        rest/travel to neutral defaults (minor impact).
        """
        squad_feat = self._team_features()
        records: list[dict[str, Any]] = []
        for _, row in rows_2026.iterrows():
            home = row["team_home"]
            away = row["team_away"]
            hf = squad_feat.get(home, {})
            af = squad_feat.get(away, {})
            home_elo = float(row.get("home_elo_pre") or 0.0)
            away_elo = float(row.get("away_elo_pre") or 0.0)
            records.append({
                "neutral_flag": 1.0,
                "host_flag": 0.0,
                "elo_diff": home_elo - away_elo,
                "form_a": hf.get("national_form", 0.5),
                "form_b": af.get("national_form", 0.5),
                "squad_quality_rank_a": hf.get("squad_quality_rank", 0.5),
                "squad_quality_rank_b": af.get("squad_quality_rank", 0.5),
                "squad_depth_rank_a": hf.get("squad_depth_rank", 0.5),
                "squad_depth_rank_b": af.get("squad_depth_rank", 0.5),
                "star_power_rank_a": hf.get("star_power_rank", 0.5),
                "star_power_rank_b": af.get("star_power_rank", 0.5),
                "star_concentration_a": hf.get("star_concentration", 0.5),
                "star_concentration_b": af.get("star_concentration", 0.5),
                "attack_value_rank_a": hf.get("attack_value_rank", 0.5),
                "attack_value_rank_b": af.get("attack_value_rank", 0.5),
                "midfield_value_rank_a": hf.get("midfield_value_rank", 0.5),
                "midfield_value_rank_b": af.get("midfield_value_rank", 0.5),
                "defense_value_rank_a": hf.get("defense_value_rank", 0.5),
                "defense_value_rank_b": af.get("defense_value_rank", 0.5),
                "club_minutes_rank_a": hf.get("club_minutes_rank", 0.5),
                "club_minutes_rank_b": af.get("club_minutes_rank", 0.5),
                "club_form_rank_a": hf.get("club_form_rank", 0.5),
                "club_form_rank_b": af.get("club_form_rank", 0.5),
                "gk_rating_a": hf.get("gk_rating", 0.5),
                "gk_rating_b": af.get("gk_rating", 0.5),
                "rest_days_a": 0.0,
                "rest_days_b": 0.0,
                "travel_km_a": 0.0,
                "travel_km_b": 0.0,
            })
        return pd.DataFrame(records)

    def retrain_with_live_results(
        self,
        n_runs: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Full retrain pipeline:
        1. Extract completed 2026 matches from live state.
        2. Merge into historical training data.
        3. Run feature ablation to select active groups.
        4. Fit model with temporal decay in production mode (all rows).
        5. Fit post-hoc WDL calibration on 2022–2025 validation rows.
        6. Save model and re-run simulation.
        """
        import pickle

        paths = _paths(self.settings)
        match_table_path = paths["match_table"]
        features_path = paths["features"]

        if not match_table_path.exists() or not features_path.exists():
            raise FileNotFoundError("Processed match table or features not found. Run `wcpredict build-data` first.")

        # --- Step 1: load historical data ---
        self.job.message = "Step 1/7 — Loading historical match data…"
        table = pd.read_parquet(match_table_path)
        features = pd.read_parquet(features_path)
        table["date"] = pd.to_datetime(table["date"])

        # --- Step 2: build 2026 rows ---
        self.job.message = "Step 2/7 — Extracting 2026 WC match rows…"
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "build_2026_training_rows",
            self.root / "scripts" / "build_2026_training_rows.py",
        )
        _mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
        build_rows = _mod.build_rows
        rows_2026 = build_rows(self.live_state_path)
        if not rows_2026.empty:
            feats_2026 = self._build_2026_feature_rows(rows_2026)
            common_match_cols = [c for c in table.columns if c in rows_2026.columns]
            table = pd.concat(
                [table, rows_2026[common_match_cols]],
                ignore_index=True,
            ).sort_values("date").reset_index(drop=True)
            features = pd.concat(
                [features, feats_2026],
                ignore_index=True,
            ).reset_index(drop=True)

        n_added = len(rows_2026)
        self.job.message = f"Step 2/7 — {n_added} 2026 row(s) added to training set"

        # --- Step 3: ablation on historical val split ---
        self.job.message = "Step 3/7 — Running feature ablation (selecting best feature groups)…"
        ablation_df, active_groups = _ablation_ladder(
            table, features, half_life_years=None
        )
        ablation_path = paths["outputs"] / "ablation_ladder.csv"
        ablation_df.to_csv(ablation_path, index=False)
        active_path = paths["outputs"] / "active_feature_groups.json"
        active_path.write_text(json.dumps({"active_groups": sorted(active_groups)}, indent=2))
        self.job.message = f"Step 3/7 — Active feature groups: {sorted(active_groups)}"

        masked_features = _mask_features(features, active_groups)

        # --- Step 4: select recency decay then fit in production mode ---
        self.job.message = "Step 4/7 — Selecting optimal recency decay weight…"
        decay_candidates: list[float | None] = [None, 3.0, 5.0, 7.0, 10.0]
        best_decay, _ = _select_recency_decay(table, masked_features, decay_candidates)
        self.job.message = f"Step 4/7 — Best decay: {best_decay}yr — fitting final model…"

        model, _, _ = _fit_with_decay(
            table,
            masked_features,
            half_life_years=best_decay,
            production=True,
        )

        # --- Step 5: post-hoc WDL calibration on 2022-2025 val rows ---
        self.job.message = "Step 5/7 — Calibrating win/draw/loss probabilities…"
        scored = table[["goals_home", "goals_away"]].notna().all(axis=1)
        val_mask = (table["date"] >= SPLIT_DATE) & (table["date"] < pd.Timestamp("2026-01-01")) & scored
        if val_mask.sum() > 20:
            fit_wdl_calibration(
                table.loc[val_mask].reset_index(drop=True),
                masked_features.loc[val_mask].reset_index(drop=True),
                model,
            )

        # --- Step 6: save model ---
        self.job.message = "Step 6/7 — Saving model…"
        paths["outputs"].mkdir(parents=True, exist_ok=True)
        with open(paths["model"], "wb") as f:
            pickle.dump(model, f)

        # --- Step 7: run simulation ---
        self.job.message = "Step 7/7 — Running Monte Carlo simulation…"
        sim_out = self.run_simulation(n_runs=n_runs, seed=seed)

        return {
            "live_rows_added": n_added,
            "active_groups": sorted(active_groups),
            "recency_half_life_years": best_decay,
            "param_cov_available": model.param_cov_ is not None,
            "wdl_calibration_fitted": model.cal_win_ is not None,
            **sim_out,
        }

    def start_retrain_job(self, n_runs: int | None = None, seed: int | None = None) -> dict[str, Any]:
        if self._future is not None and not self._future.done():
            return {"accepted": False, "job": self.job.__dict__}

        self.job = JobState(
            status="running",
            message="Starting retrain with live results…",
            started_at=time.time(),
        )

        def task() -> dict[str, Any]:
            try:
                out = self.retrain_with_live_results(n_runs=n_runs, seed=seed)
                self.job.status = "complete"
                self.job.message = f"Retrain complete — {out['live_rows_added']} live rows added"
                self.job.finished_at = time.time()
                self.job.error = None
                return out
            except Exception as exc:
                self.job.status = "failed"
                self.job.message = "Retrain failed"
                self.job.finished_at = time.time()
                self.job.error = str(exc)
                raise

        self._future = self.executor.submit(task)
        return {"accepted": True, "job": self.job.__dict__}

    def job_status(self) -> dict[str, Any]:
        if self._future is not None and self._future.done() and self.job.status == "running":
            try:
                self._future.result()
            except Exception as exc:  # pragma: no cover
                self.job.status = "failed"
                self.job.error = str(exc)
                self.job.finished_at = time.time()
        return self.job.__dict__

    # ------------------------------------------------------------------
    # Chatbot
    # ------------------------------------------------------------------

    def _build_chat_context(self) -> str:
        """Build the system prompt injected into every Claude conversation (cached 60s)."""
        if time.time() - self._chat_context_ts < 60 and self._chat_context_cache:
            return self._chat_context_cache
        ctx = self._build_chat_context_fresh()
        self._chat_context_cache = ctx
        self._chat_context_ts = time.time()
        return ctx

    def _build_chat_context_fresh(self) -> str:
        """Build the system prompt injected into every Claude conversation."""
        # --- locked results ---
        try:
            state = self.load_live_state()
            locked = [m for m in state.get("matches", []) if m.get("locked")]
            locked_lines = "\n".join(
                f"  - {m['home']} {m.get('home_score', '?')}-{m.get('away_score', '?')} {m['away']}"
                f"  (Group {m.get('group', '?')})"
                for m in locked
            ) or "  None yet"
        except Exception:
            locked_lines = "  Unavailable"

        # --- simulation probabilities ---
        champ_lines = finalist_lines = sf_lines = "  Not yet run — click Run Simulation first"
        if self.probabilities_path.exists():
            try:
                probs = pd.read_csv(self.probabilities_path).sort_values("p_champion", ascending=False)
                champ_lines = "\n".join(
                    f"  {i+1:2}. {r['team']:<28} {r['p_champion']*100:5.1f}%"
                    for i, (_, r) in enumerate(probs.iterrows())
                )
                finalist_lines = "\n".join(
                    f"  {r['team']:<28} {r['p_reach_final']*100:5.1f}%"
                    for _, r in probs.head(12).iterrows()
                )
                sf_lines = "\n".join(
                    f"  {r['team']:<28} champion {r['p_champion']*100:.1f}%"
                    f"  final {r['p_reach_final']*100:.1f}%"
                    f"  SF {r['p_reach_sf']*100:.1f}%"
                    f"  QF {r['p_reach_qf']*100:.1f}%"
                    for _, r in probs.iterrows()
                )
            except Exception:
                pass

        # --- active feature groups ---
        active_groups: list[str] = []
        ag_path = self.outputs_dir / "active_feature_groups.json"
        if ag_path.exists():
            try:
                active_groups = json.loads(ag_path.read_text()).get("active_groups", [])
            except Exception:
                pass

        # --- upcoming matches ---
        upcoming_lines = "  Unavailable"
        try:
            data = self._fetch_espn_data_cached()
            known = {team for group in self.groups.values() for team in group}
            upcoming = [m for m in data["upcoming"] if m["home"] in known or m["away"] in known]
            upcoming.sort(key=lambda m: (0 if m.get("live") else 1, m.get("date", "")))
            lines = []
            for m in upcoming[:10]:
                dt = m.get("date", "")[:16].replace("T", " ")
                badge = " [LIVE]" if m.get("live") else ""
                lines.append(f"  - {m['home']} vs {m['away']}  {dt}{badge}")
            upcoming_lines = "\n".join(lines) or "  None scheduled"
        except Exception:
            pass

        # --- groups ---
        groups_lines = "\n".join(
            f"  Group {g}: {', '.join(teams)}"
            for g, teams in sorted(self.groups.items())
        )

        return f"""You are an expert AI assistant for a 2026 FIFA World Cup prediction model called **WC Predictor**.

STRICT RULES:
- ONLY answer questions about this prediction model, the 2026 FIFA World Cup, match predictions, probabilities, team performance, and how the model works.
- If asked about anything else (general knowledge, other sports, coding, personal topics, etc.), politely decline and say you can only help with the WC 2026 predictor.
- Be concise, specific, and cite actual numbers from the data below.
- Do not invent probabilities or outcomes not shown in the data.

== MODEL OVERVIEW ==
- Type: Bivariate Poisson / Dixon-Coles with correlated score terms
- Predicts: expected goals (λ_home, λ_away) for any matchup, then derives win/draw/loss probabilities
- Training data: international matches from 1990–2025 with temporal exponential decay (recent matches weighted more)
- Features: pre-tournament Elo ratings, squad quality/depth, star power, club form, national form, attack/midfield/defense value ranks
- Active feature groups (last retrain): {active_groups or 'unknown — run retrain'}
- Tournament probabilities: Monte Carlo simulation with 50,000 independent runs

== TOURNAMENT GROUPS ==
{groups_lines}

== REAL MATCH RESULTS LOCKED INTO MODEL ==
{locked_lines}

== UPCOMING / LIVE MATCHES ==
{upcoming_lines}

== CHAMPION PROBABILITY (all 48 teams) ==
{champ_lines}

== FULL PROBABILITY TABLE (champion / final / SF / QF per team) ==
{sf_lines}
"""

    def chat(self, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        """Send a message to Claude with full model context injected as system prompt."""
        import os
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {
                "reply": (
                    "⚠️ No Anthropic API key found. "
                    "Set the ANTHROPIC_API_KEY environment variable and restart the server."
                ),
                "error": True,
            }

        system = self._build_chat_context()
        llm_cfg = self.settings.get("llm", {})
        model_name = str(
            llm_cfg.get("model_chat")
            or llm_cfg.get("model_cheap")
            or llm_cfg.get("model")
            or "claude-haiku-4-5-20251001"
        )

        messages: list[dict[str, str]] = []
        for h in history[-20:]:  # keep last 20 turns for context
            role = h.get("role", "")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            reply = response.content[0].text
            return {"reply": reply, "error": False, "model": model_name}
        except Exception as exc:
            return {"reply": f"Error calling Anthropic API with {model_name}: {exc}", "error": True}
