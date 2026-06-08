from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIRS = [
    ROOT / "src" / "wcpredict" / name
    for name in ("data", "features", "model", "llm", "adjust", "sim")
]
FORBIDDEN_TERMS = (
    "odds",
    "market",
    "bookmaker",
    "sportsbook",
    "pinnacle",
    "betfair",
    "closing_line",
)


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in PREDICTION_DIRS:
        files.extend(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def test_prediction_modules_do_not_import_market_or_eval():
    violations: list[str] = []
    for path in _source_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                lowered = name.lower()
                if ".eval" in lowered or lowered.endswith("eval"):
                    violations.append(f"{path}: imports eval module {name}")
                for term in FORBIDDEN_TERMS:
                    if term in lowered:
                        violations.append(f"{path}: imports market/odds term {name}")
    assert not violations, "\n".join(violations)


def test_prediction_modules_do_not_reference_odds_paths_or_files():
    violations: list[str] = []
    io_names = {"open", "read_csv", "read_parquet", "read_json", "read_table", "Path"}
    for path in _source_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                name = ""
            if name not in io_names:
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    continue
                lowered = arg.value.lower()
                for term in FORBIDDEN_TERMS:
                    if term in lowered:
                        violations.append(f"{path}: reads forbidden market/odds path {arg.value!r}")
    assert not violations, "\n".join(violations)
