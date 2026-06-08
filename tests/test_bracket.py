"""
test_bracket.py
Unit tests for sim/bracket.py — asserted against the official 2026 FIFA
bracket chart (FIFA Annex C / Wikipedia Template:2026 FIFA World Cup
third-place table).  No simulated data; pure lookup correctness.
"""
from itertools import combinations

import pytest

from wcpredict.sim.bracket import (
    THIRD_PLACE_TABLE,
    lookup_third_place_slots,
    build_r32,
    Matchup,
)


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------

def test_table_has_495_entries():
    assert len(THIRD_PLACE_TABLE) == 495


def test_all_keys_are_frozensets_of_8():
    for key in THIRD_PLACE_TABLE:
        assert isinstance(key, frozenset), f"Key {key!r} is not a frozenset"
        assert len(key) == 8, f"Key {key!r} has {len(key)} elements, expected 8"


def test_all_keys_use_valid_group_letters():
    valid = set("ABCDEFGHIJKL")
    for key in THIRD_PLACE_TABLE:
        for ch in key:
            assert ch in valid, f"Invalid group letter {ch!r} in key {key}"


def test_all_slot_maps_have_8_entries():
    slots = set("ABDEGIKL")
    for key, slot_map in THIRD_PLACE_TABLE.items():
        assert set(slot_map.keys()) == slots, (
            f"Key {''.join(sorted(key))} has slots {set(slot_map.keys())}, expected {slots}"
        )


def test_slot_values_are_valid_groups():
    valid = set("ABCDEFGHIJKL")
    for key, slot_map in THIRD_PLACE_TABLE.items():
        for slot, src in slot_map.items():
            assert src in valid, f"Slot {slot} -> {src!r} invalid in combo {''.join(sorted(key))}"


def test_slot_source_must_be_in_qualifying_groups():
    for key, slot_map in THIRD_PLACE_TABLE.items():
        for slot, src in slot_map.items():
            assert src in key, (
                f"Combo {''.join(sorted(key))}: slot {slot} -> {src}, "
                f"but {src} not in qualifying groups"
            )


def test_each_source_group_used_exactly_once():
    for key, slot_map in THIRD_PLACE_TABLE.items():
        sources = list(slot_map.values())
        assert len(sources) == len(set(sources)), (
            f"Combo {''.join(sorted(key))} has duplicate slot sources: {sources}"
        )


# ---------------------------------------------------------------------------
# Official spot-checks from FIFA Annex C / Wikipedia template
# Combo #1 (EFGHIJKL): slots A=E, B=J, D=I, E=F, G=H, I=G, K=L, L=K
# ---------------------------------------------------------------------------

def test_combo_EFGHIJKL():
    m = lookup_third_place_slots(set("EFGHIJKL"))
    assert m == {"A": "E", "B": "J", "D": "I", "E": "F", "G": "H", "I": "G", "K": "L", "L": "K"}


def test_combo_DFGHIJKL():
    m = lookup_third_place_slots(set("DFGHIJKL"))
    assert m == {"A": "H", "B": "G", "D": "I", "E": "D", "G": "J", "I": "F", "K": "L", "L": "K"}


def test_combo_ABCDEFGH():
    m = lookup_third_place_slots(set("ABCDEFGH"))
    assert m == {"A": "H", "B": "G", "D": "B", "E": "C", "G": "A", "I": "F", "K": "D", "L": "E"}


def test_combo_ABCDEFGK():
    m = lookup_third_place_slots(set("ABCDEFGK"))
    assert m == {"A": "C", "B": "G", "D": "B", "E": "D", "G": "A", "I": "F", "K": "E", "L": "K"}


def test_combo_CEFGHIJK():
    m = lookup_third_place_slots(set("CEFGHIJK"))
    assert m == {"A": "E", "B": "G", "D": "J", "E": "C", "G": "H", "I": "F", "K": "I", "L": "K"}


def test_combo_DEFGHIJK():
    m = lookup_third_place_slots(set("DEFGHIJK"))
    assert m == {"A": "E", "B": "G", "D": "J", "E": "D", "G": "H", "I": "F", "K": "I", "L": "K"}


# --- 6 additional spot-checks from mid/late table ---

def test_combo_ABCDEFJL():
    m = lookup_third_place_slots(set("ABCDEFJL"))
    assert m == {"A": "C", "B": "J", "D": "B", "E": "D", "G": "A", "I": "F", "K": "L", "L": "E"}


def test_combo_ABCDEFGL():
    m = lookup_third_place_slots(set("ABCDEFGL"))
    assert m == {"A": "C", "B": "G", "D": "B", "E": "D", "G": "A", "I": "F", "K": "L", "L": "E"}


def test_combo_ABCDEFHI():
    m = lookup_third_place_slots(set("ABCDEFHI"))
    assert m == {"A": "H", "B": "E", "D": "B", "E": "C", "G": "A", "I": "F", "K": "D", "L": "I"}


def test_combo_ABCDEIJK():
    m = lookup_third_place_slots(set("ABCDEIJK"))
    assert m == {"A": "E", "B": "J", "D": "B", "E": "C", "G": "A", "I": "D", "K": "I", "L": "K"}


def test_combo_BDEFGHIJ():
    m = lookup_third_place_slots(set("BDEFGHIJ"))
    assert m == {"A": "H", "B": "G", "D": "B", "E": "D", "G": "J", "I": "F", "K": "E", "L": "I"}


def test_combo_ACDEFGHI():
    m = lookup_third_place_slots(set("ACDEFGHI"))
    assert m == {"A": "H", "B": "G", "D": "E", "E": "C", "G": "A", "I": "F", "K": "D", "L": "I"}


def test_completeness_all_c12_8_combinations_present():
    """
    C(12,8) = 495.  Every valid 8-group subset of A-L must appear exactly once.
    If any is missing or duplicated, the parser in gen_bracket_data.py has a bug.
    """
    expected = {frozenset(c) for c in combinations("ABCDEFGHIJKL", 8)}
    actual   = set(THIRD_PLACE_TABLE.keys())
    missing  = expected - actual
    extra    = actual   - expected
    assert not missing, f"{len(missing)} valid combos missing: {sorted(''.join(sorted(k)) for k in list(missing)[:5])}"
    assert not extra,   f"{len(extra)} spurious entries: {sorted(''.join(sorted(k)) for k in list(extra)[:5])}"


def test_same_group_avoidance_all_495():
    """
    For every entry, the 3rd-place team from group X must NOT face the winner
    of group X.  This means slot_letter ≠ source_group for every (slot, src) pair.
    Official slot pools are designed to guarantee this; this test enforces it
    on every row of THIRD_PLACE_TABLE.
    """
    violations = []
    for key, slot_map in THIRD_PLACE_TABLE.items():
        for slot, src in slot_map.items():
            if slot == src:
                violations.append(f"combo={''.join(sorted(key))} slot={slot} src={src}")
    assert not violations, (
        f"{len(violations)} same-group violations found:\n" + "\n".join(violations[:10])
    )


def test_unknown_combination_raises_key_error():
    with pytest.raises(KeyError):
        # "X" and "Z" are not valid group letters; this frozenset won't be in the table
        lookup_third_place_slots({"A", "B", "C", "D", "E", "F", "X", "Z"})


# ---------------------------------------------------------------------------
# lookup_third_place_slots symmetry — frozenset input variants
# ---------------------------------------------------------------------------

def test_lookup_accepts_list():
    m = lookup_third_place_slots({"E", "F", "G", "H", "I", "J", "K", "L"})
    assert m["A"] == "E"


# ---------------------------------------------------------------------------
# build_r32 structural tests
# ---------------------------------------------------------------------------

def _fake_standings() -> dict[str, list[str]]:
    """Minimal group standings dict for structural tests."""
    groups = list("ABCDEFGHIJKL")
    return {
        g: [f"T{g}1", f"T{g}2", f"T{g}3", f"T{g}4"]
        for g in groups
    }


def test_build_r32_returns_16_matchups():
    standings = _fake_standings()
    matchups = build_r32(standings, set("EFGHIJKL"))
    assert len(matchups) == 16


def test_build_r32_all_matchups_are_Matchup():
    standings = _fake_standings()
    for m in build_r32(standings, set("EFGHIJKL")):
        assert isinstance(m, Matchup)
        assert m.home
        assert m.away
        assert m.label


def test_build_r32_fixed_pairings_present():
    standings = _fake_standings()
    matchups = build_r32(standings, set("EFGHIJKL"))
    labels = {m.label for m in matchups}
    assert "2A vs 2B" in labels
    assert "1C vs 2F" in labels
    assert "1F vs 2C" in labels
    assert "1J vs 2H" in labels


def test_build_r32_uses_third_place_correctly():
    standings = _fake_standings()
    matchups = build_r32(standings, set("EFGHIJKL"))
    # For EFGHIJKL combo, slot A gets 3rd from E => 1A vs 3E
    slot_a = next(m for m in matchups if m.label == "1A vs 3E")
    assert slot_a.home == "TA1"
    assert slot_a.away == "TE3"


def test_build_r32_no_team_appears_twice():
    standings = _fake_standings()
    matchups = build_r32(standings, set("EFGHIJKL"))
    all_teams = [t for m in matchups for t in (m.home, m.away)]
    assert len(all_teams) == len(set(all_teams)), "A team appears in more than one R32 match"


def test_build_r32_invalid_combo_raises():
    standings = _fake_standings()
    with pytest.raises(KeyError):
        build_r32(standings, {"A", "B", "C", "D", "E", "F", "X", "Z"})
