"""The units parser, which decides W10."""

from __future__ import annotations

import pytest

from mestra import units

PARSES = [
    ("1", {}),
    ("m", {"m": 1}),
    ("Pa", {"kg": 1, "m": -1, "s": -2}),
    ("degree", {}),
    ("s", {"s": 1}),
    ("K", {"K": 1}),
    ("W", {"kg": 1, "m": 2, "s": -3}),
    ("W m-2", {"kg": 1, "s": -3}),
    ("m2 s-1", {"m": 2, "s": -1}),
    ("m s-1", {"m": 1, "s": -1}),
    ("m/s", {"m": 1, "s": -1}),
    ("m/s2", {"m": 1, "s": -2}),
    ("m^2", {"m": 2}),
    ("m**2", {"m": 2}),
    ("kg m2 s-3", {"kg": 1, "m": 2, "s": -3}),
    ("J/(kg K)", {"m": 2, "s": -2, "K": -1}),
    ("W per m2", {"kg": 1, "s": -3}),
    ("km h-1", {"m": 1, "s": -1}),
    ("1e-3 m", {"m": 1}),
    ("%", {}),
    ("mm", {"m": 1}),
    ("degree_C", {"K": 1}),
]

REFUSES = ["kg/(m s", "", "   ", "m^", "m**", "s^^2", "kg)", "/m", "m/"]

#: Grammatical strings (section 32) that `parse` gives no dimensions
#: for: a name outside its table, a power that is not an integer.
NO_DIMENSIONS = ["fortnight", "m^-2.5", "days since 2000-01-01"]


@pytest.mark.parametrize("text, dimensions", PARSES)
def test_what_parses(text, dimensions):
    parsed = units.parse(text)
    assert parsed is not None, text
    assert parsed.dimensions == dimensions


@pytest.mark.parametrize("text", REFUSES)
def test_what_does_not(text):
    assert units.parse(text) is None, text
    assert not units.is_parseable(text)


@pytest.mark.parametrize("text", NO_DIMENSIONS)
def test_grammatical_without_dimensions(text):
    assert units.is_parseable(text), text
    assert units.parse(text) is None, text


def test_dimensions_decide_what_may_be_combined():
    assert units.same_dimensions("W m-2", "kg s-3")
    assert not units.same_dimensions("m", "s")
    # An unparseable string is never the same as anything, so a tool
    # refuses to combine it (section 3).
    assert not units.same_dimensions("kg/(m s", "kg/(m s")


def test_the_corpus_units_all_parse_except_the_one_that_must_not():
    import mestra
    from tests import corpus
    for name in corpus.valid_case_names():
        report = mestra.validate(corpus.case_path(name))
        if name == "warn_w10":
            assert "W10" in report.warning_ids
        else:
            assert "W10" not in report.warning_ids
