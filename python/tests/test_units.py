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


def test_the_corpus_units_all_parse_except_the_ones_that_must_not():
    import mestra
    from tests import corpus
    for name in corpus.valid_case_names():
        report = mestra.validate(corpus.case_path(name))
        if name.startswith("warn_w10"):
            assert "W10" in report.warning_ids
        else:
            assert "W10" not in report.warning_ids


#: The strings the four implementations were compared on, and a few
#: more that pin one reading each of section 32. Every implementation
#: carries this list and must give these verdicts.
GRAMMAR_PARSES = [
    "1", "Pa", "m s-1", "W m-2", "m2 s-2", "kg m-3", "m/s", "m s^-1",
    "m**2", "m.s-1", "m*s", "(m)", "((m))", "m (s)", "%", "%%", "m%",
    "degree", "degree_C", "degC", "K", "km", "mm", "um", "\u00b5m",
    "\u03bcm", "\u00b0", "\u00b0C", "rad", "sr", "1e3 m", "10 m",
    "0.5 m", "-1", "m-1", "m+2", "m^2", "m^-2", "m^+2", "s^0.5",
    "per s", "m per s", "days since 2000-01-01",
    "seconds since 1970-01-01T00:00:00Z",
    "hours since 2000-01-01 00:00:00", "K @ 273.15", "lg(re 1 mW)",
    "log(re 1)", "qux", "furlong", "m m", "m  s", " m", "m ", "mol",
    "cd", "A", "N", "J", "W", "V", "Ohm", "ohm", "S", "Hz", "dB",
    "count", "percent", "ppm", "1/s", "kg.m.s-2", "m2.s-1",
    # "m -1" is m times -1, "m-s" is m times s, "m2s" is one name.
    "m -1", "m-s", "m2s", "m^-2.5", "lg(re: 1 mW)", "log(m)",
]

GRAMMAR_REFUSES = [
    "m per", "/s", "s/", "m//s", "m*/s",
    # A "+" is a sign of a power, never an operator.
    "m+s",
    # The only space is U+0020.
    "m\ts", "", "  ",
]


@pytest.mark.parametrize("text", GRAMMAR_PARSES)
def test_the_grammar_accepts(text):
    assert units.is_parseable(text), text


@pytest.mark.parametrize("text", GRAMMAR_REFUSES)
def test_the_grammar_refuses(text):
    assert not units.is_parseable(text), text


def test_an_empty_units_string_is_w10_in_memory_as_on_disk():
    """A dataset read from a file whose units are "" holds "", which is
    text that does not parse. The in-memory validator took it for an
    absent attribute and said E39, where the file's validator says
    W10."""
    import mestra
    from tests import corpus
    with mestra.read(corpus.case_path("mesh_two_rows")) as ds:
        ds.keys["mach"].units = ""
        ds.scalars["cl"].units = ""
        report = mestra.validate(ds)
    assert report.error_ids == []
    assert report.warning_ids == ["W10"]
