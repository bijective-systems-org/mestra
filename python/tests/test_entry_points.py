"""One rule set decides every entry point.

Where a metadata open and a read do not name the same rule, a caller
moving between two languages finds the same invalid file opening in
one and refused in the other. Within this implementation they name
the same rule, and these tests are what holds it: over every case of
the corpus and every file of the hostile subset, the structural rule
the metadata open names is the one a read names.

The set is `mestra.reader.REFUSED`, which is the one section 2 of
`docs/api-conventions.md` fixes: E01, E16, E19, E25, E26, E29, E30,
E40 and E41. A semantic fault -- a missing unit, a support a row
references and the file does not hold, a dtype the role does not
allow -- never stops a read, so that `info` works on the files a user
most needs to inspect. That is a choice and not an accident, and it
is the same choice at every door.

Section 7 of the conventions is what makes it hold: an open reads
"attributes, dataspaces, link types, and dimension-scale structure,
and ... a category table in full", and "never a slot's data and never
a dataset inside a callable's dictionary". The nine rules are decided
from exactly that. A finding that needs a value the open did not read
-- E32 on a dictionary that holds something section 25 cannot
represent, for one -- waits for the read, along with the value, and
is not one of the nine.

Section 30 asks for more than agreement on the hostile subset:
"Opening the file for its metadata alone, and any operation that
reads a slot, must refuse with the same ids rather than return
something." The last test here is that, against each hostile case's
own `required_errors`.
"""

from __future__ import annotations

import json
import os

import pytest

import mestra
from mestra.errors import MestraError
from mestra.reader import REFUSED
from tests import corpus

HOSTILE = os.path.join(corpus.VECTORS, "hostile")


def _hostile_names() -> list[str]:
    with open(os.path.join(corpus.VECTORS, "manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    return [case["name"] for case in manifest.get("hostile", [])]


def _hostile_path(name: str) -> str:
    return os.path.join(HOSTILE, name, "case.mes")


def _hostile_expected(name: str) -> dict:
    with open(os.path.join(HOSTILE, name, "expected.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def _named(call) -> list[str]:
    """The rules one entry point names: the refusal, or the
    findings a non-refusing entry point leaves on the dataset."""
    try:
        dataset = call()
    except MestraError as exc:
        return ["refused:" + exc.rule]
    with dataset:
        return sorted({finding.rule for finding in dataset.problems})


def _structural(named: list[str]) -> list[str]:
    """The nine rules of section 2, out of what an entry point
    named, a refusal and a finding alike."""
    return sorted({rule.split(":", 1)[-1] for rule in named
                   if rule.split(":", 1)[-1] in REFUSED})


def _open(path: str) -> list[str]:
    """The metadata open: attributes and dataspaces, no array."""
    return _named(lambda: mestra.read(path))


def _read(path: str) -> list[str]:
    """The read: every value, at once."""
    return _named(lambda: mestra.read(path, lazy=False))


CASES = corpus.case_names()
HOSTILE_CASES = _hostile_names()


def test_the_hostile_subset_is_where_it_should_be():
    assert len(HOSTILE_CASES) == 15
    for name in HOSTILE_CASES:
        assert os.path.exists(_hostile_path(name)), name


@pytest.mark.parametrize("name", CASES)
def test_the_open_and_the_read_agree_on_every_corpus_case(name):
    path = corpus.case_path(name)
    assert _structural(_open(path)) == _structural(_read(path))


@pytest.mark.parametrize("name", HOSTILE_CASES)
def test_the_open_and_the_read_agree_on_every_hostile_file(name):
    path = _hostile_path(name)
    assert _structural(_open(path)) == _structural(_read(path))


@pytest.mark.parametrize("name", CASES)
def test_a_refusal_names_a_rule_from_the_documented_set(name):
    """Nothing is refused that the conventions do not list, and
    everything listed that the validator finds is refused."""
    path = corpus.case_path(name)
    named = _open(path)
    refusals = [r.split(":", 1)[1] for r in named
                if r.startswith("refused:")]
    for rule in refusals:
        assert rule in REFUSED, "%s refused under %s" % (name, rule)
    if refusals:
        return
    # Nothing was refused. The validator reads more than an open may,
    # so it may see a rule the open could not; what it must not see
    # is one the open had everything it needed to name.
    found = set(mestra.validate(path).error_ids) & set(REFUSED)
    assert not found - _NEEDS_A_VALUE_THE_OPEN_DOES_NOT_READ, (
        "%s breaks %s and opened anyway" % (name, sorted(found)))


#: E16 is the one of the nine that can depend on a value an open does
#: not read: in an unaligned file the row count of a row-varying array
#: is "the number of rows referencing that support", and an open takes
#: that from the length of the support's own `row` scale (section 21)
#: rather than from the `/row_support` column. A file where those two
#: disagree is E16 to `validate` and not to an open.
_NEEDS_A_VALUE_THE_OPEN_DOES_NOT_READ = {"E16"}


@pytest.mark.parametrize("name", HOSTILE_CASES)
def test_the_hostile_contract_holds_at_both_doors(name):
    """Section 30: the metadata open and any operation that reads a
    slot must refuse with the same ids rather than return
    something."""
    required = _hostile_expected(name)["required_errors"]
    for entry in (_open, _read):
        named = entry(_hostile_path(name))
        refusals = {r.split(":", 1)[1] for r in named
                    if r.startswith("refused:")}
        assert refusals, "%s: %s returned instead of refusing" % (
            name, entry.__name__)
        for rule in required:
            assert rule in refusals or rule in set(named), (
                "%s: %s named %s and the file requires %s"
                % (name, entry.__name__, sorted(refusals), required))


# ------------------------------------ what a metadata open may read

def _paths_read_during(call, monkeypatch) -> list[str]:
    """Every dataset one call reads, by path.

    Everything this package reads from a dataset goes through
    `h5safe.read_values`, so recording its calls records the reads.
    """
    from mestra import h5safe

    seen: list[str] = []
    real = h5safe.read_values

    def watching(dset, where="", rows=None, limit=None):
        seen.append(where or dset.name)
        return real(dset, where, rows, limit)

    monkeypatch.setattr(h5safe, "read_values", watching)
    call()
    return seen


@pytest.mark.parametrize("name", corpus.valid_case_names())
def test_a_metadata_open_reads_only_category_tables(name, monkeypatch):
    """Section 7 of docs/api-conventions.md.

    "Opening a file reads attributes, dataspaces, link types, and
    dimension-scale structure, and may read a category table in
    full ... An open never reads a slot's data and never reads a
    dataset inside a callable's dictionary; those wait for the
    read." And "/row_support is the one other dataset an open
    reads", because E16 in an unaligned file is decided from it.
    """
    path = corpus.case_path(name)
    read = _paths_read_during(
        lambda: mestra.read(path).close(), monkeypatch)
    wrong = [p for p in read
             if not p.startswith("/categories/") and p != "/row_support"]
    assert wrong == [], "%s: the open read %s" % (name, sorted(set(wrong)))


def test_an_eager_read_does_read_the_slots_and_the_dictionaries(
        monkeypatch):
    """The other half of the sentence: those wait for the read, and
    the read is where they happen."""
    path = corpus.case_path("callable_two_slots")
    read = _paths_read_during(
        lambda: mestra.read(path, lazy=False).close(), monkeypatch)
    assert any(p.startswith("/keys/") for p in read)
    assert any(p.startswith("/callables/") for p in read)


def test_a_lazy_open_leaves_a_dictionary_until_it_is_asked_for(
        monkeypatch):
    """A callable's id, type and repr are a link name and two
    attributes, so an open has them; the dictionary is datasets and
    waits."""
    path = corpus.case_path("callable_two_slots")
    with mestra.read(path) as dataset:
        read = _paths_read_during(lambda: None, monkeypatch)
        held = sorted(dataset.callables)
        assert held and len(dataset.callables) == len(held)
        assert held[0] in dataset.callables
        assert read == []
        read = _paths_read_during(
            lambda: dataset.callables[held[0]].to_dict(), monkeypatch)
        assert any(p.startswith("/callables/%s/" % held[0])
                   for p in read)


def test_an_unread_callable_is_never_handed_out_as_a_placeholder():
    """An id whose dictionary has not been read is held under its
    own name, so that listing the ids costs nothing. Every way of
    getting the values out must read them first."""
    path = corpus.case_path("callable_two_slots")
    for take in (dict, lambda c: c.copy(), lambda c: {**c},
                 lambda c: dict(c.items()),
                 lambda c: dict(zip(sorted(c), c.values()))):
        with mestra.read(path) as dataset:
            held = take(dataset.callables)
            assert held, take
            for name, obj in held.items():
                assert obj is not None, (take, name)
                assert hasattr(obj, "to_dict"), (take, name)
    with mestra.read(path) as dataset:
        assert "None" not in repr(dataset.callables)
