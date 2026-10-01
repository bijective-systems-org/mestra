"""The conformance corpus, run against this implementation.

Every case in vectors/ must give exactly the outcome its
expected.json states: the validator's errors and warnings by rule
identifier, the value at every probe bit for bit, every support_id,
every codec round trip, and every worked evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

import mestra
from tests import corpus

CASES = corpus.case_names()
VALID = corpus.valid_case_names()


def test_the_corpus_is_where_it_should_be():
    """119 cases: the band added five, of which `band_stored`,
    `affine_band` and `warn_w16` are files a reader accepts and
    `err_e12_band` and `err_e12_callable` are not; served coordinates
    added `callable_coordinates`. `wide_keys` is generated on demand,
    as vectors/README.md says: `python vectors/generate.py
    --on-demand`."""
    assert len(CASES) == 119
    assert len(VALID) == 47
    assert len(WITH_CODEC) == 12


@pytest.mark.parametrize("name", CASES)
def test_validator_outcomes(name):
    """Section 14: the errors and the warnings, by identifier."""
    want = corpus.expected(name)["validator"]
    report = mestra.validate(corpus.case_path(name))
    assert report.error_ids == want["errors"], report
    assert report.warning_ids == want["warnings"], report


@pytest.mark.parametrize("name", CASES)
def test_support_ids(name):
    """Section 24: the digest an implementation must compute.

    Every case has one, the files a reader must refuse included, so
    the digests come from the cross-file check of section 8 rather
    than from a full read.
    """
    want = corpus.expected(name)["support_ids"]
    assert mestra.support_ids(corpus.case_path(name)) == want


@pytest.mark.parametrize("name", VALID)
def test_support_ids_from_the_model(name):
    """The same digests, computed from the dataset's own arrays."""
    want = corpus.expected(name)["support_ids"]
    with mestra.read(corpus.case_path(name)) as ds:
        got = {sname: support.computed_support_id()
               for sname, support in ds.supports.items()}
    assert got == want


@pytest.mark.parametrize("name", VALID)
def test_probes(name):
    """Section 30: the value at each named coordinate, bit for bit."""
    probes = corpus.expected(name)["probes"]
    with mestra.read(corpus.case_path(name)) as ds:
        for probe in probes:
            got = corpus.probe_value(ds, probe)
            if np.asarray(got).dtype.kind == "f":
                assert corpus.bits_equal(
                    got, corpus.as_float(probe["value"])), probe
            else:
                assert str(int(got)) == probe["value"], probe


WITH_CODEC = [n for n in CASES if corpus.expected(n)["codec"]]


@pytest.mark.parametrize("name", WITH_CODEC)
def test_codec_round_trip(name):
    """Sections 17 and 25: the dictionary, out and back.

    Two files that break a rule elsewhere still state a round trip,
    so this runs on every case that has one and not only on the
    valid ones.
    """
    want = corpus.expected(name)["codec"]
    with mestra.read(corpus.case_path(name)) as ds:
        for identifier, tagged in want.items():
            got = corpus.tagged(ds.callables[identifier].to_dict())
            assert corpus.same_tagged(got, tagged) == []


@pytest.mark.parametrize("name", VALID)
def test_evaluations(name):
    """Section 27: the worked evaluation of each callable."""
    entries = corpus.expected(name)["evaluation"]
    if not entries:
        return
    with mestra.read(corpus.case_path(name)) as ds:
        for entry in entries:
            table = {key: np.array([corpus.as_float(v) for v in values])
                     for key, values in entry["keys"].items()}
            out = mestra.evaluate(ds, table)
            for probe in entry["probes"]:
                got = corpus.probe_value(out, probe)
                assert corpus.bits_equal(
                    got, corpus.as_float(probe["value"])), probe


@pytest.mark.parametrize("name", VALID)
def test_evaluation_through_the_callable(name):
    """The same numbers, straight from `__call__`."""
    entries = corpus.expected(name)["evaluation"]
    if not entries:
        return
    with mestra.read(corpus.case_path(name)) as ds:
        for entry in entries:
            table = {key: np.array([corpus.as_float(v) for v in values])
                     for key, values in entry["keys"].items()}
            values = ds.callables[entry["callable"]](table)
            for probe in entry["probes"]:
                slot = _slot_of(ds, probe["slot"])
                record = values[slot.output]
                array = (record.uncertainty if slot.statistic == "band"
                         else record.mean)
                index = [probe["row"]]
                if "node" in probe:
                    index += [probe["node"], probe["component"]]
                assert corpus.bits_equal(
                    array[tuple(index)],
                    corpus.as_float(probe["value"])), probe


def _slot_of(ds, path):
    parts = path.strip("/").split("/")
    if parts[0] == "scalars":
        return ds.scalars[parts[1]]
    support = ds.supports[parts[1]]
    if parts[2] == "coordinates":
        return support.coordinates
    arrays = (support.node_arrays if parts[2] == "node_arrays"
              else support.cell_arrays)
    return arrays[parts[3]]


@pytest.mark.parametrize("name", CASES)
def test_the_description_matches_the_manifest(name):
    """The corpus states the same description in both places."""
    import json
    import os
    with open(os.path.join(corpus.VECTORS, "manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    described = {case["name"]: case["description"]
                 for case in manifest["cases"]}
    assert described[name] == corpus.expected(name)["description"]
