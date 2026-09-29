"""Files nobody vouches for.

Every file under tests/hostile/ must be handled by `read`,
`read(lazy=False)`, `validate`, `info` and `support_ids` in one of
two ways: findings, or a `MestraError` that names the rule. Never a
crash, never a hang, never an allocation that takes the machine
down.

Each file is driven in a subprocess with a timeout, because the only
defence against a hang inside a C library is a process that can be
killed. The same calls are then made in this process, where their
findings can be looked at.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import mestra
from mestra import limits

HERE = os.path.dirname(os.path.abspath(__file__))
HOSTILE = os.path.join(HERE, "hostile")
MAKER = os.path.join(HOSTILE, "make_hostile.py")

#: Every call must finish in this long, for any file.
TIMEOUT = 60

#: The identifiers a refusal may name (sections 14 and 29).
RULES = ("E01", "E16", "E19", "E25", "E26", "E29", "E30", "E40",
         "E41")

#: What the driver runs against one file.
DRIVER = r"""
import json, sys
import mestra
from mestra.cli import main as cli

path = sys.argv[1]
out = {}
for label in ("read", "read_eager", "validate", "support_ids", "info"):
    try:
        if label == "read":
            with mestra.read(path) as ds:
                _ = ds.n_rows, ds.aligned, ds.key_names()
                for key in ds.keys.values():
                    _ = key.role, key.units, key.lower, key.upper
                for support in ds.supports.values():
                    _ = support.kind, support.stored_support_id
                    for slot in support.arrays().values():
                        _ = slot.role, slot.dims, slot.units
                out[label] = ["problems", len(ds.problems)]
        elif label == "read_eager":
            ds = mestra.read(path, lazy=False)
            out[label] = ["problems", len(ds.problems)]
        elif label == "validate":
            report = mestra.validate(path)
            out[label] = ["report", report.error_ids,
                          report.warning_ids, len(report.unclassified)]
        elif label == "support_ids":
            out[label] = ["ids", sorted(mestra.support_ids(path))]
        else:
            cli(["info", path])
            out[label] = ["ok"]
    except mestra.MestraError as exc:
        out[label] = ["MestraError", exc.rule]
sys.stderr.write(json.dumps(out))
"""


def hostile_files() -> list[str]:
    return sorted(n for n in os.listdir(HOSTILE) if n.endswith(".mes"))


CASES = hostile_files()


def drive(path: str, timeout: int = TIMEOUT) -> dict:
    """Run every entry point against one file, in its own process."""
    done = subprocess.run(
        [sys.executable, "-c", DRIVER, path],
        capture_output=True, timeout=timeout, text=True,
        cwd=os.path.dirname(HERE))
    assert done.returncode == 0, (
        "the process did not survive %s (exit %d)\n%s"
        % (path, done.returncode, done.stderr[-2000:]))
    tail = done.stderr[done.stderr.rfind("{"):]
    return json.loads(tail)


def test_there_are_hostile_files():
    assert len(CASES) >= 13
    assert os.path.exists(MAKER)


@pytest.mark.parametrize("name", CASES)
def test_every_entry_point_survives(name):
    """Findings or a MestraError, in one process that comes back."""
    got = drive(os.path.join(HOSTILE, name))
    assert sorted(got) == ["info", "read", "read_eager", "support_ids",
                           "validate"]
    for label, answer in got.items():
        if answer[0] == "MestraError":
            assert answer[1] in RULES, (name, label, answer)
    # Opening one of these must refuse rather than hand back
    # something this reader cannot vouch for (section 30).
    assert got["read"][0] == "MestraError", name
    assert got["read_eager"][0] == "MestraError", name


@pytest.mark.parametrize("name", CASES)
def test_validate_says_something_about_every_one(name):
    """A hostile file is never quietly accepted."""
    report = mestra.validate(os.path.join(HOSTILE, name))
    assert not report.ok, name
    assert report.findings, name


# ------------------------------------------------- what each one gives

def validate(name):
    return mestra.validate(os.path.join(HOSTILE, name))


def read(name, lazy=True, strict=False):
    """Open one anyway, to look at what came back.

    `strict=False` because `read` refuses these files by default;
    that refusal is what test_every_entry_point_survives checks.
    """
    return mestra.read(os.path.join(HOSTILE, name), lazy=lazy,
                       strict=strict)


@pytest.mark.parametrize("name", CASES)
def test_info_prints_something_about_every_one(name, capsys):
    """`mestra info` on a stranger says what it found or why not."""
    from mestra.cli import main as cli
    status = cli(["info", os.path.join(HOSTILE, name)])
    out = capsys.readouterr().out
    assert status in (0, 1)
    assert out.strip()


def test_array_attributes_are_e19():
    """Section 18 gives every attribute a scalar dataspace."""
    report = validate("attrs_as_arrays.mes")
    assert "E19" in report.error_ids
    where = [f.where for f in report.errors if f.rule == "E19"]
    assert "/" in where and "/keys/mach" in where
    assert "/scalars/cl" in where
    with read("attrs_as_arrays.mes") as ds:
        # The file still reads: an array attribute comes back as its
        # first element rather than as a crash.
        assert ds.format == "mestra/0"
        assert ds.keys["mach"].lower == 0.1


@pytest.mark.parametrize("name", ["filter_many_client_values.mes",
                                  "filter_unknown_id.mes"])
def test_odd_filters_are_e29(name):
    assert "E29" in validate(name).error_ids


def test_deep_groups_stop_at_the_limit():
    report = validate("deep_groups.mes")
    assert any("nests more than" in f.message
               for f in report.unclassified)
    with read("deep_groups.mes") as ds:
        assert ds.lossy, "a group too deep to copy is not rewritable"
        with pytest.raises(mestra.MestraError) as caught:
            mestra.write(ds, os.devnull)
        assert caught.value.rule == "E41"


def test_a_group_that_is_its_own_ancestor():
    report = validate("cyclic_groups.mes")
    assert "E40" in report.error_ids
    assert all(f.rule in ("E40", "E41") for f in report.unclassified)
    with read("cyclic_groups.mes") as ds:
        assert any("soft link" in f.message for f in ds.problems)


def test_no_link_is_followed():
    report = validate("links.mes")
    messages = " ".join(f.message for f in report.unclassified)
    assert "external link" in messages
    assert "does not follow" in messages
    with read("links.mes") as ds:
        external = [f for f in ds.problems if "external link" in f.message]
        assert external
        # An external link names another file, so a rewrite would
        # lose it rather than copy it.
        assert ds.lossy


def test_members_of_the_wrong_kind():
    report = validate("member_kinds.mes")
    with read("member_kinds.mes") as ds:
        messages = " ".join(f.message for f in ds.problems)
    assert "a key is a dataset" in messages
    assert "a support is a group" in messages
    assert "a callable is a group" in messages
    # An object of the wrong kind is one the reader cannot read as
    # what its place says it is (E41), in every language; a callable
    # that is a dataset is not a callable group without a type (E15).
    wrong_kind = {f.where for f in report.errors if f.rule == "E41"}
    assert {"/keys/alpha", "/supports/s1", "/callables/m1"} <= wrong_kind
    assert "E15" not in report.error_ids
    # The scalar that is a group with `source = data` is E30.
    assert "E30" in report.error_ids


def test_an_enormous_declared_shape():
    """Lazy reading does not touch it; eager reading refuses."""
    with read("enormous_shape.mes", strict=False) as ds:
        slot = ds.supports["s0"].node_arrays["enormous"]
        assert slot.data.shape == (1000000, 1000000)
        assert slot.data.reads == 0
        with pytest.raises(mestra.MestraError) as caught:
            slot.read()
        assert caught.value.rule == "E41"
        assert str(limits.MAX_READ_ELEMENTS) in str(caught.value)
        # A row range of it is a normal read.
        part = slot.read(slice(0, 1))
        assert part.shape == (1, 1000000)
    with pytest.raises(mestra.MestraError) as caught:
        read("enormous_shape.mes", lazy=False, strict=False)
    assert caught.value.rule == "E41"


def test_strings_that_are_not_utf8():
    report = validate("bad_strings.mes")
    assert "E26" in report.error_ids
    with read("bad_strings.mes") as ds:
        # Replaced, not raised, and the empty entry is still an entry.
        assert len(ds.categories["region"]) == 2
        assert ds.categories["region"][1] == ""


def test_scales_attached_twice_and_without_a_name():
    report = validate("scale_trouble.mes")
    assert "E25" in report.error_ids
    with read("scale_trouble.mes") as ds:
        # An axis with two scales falls back to the names the slot's
        # own attributes imply, and the reader keeps working.
        assert ds.supports["s0"].node_arrays["pressure"].dims == (
            "row", "node", "component")


def test_the_validator_carries_on_past_what_it_cannot_read():
    """The rules after the unreadable object are still reported."""
    report = validate("unreadable_dataset.mes")
    assert report.unclassified
    assert any("/categories/" in f.where for f in report.unclassified)
    # E11 is about a field that sits after the unreadable objects.
    assert "E11" in report.error_ids
    assert any(f.where.endswith("zeta") for f in report.errors)


@pytest.mark.parametrize("name", ["not_hdf5.mes", "truncated.mes"])
def test_files_that_are_not_hdf5(name):
    report = validate(name)
    assert report.error_ids == ["E01"]
    with pytest.raises(mestra.MestraError) as caught:
        read(name)
    assert caught.value.rule == "E01"
    with pytest.raises(mestra.MestraError):
        mestra.support_ids(os.path.join(HOSTILE, name))


def test_a_file_that_is_not_there():
    with pytest.raises(mestra.MestraError) as caught:
        mestra.read(os.path.join(HOSTILE, "no_such_file.mes"))
    assert caught.value.rule == "E01"


# ------------------------------------------------- the big deep file

@pytest.mark.slow
def test_thirty_thousand_levels(tmp_path):
    """The case that is too big to commit, made on demand.

    Asking HDF5 for the name of a scale reached by object reference
    used to take the process down on this file. It is here so that
    it cannot come back.
    """
    made = subprocess.run(
        [sys.executable, MAKER, "--deep", str(tmp_path)],
        capture_output=True, timeout=300, text=True)
    assert made.returncode == 0, made.stderr[-2000:]
    path = made.stdout.strip()
    assert os.path.exists(path)
    got = drive(path, timeout=300)
    assert got["validate"][0] == "report"
    assert got["read"] == ["MestraError", "E41"]
    report = mestra.validate(path)
    assert any("nests more than" in f.message
               for f in report.unclassified)


# ---------------------------------------------- the recursive routines

def test_the_units_parser_has_a_depth_cap():
    from mestra import units
    deep = "(" * (limits.MAX_UNITS_DEPTH + 10) + "m" + ")" * (
        limits.MAX_UNITS_DEPTH + 10)
    assert units.parse(deep) is None
    assert units.parse("(" * 3 + "m" + ")" * 3) is not None
    assert units.parse("m " * (limits.MAX_UNITS_LENGTH)) is None


def test_the_codec_has_a_depth_cap(tmp_path):
    import h5py

    from mestra.codec import decode_dict, encode_dict
    nest: dict = {"x": 1}
    for _ in range(limits.MAX_DEPTH + 5):
        nest = {"g": nest}
    path = str(tmp_path / "deep.h5")
    with h5py.File(path, "w") as f, pytest.raises(mestra.MestraError) as caught:
        encode_dict(f.create_group("m"), nest)
    assert caught.value.rule == "E32"

    # And reading one that deep stops rather than recurses.
    with h5py.File(path, "w") as f:
        group = f.create_group("m")
        for _ in range(limits.MAX_DEPTH + 5):
            group = group.create_group("g")
        group.attrs.create("x", 1)
    with h5py.File(path, "r") as f:
        problems: list = []
        decode_dict(f["m"], problems=problems)
        assert any("nests more than" in p.message for p in problems)


def test_the_affine_order_is_not_numpy_s_to_choose():
    """Section 27: the dot product is accumulated over the keys in
    the declared order and b is added last, with no fused
    multiply-add.

    The check is against plain Python float arithmetic, which has no
    fused operations and no reassociation: every output must agree
    bit for bit.
    """
    import numpy as np

    rng = np.random.default_rng(20260920)
    for _ in range(50):
        n_keys = int(rng.integers(1, 6))
        n_out = int(rng.integers(1, 8))
        rows = int(rng.integers(1, 4))
        matrix = rng.normal(size=(n_out, n_keys)) * float(
            10.0 ** int(rng.integers(-6, 6)))
        offset = rng.normal(size=n_out) * float(
            10.0 ** int(rng.integers(-6, 6)))
        table = {"k%d" % j: rng.normal(size=rows) for j in range(n_keys)}
        keys = list(table)
        model = mestra.Affine(keys, {"y": {"A": matrix, "b": offset,
                                           "shape": [n_out]}})
        got = model(table)["y"].mean
        for row in range(rows):
            for out in range(n_out):
                total = 0.0
                for at, key in enumerate(keys):
                    total += float(matrix[out, at]) * float(
                        table[key][row])
                total += float(offset[out])
                assert got[row, out].tobytes() == np.float64(
                    total).tobytes(), (row, out)
