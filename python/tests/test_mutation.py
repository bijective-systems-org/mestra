"""Corpus files with their bytes damaged.

The hostile subset is files somebody built to be wrong. This is the
other kind: a good file that lost bytes in transit, or had a few
changed, which is what a reader meets far more often. Each corpus
case here is truncated at a sweep of lengths, has single bits
flipped at seeded positions, and has short runs zeroed, and every
one of those files is driven through `read`, `read(lazy=False)` and
`validate` in its own process.

The contract is the one the hostile tests state: findings, or a
`MestraError` that names a rule. Any other exception is a failure
here, because it means a caller would see an h5py or numpy error
this package never documented.

What this test does not promise is that the process comes back. The
HDF5 library parses the damaged metadata before any code of this
package runs, and on a crafted single-byte change it can fault or
spin instead of returning an error: measured 2026-09-27 as 4 of 424
mutations across two corpus cases (three faults and one hang, in
attribute and dimension-scale reads, and in one case at file
close), in the library itself, and the same happens to libhdf5 on
the C++ side. A lost child is therefore reported as a warning that
names the mutation, so it stays visible, and never as a pass or a
failure of this package. The defence is a process boundary, which is
why the command line is one, and docs/compatibility.md says so.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import warnings

import pytest

from . import corpus

#: The corpus cases damaged here: small, and between them a mesh
#: support, keys, scales, a callable and a label.
CASES = ("derived_displacement", "err_e11")

#: The seed every run uses, so a finding repeats.
SEED = 20260927

#: Every child must answer in this long, for any file.
TIMEOUT = 60

#: The identifiers a refusal may name (sections 14 and 29), plus the
#: ones a damaged file can legitimately earn on its way in.
RULES = ("E01", "E16", "E19", "E25", "E26", "E29", "E30", "E40", "E41")

DRIVER = r"""
import json, sys
import mestra
path = sys.argv[1]
out = {}
for label in ("read", "read_eager", "validate"):
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
        else:
            report = mestra.validate(path)
            out[label] = ["report", report.error_ids, report.warning_ids,
                          len(report.unclassified)]
    except mestra.MestraError as exc:
        out[label] = ["MestraError", exc.rule]
    except Exception as exc:  # the contract violation this test is for
        out[label] = ["other", type(exc).__name__, str(exc)[:200]]
sys.stderr.write("\nMUTATION " + json.dumps(out))
"""


def case_bytes(name: str) -> bytes:
    with open(os.path.join(corpus.CASES, name, "case.mes"), "rb") as fh:
        return fh.read()


def mutations(data: bytes, rng: random.Random, flips: int, zeroed: int):
    """(label, bytes) for one file: a truncation sweep, bit flips, zeroed runs."""
    n = len(data)
    for k in range(1, 17):
        yield "truncated to %d of %d bytes" % (n * k // 17, n), data[: n * k // 17]
    for _ in range(flips):
        pos = rng.randrange(n)
        bit = rng.randrange(8)
        damaged = bytearray(data)
        damaged[pos] ^= 1 << bit
        yield "bit %d of byte %d flipped" % (bit, pos), bytes(damaged)
    for _ in range(zeroed):
        pos = rng.randrange(n)
        damaged = bytearray(data)
        damaged[pos:pos + 16] = b"\0" * 16
        yield "16 bytes zeroed at %d" % pos, bytes(damaged)


def drive(path: str) -> tuple[str, dict | None]:
    """One child per file: ("answered", what) or ("lost", how)."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", DRIVER, path],
            capture_output=True, timeout=TIMEOUT, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except subprocess.TimeoutExpired:
        return "lost", {"how": "no answer in %d s" % TIMEOUT}
    if done.returncode < 0:
        return "lost", {"how": "signal %d" % -done.returncode}
    marker = done.stderr.rfind("MUTATION ")
    if done.returncode != 0 or marker < 0:
        return "lost", {"how": "exit %d" % done.returncode,
                        "stderr": done.stderr[-500:]}
    return "answered", json.loads(done.stderr[marker + len("MUTATION "):])


def check(name: str, label: str, answer: dict) -> None:
    for entry, got in answer.items():
        assert got[0] in ("problems", "report", "MestraError"), (
            "%s, %s: %s raised %s: %s -- not a MestraError and not findings"
            % (name, label, entry, got[1], got[2]))
        if got[0] == "MestraError":
            assert got[1] in RULES, (name, label, entry, got)


@pytest.mark.parametrize("name", CASES)
def test_damaged_files_get_an_answer_this_package_documents(name, tmp_path):
    """Sixty-four damaged copies of each case, each in its own process."""
    rng = random.Random(SEED)
    data = case_bytes(name)
    lost = []
    answered = 0
    for label, blob in mutations(data, rng, flips=40, zeroed=8):
        path = str(tmp_path / "damaged.mes")
        with open(path, "wb") as fh:
            fh.write(blob)
        outcome, detail = drive(path)
        if outcome == "lost":
            lost.append("%s: %s (%s)" % (name, label, detail["how"]))
            continue
        answered += 1
        check(name, label, detail)
    assert answered > 0
    if lost:
        warnings.warn(
            "libhdf5 did not return control on %d damaged cop%s of %s "
            "(a fault or a hang in the library, before this package "
            "ran; see docs/compatibility.md):\n  %s"
            % (len(lost), "y" if len(lost) == 1 else "ies", name,
               "\n  ".join(lost)), stacklevel=1)
