"""What the validator says, as section 5 of the conventions fixes it.

One finding per rule per object; a rule that can be broken per row
says so once, with the count and the first three rows; and a finding
prints as `<id> <path>: <message>`.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

import mestra
from mestra.errors import Finding
from tests import corpus

XY = np.array([[0., 0.], [1., 0.], [2., 0.],
               [0., 1.], [1., 1.], [2., 1.]])
CELLS = (np.array([9, 9]), np.array([0, 4, 8]),
         np.array([0, 1, 4, 3, 1, 2, 5, 4]))


def pairs(report):
    return collections.Counter((f.rule, f.where) for f in report.findings)


@pytest.mark.parametrize("name", corpus.case_names())
def test_one_finding_per_rule_per_object(name):
    """No object draws the same rule twice, anywhere in the corpus."""
    repeated = {at: n for at, n in
                pairs(mestra.validate(corpus.case_path(name))).items()
                if n > 1}
    assert repeated == {}


def test_a_finding_prints_as_the_conventions_say():
    found = Finding("E11", "/scalars/cl", "a scalar carries units")
    assert str(found) == "E11 /scalars/cl: a scalar carries units"
    assert str(Finding("E17", "", "no writer")) == "E17: no writer"


def with_status(words, rows):
    ds = mestra.Dataset(writer="t", created="2026-09-19T00:00:00Z")
    ds.add_key("mach", np.linspace(0.1, 0.9, len(rows)),
               role="condition", units="1")
    ds.add_key("status", rows, role="status", categories=words)
    return ds


def test_w02_reports_once_with_the_count_and_the_first_three():
    """P6: six rows gave six findings, one of them about a good row."""
    ds = with_status(["converged", "failed"],
                     [0, 1, 0, 1, 1, 1, 0, 1])
    found = [f for f in mestra.validate(ds).warnings if f.rule == "W02"]
    assert len(found) == 1
    assert "5 rows" in found[0].message
    assert "of 8" in found[0].message
    assert "at row 1, 3, 4 and 2 more" in found[0].message


def test_w02_says_which_word_it_expected():
    """P7: nothing said the word the validator wants is `converged`."""
    ds = with_status(["ok", "failed"], [0, 0, 1])
    found = [f for f in mestra.validate(ds).warnings if f.rule == "W02"]
    assert "converged" in found[0].message
    assert "converged, failed, partial" in found[0].message
    assert "'ok'" in found[0].message


def test_w02_is_silent_when_every_row_converged():
    ds = with_status(["converged", "failed"], [0, 0, 0])
    assert [f for f in mestra.validate(ds).warnings
            if f.rule == "W02"] == []


def test_w03_reports_once_with_the_count_and_the_first_three():
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", np.linspace(0.1, 0.9, 6), role="condition",
               units="1")
    ds.add_scalar("cl", [np.nan, 1.0, np.nan, 2.0, np.nan, np.inf],
                  units="1")
    found = [f for f in mestra.validate(ds).warnings if f.rule == "W03"]
    assert len(found) == 1
    assert "4 non-finite values" in found[0].message
    assert "at row 0, 2, 4 and 1 more" in found[0].message


def test_w03_on_a_field_counts_the_rows_and_not_the_values(tmp_path):
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    values = np.zeros((2, 6))
    values[1, :] = np.nan
    support.add_node_array("p", values, units="Pa",
                           dims=("row", "node"))
    path = str(tmp_path / "nan.mes")
    mestra.write(ds, path)
    found = [f for f in mestra.validate(path).warnings
             if f.rule == "W03"]
    assert len(found) == 1
    assert "6 non-finite values" in found[0].message
    assert "at row 1" in found[0].message


def test_w04_reports_once_with_the_count_and_the_first_three(tmp_path):
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.0, 0.5, 2.0, 3.0, 4.0], role="condition",
               units="1", lower=0.1, upper=0.9)
    path = str(tmp_path / "bounds.mes")
    mestra.write(ds, path, check=False)
    found = [f for f in mestra.validate(path).warnings
             if f.rule == "W04"]
    assert len(found) == 1
    assert "4 values" in found[0].message
    assert "[0.1, 0.9]" in found[0].message
    assert "at row 0, 2, 3 and 1 more" in found[0].message


def test_e09_names_the_trajectory_the_file_names_it(tmp_path):
    """P12: the file knows this trajectory is r000."""
    ds = mestra.Dataset(writer="t")
    ds.add_key("run", [0, 0, 0], role="group",
               categories=["r000", "r001"], generalisation=True)
    ds.add_key("t", [0.0, 0.3, 0.1], role="time", units="s",
               trajectory_group="run")
    path = str(tmp_path / "time.mes")
    mestra.write(ds, path, check=False)
    found = [f for f in mestra.validate(path).errors if f.rule == "E09"]
    assert len(found) == 1
    assert "r000" in found[0].message


def test_findings_from_opening_a_file_reach_the_dataset_report():
    """`unclassified` is a view of `errors`, so E40 must land there."""
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4], role="condition", units="1")
    ds.problems.append(
        Finding("E40", "/keys/elsewhere", "an external link"))
    report = mestra.validate(ds)
    assert "E40" in report.error_ids
    assert [f.where for f in report.unclassified] == ["/keys/elsewhere"]


def test_a_support_may_carry_weights_at_both_locations(tmp_path):
    """Section 3 counts weight 0..1 *per location*."""
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    support.add_node_array("p", np.zeros((2, 6)), units="Pa",
                           dims=("row", "node"))
    mestra.compute_weights(support, "node")
    mestra.compute_weights(support, "cell")
    assert mestra.validate(ds).error_ids == []
    path = str(tmp_path / "both.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []


def test_two_arrays_of_one_role_at_one_location_are_still_e03():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    mestra.compute_weights(support, "node")
    mestra.compute_weights(support, "node", name="lumped")
    assert "E03" in mestra.validate(ds).error_ids


def test_a_finding_about_an_illegal_name_stays_on_one_line():
    """Every tool prints one finding per line (conventions section
    5), and a name that breaks E33 may hold a newline; it is written
    as \\x0a rather than ending the line."""
    report = mestra.validate(corpus.case_path("err_e33_newline"))
    line = str(report.errors[0])
    assert "\n" not in line
    assert line.startswith("E33 /keys/mach\\x0a: ")
