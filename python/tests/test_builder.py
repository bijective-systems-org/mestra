"""Building a dataset from arrays, and the messages a mistake gets.

A user should reach a valid file in a handful of calls, and a
mistake should name the rule it breaks.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mestra
from mestra.errors import MestraError
from tests import corpus

XY = np.array([[0., 0.], [1., 0.], [2., 0.],
               [0., 1.], [1., 1.], [2., 1.]])
CELLS = (np.array([9, 9]), np.array([0, 4, 8]),
         np.array([0, 1, 4, 3, 1, 2, 5, 4]))


def build_example():
    """docs/example.md file one, built from arrays."""
    ds = mestra.Dataset(writer="mestra examples 0",
                        created="2026-09-19T00:00:00Z")
    ds.add_key("mach", [0.40, 0.80], role="condition", units="1",
               lower=0.1, upper=0.9)
    ds.add_key("member", [0, 1], role="group",
               categories=["wing_a", "wing_b"], generalisation=True)
    ds.add_scalar("cl", [0.25, 0.55], units="1")
    support = ds.add_support(
        "s0", coordinates=np.stack([XY, XY * [1.5, 1.0]]),
        varies="group:member", cells=CELLS)
    support.add_node_array(
        "pressure", [[101., 102., 103., 104., 105., 106.],
                     [201., 202., 203., 204., 205., 206.]], units="Pa")
    support.add_cell_array("region", [0, 1], role="label",
                           categories=["inlet", "outlet"])
    return ds


def test_a_built_file_is_the_worked_example(tmp_path):
    """Eight calls reach the file docs/example.md lists."""
    path = str(tmp_path / "built.mes")
    mestra.write(build_example(), path)
    assert mestra.validate(path).error_ids == []
    assert corpus.structural_diff(corpus.case_path("mesh_two_rows"),
                                  path) == []


def test_the_builder_fills_in_what_it_can():
    ds = build_example()
    support = ds.supports["s0"]
    assert ds.aligned is True
    assert ds.n_rows == 2
    assert support.n_nodes == 6 and support.n_cells == 2
    assert support.support_id.startswith("96df395d")
    assert support.node_arrays["pressure"].components == 1
    assert support.node_arrays["pressure"].dims == (
        "row", "node", "component")
    assert support.coordinates.dims == (
        "group:member", "node", "component")


def test_bounds_default_to_the_observed_range():
    ds = mestra.Dataset(writer="t")
    key = ds.add_key("alpha", [-2.0, 4.0, 10.0], role="condition",
                     units="degree")
    assert (key.lower, key.upper) == (-2.0, 10.0)
    other = ds.add_key("beta", [1.0, 2.0, 3.0], role="condition",
                       units="1", bounds=None)
    assert (other.lower, other.upper) == (None, None)


def test_bounds_are_the_observed_finite_range():
    """A missing value must not become the domain of validity."""
    ds = mestra.Dataset(writer="t")
    key = ds.add_key("alpha", [-2.0, np.nan, 10.0, np.inf],
                     role="condition", units="degree")
    assert (key.lower, key.upper) == (-2.0, 10.0)


def test_a_bound_the_caller_gave_is_kept_and_the_other_observed():
    ds = mestra.Dataset(writer="t")
    key = ds.add_key("alpha", [1.0, 2.0, 3.0], role="condition",
                     units="degree", lower=0.0)
    assert (key.lower, key.upper) == (0.0, 3.0)


def test_dims_names_the_axes_of_the_array_you_are_passing():
    """P3: `dims` settles what no shape can settle."""
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", np.linspace(0.1, 0.6, 6), role="condition",
               units="1")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    slot = support.add_node_array("p", np.arange(36.).reshape(6, 6),
                                  units="Pa", dims=("row", "node"))
    assert slot.varies == "row"
    assert slot.components == 1
    assert slot.dims == ("row", "node", "component")
    assert slot.values.shape == (6, 6, 1)


def test_dims_may_be_in_the_callers_own_axis_order():
    """The builder stores it in the order section 19 requires."""
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    given = np.arange(12.).reshape(2, 6)
    slot = support.add_node_array("q", given, units="Pa",
                                  dims=("component", "node"))
    assert slot.varies == "none"
    assert slot.components == 2
    assert slot.dims == ("node", "component")
    assert slot.values.at(node=1, component=1) == given[1, 1]


def test_dims_and_varies_that_disagree_are_refused():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("p", np.zeros((2, 6)), units="Pa",
                               dims=("row", "node"), varies="none")
    assert caught.value.rule == "E04"
    assert "varies" in str(caught.value) and "dims" in str(caught.value)


def test_dims_that_names_the_wrong_number_of_axes():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("p", np.zeros((2, 6)), units="Pa",
                               dims=("node",))
    assert caught.value.rule == "E04"


def test_dims_that_names_an_axis_the_format_does_not_have():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("p", np.zeros((2, 6)), units="Pa",
                               dims=("sample", "node"))
    assert caught.value.rule == "E04"
    with pytest.raises(MestraError) as other:
        support.add_node_array("p", np.zeros((2, 6)), units="Pa",
                               dims=("instance", "node"))
    assert "group:" in str(other.value)


def test_a_group_varying_array_from_dims(tmp_path):
    ds = mestra.Dataset(writer="t")
    ds.add_category_table("member", ["wing_a", "wing_b"])
    ds.add_key("member", [0, 1], role="group", category="member")
    ds.set_generalisation_group("member")
    support = ds.add_support(
        "s0", coordinates=np.stack([XY, XY * [1.5, 1.0]]),
        varies="group:member", cells=CELLS)
    slot = support.add_node_array(
        "pressure", np.zeros((6, 2)), units="Pa",
        dims=("node", "group:member"))
    assert slot.varies == "group:member"
    assert slot.dims == ("group:member", "node", "component")
    path = str(tmp_path / "group.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []


def test_a_group_varying_array_of_the_wrong_length():
    ds = mestra.Dataset(writer="t")
    ds.add_category_table("member", ["wing_a", "wing_b"])
    ds.add_key("member", [0, 1], role="group", category="member")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("pressure", np.zeros((3, 6)),
                               units="Pa", varies="group:member")
    assert caught.value.rule == "E34"


def test_a_varies_naming_a_group_the_file_does_not_declare():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("pressure", np.zeros((2, 6)),
                               units="Pa", varies="group:member")
    assert caught.value.rule == "E04"


def test_a_category_table_comes_before_the_key_that_names_it():
    ds = mestra.Dataset(writer="t")
    ds.add_category_table("member", ["wing_a", "wing_b"])
    key = ds.add_key("member", [0, 1], role="group", category="member")
    assert key.category == "member"
    assert ds.categories["member"][1] == "wing_b"


def test_a_key_whose_table_is_not_there_yet():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("member", [0, 1], role="group", category="member")
    assert caught.value.rule == "E39"
    assert "add_category_table" in str(caught.value)


def test_a_category_id_outside_the_table():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("member", [0, 2], role="group",
                   categories=["wing_a", "wing_b"])
    assert caught.value.rule == "E10"


def test_the_unit_of_generalisation_is_a_dataset_property():
    ds = mestra.Dataset(writer="t")
    ds.add_key("member", [0, 1], role="group",
               categories=["wing_a", "wing_b"])
    assert ds.generalisation_group is None
    ds.set_generalisation_group("member")
    assert ds.generalisation_group == "member"
    ds.set_generalisation_group(None)
    assert ds.generalisation_group is None


def test_the_unit_of_generalisation_is_a_group_key():
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    with pytest.raises(MestraError) as caught:
        ds.set_generalisation_group("mach")
    assert caught.value.rule == "E03"
    with pytest.raises(MestraError):
        ds.set_generalisation_group("nothing_of_the_sort")


def test_units_are_required_where_section_3_requires_them():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("mach", [0.4], role="condition")
    assert caught.value.rule == "E39"
    with pytest.raises(MestraError) as other:
        ds.add_scalar("cl", [0.25])
    assert other.value.rule == "E11"
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as third:
        support.add_node_array("p", np.zeros(6))
    assert third.value.rule == "E11"


def test_a_key_that_carries_units_it_should_not():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("member", [0, 1], role="group", units="m",
                   categories=["wing_a", "wing_b"])
    assert caught.value.rule == "E39"


def test_a_second_key_of_a_role_that_allows_one():
    ds = mestra.Dataset(writer="t")
    ds.add_key("t", [0.0, 1.0], role="time", units="s")
    with pytest.raises(MestraError) as caught:
        ds.add_key("t2", [0.0, 1.0], role="time", units="s")
    assert caught.value.rule == "E03"


def test_a_band_states_its_level_and_method():
    """E12 at build time: a stored band carries level in (0, 1) and a
    method; a slot a callable serves is value, mean or band."""
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    ds.add_scalar("cl", [0.25, 0.55], units="1")
    for rule, kwargs in [
            ("E12", {}),
            ("E12", {"level": 1.96, "method": "m"}),
            ("E12", {"level": 0.95}),
            ("E12", {"level": 0.95, "method": "m"})]:
        with pytest.raises(MestraError) as caught:
            ds.add_scalar("cl_band", [0.1, 0.1], units="1",
                          statistic="band", **kwargs)
        assert caught.value.rule == rule
    ds.add_scalar("cl_band", [0.1, 0.1], units="1", statistic="band",
                  of="cl", level=0.95, method="m")
    with pytest.raises(MestraError) as caught:      # level without a band
        ds.add_scalar("cl_mean", [0.1, 0.1], units="1", level=0.95)
    assert caught.value.rule == "E12"
    ds.add_callable("m1", mestra.Affine(
        ["mach"], {"cl": {"A": [[2.0]], "b": [0.05], "shape": []}}))
    with pytest.raises(MestraError) as caught:
        ds.add_callable_slot("cl_draws", units="1", callable="m1",
                             output="cl", statistic="draw")
    assert caught.value.rule == "E12"
    served = ds.add_callable_slot("cl_served_band", units="1",
                                  callable="m1", output="cl",
                                  statistic="band", of="cl")
    assert served.level is None      # written at evaluation, not before


def test_a_derived_array_carries_what_section_3_requires():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("dx", np.zeros(6), role="derived",
                               units="m")
    assert caught.value.rule == "E13"


def test_an_axis_supports_coordinates_do_not_vary():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_support("s0", kind="axis",
                       coordinates=np.zeros((2, 4)), units="s",
                       varies="row")
    assert caught.value.rule == "E35"


def test_a_mesh_support_carries_cells():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_support("s0", kind="mesh", coordinates=XY)
    assert caught.value.rule == "E38"


def test_a_quantile_slot_carries_its_quantile():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("p_q90", np.zeros(6), units="Pa",
                               statistic="quantile", of="p")
    assert caught.value.rule == "E12"


def test_a_callable_slot_names_a_callable_that_is_there(tmp_path):
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [], role="condition", units="1", lower=0.1,
               upper=0.9)
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_callable_slot("pressure", units="Pa",
                                  callable="m1", components=1)
    assert caught.value.rule == "E14"

    model = mestra.Affine(
        ["mach"], {"pressure": {"A": [[1.0]] * 6, "b": [0.0] * 6,
                                "shape": [6, 1]}})
    ds.add_callable("m1", model)
    slot = support.add_callable_slot("pressure", units="Pa",
                                     callable=model, components=1)
    assert slot.source == "callable:m1"
    assert slot.output == "pressure"
    assert slot.varies == "row"
    ds.add_callable_slot("cl", units="1", callable="m1", output="cl")
    assert ds.scalars["cl"].source == "callable:m1"
    path = str(tmp_path / "model.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []


def test_a_callable_slot_declares_its_shape():
    ds = mestra.Dataset(writer="t")
    ds.add_callable("m1", mestra.Affine(["mach"], {}))
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_callable_slot("pressure", units="Pa",
                                  callable="m1")
    assert caught.value.rule == "E31"


def test_a_small_array_prints_its_values():
    """P11: a one-element scalar should print the number."""
    ds = mestra.Dataset(writer="t")
    ds.add_scalar("cl", [1.45], units="1")
    assert "1.45" in repr(ds.scalars["cl"].values)
    assert "dims=row" in repr(ds.scalars["cl"].values)
    big = mestra.NamedArray(np.zeros((6, 6, 1)), ("row", "node",
                                                  "component"))
    assert "6x6x1" in repr(big)


def test_coordinates_are_on_the_support_and_the_error_says_so():
    """P9: the natural first guess gets an answer, not a KeyError."""
    ds = build_example()
    support = ds.supports["s0"]
    with pytest.raises(KeyError) as caught:
        support.node_arrays["coordinates"]
    assert "support.coordinates" in str(caught.value)
    with pytest.raises(KeyError) as other:
        support.node_arrays["pressur"]
    assert "pressure" in str(other.value)


def test_a_scalar_only_file_needs_no_support(tmp_path):
    ds = mestra.Dataset(writer="t")
    ds.add_key("geometry", [0, 0, 1, 1], role="group",
               categories=["g000", "g001"], generalisation=True)
    ds.add_key("incidence", [2.0, 8.0, 2.0, 8.0], role="condition",
               units="degree")
    ds.add_scalar("cl", [0.2, 0.8, 0.3, 0.9], units="1")
    path = str(tmp_path / "scalars.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []


def test_an_axis_support(tmp_path):
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [1.4, 1.6], role="condition", units="1")
    support = ds.add_support("s0", kind="axis",
                             coordinates=[0.0, 0.25, 0.5, 0.75],
                             units="s")
    support.add_node_array("overpressure",
                           [[10., 11., 12., 13.],
                            [20., 21., 22., 23.]], units="Pa")
    path = str(tmp_path / "axis.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []
    with mestra.read(path) as back:
        assert back.supports["s0"].kind == "axis"
        assert back.supports["s0"].n_nodes == 4
        assert back.supports["s0"].coordinates.dims == ("node",
                                                        "component")


def test_a_shape_that_fits_two_readings_is_refused():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("p", np.zeros((6, 6)), units="Pa")
    assert caught.value.rule == "E04"
    assert "varies" in str(caught.value)


def test_a_shape_that_fits_nothing_says_so():
    ds = mestra.Dataset(writer="t")
    support = ds.add_support("s0", coordinates=XY, cells=CELLS)
    with pytest.raises(MestraError) as caught:
        support.add_node_array("p", np.zeros((2, 5)), units="Pa")
    assert caught.value.rule == "E05"


def test_a_role_that_is_not_in_section_3():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("x", [1.0], role="input", units="1")
    assert caught.value.rule == "E02"


def test_a_reserved_name():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("mestra_mach", [1.0], role="condition", units="1")
    assert caught.value.rule == "E33"


def test_a_name_netcdf4_would_refuse():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("mach number", [1.0], role="condition", units="1")
    assert caught.value.rule == "E33"


@pytest.mark.parametrize("name", [".", ".mach", "-mach", "+mach"])
def test_a_name_must_begin_with_a_letter_a_digit_or_an_underscore(name):
    """Section 18. "." was a legal name, and h5py reads it as the group
    itself."""
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key(name, [1.0], role="condition", units="1")
    assert caught.value.rule == "E33"
    with pytest.raises(MestraError) as caught:
        ds.add_category_table(name, ["a"])
    assert caught.value.rule == "E33"


@pytest.mark.parametrize("check", [True, False])
def test_the_writer_refuses_a_name_hdf5_cannot_hold(tmp_path, check):
    """A key renamed to "." after it was built reached h5py, which
    failed halfway through the file. The writer refuses it first, even
    when told not to check, and leaves nothing behind."""
    ds = mestra.Dataset(writer="t", created="2026-01-01T00:00:00Z")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    ds.add_scalar("cl", [0.1, 0.2], units="1")
    key = ds.keys.pop("mach")
    key.name = "."
    ds.keys["."] = key
    out = tmp_path / "out.mes"
    with pytest.raises(MestraError) as caught:
        mestra.write(ds, str(out), check=check)
    assert caught.value.rule == "E33"
    assert os.listdir(tmp_path) == []


def test_rows_must_agree():
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    with pytest.raises(MestraError) as caught:
        ds.add_scalar("cl", [0.1, 0.2, 0.3], units="1")
    assert caught.value.rule == "E16"


def test_a_categorical_key_is_an_integer():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("split", [0.0, 1.0], role="split",
                   categories=["train", "test"])
    assert caught.value.rule == "E20"


def test_an_id_key_is_integers_or_text():
    # A float id was truncated to an integer in silence; the Julia
    # builder refuses it, and so does this one.
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("run", [1.0, 2.5], role="id")
    assert caught.value.rule == "E20"
    assert "row identifiers" in caught.value.message
    ds.add_key("run", [1, 2], role="id")
    other = mestra.Dataset(writer="t")
    other.add_key("case", ["a", "b"], role="id")


def test_two_supports_need_a_row_support(tmp_path):
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4, 0.5, 0.6], role="condition", units="1")
    ds.add_support("s0", coordinates=XY, cells=CELLS)
    ds.add_support("s1", coordinates=XY[:4],
                   cells=(np.array([9]), np.array([0, 4]),
                          np.array([0, 1, 3, 2])))
    assert ds.aligned is False
    report = mestra.validate(ds)
    assert "E28" in report.error_ids
    ds.set_row_support([0, 1, 0])
    assert mestra.validate(ds).error_ids == []
    assert list(ds.rows_on("s0")) == [0, 2]
    path = str(tmp_path / "two.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []
    assert mestra.validate(path).warning_ids == ["W05"]


def test_a_row_varying_array_in_an_unaligned_file(tmp_path):
    """Section 22: one entry per row that references this support."""
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4, 0.5, 0.6], role="condition", units="1")
    first = ds.add_support("s0", coordinates=XY, cells=CELLS)
    second = ds.add_support("s1", coordinates=XY[:4],
                            cells=(np.array([9]), np.array([0, 4]),
                                   np.array([0, 1, 3, 2])))
    ds.set_row_support([0, 1, 0])
    first.add_node_array("p", np.zeros((2, 6)), units="Pa",
                         dims=("row", "node"))
    second.add_node_array("p", np.zeros((1, 4)), units="Pa",
                          dims=("row", "node"))
    with pytest.raises(MestraError) as caught:
        first.add_node_array("q", np.zeros((3, 6)), units="Pa",
                             dims=("row", "node"))
    assert caught.value.rule == "E16"
    assert "section 22" in str(caught.value)
    # A refused call leaves the support as it found it.
    assert "q" not in first.node_arrays
    path = str(tmp_path / "two.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []


def test_a_dataset_validates_in_memory():
    ds = build_example()
    assert mestra.validate(ds).ok
    ds.scalars["cl"].units = None
    assert "E11" in mestra.validate(ds).error_ids


def test_named_arrays_permute_by_name():
    ds = build_example()
    pressure = ds.supports["s0"].node_arrays["pressure"].values
    assert pressure.at(row=1, node=3, component=0) == 204.0
    other = pressure.transpose("component", "node", "row")
    assert other.shape == (1, 6, 2)
    assert other.at(row=1, node=3, component=0) == 204.0
    with pytest.raises(KeyError):
        pressure.at(cell=0)


def test_the_support_id_is_the_worked_example():
    """Section 24: six nodes and two quadrilaterals."""
    assert mestra.support_digest(6, [9, 9], [0, 4, 8],
                                 [0, 1, 4, 3, 1, 2, 5, 4]) == (
        "96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c943936c7")
    assert mestra.support_digest(
        4, axis_coordinates=[0.0, 0.5, 1.0, 1.5]) == (
        "57467fe7370808bdb0ad01b95d963f59e8bc6762f90f96049453ae33bb05a54c")
    assert mestra.support_digest(0) == (
        "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc")


def test_a_callable_may_serve_a_mesh_supports_coordinates(tmp_path):
    """Section 10: a model of the geometry itself. The support is
    added with its node count and no coordinates, and the coordinates
    builder with the values dropped names the callable."""
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [], role="condition", units="1", lower=0.1,
               upper=0.9)
    with pytest.raises(MestraError) as caught:
        ds.add_support("s0", cells=CELLS)
    assert caught.value.rule == "E03"
    assert "set_callable_coordinates" in caught.value.message
    support = ds.add_support("s0", cells=CELLS, n_nodes=6)
    # x = x0 (1 + mach): the mesh stretches along x with mach.
    stretch = np.zeros((12, 1))
    stretch[0::2, 0] = XY[:, 0]
    model = mestra.Affine(["mach"], {
        "coordinates": {"A": stretch, "b": XY.ravel(), "shape": [6, 2]}})
    with pytest.raises(MestraError) as caught:
        support.set_callable_coordinates(units="m", callable="m1",
                                         components=2)
    assert caught.value.rule == "E14"
    ds.add_callable("m1", model)
    with pytest.raises(MestraError) as caught:
        support.set_callable_coordinates(units="m", callable="m1")
    assert caught.value.rule == "E31"
    slot = support.set_callable_coordinates(units="m", callable=model,
                                            components=2)
    assert slot.source == "callable:m1"
    assert slot.output == "coordinates"
    assert slot.varies == "row"
    assert slot.role == "coordinates"
    with pytest.raises(MestraError) as caught:
        support.set_callable_coordinates(units="m", callable="m1",
                                         components=2)
    assert caught.value.rule == "E03"

    path = str(tmp_path / "geometry.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []
    with mestra.read(path) as back:
        served = back.supports["s0"].coordinates
        assert served.is_callable and served.data is None
        assert served.components == 2
        with pytest.raises(MestraError) as caught:
            mestra.compute_weights(back.supports["s0"], "cell")
        assert "evaluate the file first" in caught.value.message
        out = mestra.evaluate(back, {"mach": np.array([0.5])})
    got = out.supports["s0"].coordinates
    assert not got.is_callable
    assert got.dims == ("row", "node", "component")
    want = XY.copy()
    want[:, 0] *= 1.5
    np.testing.assert_array_equal(np.asarray(got.read().values)[0], want)


def test_an_axis_supports_coordinates_are_never_served():
    ds = mestra.Dataset(writer="t")
    ds.add_callable("m1", mestra.Affine(["mach"], {}))
    with pytest.raises(MestraError) as caught:
        ds.add_support("t", kind="axis", n_nodes=3, units="s")
    assert caught.value.rule == "E03"
    axis = ds.add_support("t", coordinates=[0.0, 1.0, 2.0], units="s")
    with pytest.raises(MestraError) as caught:
        axis.set_callable_coordinates(units="s", callable="m1",
                                      components=1)
    assert caught.value.rule == "E03"
    assert "identity" in caught.value.message or "stored" in caught.value.message


# ------------------------------------------- data that cannot be written

def test_a_ragged_array_is_refused_by_name():
    """numpy's refusal names no argument; the builder's names the
    slot and says what shape to pass (E20)."""
    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    with pytest.raises(MestraError) as caught:
        ds.add_scalar("cl", [[0.1, 0.2], [0.3]], units="1")
    assert caught.value.rule == "E20"
    assert caught.value.where == "cl"
    assert "rectangular" in caught.value.message
    axis = ds.add_support("t", coordinates=[0.0, 1.0, 2.0], units="s")
    with pytest.raises(MestraError) as caught:
        axis.add_node_array("p", [[[1.0], [2.0], [3.0]], [[1.0], [2.0]]],
                            units="Pa", dims=("row", "node", "component"))
    assert caught.value.rule == "E20"
    assert caught.value.where == "p"
    with pytest.raises(MestraError) as caught:
        ds.add_support("u", coordinates=[[0.0], [1.0, 2.0]], units="s")
    assert caught.value.rule == "E20"
    assert caught.value.where == "u/coordinates"


def test_text_where_a_number_belongs_is_refused_by_name():
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("mach", ["fast", "faster"], role="condition", units="1")
    assert caught.value.rule == "E20"
    assert caught.value.where == "mach"


def test_a_bound_that_is_not_finite_is_refused(tmp_path):
    """Section 18: a bound is finite. At build time by the builder,
    and at write time when it was assigned afterwards, so that the
    writer never emits what its validator would refuse (E19)."""
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("mach", [0.4, 0.8], role="condition", units="1",
                   lower=float("nan"), upper=1.0)
    assert caught.value.rule == "E19"
    assert caught.value.where == "mach"
    assert "lower" in caught.value.message
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    ds.keys["mach"].upper = float("inf")
    path = str(tmp_path / "bound.mes")
    with pytest.raises(MestraError) as caught:
        mestra.write(ds, path)
    assert caught.value.rule == "E19"
    assert "/keys/mach" in str(caught.value)
    assert not os.path.exists(path)


def test_units_that_are_not_text_are_refused(tmp_path):
    ds = mestra.Dataset(writer="t")
    with pytest.raises(MestraError) as caught:
        ds.add_key("mach", [0.4, 0.8], role="condition", units=1)
    assert caught.value.rule == "E19"
    assert caught.value.where == "mach"
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    with pytest.raises(MestraError) as caught:
        ds.add_scalar("cl", [0.1, 0.2], units=1)
    assert caught.value.rule == "E19"
    ds.add_scalar("cl", [0.1, 0.2], units="1")
    ds.scalars["cl"].units = 1
    path = str(tmp_path / "units.mes")
    with pytest.raises(MestraError) as caught:
        mestra.write(ds, path)
    assert caught.value.rule == "E19"
    assert "/scalars/cl" in str(caught.value)
    assert not os.path.exists(path)


def test_a_name_ending_in_a_newline_is_refused():
    """A legal name is built from letters, digits, underscore, hyphen,
    "." and "+" (section 18). The check matched with `$`, which also
    matches before a final newline, so "mach\\n" was accepted by the
    builder and by the validator."""
    ds = mestra.Dataset(writer="test", created="2026-10-01T00:00:00Z")
    with pytest.raises(mestra.MestraError) as caught:
        ds.add_key("mach\n", [0.4], role="condition", units="1")
    assert caught.value.rule == "E33"
