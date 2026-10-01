"""The callable protocol, the registry, and the dictionary codec."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import h5py
import numpy as np
import pytest

import mestra
from mestra.codec import decode_dict, encode_dict
from mestra.errors import MestraError
from tests import corpus


def round_trip(tmp_path, dictionary):
    """Write a dictionary and read it back (sections 17 and 25)."""
    path = str(tmp_path / "codec.h5")
    with h5py.File(path, "w") as f:
        encode_dict(f.create_group("m"), dictionary)
    with h5py.File(path, "r") as f:
        return decode_dict(f["m"])


def test_a_dictionary_of_everything(tmp_path):
    original = {
        "an_int": 3,
        "a_float": 3.0,
        "a_bool": True,
        "a_string": "café",
        "nothing": None,
        "numbers": [1.0, 2.0, 3.0],
        "names": ["mach", "alpha"],
        "matrix": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "counts": np.array([1, 2, 3], dtype="int32"),
        "flags": np.array([True, False]),
        "nested": {"type": "this is allowed below the top level",
                   "repr": "so is this", "deeper": {"x": 1}},
    }
    back = round_trip(tmp_path, original)
    assert back["an_int"] == 3 and isinstance(back["an_int"], int)
    assert back["a_float"] == 3.0 and isinstance(back["a_float"], float)
    assert back["a_bool"] is True
    assert back["a_string"] == "café"
    assert back["nothing"] is None
    assert back["numbers"].tolist() == [1.0, 2.0, 3.0]
    assert back["names"] == ["mach", "alpha"]
    assert back["matrix"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert back["counts"].dtype == np.int32
    assert back["flags"].dtype == np.bool_
    assert back["nested"]["deeper"]["x"] == 1
    assert back["nested"]["type"].startswith("this is allowed")


def test_an_integer_and_a_float_stay_apart(tmp_path):
    back = round_trip(tmp_path, {"a": 2, "b": 2.0})
    assert isinstance(back["a"], int)
    assert isinstance(back["b"], float)


def test_a_zero_dimensional_array_becomes_a_number(tmp_path):
    """Section 25: it must not be written as a dataset."""
    back = round_trip(tmp_path, {"tolerance": np.float64(1e-9)})
    assert back["tolerance"] == 1e-9
    assert not isinstance(back["tolerance"], np.ndarray)


def test_an_empty_array_keeps_its_dtype(tmp_path):
    back = round_trip(tmp_path, {
        "floats": np.zeros((0,), dtype="float64"),
        "integers": np.zeros((0,), dtype="int64"),
        "unknown": []})
    assert back["floats"].dtype == np.float64
    assert back["integers"].dtype == np.int64
    assert back["unknown"].dtype == np.float64
    assert back["unknown"].shape == (0,)


def test_an_empty_axis_is_unlimited(tmp_path):
    """Section 25, so that the dimension is legal in netCDF-4."""
    path = str(tmp_path / "codec.h5")
    with h5py.File(path, "w") as f:
        encode_dict(f.create_group("m"),
                    {"shape": np.zeros((0,), dtype="int64")})
    with h5py.File(path, "r") as f:
        assert f["m/shape"].maxshape == (None,)
        assert f["m/mestra_shape_d0"].maxshape == (None,)


@pytest.mark.parametrize("value, rule", [
    ([[1.0], [2.0, 3.0]], "E32"),
    ([1, "two"], "E32"),
    ([{"a": 1}], "E32"),
    ([None, None], "E32"),
    (np.zeros(3, dtype="float32"), "E32"),
    (np.zeros(3, dtype="uint16"), "E32"),
    ("a string with a \x00 in it", "E32"),
])
def test_what_the_writer_must_refuse(tmp_path, value, rule):
    with pytest.raises(MestraError) as caught:
        round_trip(tmp_path, {"x": value})
    assert caught.value.rule == rule


@pytest.mark.parametrize("name", ["type", "repr"])
def test_type_and_repr_are_taken_at_the_top_level(tmp_path, name):
    with pytest.raises(MestraError) as caught:
        round_trip(tmp_path, {name: "x"})
    assert caught.value.rule == "E32"


def test_the_reserved_prefix_is_refused(tmp_path):
    with pytest.raises(MestraError) as caught:
        round_trip(tmp_path, {"mestra_x": 1})
    assert caught.value.rule == "E33"


def test_a_keys_table_the_caller_got_wrong_carries_no_rule_id():
    """P4: a rule identifier is a finding about a file."""
    model = mestra.Affine(
        ["mach", "alpha"],
        {"cl": {"A": [[2.0, 0.1]], "b": [0.05], "shape": []}})
    with pytest.raises(MestraError) as caught:
        model({"mach": [0.5]})
    assert caught.value.rule == ""
    assert "alpha" in str(caught.value)
    with pytest.raises(MestraError) as other:
        mestra.keys_table({"mach": [0.5], "alpha": [1.0, 2.0]})
    assert other.value.rule == ""


# ------------------------------------------------------------- affine

def test_the_worked_example_of_section_27():
    model = mestra.Affine(
        ["mach", "alpha"],
        {"cl": {"A": [[2.0, 0.1]], "b": [0.05], "shape": []},
         "pressure": {"A": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.5],
                            [4.0, 0.5], [5.0, 1.0], [6.0, 1.0]],
                      "b": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                      "shape": [6, 1]}})
    out = model({"mach": [0.5], "alpha": [4.0]})
    assert corpus.fnum(out["cl"].mean[0]) == "1.44999999999999996e+00"
    assert out["pressure"].mean.shape == (1, 6, 1)
    assert out["pressure"].mean[0, :, 0].tolist() == [0.5, 1.1, 3.7, 4.3,
                                                      6.9, 7.5]
    assert not out["cl"].has_uncertainty
    assert out["pressure"].uncertainty is None


def test_a_prediction_is_a_mean_and_at_most_a_band():
    """Section 10: the record, and what it refuses."""
    mean = np.array([1.0, 2.0])
    plain = mestra.Prediction(mean)
    assert plain.uncertainty is None and plain.level is None
    banded = mestra.Prediction(mean, [0.1, 0.2], level=0.95,
                               method="constant band")
    assert banded.uncertainty.tolist() == [0.1, 0.2]
    assert banded.level == 0.95
    with pytest.raises(MestraError):            # a band needs a level
        mestra.Prediction(mean, [0.1, 0.2], method="m")
    with pytest.raises(MestraError):            # in (0, 1), not sigma
        mestra.Prediction(mean, [0.1, 0.2], level=1.96, method="m")
    with pytest.raises(MestraError):            # and a method
        mestra.Prediction(mean, [0.1, 0.2], level=0.95)
    with pytest.raises(MestraError):            # of the mean's shape
        mestra.Prediction(mean, [0.1], level=0.95, method="m")
    with pytest.raises(MestraError):            # never negative
        mestra.Prediction(mean, [-0.1, 0.2], level=0.95, method="m")
    with pytest.raises(MestraError):            # level without a band
        mestra.Prediction(mean, level=0.95)


def test_an_affine_output_may_carry_a_constant_band():
    """Section 27: uncertainty, level and method, all three or none,
    and the band is the same in every row."""
    model = mestra.Affine(
        ["mach"],
        {"cl": {"A": [[2.0]], "b": [0.05], "shape": [],
                "uncertainty": [0.02], "level": 0.95,
                "method": "constant band"},
         "p": {"A": [[1.0], [2.0]], "b": [0.0, 0.1], "shape": [2, 1],
               "uncertainty": [0.5, 0.6], "level": 0.68,
               "method": "constant band"}})
    out = model({"mach": [0.5, 0.7]})
    assert out["cl"].uncertainty.tolist() == [0.02, 0.02]
    assert out["cl"].level == 0.95 and out["cl"].method == "constant band"
    assert out["p"].uncertainty.shape == (2, 2, 1)
    assert out["p"].uncertainty[1, :, 0].tolist() == [0.5, 0.6]
    assert out["p"].level == 0.68
    back = mestra.Affine.from_dict(model.to_dict())
    assert back.to_dict()["outputs"]["cl"]["level"] == 0.95
    assert back.to_dict()["outputs"]["p"]["method"] == "constant band"
    with pytest.raises(MestraError):            # one of three is not
        mestra.Affine(["mach"], {"cl": {"A": [[2.0]], "b": [0.05],
                                        "shape": [], "level": 0.95}})
    with pytest.raises(MestraError):            # the band's length
        mestra.Affine(["mach"], {"cl": {"A": [[2.0]], "b": [0.05],
                                        "shape": [], "uncertainty": [1, 2],
                                        "level": 0.95, "method": "m"}})


def test_a_two_dimensional_keys_table():
    """Section 26: rows by keys, with the names alongside."""
    model = mestra.Affine(["mach", "alpha"],
                          {"cl": {"A": [[2.0, 0.1]], "b": [0.05],
                                  "shape": []}})
    table = mestra.keys_table(np.array([[0.5, 4.0]]),
                              ["mach", "alpha"])
    assert corpus.fnum(model(table)["cl"].mean[0]) == \
        "1.44999999999999996e+00"


def test_a_missing_column_is_an_error():
    model = mestra.Affine(["mach"], {"cl": {"A": [[2.0]], "b": [0.05],
                                            "shape": []}})
    with pytest.raises(MestraError):
        model({"alpha": [1.0]})


def test_columns_of_different_lengths_are_an_error():
    with pytest.raises(MestraError):
        mestra.keys_table({"a": [1.0, 2.0], "b": [1.0]})


def test_an_affine_dictionary_holds_nothing_else():
    with pytest.raises(MestraError):
        mestra.Affine.from_dict({"keys": ["m"], "outputs": {},
                                 "extra": 1})


def test_a_shape_that_does_not_match_a_is_refused():
    with pytest.raises(MestraError):
        mestra.Affine(["mach"], {"cl": {"A": [[1.0, 2.0]], "b": [0.0],
                                        "shape": []}})


# ----------------------------------------------------------- registry

class Constant(mestra.Callable):
    """A callable that answers the same value for every row."""

    type: ClassVar[str] = "example.constant"

    def __init__(self, value: float, width: int) -> None:
        self.value = float(value)
        self.width = int(width)

    def __call__(self, keys: Any) -> dict[str, mestra.Prediction]:
        rows = mestra.keys_table(keys)
        length = len(next(iter(rows.values())))
        return {"pressure": mestra.Prediction(
            np.full((length, self.width, 1), self.value))}

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "width": self.width}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Constant:
        return cls(d["value"], d["width"])

    def __repr__(self) -> str:
        return "constant(%g on %d nodes)" % (self.value, self.width)


def test_adding_a_type_is_a_subclass_and_one_call(tmp_path):
    mestra.register_callable(Constant)
    assert mestra.callable_types()["example.constant"] is Constant

    ds = mestra.Dataset(writer="t")
    ds.add_key("mach", [], role="condition", units="1", lower=0.1,
               upper=0.9)
    ds.add_callable("m1", Constant(3.5, 6))
    support = ds.add_support(
        "s0", coordinates=np.zeros((6, 2)),
        cells=(np.array([9, 9]), np.array([0, 4, 8]),
               np.array([0, 1, 4, 3, 1, 2, 5, 4])))
    support.add_node_array("pressure", callable_id="m1",
                           output="pressure", components=1, units="Pa")
    path = str(tmp_path / "constant.mes")
    mestra.write(ds, path)
    assert mestra.validate(path).error_ids == []

    with mestra.read(path) as back:
        model = back.callables["m1"]
        assert isinstance(model, Constant)
        assert repr(model) == "constant(3.5 on 6 nodes)"
        out = mestra.evaluate(back, {"mach": [0.4, 0.8]})
    values = out.supports["s0"].node_arrays["pressure"]
    assert values.source == "data"
    assert values.read().at(row=1, node=2, component=0) == 3.5


def test_an_unknown_type_is_copied_and_not_interpreted(tmp_path):
    """Section 12: a reader may copy what it does not own."""
    source = corpus.case_path("affine_zero_rows")
    with mestra.read(source) as ds:
        held = ds.callables["m1"].to_dict()
    kinds = mestra.callable_types()
    try:
        del mestra.callables._REGISTRY["affine"]
        with mestra.read(source) as ds:
            model = ds.callables["m1"]
            assert isinstance(model, mestra.OpaqueCallable)
            assert model.type == "affine"
            assert sorted(model.to_dict()) == sorted(held)
            with pytest.raises(MestraError):
                model({"mach": [0.5], "alpha": [4.0]})
            path = str(tmp_path / "copied.mes")
            mestra.write(ds, path)
        assert corpus.structural_diff(source, path) == []
    finally:
        mestra.callables._REGISTRY.update(kinds)


def test_a_callable_class_needs_a_type():
    class Nameless(mestra.Callable):
        def __call__(self, keys):
            return {}

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, d):
            return cls()

    with pytest.raises(MestraError):
        mestra.register_callable(Nameless)


def test_an_affine_of_no_keys_writes_its_keys_as_strings(tmp_path):
    """Section 27 makes `keys` a fixed-length string dataset. With no
    keys the list was empty, and section 25 writes an empty list whose
    element type nobody knows as float64, so a constant was written
    with float64 keys and read back in C++, Julia and MATLAB as a
    number array where the type says strings."""
    ds = mestra.Dataset(writer="test", created="2026-10-01T00:00:00Z")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    ds.add_callable("m1", mestra.Affine(
        [], {"cl": {"A": np.zeros((1, 0)), "b": [0.05], "shape": []}}))
    ds.add_callable_slot("cl", units="1", callable="m1")
    path = str(tmp_path / "constant.mes")
    mestra.write(ds, path)
    with h5py.File(path, "r") as f:
        keys = f["/callables/m1/keys"]
        assert keys.shape == (0,)
        assert h5py.check_string_dtype(keys.dtype) is not None
    with mestra.read(path) as back:
        assert back.callables["m1"].keys == []
        out = back.callables["m1"]({"mach": [0.4]})
        assert out["cl"].mean.tolist() == [0.05]
