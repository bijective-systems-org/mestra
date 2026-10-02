"""Malformed but openable: the rule each fault is named by.

Every file here is a corpus case with one thing changed, made on
demand with h5py so that the change is stated next to the rule it
draws. The point is agreement: the same file was put through the
Python, C++, MATLAB and Julia validators, and each test pins the
identifiers the four settled on, with the rule text that decides it.
A reader that named a different rule for one of these, or none, would
be found here before a user found it.

The files that must also refuse a strict read are the ones that break
a structural rule (section 2 of docs/api-conventions.md); a semantic
fault leaves the file readable, so that `info` works on it.
"""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

import mestra
from tests import corpus


def _case(tmp_path, base: str, name: str) -> str:
    path = str(tmp_path / (name + ".mes"))
    shutil.copy(corpus.case_path(base), path)
    return path


def _fixed(text: str):
    body = text.encode("utf-8")
    return body, h5py.string_dtype(encoding="utf-8", length=len(body))


def _refuses(path: str, rule: str) -> None:
    with pytest.raises(mestra.MestraError) as caught:
        mestra.read(path)
    assert caught.value.rule == rule


def _reads(path: str) -> None:
    with mestra.read(path) as ds:
        assert ds.n_rows >= 0


def test_a_key_of_the_wrong_dtype(tmp_path):
    """Section 19: a condition key is float64, and float32 is E20
    anywhere. A semantic rule, so the file still reads."""
    path = _case(tmp_path, "mesh_two_rows", "key_wrong_dtype")
    with h5py.File(path, "r+") as f:
        key = f["/keys/mach"]
        attrs = {k: (key.attrs[k], h5py.h5a.open(key.id, k.encode()).dtype)
                 for k in key.attrs if k != "DIMENSION_LIST"}
        values = key[...]
        del f["/keys/mach"]
        fresh = f.create_dataset("/keys/mach", data=values.astype("<f4"),
                                 chunks=key.chunks, maxshape=(None,),
                                 track_times=False)
        for name, (value, dtype) in attrs.items():
            fresh.attrs.create(name, value, dtype=dtype)
        fresh.dims[0].attach_scale(f["/row"])
    report = mestra.validate(path)
    assert report.error_ids == ["E20"]
    assert [f.where for f in report.errors] == ["/keys/mach"]
    _reads(path)


def test_a_support_declaring_no_nodes(tmp_path):
    """E05 on every node-indexed array against the declaration, E24
    on the connectivity that now points past it, E08 because the
    node count is hashed. Nothing structural; the file reads."""
    path = _case(tmp_path, "mesh_two_rows", "support_zero_nodes")
    with h5py.File(path, "r+") as f:
        f["/supports/s0"].attrs.modify("n_nodes", np.int64(0))
    report = mestra.validate(path)
    assert report.error_ids == ["E05", "E08", "E24"]
    assert sorted(f.where for f in report.errors if f.rule == "E05") == [
        "/supports/s0/coordinates", "/supports/s0/node_arrays/pressure"]
    _reads(path)


def test_a_mesh_declaring_more_cells_than_it_has(tmp_path):
    """The cell count a mesh declares is the length of its cell_types;
    with the cell array gone nothing else would hold it, so E05 is
    reported on cell_types itself."""
    path = _case(tmp_path, "mesh_two_rows", "n_cells_declared")
    with h5py.File(path, "r+") as f:
        del f["/supports/s0/cell_arrays/region"]
        f["/supports/s0"].attrs.modify("n_cells", np.int64(3))
    report = mestra.validate(path)
    assert report.error_ids == ["E05"]
    assert [f.where for f in report.errors] == ["/supports/s0/cell_types"]
    assert "declares 3 cells" in report.errors[0].message


def test_a_negative_cell_count(tmp_path):
    """A count below zero disagrees with every cell-indexed dataset,
    and each says so once (E05)."""
    path = _case(tmp_path, "mesh_two_rows", "negative_count")
    with h5py.File(path, "r+") as f:
        f["/supports/s0"].attrs.modify("n_cells", np.int64(-1))
    report = mestra.validate(path)
    assert report.error_ids == ["E05"]
    assert sorted(f.where for f in report.errors) == [
        "/supports/s0/cell_arrays/region", "/supports/s0/cell_types"]
    _reads(path)


def test_a_slot_axis_carrying_a_strangers_scale(tmp_path):
    """Section 21 puts `node` on the node axis of a node array. A
    scale of another name there is E25 on the slot, and not on the
    scale dataset, which section 14 exempts from the rule; the scale
    is also E42 for the properties it was made with. A strict read
    refuses (E25)."""
    path = _case(tmp_path, "mesh_two_rows", "slot_axis_unknown")
    with h5py.File(path, "r+") as f:
        stranger = f.create_dataset("/supports/s0/elsewhere",
                                    data=np.zeros(6, ">f4"),
                                    track_times=False)
        stranger.make_scale("elsewhere")
        slot = f["/supports/s0/node_arrays/pressure"]
        slot.dims[1].detach_scale(f["/supports/s0/node"])
        slot.dims[1].attach_scale(stranger)
    report = mestra.validate(path)
    assert report.error_ids == ["E25", "E42"]
    assert [f.where for f in report.errors if f.rule == "E25"] == [
        "/supports/s0/node_arrays/pressure"]
    _refuses(path, "E25")


def test_units_that_are_not_a_string(tmp_path):
    """E19 on the attribute, and not W10 as well: a units that is not
    a string is not a string that does not parse. A strict read
    refuses (E19)."""
    path = _case(tmp_path, "mesh_two_rows", "units_not_string")
    with h5py.File(path, "r+") as f:
        slot = f["/supports/s0/node_arrays/pressure"]
        del slot.attrs["units"]
        slot.attrs.create("units", np.float64(1.0))
    report = mestra.validate(path)
    assert report.error_ids == ["E19"]
    assert report.warning_ids == []
    assert [f.where for f in report.errors] == [
        "/supports/s0/node_arrays/pressure"]
    _refuses(path, "E19")


def test_an_affine_with_no_keys(tmp_path):
    """Section 27 says a reader refuses an affine dictionary that is
    not exactly the one it describes. The validator does not
    interpret a dictionary (section 25), so the file validates. The
    reader refuses to build the affine, says what is missing in the
    dataset's problems, and keeps the dictionary opaque, so the file
    still round-trips and only evaluation is refused. A KeyError
    from inside the constructor used to escape that arrangement."""
    path = _case(tmp_path, "callable_two_slots", "callable_missing_field")
    with h5py.File(path, "r+") as f:
        del f["/callables/m2/keys"]
        del f["/callables/m2/mestra_keys_d0"]
    assert mestra.validate(path).ok
    with mestra.read(path) as ds:
        kept = ds.callables["m2"]
        assert isinstance(kept, mestra.OpaqueCallable)
        assert any("no keys" in p.message for p in ds.problems)
        with pytest.raises(mestra.MestraError):
            mestra.evaluate(ds, {"mach": [0.5]})
    with pytest.raises(mestra.MestraError) as caught:
        mestra.Affine.from_dict({"outputs": {}})
    assert caught.value.rule == "section 27"
    assert "no keys" in caught.value.message


def test_an_affine_output_with_no_offset():
    """The same rule one level down: an output entry holds A, b and
    shape, and the constructor says which is missing."""
    with pytest.raises(mestra.MestraError) as caught:
        mestra.Affine(["mach"], {"cl": {"A": [[1.0]], "shape": []}})
    assert caught.value.rule == "section 27"
    assert "no b" in caught.value.message
    assert caught.value.where == "cl"


def test_duplicated_entries_in_a_category_table(tmp_path):
    """No rule of section 14 names two entries with one text, and no
    implementation invents one: the table is accepted, with the W13
    the shortened text now draws. Pinned so that a reader does not
    quietly start refusing what the others accept."""
    path = _case(tmp_path, "mesh_two_rows",
                 "duplicate_category_entries")
    with h5py.File(path, "r+") as f:
        table = f["/categories/region"]
        entries = table[...]
        entries[1] = entries[0]
        table[...] = entries
    report = mestra.validate(path)
    assert report.error_ids == []
    assert report.warning_ids == ["W13"]
    with mestra.read(path) as ds:
        assert list(ds.categories["region"]) == ["inlet", "inlet"]


def test_an_attribute_where_a_dataset_belongs(tmp_path):
    """cell_types as an attribute on the support: the dataset is
    missing (E38), the digest no longer matches (E08), and the
    attribute is one this version does not know (W11). Section 18
    constrains the attributes it names, so the array-valued stranger
    is not E19."""
    path = _case(tmp_path, "mesh_two_rows", "attribute_for_dataset")
    with h5py.File(path, "r+") as f:
        types = f["/supports/s0/cell_types"][...]
        del f["/supports/s0/cell_types"]
        f["/supports/s0"].attrs.create("cell_types", types)
    report = mestra.validate(path)
    assert report.error_ids == ["E08", "E38"]
    assert report.warning_ids == ["W11"]
    _reads(path)


def test_a_key_one_group_too_deep(tmp_path):
    """A group under /keys is not a key column and cannot be read as
    one: E41 with its path, and not a missing attribute (E39), a slot
    stored wrongly (E30) or a callable without a type (E15). A strict
    read refuses (E41)."""
    path = _case(tmp_path, "mesh_two_rows", "group_too_deep")
    with h5py.File(path, "r+") as f:
        f["/keys"].create_group("nested")
        f.move("/keys/mach", "/keys/nested/mach")
    report = mestra.validate(path)
    assert report.error_ids == ["E41"]
    assert [f.where for f in report.errors] == ["/keys/nested"]
    _refuses(path, "E41")


def test_a_future_major_version(tmp_path):
    """Section 28: E01 and nothing else, because a reader must not
    read such a file even partially."""
    path = _case(tmp_path, "mesh_two_rows", "future_major")
    with h5py.File(path, "r+") as f:
        del f.attrs["format"]
        body, dtype = _fixed("mestra/1")
        f.attrs.create("format", body, dtype=dtype)
    report = mestra.validate(path)
    assert report.error_ids == ["E01"]
    assert report.warning_ids == []
    _refuses(path, "E01")


def test_a_bound_that_is_not_finite(tmp_path):
    """Section 18: an attribute that declares a bound, a level or a
    quantile must be finite (E19). A file with a NaN lower bound was
    accepted outright by two of the four validators, because the
    check lived with the bounds rules of one and the encoding rules
    of another; it is E19 in all four now, and a level of inf draws
    E12 for its range as well."""
    path = _case(tmp_path, "mesh_two_rows", "nan_bound")
    with h5py.File(path, "r+") as f:
        f["/keys/mach"].attrs.modify("lower", np.float64("nan"))
    report = mestra.validate(path)
    assert report.error_ids == ["E19"]
    assert [f.where for f in report.errors] == ["/keys/mach"]
    _refuses(path, "E19")
    path = _case(tmp_path, "band_stored", "inf_level")
    with h5py.File(path, "r+") as f:
        f["/supports/s0/node_arrays/pressure_band"].attrs.modify(
            "level", np.float64("inf"))
    report = mestra.validate(path)
    assert report.error_ids == ["E12", "E19"]


def test_a_big_endian_array_in_a_dictionary(tmp_path):
    """Section 25 stores a numeric array in a dictionary
    little-endian. A big-endian float64 has the name float64 in numpy,
    so a check of the dtype's name alone let it through; it is E32,
    and the file still reads, since E32 is not structural."""
    path = _case(tmp_path, "callable_two_slots", "dictionary_big_endian")
    with h5py.File(path, "r+") as f:
        group = f["/callables/m2"]
        data = group.create_dataset("weights", data=[0.5, 0.25],
                                    dtype=">f8", track_times=False)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_attr_creation_order(h5py.h5p.CRT_ORDER_TRACKED |
                                     h5py.h5p.CRT_ORDER_INDEXED)
        dcpl.set_obj_track_times(False)
        dcpl.set_chunk((2,))
        dim = h5py.Dataset(h5py.h5d.create(
            group.id, b"mestra_weights_d0", h5py.h5t.IEEE_F32BE,
            h5py.h5s.create_simple((2,), (2,)), dcpl=dcpl))
        dim.make_scale("This is a netCDF dimension but not a netCDF "
                       "variable.         2")
        data.dims[0].attach_scale(dim)
    report = mestra.validate(path)
    assert report.error_ids == ["E32"]
    assert [f.where for f in report.errors] == ["/callables/m2/weights"]


@pytest.mark.parametrize("name", ["err_e16_row_support",
                                  "err_e16_support_rows"])
def test_a_row_count_disagreeing_with_row_support_refuses_a_read(name):
    """E16 is structural, so a strict read refuses it. In an unaligned
    file it is decided from /row_support, which a metadata open reads
    in full (conventions section 7); the open used to leave it to the
    full validator, and the read returned the file."""
    with pytest.raises(mestra.MestraError) as caught:
        mestra.read(corpus.case_path(name))
    assert caught.value.rule == "E16"


def test_a_mesh_without_its_kind_is_e39_alone(tmp_path):
    """A support that does not say what kind it is is E39, and no rule
    that depends on the kind is decided for it. Python took it for a
    mesh and then for not one: E08 for the digest. C++ said E39 alone;
    Julia and MATLAB added E03, E08 and E38."""
    path = _case(tmp_path, "mesh_two_rows", "no_kind")
    with h5py.File(path, "r+") as f:
        del f["/supports/s0"].attrs["kind"]
    assert mestra.validate(path).error_ids == ["E39"]


def test_a_support_without_a_kind_it_declares_has_no_digest():
    """Section 24: the declared kind decides which arrays are hashed,
    so a support that declares none, or one section 6 does not have,
    has no digest. The reader's digest took a kind-less support for a
    mesh, and the model's hashed an unknown kind's node count alone."""
    from mestra.model import Support
    cells = dict(n_nodes=6, cell_types=[9, 9], cell_offsets=[0, 4, 8],
                 cell_connectivity=[0, 1, 4, 3, 1, 2, 5, 4])
    with pytest.raises(mestra.MestraError) as refused:
        Support("s0", None, **cells).computed_support_id()
    assert refused.value.rule == "E39"
    with pytest.raises(mestra.MestraError) as refused:
        Support("s0", "grid", **cells).computed_support_id()
    assert refused.value.rule == "E02"
    assert mestra.support_ids(corpus.case_path("err_e39_kind_mesh")) == {}
    assert mestra.support_ids(corpus.case_path("err_e02_kind")) == {}
