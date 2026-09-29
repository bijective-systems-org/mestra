"""Read, write, compare: every valid case of the corpus.

The normative comparison is the structural equality of section 30.
Byte identity across HDF5 versions is not required and must not be
tested, but this writer must be deterministic: two runs of it on one
machine produce the same bytes.
"""

from __future__ import annotations

import os
import stat

import pytest

import mestra
from mestra import writer
from mestra.model import FileSource
from tests import corpus

VALID = corpus.valid_case_names()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_replacement_preserves_permissions_and_new_files_use_umask(tmp_path):
    ds = mestra.read(corpus.case_path("mesh_two_rows"), lazy=False)
    control = tmp_path / "normal-creation"
    control.touch()
    target = tmp_path / "result.mes"
    mestra.write(ds, str(target))
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)
    target.chmod(0o640)
    mestra.write(ds, str(target))
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert not mestra.validate(str(target)).errors


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("phase", ["write", "replace"])
def test_failed_replacement_preserves_the_destination(tmp_path, monkeypatch,
                                                     existing, phase):
    ds = mestra.read(corpus.case_path("mesh_two_rows"), lazy=False)
    target = tmp_path / "result.mes"
    if existing:
        mestra.write(ds, str(target))
    before = target.read_bytes() if existing else None

    def interrupted_write(dataset, handle):
        handle.create_group("unfinished")
        raise OSError("injected write failure")

    def interrupted_replace(source, destination):
        # Publication receives a closed, complete Mestra file.
        assert not mestra.validate(source).errors
        raise OSError("injected replace failure")

    if phase == "write":
        monkeypatch.setattr(writer, "_write", interrupted_write)
    else:
        monkeypatch.setattr(writer.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match=f"injected {phase} failure"):
        mestra.write(ds, str(target))
    assert (target.read_bytes() if target.exists() else None) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == (
        ["result.mes"] if existing else [])
    if existing:
        assert not mestra.validate(str(target)).errors


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_a_destination_that_cannot_take_a_file_is_refused_first(tmp_path):
    """The staging and the publish both happen in the destination's
    directory (docs/compatibility.md, "Replacing files"), so a
    destination that cannot take a file is refused before anything
    is validated or created, with a MestraError that names the path
    and nothing left behind."""
    ds = mestra.read(corpus.case_path("mesh_two_rows"), lazy=False)
    a_directory = tmp_path / "taken.mes"
    a_directory.mkdir()
    read_only = tmp_path / "sealed"
    read_only.mkdir()
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        for path, said in ((str(a_directory), "names a directory"),
                           (str(read_only / "x.mes"), "not writable"),
                           (str(tmp_path / "gone" / "x.mes"),
                            "does not exist"),
                           ("", "path is empty")):
            with pytest.raises(mestra.MestraError) as caught:
                mestra.write(ds, path)
            assert caught.value.rule == ""
            assert caught.value.where == path
            assert said in caught.value.message
    finally:
        read_only.chmod(stat.S_IRWXU)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sealed",
                                                          "taken.mes"]
    assert list(a_directory.iterdir()) == []
    assert list(read_only.iterdir()) == []


def test_write_validates_first_and_refuses_on_an_error(tmp_path):
    """Section 2 of the conventions: no writer emits a file its own
    validator rejects."""
    ds = mestra.read(corpus.case_path("mesh_two_rows"), lazy=False)
    ds.scalars["cl"].units = None
    path = str(tmp_path / "bad.mes")
    with pytest.raises(mestra.MestraError) as caught:
        mestra.write(ds, path)
    assert caught.value.rule == "E11"
    assert "check=False" in str(caught.value)
    assert "/scalars/cl" in str(caught.value)
    assert not os.path.exists(path)

    mestra.write(ds, path, check=False)
    assert mestra.validate(path).error_ids == ["E11"]


def test_a_warning_does_not_stop_a_write(tmp_path):
    """W05 is the file saying something true about itself."""
    source = corpus.case_path("warn_w05")
    written = str(tmp_path / "again.mes")
    with mestra.read(source) as ds:
        mestra.write(ds, written)
    assert mestra.validate(written).warning_ids == ["W05"]


@pytest.mark.parametrize("name", VALID)
def test_read_write_compare(tmp_path, name):
    """Section 30: structurally equal to the file it came from."""
    source = corpus.case_path(name)
    written = str(tmp_path / "again.mes")
    dataset = mestra.read(source, lazy=False)
    mestra.write(dataset, written)
    assert corpus.structural_diff(source, written) == []


@pytest.mark.parametrize("name", VALID)
def test_written_files_still_validate(tmp_path, name):
    """What this writer produces says what the original said."""
    source = corpus.case_path(name)
    written = str(tmp_path / "again.mes")
    with mestra.read(source) as dataset:
        mestra.write(dataset, written)
    want = corpus.expected(name)["validator"]
    report = mestra.validate(written)
    assert report.error_ids == want["errors"], report
    assert report.warning_ids == want["warnings"], report


@pytest.mark.parametrize("name", VALID)
def test_the_writer_is_deterministic(tmp_path, name):
    """Two runs on one machine produce the same bytes (section 30)."""
    dataset = mestra.read(corpus.case_path(name), lazy=False)
    first = str(tmp_path / "one.mes")
    second = str(tmp_path / "two.mes")
    mestra.write(dataset, first)
    mestra.write(dataset, second)
    with open(first, "rb") as a, open(second, "rb") as b:
        assert a.read() == b.read()


@pytest.mark.parametrize("name", VALID)
def test_netcdf4_opens_what_we_write(tmp_path, name):
    """A valid file is also a valid netCDF-4 file (section 13).

    Two readers that share no code, and each variable's dimension
    names must be the link names of the scales attached to it, which
    is the check that catches a reader taking a name from the NAME
    attribute instead.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    h5netcdf = pytest.importorskip("h5netcdf")
    written = str(tmp_path / "again.mes")
    with mestra.read(corpus.case_path(name)) as dataset:
        mestra.write(dataset, written)
    wanted = _dimensions_on_disk(written)

    handle = netCDF4.Dataset(written, "r")
    try:
        seen = _dimensions(handle)
    finally:
        handle.close()
    for path, dims in wanted.items():
        assert path in seen, path
        assert seen[path] == dims, path

    other = h5netcdf.File(written, "r")
    try:
        seen = _h5netcdf_dimensions(other, "")
    finally:
        other.close()
    for path, dims in wanted.items():
        assert path in seen, path
        assert [d.rsplit("/", 1)[-1] for d in seen[path]] == dims, path


def _h5netcdf_dimensions(group, prefix):
    out = {}
    for name, variable in group.variables.items():
        out[prefix + "/" + name] = list(variable.dimensions)
    for name, sub in group.groups.items():
        out.update(_h5netcdf_dimensions(sub, prefix + "/" + name))
    return out


def _dimensions(group, prefix=""):
    """Every variable's dimension names, as netCDF-4 gives them."""
    out = {}
    for name, variable in group.variables.items():
        out[prefix + "/" + name] = list(variable.dimensions)
    for name, sub in group.groups.items():
        out.update(_dimensions(sub, prefix + "/" + name))
    return out


def _dimensions_on_disk(path):
    """Every variable's dimension names, as the file's scales give
    them: the link name of the scale attached to each axis."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.attrs.get("CLASS", b"") == b"DIMENSION_SCALE":
                return
            out["/" + name] = [dim[0].name.rsplit("/", 1)[-1]
                               for dim in obj.dims]
        f.visititems(visit)
    return out


@pytest.mark.parametrize("name", VALID)
def test_opening_a_file_reads_no_array(name):
    """Section 29: attributes and dataspaces only."""
    with mestra.read(corpus.case_path(name)) as dataset:
        _ = dataset.n_rows, dataset.aligned
        for key in dataset.keys.values():
            _ = key.role, key.lower, key.upper, key.units
        for support in dataset.supports.values():
            _ = support.stored_support_id, support.kind
            for slot in support.arrays().values():
                _ = slot.role, slot.units, slot.dims, slot.source
        for source in _sources(dataset):
            assert source.reads == 0, source.path


def test_a_lazy_read_touches_one_slot_only():
    """Section 29: one slot, one row range, nothing else."""
    with mestra.read(corpus.case_path("mesh_two_rows")) as dataset:
        pressure = dataset.supports["s0"].node_arrays["pressure"]
        part = pressure.read(slice(1, 2))
        assert part.dims == ("row", "node", "component")
        assert part.shape == (1, 6, 1)
        assert part.at(row=0, node=3, component=0) == 204.0
        touched = [s.path for s in _sources(dataset) if s.reads]
        assert touched == ["/supports/s0/node_arrays/pressure"]


def _sources(dataset):
    """Every lazy source a dataset holds."""
    out = []
    for key in dataset.keys.values():
        if isinstance(key.data, FileSource):
            out.append(key.data)
    for slot in dataset.slots().values():
        if isinstance(slot.data, FileSource):
            out.append(slot.data)
    for support in dataset.supports.values():
        for source in support._cells.values():
            if isinstance(source, FileSource):
                out.append(source)
    if isinstance(dataset._row_support, FileSource):
        out.append(dataset._row_support)
    return out


def test_reading_it_all_closes_the_file():
    dataset = mestra.read(corpus.case_path("mesh_two_rows"), lazy=False)
    assert dataset._file is None
    assert dataset.scalars["cl"].values.at(row=0) == 0.25
