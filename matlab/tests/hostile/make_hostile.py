#!/usr/bin/env python3
"""Write the hostile files the MATLAB reader must survive.

These are not conformance vectors.  Every one of them is a file no
conforming writer would produce, built to break a reader that trusts
what it opens: attributes with the wrong dataspace, filters a reader
cannot run, nesting deeper than a recursive walk can follow, links
that dangle, loop or point at another file, objects that are the
wrong kind, a shape far larger than memory, strings that are not
valid UTF-8, dimension scales that break their own rules, and an
object whose read fails in the middle of a group that must still be
validated to the end.

Written with h5py, like the conformance corpus, because h5py can say
things MATLAB's HDF5 interface cannot say at all: a variable-length
string attribute, an unregistered filter, a non-scalar attribute, a
soft or external link, and an enormous unwritten shape.  The MATLAB
side only has to survive them.

    python make_hostile.py [outdir]

Nothing here depends on the mestra package in any language.
"""

import os
import sys

import h5py
import numpy as np

CREATED = "2026-09-19T00:00:00Z"
WRITER = "mestra hostile 0"
DIM_NAME = "This is a netCDF dimension but not a netCDF variable."

# 30000 levels of nesting is 4.4 MB even with the most compact object
# headers HDF5 offers, which is too much to keep in a repository whose
# conformance files are 35 kB each.  The committed files use a depth
# that is still several times any recursion limit a language imposes,
# and the MATLAB test builds a 30000-level file of its own at run time,
# so the full depth is exercised on every run without being committed.
COMMITTED_DEPTH = 1000


def sdtype(n):
    return h5py.string_dtype("utf-8", max(n, 1))


def sattr(obj, name, value):
    raw = value.encode("utf-8")
    obj.attrs.create(name, raw, dtype=sdtype(len(raw)))


def battr(obj, name, value):
    obj.attrs.create(name, np.int8(1 if value else 0))


def iattr(obj, name, value):
    obj.attrs.create(name, np.int64(value))


def fattr(obj, name, value):
    obj.attrs.create(name, np.float64(value))


def scale(group, name, length, unlimited=False):
    # track_order is attribute creation order, tracked and indexed,
    # which section 21 requires of a scale (E42); a hostile file is
    # about its one fault and not about that.
    d = group.create_dataset(
        name, shape=(length,), dtype=">f4",
        maxshape=(None,) if unlimited else (length,),
        chunks=(1,) if unlimited else None, track_times=False,
        track_order=True)
    d.make_scale("%s%10d" % (DIM_NAME, length))
    return d


def base(f, n_rows=2):
    """A small, otherwise plausible file: root attributes, the row
    scale, one condition key and one mesh support."""
    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)

    keys = f.create_group("keys")
    mach = keys.create_dataset(
        "mach", shape=(n_rows,), dtype="<f8",
        data=np.linspace(0.4, 0.8, n_rows), maxshape=(None,),
        chunks=(max(n_rows, 1),), track_times=False)
    mach.dims[0].attach_scale(row)
    sattr(mach, "role", "condition")
    sattr(mach, "units", "1")

    scalars = f.create_group("scalars")
    cl = scalars.create_dataset(
        "cl", shape=(n_rows,), dtype="<f8", data=np.linspace(0.25, 0.55, n_rows),
        maxshape=(None,), chunks=(max(n_rows, 1),), track_times=False)
    cl.dims[0].attach_scale(row)
    sattr(cl, "units", "1")
    sattr(cl, "source", "data")

    sup = f.create_group("supports").create_group("s0")
    sattr(sup, "kind", "mesh")
    iattr(sup, "n_nodes", 6)
    iattr(sup, "n_cells", 2)
    sattr(sup, "support_id",
          "96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c943936c7")
    node = scale(sup, "node", 6)
    cell = scale(sup, "cell", 2)
    cell_plus_one = scale(sup, "cell_plus_one", 3)
    index = scale(sup, "index", 8)
    for name, data, dtype, sc in (
            ("cell_types", [9, 9], "|u1", cell),
            ("cell_offsets", [0, 4, 8], "<i8", cell_plus_one),
            ("cell_connectivity", [0, 1, 4, 3, 1, 2, 5, 4], "<i8", index)):
        d = sup.create_dataset(name, data=np.asarray(data, dtype=dtype),
                               track_times=False)
        d.dims[0].attach_scale(sc)
    coords = sup.create_dataset(
        "coordinates",
        data=np.array([[0., 0.], [1., 0.], [2., 0.],
                       [0., 1.], [1., 1.], [2., 1.]]), track_times=False)
    coords.dims[0].attach_scale(node)
    coords.dims[1].attach_scale(component_2)
    sattr(coords, "role", "coordinates")
    sattr(coords, "varies", "none")
    sattr(coords, "units", "m")
    iattr(coords, "components", 2)
    sattr(coords, "source", "data")

    node_arrays = sup.create_group("node_arrays")
    pressure = node_arrays.create_dataset(
        "pressure", shape=(n_rows, 6, 1), dtype="<f8",
        data=np.arange(n_rows * 6, dtype="<f8").reshape(n_rows, 6, 1),
        maxshape=(None, 6, 1), chunks=(max(n_rows, 1), 6, 1),
        track_times=False)
    for axis, sc in enumerate((row, node, component_1)):
        pressure.dims[axis].attach_scale(sc)
    sattr(pressure, "role", "field")
    sattr(pressure, "varies", "row")
    sattr(pressure, "units", "Pa")
    iattr(pressure, "components", 1)
    sattr(pressure, "source", "data")
    return {"row": row, "node": node, "component_1": component_1,
            "component_2": component_2, "support": sup,
            "keys": keys, "scalars": scalars, "pressure": pressure,
            "mach": mach, "cl": cl}


# ------------------------------------------------- non-scalar attributes

def array_attrs(obj):
    """The three encodings section 18 names, each with a dataspace that
    is not scalar.  A reader that takes the value without looking at
    the dataspace gets an array where it expects one number."""
    obj.attrs.create("mestra_hostile_i64", np.array([1, 2, 3], dtype="<i8"))
    obj.attrs.create("mestra_hostile_f64", np.array([1.0, 2.0], dtype="<f8"))
    obj.attrs.create("mestra_hostile_vlen",
                     np.array(["one", "two"], dtype=h5py.string_dtype("utf-8")))


def case_attr_array_root(f):
    b = base(f)
    # The named attributes themselves, with the wrong dataspace.
    del f.attrs["aligned"]
    f.attrs.create("aligned", np.array([1, 1], dtype="|i1"))
    f.attrs.create("generalisation_group",
                   np.array([b"a", b"b"], dtype=sdtype(1)))
    array_attrs(f)


def case_attr_array_key(f):
    b = base(f)
    mach = b["mach"]
    del mach.attrs["units"]
    mach.attrs.create("units", np.array([b"1", b"2"], dtype=sdtype(1)))
    mach.attrs.create("lower", np.array([0.1, 0.2], dtype="<f8"))
    mach.attrs.create("upper", np.array([[0.9], [1.0]], dtype="<f8"))
    array_attrs(mach)


def case_attr_array_slot(f):
    b = base(f)
    p = b["pressure"]
    del p.attrs["components"]
    p.attrs.create("components", np.array([1, 1], dtype="<i8"))
    p.attrs.create("quantile", np.array([0.5, 0.9], dtype="<f8"))
    array_attrs(p)
    sup = b["support"]
    array_attrs(sup)


# --------------------------------------------------------------- filters

def filtered(group, name, filter_id, cd_values, flags, scales):
    """A dataset carrying a filter this reader cannot run."""
    space = h5py.h5s.create_simple((2, 6, 1), (h5py.h5s.UNLIMITED, 6, 1))
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((2, 6, 1))
    dcpl.set_obj_track_times(False)
    dcpl.set_filter(filter_id, flags, tuple(cd_values))
    dsid = h5py.h5d.create(group.id, name.encode(), h5py.h5t.IEEE_F64LE,
                           space, dcpl)
    d = h5py.Dataset(dsid)
    for axis, sc in enumerate(scales):
        d.dims[axis].attach_scale(sc)
    return d


def case_filter_many_cd(f):
    b = base(f)
    d = filtered(b["support"]["node_arrays"], "heat_flux", 32017,
                 list(range(1, 21)), h5py.h5z.FLAG_OPTIONAL,
                 (b["row"], b["node"], b["component_1"]))
    sattr(d, "role", "field")
    sattr(d, "varies", "row")
    sattr(d, "units", "W m-2")
    iattr(d, "components", 1)
    sattr(d, "source", "data")


def case_filter_unknown_id(f):
    b = base(f)
    # HDF5 refuses to create a dataset whose mandatory filter is not
    # registered, so the filter is marked optional.  What the reader
    # sees is the same: a filter identifier that is neither gzip nor
    # shuffle, which section 23 does not allow.
    d = filtered(b["support"]["node_arrays"], "heat_flux", 32018,
                 [7], h5py.h5z.FLAG_OPTIONAL,
                 (b["row"], b["node"], b["component_1"]))
    sattr(d, "role", "field")
    sattr(d, "varies", "row")
    sattr(d, "units", "W m-2")
    iattr(d, "components", 1)
    sattr(d, "source", "data")


# ---------------------------------------------------------------- depth

def nest(group, depth):
    g = group
    for _ in range(depth):
        g = g.create_group("g")
    return g


def case_deep_callables(f):
    base(f)
    m1 = f.create_group("callables").create_group("m1")
    sattr(m1, "type", "affine")
    nest(m1, COMMITTED_DEPTH)


def case_deep_root(f):
    base(f)
    nest(f.create_group("extras"), COMMITTED_DEPTH)


def case_hard_link_cycle(f):
    """Two groups and one hard link, and the tree is infinitely deep.
    The same defence as the deep case, in 300 bytes."""
    base(f)
    extras = f.create_group("extras")
    extras.create_group("down")
    extras["down/up"] = extras


# ---------------------------------------------------------------- links

def case_link_soft_dangling(f):
    b = base(f)
    b["keys"]["ghost"] = h5py.SoftLink("/keys/nowhere")
    b["scalars"]["ghost"] = h5py.SoftLink("/scalars/nowhere")
    f["supports"]["ghost"] = h5py.SoftLink("/supports/nowhere")
    f.create_group("callables")["ghost"] = h5py.SoftLink("/callables/nowhere")


def case_link_soft_cycle(f):
    b = base(f)
    b["keys"]["loop"] = h5py.SoftLink("/keys/loop")
    b["scalars"]["loop"] = h5py.SoftLink("/scalars/loop")
    f["supports"]["loop"] = h5py.SoftLink("/supports/loop")
    f.create_group("callables")["loop"] = h5py.SoftLink("/callables/loop")


def case_link_external(f, target):
    b = base(f)
    for holder in (b["keys"], b["scalars"], f["supports"],
                   f.create_group("callables")):
        holder["elsewhere"] = h5py.ExternalLink(target, "/")


def write_external_target(path):
    with h5py.File(path, "w") as f:
        base(f)


# ------------------------------------------------------ the wrong kind

def case_kind_swap(f):
    b = base(f)
    # A key that is a group, and a category table that is a group.
    g = b["keys"].create_group("regime")
    sattr(g, "role", "categorical")
    sattr(g, "category", "regime")
    f.create_group("categories").create_group("regime")
    # A support that is a dataset, and a callable that is a dataset.
    f["supports"].create_dataset("s1", data=np.zeros(3), track_times=False)
    f.create_group("callables").create_dataset("m1", data=np.zeros(3),
                                               track_times=False)
    # node_arrays as a dataset rather than a group, on a second support.
    s2 = f["supports"].create_group("s2")
    sattr(s2, "kind", "none")
    iattr(s2, "n_nodes", 0)
    iattr(s2, "n_cells", 0)
    sattr(s2, "support_id",
          "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc")
    s2.create_dataset("node_arrays", data=np.zeros(2), track_times=False)


# ----------------------------------------------------------- huge shape

def case_huge_shape(f):
    b = base(f)
    na = b["support"]["node_arrays"]
    # 10^12 elements, chunked, never written: the file is tiny and a
    # reader that materialises it is not.
    big = na.create_dataset("enormous", shape=(10 ** 6, 10 ** 6, 1),
                            dtype="<f8", maxshape=(None, 10 ** 6, 1),
                            chunks=(1, 10 ** 6, 1), track_times=False)
    # The scales are attached so that a reader can see the leading axis
    # is `row` and try to read a row range of it.  One row is eight
    # megabytes, which a reader may read; the whole thing is eight
    # terabytes, which it may not.
    for axis, sc in enumerate((b["row"], b["node"], b["component_1"])):
        big.dims[axis].attach_scale(sc)
    sattr(big, "role", "field")
    sattr(big, "varies", "row")
    sattr(big, "units", "Pa")
    iattr(big, "components", 1)
    sattr(big, "source", "data")
    # One row of this one is itself far larger than memory, so a reader
    # that caps only the whole dataset is still caught.
    wide = na.create_dataset("wide_row", shape=(2, 10 ** 11, 1),
                             dtype="<f8", maxshape=(None, 10 ** 11, 1),
                             chunks=(1, 10 ** 6, 1), track_times=False)
    for axis, sc in enumerate((b["row"], b["node"], b["component_1"])):
        wide.dims[axis].attach_scale(sc)
    sattr(wide, "role", "field")
    sattr(wide, "varies", "row")
    sattr(wide, "units", "Pa")
    iattr(wide, "components", 1)
    sattr(wide, "source", "data")


# -------------------------------------------------------------- strings

def raw_strings(group, name, raw, size, scale_):
    d = group.create_dataset(name, shape=(len(raw),), dtype=sdtype(size),
                             data=[r.ljust(size, b"\0") for r in raw],
                             track_times=False)
    if scale_ is not None:
        d.dims[0].attach_scale(scale_)
    return d


def case_string_bad_utf8(f):
    b = base(f)
    cats = f.create_group("categories")
    cat = scale(f, "category_region", 3)
    raw_strings(cats, "region",
                [b"\xff\xfe\xfd", b"ok", b"\xc3"], 3, cat)
    label = b["support"].create_group("cell_arrays")
    d = label.create_dataset("region", shape=(2, 1), dtype="<i4",
                             data=np.array([[0], [1]], dtype="<i4"),
                             track_times=False)
    d.dims[0].attach_scale(f["supports/s0/cell"])
    d.dims[1].attach_scale(b["component_1"])
    sattr(d, "role", "label")
    sattr(d, "varies", "none")
    iattr(d, "components", 1)
    sattr(d, "source", "data")
    sattr(d, "category", "region")
    # A string attribute whose bytes are not valid UTF-8 either.
    f["supports/s0"].attrs.create("mestra_hostile_bad",
                                  np.bytes_(b"\xff\xfe"), dtype=sdtype(2))


def case_category_empty(f):
    base(f)
    cats = f.create_group("categories")
    cat = scale(f, "category_member", 3)
    raw_strings(cats, "member", [b"", b"wing_b", b""], 6, cat)
    empty = scale(f, "category_blank", 2)
    raw_strings(cats, "blank", [b"", b""], 1, empty)


# --------------------------------------------------------------- scales

def case_scale_twice(f):
    b = base(f)
    # The same scale attached twice to one axis, and two different
    # scales attached to another.
    p = b["pressure"]
    p.dims[1].attach_scale(b["node"])
    p.dims[2].attach_scale(b["component_2"])


def case_scale_no_name(f):
    b = base(f)
    del b["node"].attrs["NAME"]
    del b["row"].attrs["NAME"]


# --------------------------------------------- a read that fails midway

def case_read_fails_midway(f):
    """A key whose read cannot succeed, sorted before a key that has a
    finding of its own.  A validator that abandons the pass on the
    first failure never reports the second.

    The read is made to fail with a checksum: the key carries
    fletcher32, and corrupt_first_chunk flips a byte of its stored data
    afterwards, so HDF5 refuses to hand the data over."""
    b = base(f)
    broken = b["keys"].create_dataset(
        "aaa_broken", shape=(2,), dtype="<f8", data=[0.4, 0.8],
        maxshape=(None,), chunks=(2,), fletcher32=True, track_times=False)
    broken.dims[0].attach_scale(b["row"])
    sattr(broken, "role", "condition")
    sattr(broken, "units", "1")
    # zzz_late is missing its units, which is E39 and must still be found.
    late = b["keys"].create_dataset(
        "zzz_late", shape=(2,), dtype="<f8", data=[0.1, 0.2],
        maxshape=(None,), chunks=(2,), track_times=False)
    late.dims[0].attach_scale(b["row"])
    sattr(late, "role", "condition")


def corrupt_first_chunk(path, dataset):
    """Flip one byte of a dataset's first stored chunk, so that its
    fletcher32 checksum no longer matches and HDF5 refuses the read."""
    with h5py.File(path, "r") as f:
        info = f[dataset].id.get_chunk_info(0)
        offset = info.byte_offset
    with open(path, "r+b") as fh:
        fh.seek(offset)
        byte = fh.read(1)
        fh.seek(offset)
        fh.write(bytes([byte[0] ^ 0xFF]))


CASES = [
    ("attr_array_root", case_attr_array_root),
    ("attr_array_key", case_attr_array_key),
    ("attr_array_slot", case_attr_array_slot),
    ("category_empty", case_category_empty),
    ("deep_callables", case_deep_callables),
    ("deep_root", case_deep_root),
    ("filter_many_cd", case_filter_many_cd),
    ("filter_unknown_id", case_filter_unknown_id),
    ("hard_link_cycle", case_hard_link_cycle),
    ("huge_shape", case_huge_shape),
    ("kind_swap", case_kind_swap),
    ("link_soft_cycle", case_link_soft_cycle),
    ("link_soft_dangling", case_link_soft_dangling),
    ("read_fails_midway", case_read_fails_midway),
    ("scale_no_name", case_scale_no_name),
    ("scale_twice", case_scale_twice),
    ("string_bad_utf8", case_string_bad_utf8),
]


def main(argv):
    outdir = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cases")
    os.makedirs(outdir, exist_ok=True)

    target = os.path.join(outdir, "external_target.mes")
    write_external_target(target)

    for name, builder in CASES:
        path = os.path.join(outdir, name + ".mes")
        if os.path.exists(path):
            os.remove(path)
        with h5py.File(path, "w", libver=("v110", "v110")) as f:
            builder(f)
        if name == "read_fails_midway":
            corrupt_first_chunk(path, "/keys/aaa_broken")
        print("%-22s %8.1f kB" % (name, os.path.getsize(path) / 1024))

    path = os.path.join(outdir, "link_external.mes")
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w", libver=("v110", "v110")) as f:
        case_link_external(f, "external_target.mes")
    print("%-22s %8.1f kB" % ("link_external", os.path.getsize(path) / 1024))

    # Not a case: a file that is not HDF5 at all.
    path = os.path.join(outdir, "not_hdf5.mes")
    with open(path, "wb") as fh:
        fh.write(b"this is not an HDF5 file\n" * 8)
    print("%-22s %8.1f kB" % ("not_hdf5", os.path.getsize(path) / 1024))

    # Not a case: a truncated HDF5 file.
    with open(os.path.join(outdir, "attr_array_root.mes"), "rb") as fh:
        head = fh.read(600)
    with open(os.path.join(outdir, "truncated.mes"), "wb") as fh:
        fh.write(head)
    print("%-22s %8.1f kB" % ("truncated", 600 / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
