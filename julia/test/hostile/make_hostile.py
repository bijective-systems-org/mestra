"""Write the hostile files under julia/test/hostile/.

These are not conformance cases.  The conformance corpus under
vectors/ is what a conforming file looks like; this directory is the
opposite, and none of it is a mestra file in any useful sense.  Every
file here is something a reader may be handed by somebody else: an
attribute stored as an array where the specification says scalar, a
filter nothing can decode, thirty thousand nested groups, a link that
points nowhere or at another file, a dataset that declares a trillion
elements and holds none.  A reader must answer each one with a
finding or with a MestraError and must not crash, hang, or try to
allocate what the file claims.

Nothing here depends on any mestra implementation.  It is h5py and
numpy, chosen over HDF5.jl because h5py reaches the parts of the C
API these files need in the fewest lines: soft, external and dangling
links, a filter declared with client data nothing will read, external
raw storage pointing at a file that does not exist, and a fixed-length
string holding bytes that are not UTF-8.

Run it with any Python that has h5py and numpy:

    python make_hostile.py [output_directory]

The default output directory is the directory holding this file.
Two files, deep_keys.mes and deep_callables.mes, are large for what
they hold and are written only when --deep is passed; the test suite
builds them itself in a temporary directory, so they are not
committed.
"""

import os
import sys

import h5py
import numpy as np

CREATED = "2026-09-19T00:00:00Z"
WRITER = "mestra hostile 0"
DIM_NAME = "This is a netCDF dimension but not a netCDF variable."
DEEP_LEVELS = 30000


def sdtype(n):
    return h5py.string_dtype(encoding="utf-8", length=max(1, n))


def sattr(obj, name, value):
    raw = value.encode("utf-8")
    n = max(1, len(raw))
    obj.attrs.create(name, raw.ljust(n, b"\0"), dtype=sdtype(n))


def rattr(obj, name, raw):
    n = max(1, len(raw))
    obj.attrs.create(name, raw.ljust(n, b"\0"), dtype=sdtype(n))


def battr(obj, name, value):
    obj.attrs.create(name, np.int8(1 if value else 0))


def iattr(obj, name, value):
    obj.attrs.create(name, np.int64(value))


def fattr(obj, name, value):
    obj.attrs.create(name, np.float64(value))


def scale(group, name, length, unlimited=False, with_name=True):
    # track_order is attribute creation order, tracked and indexed,
    # which section 21 requires of a scale (E42); a hostile file is
    # about its one fault and not about that.
    d = group.create_dataset(
        name, shape=(length,), dtype=">f4",
        maxshape=(None,) if unlimited else (length,),
        chunks=(1,) if unlimited else None, track_times=False,
        track_order=True)
    if with_name:
        d.make_scale("%s%10d" % (DIM_NAME, length))
    else:
        # CLASS without NAME: still a dimension scale to H5DSis_scale,
        # and a reader that took the dimension's name from NAME rather
        # than from the link would have nothing to take.
        d.attrs.create("CLASS", b"DIMENSION_SCALE",
                       dtype=h5py.string_dtype(encoding="ascii",
                                               length=16))
    return d


def root(f, **kw):
    sattr(f, "created", kw.get("created", CREATED))
    sattr(f, "format", kw.get("format", "mestra/0"))
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)


def minimal(f, n_rows=2):
    """A root, a row scale, one key and one scalar: enough of a file
    that a reader gets far enough to meet whatever is wrong with it."""
    root(f)
    row = scale(f, "row", n_rows, unlimited=True)
    keys = f.create_group("keys")
    mach = keys.create_dataset("mach", shape=(n_rows,), dtype="<f8",
                               data=np.linspace(0.4, 0.8, n_rows),
                               maxshape=(None,), chunks=(n_rows,),
                               track_times=False)
    mach.dims[0].attach_scale(row)
    sattr(mach, "role", "condition")
    sattr(mach, "units", "1")
    scalars = f.create_group("scalars")
    cl = scalars.create_dataset("cl", shape=(n_rows,), dtype="<f8",
                                data=np.linspace(0.25, 0.55, n_rows),
                                maxshape=(None,), chunks=(n_rows,),
                                track_times=False)
    cl.dims[0].attach_scale(row)
    sattr(cl, "units", "1")
    sattr(cl, "source", "data")
    return row, keys, scalars


# ------------------------------------------- attributes that are arrays

def case_attr_array_root(f):
    """Section 18 gives every attribute a scalar dataspace.  Here the
    root attributes are arrays, and one is a variable-length string,
    which section 18 forbids anywhere in the file."""
    f.attrs.create("format", np.array([1, 2, 3], dtype="<i8"))
    f.attrs.create("writer", np.array([1.5, 2.5], dtype="<f8"))
    f.attrs.create("created", np.array(["a", "b"],
                                       dtype=h5py.string_dtype()))
    battr(f, "aligned", True)
    minimal_tail(f)


def minimal_tail(f, n_rows=2):
    row = scale(f, "row", n_rows, unlimited=True)
    keys = f.create_group("keys")
    mach = keys.create_dataset("mach", shape=(n_rows,), dtype="<f8",
                               data=np.linspace(0.4, 0.8, n_rows),
                               maxshape=(None,), chunks=(n_rows,),
                               track_times=False)
    mach.dims[0].attach_scale(row)
    sattr(mach, "role", "condition")
    sattr(mach, "units", "1")
    f.create_group("scalars")
    return row, keys


def case_attr_array_key(f):
    """An array where a key's role, units and bounds should be."""
    row, keys = minimal_tail(f)
    k = keys["mach"]
    del k.attrs["role"]
    del k.attrs["units"]
    k.attrs.create("role", np.array(["condition", "design"],
                                    dtype=h5py.string_dtype()))
    k.attrs.create("units", np.array([b"1", b"m"], dtype=sdtype(1)))
    k.attrs.create("lower", np.array([0.1, 0.2], dtype="<f8"))
    k.attrs.create("upper", np.zeros((0,), dtype="<f8"))


def case_attr_array_slot(f):
    """An array where a slot's components, units and source should
    be, on a node array of a mesh support."""
    row, keys = minimal_tail(f)
    comp = scale(f, "component_1", 1)
    sup = f.create_group("supports").create_group("s0")
    node = scale(sup, "node", 3)
    sattr(sup, "kind", "axis")
    iattr(sup, "n_nodes", 3)
    iattr(sup, "n_cells", 0)
    sattr(sup, "support_id", "0" * 64)
    c = sup.create_dataset("coordinates", data=np.zeros((3, 1)),
                           dtype="<f8", track_times=False)
    c.dims[0].attach_scale(node)
    c.dims[1].attach_scale(comp)
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "s")
    iattr(c, "components", 1)
    sattr(c, "source", "data")
    na = sup.create_group("node_arrays")
    p = na.create_dataset("p", shape=(2, 3, 1), dtype="<f8",
                          data=np.zeros((2, 3, 1)), maxshape=(None, 3, 1),
                          chunks=(2, 3, 1), track_times=False)
    p.dims[0].attach_scale(row)
    p.dims[1].attach_scale(node)
    p.dims[2].attach_scale(comp)
    sattr(p, "role", "field")
    sattr(p, "varies", "row")
    p.attrs.create("units", np.array([b"Pa", b"K"], dtype=sdtype(2)))
    p.attrs.create("components", np.array([1, 1], dtype="<i8"))
    p.attrs.create("source", np.array(["data"], dtype=h5py.string_dtype()))


# ------------------------------------------------------------ filters

UNKNOWN_FILTER = 32109


def case_filter_many_cd(f):
    """A filter declared with twelve client-data values.  A reader
    that asked the library for them into a buffer of eight and then
    trusted the returned count would write past the end of it.  The
    filter is optional, so the data is stored unfiltered and can be
    read; only the declaration is hostile."""
    row, keys, scalars = minimal(f)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((2,))
    dcpl.set_filter(UNKNOWN_FILTER, h5py.h5z.FLAG_OPTIONAL,
                    tuple(range(12)))
    dcpl.set_obj_track_times(False)
    space = h5py.h5s.create_simple((2,), (h5py.h5s.UNLIMITED,))
    tid = h5py.h5t.NATIVE_DOUBLE
    dsid = h5py.h5d.create(scalars.id, b"cd", tid, space, dcpl)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL,
               np.array([1.0, 2.0], dtype="<f8"))
    d = scalars["cd"]
    d.dims[0].attach_scale(row)
    sattr(d, "units", "1")
    sattr(d, "source", "data")


def case_filter_unknown(f):
    """A filter identifier no library has, on a dataset that does
    hold data.  The declaration is readable and the data is not."""
    row, keys, scalars = minimal(f)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((2,))
    dcpl.set_filter(UNKNOWN_FILTER + 1, h5py.h5z.FLAG_OPTIONAL, (7,))
    dcpl.set_obj_track_times(False)
    space = h5py.h5s.create_simple((2,), (h5py.h5s.UNLIMITED,))
    dsid = h5py.h5d.create(scalars.id, b"weird", h5py.h5t.NATIVE_DOUBLE,
                           space, dcpl)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL,
               np.array([3.0, 4.0], dtype="<f8"))
    d = scalars["weird"]
    d.dims[0].attach_scale(row)
    sattr(d, "units", "1")
    sattr(d, "source", "data")


# -------------------------------------------------------------- links

def add_links(f, maker):
    """One link of the given kind under each of the four groups a
    reader walks."""
    for parent in ("keys", "scalars", "supports", "callables"):
        if parent not in f:
            f.create_group(parent)
        f[parent]["elsewhere"] = maker(parent)


def case_link_dangling(f):
    minimal(f)
    add_links(f, lambda p: h5py.SoftLink("/%s/nothing_is_here" % p))


def case_link_cycle(f):
    minimal(f)
    for parent in ("keys", "scalars", "supports", "callables"):
        if parent not in f:
            f.create_group(parent)
        f[parent]["there"] = h5py.SoftLink("/%s/back" % parent)
        f[parent]["back"] = h5py.SoftLink("/%s/there" % parent)
    # and one that points at the group holding it
    f["scalars"]["itself"] = h5py.SoftLink("/scalars")


def case_link_external(f):
    minimal(f)
    add_links(f, lambda p: h5py.ExternalLink("somewhere_else.mes", "/"))
    # one pointing at a file that does exist, so that a reader that
    # follows external links succeeds and is wrong rather than failing
    f["scalars"]["neighbour"] = h5py.ExternalLink(
        "link_dangling.mes", "/scalars/cl")


# ------------------------------------------------------ kind confusion

def case_kind_confusion(f):
    """A key that is a group, a scalar that is a group with data-ish
    children, and a support that is a dataset."""
    row, keys, scalars = minimal(f)
    g = keys.create_group("asgroup")
    sattr(g, "role", "condition")
    sattr(g, "units", "1")
    g.create_dataset("surprise", data=np.zeros(3), track_times=False)
    supports = f.create_group("supports")
    d = supports.create_dataset("s0", data=np.arange(4, dtype="<i8"),
                                track_times=False)
    sattr(d, "kind", "mesh")
    iattr(d, "n_nodes", 4)
    iattr(d, "n_cells", 0)
    sattr(d, "support_id", "0" * 64)
    cal = f.create_group("callables")
    cal.create_dataset("m1", data=np.zeros(2), track_times=False)


# ------------------------------------------------------ a huge claim

def case_huge_declared(f):
    """A chunked dataset that declares a trillion elements and holds
    none.  Nothing in the file is eight terabytes; the number is.  A
    reader that sized a buffer from the dataspace would ask the
    operating system for all of it."""
    row, keys, scalars = minimal(f)
    d = scalars.create_dataset("huge", shape=(10 ** 12,), dtype="<f8",
                               chunks=(1024,), maxshape=(None,),
                               track_times=False)
    d.dims[0].attach_scale(row)
    sattr(d, "units", "1")
    sattr(d, "source", "data")
    # and one with a chunk as large as the claim, so that even a
    # single-row read would pull a chunk of it
    e = scalars.create_dataset("huge_chunk", shape=(4 * 10 ** 8,),
                               dtype="<f8", chunks=(4 * 10 ** 8,),
                               maxshape=(None,), track_times=False)
    e.dims[0].attach_scale(row)
    sattr(e, "units", "1")
    sattr(e, "source", "data")


def case_huge_strings(f):
    """The same claim on a fixed-length string dataset, which a
    reader has to walk element by element for E26."""
    row, keys, scalars = minimal(f)
    cats = f.create_group("categories")
    d = cats.create_dataset("region", shape=(10 ** 11,), dtype=sdtype(64),
                            chunks=(1024,), maxshape=(None,),
                            track_times=False)
    s = scale(f, "category_region", 1)
    d.dims[0].attach_scale(s)


# ---------------------------------------------------------- strings

def case_bad_utf8(f):
    """A fixed-length string that is not UTF-8, in a category table
    and in an attribute."""
    row, keys, scalars = minimal(f)
    cats = f.create_group("categories")
    s = scale(f, "category_region", 2)
    d = cats.create_dataset("region", shape=(2,), dtype=sdtype(6),
                            data=[b"\xff\xfe\x00\x00\x00\x00",
                                  b"outlet"], track_times=False)
    d.dims[0].attach_scale(s)
    rattr(f["keys"]["mach"], "category", b"\xc3\x28bad")


def case_empty_category(f):
    """A category table with an empty entry, which is a legal
    fixed-length string of nothing at all."""
    row, keys, scalars = minimal(f)
    cats = f.create_group("categories")
    s = scale(f, "category_region", 3)
    d = cats.create_dataset("region", shape=(3,), dtype=sdtype(6),
                            data=[b"", b"inlet", b""], track_times=False)
    d.dims[0].attach_scale(s)
    g = keys.create_dataset("region", shape=(2,), dtype="<i4",
                            data=np.array([0, 2], dtype="<i4"),
                            maxshape=(None,), chunks=(2,),
                            track_times=False)
    g.dims[0].attach_scale(row)
    sattr(g, "role", "categorical")
    sattr(g, "category", "region")


# ------------------------------------------------- dimension scales

def case_scale_twice(f):
    """Two different scales on one axis.  Section 21 allows exactly
    one, and a reader that took the first it found would take a name
    that is not the dimension's."""
    row, keys, scalars = minimal(f)
    other = scale(f, "component_1", 2)
    scalars["cl"].dims[0].attach_scale(other)


def case_scale_no_name(f):
    """A dimension scale with CLASS and no NAME."""
    root(f)
    row = scale(f, "row", 2, unlimited=True, with_name=False)
    keys = f.create_group("keys")
    mach = keys.create_dataset("mach", shape=(2,), dtype="<f8",
                               data=np.array([0.4, 0.8]),
                               maxshape=(None,), chunks=(2,),
                               track_times=False)
    mach.dims[0].attach_scale(row)
    sattr(mach, "role", "condition")
    sattr(mach, "units", "1")
    f.create_group("scalars")


# ------------------------------------------------- an unreadable object

def case_unreadable_continues(f):
    """A dataset whose raw data lives in a file that is not there,
    named so that it sorts before two objects that are perfectly
    readable.  A validator that abandoned the pass on the first
    failure would never look at the other two."""
    row, keys, scalars = minimal(f)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_external(b"no_such_raw_file.bin", 0, 16)
    dcpl.set_obj_track_times(False)
    space = h5py.h5s.create_simple((2,))
    h5py.h5d.create(scalars.id, b"a_broken", h5py.h5t.NATIVE_DOUBLE,
                    space, dcpl)
    d = scalars["a_broken"]
    sattr(d, "units", "1")
    sattr(d, "source", "data")
    # The two that must still be reached, each with something a
    # validator has to say about it, so that reaching them is visible
    # in the report and not merely assumed.
    y = scalars.create_dataset("y_warns", shape=(2,), dtype="<f8",
                               data=np.array([1.0, 2.0]),
                               maxshape=(None,), chunks=(2,),
                               track_times=False)
    y.dims[0].attach_scale(row)
    sattr(y, "units", "kg/(m s")          # W10, unparseable
    sattr(y, "source", "data")
    z = scalars.create_dataset("z_errors", shape=(2,), dtype="<f8",
                               data=np.array([1.0, float("nan")]),
                               maxshape=(None,), chunks=(2,),
                               track_times=False)
    z.dims[0].attach_scale(row)
    sattr(z, "units", "1")
    sattr(z, "source", "model")           # E36, and W03 for the NaN
    # and a key with something wrong that must still be reported
    bad = keys.create_dataset("no_units", shape=(2,), dtype="<f8",
                              data=np.array([1.0, 2.0]),
                              maxshape=(None,), chunks=(2,),
                              track_times=False)
    bad.dims[0].attach_scale(row)
    sattr(bad, "role", "condition")


# -------------------------------------------------------- not a file

def case_not_hdf5(path):
    with open(path, "wb") as fh:
        fh.write(b"this is not an HDF5 file, whatever the extension "
                 b"says\n" + bytes(range(256)) * 4)


# ------------------------------------------------------ deep nesting

def write_deep(path, parent, levels=DEEP_LEVELS):
    """`levels` nested groups under one of the groups a reader walks.
    Every recursive walk in a reader has to be capped or iterative to
    survive this, and the depth is the only thing the file holds."""
    with h5py.File(path, "w") as f:
        minimal(f)
        if parent not in f:
            f.create_group(parent)
        g = f[parent]
        if parent == "callables":
            g = g.create_group("m1")
            sattr(g, "type", "deep")
        for i in range(levels):
            g = g.create_group("g")
        g.attrs.create("bottom", np.int64(1))


CASES = {
    "attr_array_root": case_attr_array_root,
    "attr_array_key": case_attr_array_key,
    "attr_array_slot": case_attr_array_slot,
    "filter_many_cd": case_filter_many_cd,
    "filter_unknown": case_filter_unknown,
    "link_dangling": case_link_dangling,
    "link_cycle": case_link_cycle,
    "link_external": case_link_external,
    "kind_confusion": case_kind_confusion,
    "huge_declared": case_huge_declared,
    "huge_strings": case_huge_strings,
    "bad_utf8": case_bad_utf8,
    "empty_category": case_empty_category,
    "scale_twice": case_scale_twice,
    "scale_no_name": case_scale_no_name,
    "unreadable_continues": case_unreadable_continues,
}


def main(argv):
    out = (argv[1] if len(argv) > 1 and not argv[1].startswith("-")
           else os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(out):
        os.makedirs(out)
    for name in sorted(CASES):
        path = os.path.join(out, name + ".mes")
        with h5py.File(path, "w") as f:
            CASES[name](f)
        print("%-24s %8d bytes" % (name, os.path.getsize(path)))
    case_not_hdf5(os.path.join(out, "not_hdf5.mes"))
    print("%-24s %8d bytes" % ("not_hdf5",
                               os.path.getsize(os.path.join(out,
                                                            "not_hdf5.mes"))))
    if "--deep" in argv:
        for name, parent in (("deep_keys", "keys"),
                             ("deep_callables", "callables")):
            path = os.path.join(out, name + ".mes")
            write_deep(path, parent)
            print("%-24s %8d bytes" % (name, os.path.getsize(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
