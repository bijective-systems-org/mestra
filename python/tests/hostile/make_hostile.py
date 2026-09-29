"""Write the hostile files, with h5py and nothing else.

    python make_hostile.py [output_directory]
    python make_hostile.py --deep DIRECTORY      # the big one, alone

These are not mestra files and most of them are not valid HDF5
practice either. Each one is a small file that is legal enough for a
library to open and hostile enough to break a reader that trusts
what it finds: an attribute where a scalar belongs and an array
sits, a filter no library has, a link that points at another file, a
group that is its own ancestor, a dataset that declares a thousand
billion elements, a string that is not UTF-8.

Every file starts from the same small valid skeleton so that what is
wrong with it is the one thing its name says, and so that a reader
has something to get right as well as something to refuse.

The deep-nesting case is not committed: thirty thousand groups is
four megabytes however they are stored, which is no size for a test
fixture. `--deep` writes it on demand and the test that needs it
calls this script.
"""

import os
import sys

import h5py
import numpy as np

CREATED = "2026-09-20T00:00:00Z"
WRITER = "mestra hostile 0"
DIM_NAME = "This is a netCDF dimension but not a netCDF variable."

#: The nesting the deep case builds. Thirty thousand levels costs a
#: naive walker a quadratic amount of path building long before it
#: costs it a stack.
DEEP = 30000

#: An HDF5 filter identifier no library implements.
UNKNOWN_FILTER = 61000

N_NODES = 6
CELL_TYPES = np.array([9, 9], dtype="<u1")
CELL_OFFSETS = np.array([0, 4, 8], dtype="<i8")
CELL_CONNECTIVITY = np.array([0, 1, 4, 3, 1, 2, 5, 4], dtype="<i8")
COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                   [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
PRESSURE = np.array([[[101.0], [102.0], [103.0],
                      [104.0], [105.0], [106.0]],
                     [[201.0], [202.0], [203.0],
                      [204.0], [205.0], [206.0]]])


# --------------------------------------------------------------- bits

def sdtype(nbytes):
    return h5py.string_dtype(encoding="utf-8", length=max(1, nbytes))


def sattr(obj, name, value):
    raw = value.encode("utf-8")
    n = max(1, len(raw))
    obj.attrs.create(name, raw.ljust(n, b"\0"), dtype=sdtype(n))


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


def dataset(group, name, data, dtype, scales, n_rows=None):
    data = np.asarray(data, dtype=dtype)
    kw = {}
    if n_rows is not None:
        kw["maxshape"] = (None,) + data.shape[1:]
        kw["chunks"] = (max(1, n_rows),) + data.shape[1:]
    d = group.create_dataset(name, shape=data.shape, dtype=dtype,
                             data=data, track_times=False, **kw)
    for axis, s in enumerate(scales):
        if s is not None:
            d.dims[axis].attach_scale(s)
    return d


def strings(group, name, values, scale_, size=None, raw=None):
    if raw is None:
        raw = [v.encode("utf-8") for v in values]
    n = size if size is not None else max([len(r) for r in raw] + [1])
    d = group.create_dataset(name, shape=(len(raw),), dtype=sdtype(n),
                             data=[r.ljust(n, b"\0") for r in raw],
                             track_times=False)
    if scale_ is not None:
        d.dims[0].attach_scale(scale_)
    return d


def skeleton(f):
    """A small valid file: two rows, one mesh support, one field.

    Everything else in this directory is this with one thing wrong.
    """
    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", 2, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)
    category_region = scale(f, "category_region", 2)

    categories = f.create_group("categories")
    strings(categories, "region", ["inlet", "outlet"], category_region)

    keys = f.create_group("keys")
    mach = dataset(keys, "mach", [0.40, 0.80], "<f8", [row], n_rows=2)
    sattr(mach, "role", "condition")
    sattr(mach, "units", "1")
    fattr(mach, "lower", 0.1)
    fattr(mach, "upper", 0.9)

    scalars = f.create_group("scalars")
    cl = dataset(scalars, "cl", [0.25, 0.55], "<f8", [row], n_rows=2)
    sattr(cl, "units", "1")
    sattr(cl, "source", "data")

    sup = f.create_group("supports").create_group("s0")
    node = scale(sup, "node", N_NODES)
    cell = scale(sup, "cell", len(CELL_TYPES))
    cell_plus_one = scale(sup, "cell_plus_one", len(CELL_OFFSETS))
    index = scale(sup, "index", len(CELL_CONNECTIVITY))
    dataset(sup, "cell_types", CELL_TYPES, "<u1", [cell])
    dataset(sup, "cell_offsets", CELL_OFFSETS, "<i8", [cell_plus_one])
    dataset(sup, "cell_connectivity", CELL_CONNECTIVITY, "<i8", [index])
    sattr(sup, "kind", "mesh")
    iattr(sup, "n_nodes", N_NODES)
    iattr(sup, "n_cells", len(CELL_TYPES))
    sattr(sup, "support_id",
          "96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c9439"
          "36c7")
    c = dataset(sup, "coordinates", COORDS, "<f8", [node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    p = dataset(sup.create_group("node_arrays"), "pressure", PRESSURE,
                "<f8", [row, node, component_1], n_rows=2)
    sattr(p, "role", "field")
    sattr(p, "varies", "row")
    sattr(p, "units", "Pa")
    iattr(p, "components", 1)
    sattr(p, "source", "data")
    return {"row": row, "component_1": component_1,
            "component_2": component_2, "node": node, "support": sup,
            "keys": keys, "scalars": scalars}


# ------------------------------------------------------------- the cases

def attrs_as_arrays(f):
    """Attributes where section 18 has a scalar and an array sits,
    on the root, on a key and on a slot, in all three encodings."""
    parts = skeleton(f)
    f.attrs.create("format", np.array([b"mestra/0", b"mestra/0"],
                                      dtype=sdtype(8)))
    f.attrs.create("aligned", np.array([1, 1], dtype="int8"))
    f.attrs.create("spread", np.array([1, 2, 3], dtype="int64"))
    mach = parts["keys"]["mach"]
    mach.attrs.create("lower", np.array([0.1, 0.2], dtype="float64"))
    mach.attrs.create("role", ["condition", "design"],
                      dtype=h5py.string_dtype())
    cl = parts["scalars"]["cl"]
    cl.attrs.create("units", np.array([b"1", b"1"], dtype=sdtype(1)))
    cl.attrs.create("empty", np.zeros((0,), dtype="float64"))


def filter_many_client_values(f):
    """A filter with twelve client-data values, which h5py's own
    properties do not report at all."""
    parts = skeleton(f)
    _filtered(f, parts, 305, tuple(range(12)), optional=True)


def filter_unknown_id(f):
    """A filter identifier no library implements.

    Marked optional because HDF5 refuses at creation time to make a
    dataset whose mandatory filter is not registered; the file still
    declares an identifier a reader cannot know, which is the point.
    """
    parts = skeleton(f)
    _filtered(f, parts, UNKNOWN_FILTER, (7,), optional=True)


def _filtered(f, parts, code, values, optional):
    space = h5py.h5s.create_simple((2, 6, 1), (h5py.h5s.UNLIMITED, 6, 1))
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    plist.set_chunk((2, 6, 1))
    plist.set_filter(code,
                     h5py.h5z.FLAG_OPTIONAL if optional else 0, values)
    node_arrays = parts["support"]["node_arrays"]
    h5py.h5d.create(node_arrays.id, b"filtered", h5py.h5t.IEEE_F64LE,
                    space, plist)
    d = node_arrays["filtered"]
    d.dims[0].attach_scale(parts["row"])
    d.dims[1].attach_scale(parts["node"])
    d.dims[2].attach_scale(parts["component_1"])
    sattr(d, "role", "field")
    sattr(d, "varies", "row")
    sattr(d, "units", "Pa")
    iattr(d, "components", 1)
    sattr(d, "source", "data")


def deep_groups(f, levels=200):
    """Groups nested far deeper than any file of this format.

    The committed copy is shallow enough to commit and deeper than
    the reader's limit; `--deep` writes the thirty thousand the
    review asked for.
    """
    skeleton(f)
    group = f.create_group("callables").create_group("m1")
    sattr(group, "type", "example.deep")
    for _ in range(levels):
        group = group.create_group("g")
    other = f.create_group("extras")
    for _ in range(levels):
        other = other.create_group("g")


def cyclic_groups(f):
    """A group that is its own ancestor, by a soft link.

    Cheaper than deep nesting and worse: a walker that follows it
    never stops.
    """
    skeleton(f)
    m1 = f.create_group("callables").create_group("m1")
    sattr(m1, "type", "example.cycle")
    inner = m1.create_group("inner")
    inner["back"] = h5py.SoftLink("/callables/m1")
    f["extras"] = h5py.SoftLink("/")


def links(f):
    """A dangling link, a cycle, and an external link, under each of
    the four groups that hold producer-chosen names."""
    parts = skeleton(f)
    for group in (f["keys"], f["scalars"], f["supports"],
                  f.create_group("callables")):
        group["dangling"] = h5py.SoftLink("/no/such/object")
        group["cycle_a"] = h5py.SoftLink(group.name + "/cycle_b")
        group["cycle_b"] = h5py.SoftLink(group.name + "/cycle_a")
        group["elsewhere"] = h5py.ExternalLink("other.mes", "/keys/mach")
    parts["support"]["node_arrays"]["outside"] = h5py.ExternalLink(
        "other.mes", "/supports/s0/node_arrays/pressure")


def member_kinds(f):
    """A group where a dataset belongs and a dataset where a group
    belongs, in both directions."""
    skeleton(f)
    group = f["keys"].create_group("alpha")
    sattr(group, "role", "condition")
    sattr(group, "units", "degree")
    f["scalars"].create_dataset("cd", data=np.zeros(2), dtype="<f8")
    sattr(f["scalars"]["cd"], "units", "1")
    sattr(f["scalars"]["cd"], "source", "callable:m1")
    f["supports"].create_dataset("s1", data=np.zeros(3), dtype="<f8")
    f.create_group("callables").create_dataset("m1", data=np.zeros(1),
                                               dtype="<f8")
    f.create_dataset("categories_as_dataset", data=np.zeros(1),
                     dtype="<f8")


def enormous_shape(f):
    """A dataset of 10**12 elements that was never written.

    The file is a few kilobytes; a reader that materialises it needs
    eight terabytes.
    """
    parts = skeleton(f)
    big = parts["support"]["node_arrays"].create_dataset(
        "enormous", shape=(1000000, 1000000), dtype="<f8",
        chunks=(1, 1000000), maxshape=(None, 1000000),
        track_times=False)
    big.dims[0].attach_scale(parts["row"])
    sattr(big, "role", "field")
    sattr(big, "varies", "row")
    sattr(big, "units", "Pa")
    iattr(big, "components", 1)
    sattr(big, "source", "data")
    wide = f["keys"].create_dataset(
        "enormous_key", shape=(10 ** 12,), dtype="<f8",
        chunks=(1024,), maxshape=(None,), track_times=False)
    wide.dims[0].attach_scale(parts["row"])
    sattr(wide, "role", "condition")
    sattr(wide, "units", "1")


def bad_strings(f):
    """Fixed-length strings that are not UTF-8, and an empty one."""
    parts = skeleton(f)
    del f["categories"]["region"]
    strings(f["categories"], "region", None,
            f["category_region"], size=6,
            raw=[b"\xff\xfe\x00\x00\x00\x00", b""])
    ids = f["keys"].create_dataset(
        "id", shape=(2,), dtype=sdtype(4),
        data=[b"\xc3\x28ab", b"ok"], maxshape=(None,), chunks=(2,),
        track_times=False)
    ids.dims[0].attach_scale(parts["row"])
    sattr(ids, "role", "id")
    f.attrs.create("writer", b"\xff\xfe", dtype=sdtype(2))


def scale_trouble(f):
    """A scale attached twice, and a scale with no NAME attribute."""
    parts = skeleton(f)
    other = scale(f, "component_1_again", 1)
    pressure = parts["support"]["node_arrays"]["pressure"]
    pressure.dims[2].attach_scale(other)
    nameless = scale(f, "nameless", 2)
    del nameless.attrs["NAME"]
    cl = f["scalars"]["cl"]
    cl.dims[0].detach_scale(parts["row"])
    cl.dims[0].attach_scale(nameless)


def unreadable_dataset(f):
    """A dataset with no conversion path, between two good ones.

    An opaque type reads as an OSError from inside the library. It
    sits under /scalars between `cl` and `cm` so that a validator
    that abandons the pass on it is caught by the rules it then
    misses.
    """
    parts = skeleton(f)
    space = h5py.h5s.create_simple((2,), (h5py.h5s.UNLIMITED,))
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    plist.set_chunk((2,))
    opaque = h5py.h5t.create(h5py.h5t.OPAQUE, 8)
    opaque.set_tag(b"nothing a reader knows")
    # A category table the library will not convert, and one that
    # declares more elements than a reader will materialise. Both
    # sit before the rules below, so a validator that abandons the
    # pass on them reports none of those.
    flat = h5py.h5s.create_simple((2,))
    h5py.h5d.create(f["categories"].id, b"broken", opaque, flat,
                    h5py.h5p.create(h5py.h5p.DATASET_CREATE))
    f["categories"].create_dataset(
        "enormous", shape=(10 ** 10,), dtype=sdtype(1),
        chunks=(4096,), maxshape=(None,), track_times=False)
    h5py.h5d.create(f["scalars"].id, b"cm", opaque, space, plist)
    cm = f["scalars"]["cm"]
    cm.dims[0].attach_scale(parts["row"])
    sattr(cm, "units", "1")
    sattr(cm, "source", "data")
    # A field with no units, after the unreadable one: a validator
    # that stopped would never report E11.
    after = dataset(parts["support"]["node_arrays"], "zeta", PRESSURE,
                    "<f8", [parts["row"], parts["node"],
                            parts["component_1"]], n_rows=2)
    sattr(after, "role", "field")
    sattr(after, "varies", "row")
    iattr(after, "components", 1)
    sattr(after, "source", "data")


def not_hdf5(path):
    """Not an HDF5 file at all, with the extension of one."""
    with open(path, "wb") as fh:
        fh.write(b"format = mestra/0\nthis is a text file\n")


def truncated(path):
    """An HDF5 superblock and then nothing."""
    with h5py.File(path + ".whole", "w") as f:
        skeleton(f)
    with open(path + ".whole", "rb") as fh:
        head = fh.read(2048)
    os.remove(path + ".whole")
    with open(path, "wb") as fh:
        fh.write(head)


CASES = {
    "attrs_as_arrays": attrs_as_arrays,
    "filter_many_client_values": filter_many_client_values,
    "filter_unknown_id": filter_unknown_id,
    "deep_groups": deep_groups,
    "cyclic_groups": cyclic_groups,
    "links": links,
    "member_kinds": member_kinds,
    "enormous_shape": enormous_shape,
    "bad_strings": bad_strings,
    "scale_trouble": scale_trouble,
    "unreadable_dataset": unreadable_dataset,
}

#: Cases written with the latest superblock, because a group costs
#: a seventh as many bytes there and these hold hundreds of them.
LATEST = ("deep_groups",)

#: Cases that are not HDF5 files, so they are written byte by byte.
RAW_CASES = {
    "not_hdf5": not_hdf5,
    "truncated": truncated,
}


def write_all(out):
    if not os.path.isdir(out):
        os.makedirs(out)
    for name, build in sorted(CASES.items()):
        path = os.path.join(out, name + ".mes")
        f = h5py.File(path, "w",
                      libver="latest" if name in LATEST else None)
        try:
            build(f)
        finally:
            f.close()
    for name, build in sorted(RAW_CASES.items()):
        build(os.path.join(out, name + ".mes"))
    return sorted(CASES) + sorted(RAW_CASES)


def write_deep(out, levels=DEEP):
    """The thirty-thousand-level file, which is too big to commit."""
    if not os.path.isdir(out):
        os.makedirs(out)
    path = os.path.join(out, "deep_groups_30000.mes")
    with h5py.File(path, "w", libver="latest") as f:
        deep_groups(f, levels)
    return path


def main(argv):
    if len(argv) > 1 and argv[1] == "--deep":
        where = argv[2] if len(argv) > 2 else "."
        print(write_deep(where))
        return 0
    out = argv[1] if len(argv) > 1 else os.path.dirname(
        os.path.abspath(__file__))
    names = write_all(out)
    print("%d hostile files written under %s" % (len(names), out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
