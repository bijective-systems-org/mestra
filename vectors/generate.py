"""Generate the mestra conformance corpus.

Writes one directory per case under vectors/cases/, each holding
case.mes and expected.json, and writes vectors/manifest.json.

Nothing here is a mestra implementation.  It is SPEC.md sections 13
to 30 read literally with h5py and numpy alone, so that a reader in
any language can be checked against files whose every byte was
chosen on purpose.  There is no mestra package and this file must
not become one.

Run it with any Python that has h5py and numpy:

    python generate.py [output_directory]
    python generate.py --on-demand

The default output directory is the directory holding this file.
The first form rewrites every case. The second writes only the three
files that are too large to commit -- cases/wide_keys and the two
deep hostile files -- into place, and touches nothing that is
committed; it is what a checkout needs before running a suite.

Byte reproducibility (section 30) rests on three things and all
three are honoured here: track_times is off on every dataset,
`created` and `writer` are fixed strings rather than the time and
the version of the run, and the order in which links, attributes
and dimension scales are created is the order written in this file.
"""

import hashlib
import json
import os
import struct
import sys

import h5py
import numpy as np

CREATED = "2026-09-19T00:00:00Z"
WRITER = "mestra corpus 0"

# The two files of docs/example.md were written by a different
# script under a different writer string.  The corpus reproduces
# them exactly, so it must reproduce that string too.
EXAMPLE_WRITER = "mestra examples 0"

# Section 21: the NAME attribute of a dimension scale that has no
# coordinate variable.  53 characters, then the length in ten
# columns, which is the C format "%s%10d".
DIM_NAME = "This is a netCDF dimension but not a netCDF variable."


# --------------------------------------------------------- encodings

def sdtype(nbytes):
    """Fixed-length UTF-8, NUL-padded (sections 18 and 19)."""
    return h5py.string_dtype(encoding="utf-8", length=max(1, nbytes))


def sattr(obj, name, value):
    """A string attribute, encoded as section 18 requires."""
    raw = value.encode("utf-8")
    n = max(1, len(raw))
    obj.attrs.create(name, raw.ljust(n, b"\0"), dtype=sdtype(n))


def rattr(obj, name, raw):
    """A string attribute from raw bytes, for the null sentinel."""
    n = max(1, len(raw))
    obj.attrs.create(name, raw.ljust(n, b"\0"), dtype=sdtype(n))


def vattr(obj, name, value):
    """A variable-length string attribute.  Section 18 forbids this
    everywhere; it exists so that the corpus can hold one file that
    breaks the rule (E19)."""
    obj.attrs.create(name, value,
                     dtype=h5py.string_dtype(encoding="utf-8"))


def battr(obj, name, value):
    obj.attrs.create(name, np.int8(1 if value else 0))


def iattr(obj, name, value):
    obj.attrs.create(name, np.int64(value))


def i32attr(obj, name, value):
    """An integer attribute in the wrong width, for E19."""
    obj.attrs.create(name, np.int32(value))


def fattr(obj, name, value):
    obj.attrs.create(name, np.float64(value))


def strings(group, name, values, scale_, size=None, raw=None):
    """A one-dimensional fixed-length UTF-8 string dataset."""
    if raw is None:
        raw = [v.encode("utf-8") for v in values]
    n = size if size is not None else max([len(r) for r in raw] + [1])
    d = group.create_dataset(name, shape=(len(raw),), dtype=sdtype(n),
                             data=[r.ljust(n, b"\0") for r in raw],
                             track_times=False)
    if scale_ is not None:
        d.dims[0].attach_scale(scale_)
    return d


def row_strings(group, name, values, scale_, n_rows):
    """A fixed-length string dataset over the unlimited row axis."""
    raw = [v.encode("utf-8") for v in values]
    n = max([len(r) for r in raw] + [1])
    d = group.create_dataset(
        name, shape=(len(raw),), dtype=sdtype(n),
        data=[r.ljust(n, b"\0") for r in raw], track_times=False,
        maxshape=(None,), chunks=(chunk_rows(n, (), n_rows),))
    d.dims[0].attach_scale(scale_)
    return d


def scale(group, name, length, unlimited=False):
    """A dimension scale, written as netCDF-C writes one (21).

    The creation property list is the point of this function. A scale
    is created with attribute creation order tracked and indexed,
    which gives it a version 2 object header, which is what lets its
    REFERENCE_LIST live in the file's heap. Without it a scale takes
    at most 4085 attachments: REFERENCE_LIST grows sixteen bytes each
    time and an object header message may not exceed 64 KiB. Object
    time tracking has to be turned off in the same list, because a
    version 2 object header records four timestamps unless it is told
    not to, and a file that records when it was written is not byte
    reproducible. No other object's property list is touched.
    """
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_attr_creation_order(h5py.h5p.CRT_ORDER_TRACKED |
                                 h5py.h5p.CRT_ORDER_INDEXED)
    dcpl.set_obj_track_times(False)
    dcpl.set_chunk((1,) if unlimited else (max(1, length),))
    space = h5py.h5s.create_simple(
        (length,), (h5py.h5s.UNLIMITED,) if unlimited else (length,))
    tid = h5py.h5t.py_create(np.dtype(">f4"), logical=True)
    d = h5py.Dataset(h5py.h5d.create(group.id, name.encode("utf-8"),
                                     tid, space, dcpl=dcpl))
    d.make_scale("%s%10d" % (DIM_NAME, length))
    return d


def scale_plain(group, name, length, unlimited=False):
    """A dimension scale created with the library's default property
    list, which is what every writer did before the rule of section
    21. Only the E42 case wants one."""
    d = group.create_dataset(
        name, shape=(length,), dtype=">f4",
        maxshape=(None,) if unlimited else (length,),
        chunks=(1,) if unlimited else None, track_times=False)
    d.make_scale("%s%10d" % (DIM_NAME, length))
    return d


def chunk_rows(itemsize, rest, n_rows):
    """The default of section 23: about 1 MiB of rows, at least one,
    capped at the row count, and 1 when there are no rows."""
    if n_rows == 0:
        return 1
    b = itemsize
    for extent in rest:
        b *= max(1, extent)
    c = 1048576 // b
    if c < 1:
        c = 1
    if c > n_rows:
        c = n_rows
    return c


def dataset(group, name, data, dtype, scales, n_rows=None,
            chunks=None, contiguous=False, fletcher32=False,
            gzip=None, shuffle=False):
    """A data slot.  `scales` is one scale per axis, in order; a None
    entry leaves that axis unattached, which only the E25 case
    wants."""
    data = np.asarray(data, dtype=dtype)
    kw = {}
    if n_rows is not None and not contiguous:
        kw["maxshape"] = (None,) + data.shape[1:]
        kw["chunks"] = (chunks if chunks is not None else
                        (chunk_rows(data.dtype.itemsize, data.shape[1:],
                                    n_rows),) + data.shape[1:])
    elif chunks is not None:
        kw["chunks"] = chunks
    if fletcher32:
        kw["fletcher32"] = True
    if gzip is not None:
        kw["compression"] = "gzip"
        kw["compression_opts"] = gzip
    if shuffle:
        kw["shuffle"] = True
    d = group.create_dataset(name, shape=data.shape, dtype=dtype,
                             data=data, track_times=False, **kw)
    for axis, s in enumerate(scales):
        if s is not None:
            d.dims[axis].attach_scale(s)
    return d


# --------------------------------------------------------- support_id

def support_id(n_nodes, cell_types=(), cell_offsets=(),
               connectivity=(), axis_coordinates=None):
    """Section 24, written from the text rather than from any other
    implementation: n_nodes as one int64 little-endian, then
    cell_types as uint8, then cell_offsets and cell_connectivity as
    int64 little-endian, then, for an axis support only, the
    coordinates as float64 little-endian, all in storage order with
    nothing between them."""
    h = hashlib.sha256()
    h.update(struct.pack("<q", int(n_nodes)))
    for v in np.asarray(cell_types, dtype="<u1").ravel():
        h.update(struct.pack("<B", int(v)))
    for v in np.asarray(cell_offsets, dtype="<i8").ravel():
        h.update(struct.pack("<q", int(v)))
    for v in np.asarray(connectivity, dtype="<i8").ravel():
        h.update(struct.pack("<q", int(v)))
    if axis_coordinates is not None:
        for v in np.asarray(axis_coordinates, dtype="<f8").ravel():
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


# ------------------------------------------------- expected.json bits

def fnum(value):
    """A float64 as section 30 requires it: the C format %.17e, or
    one of the three non-finite spellings."""
    value = float(value)
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return "%.17e" % value


def probe(slot, array, row=None, instance=None, draw=None, node=None,
          component=None, index=None):
    """One probe object.  The index order follows the stored order,
    (row | group, [draw], node | cell, component), so a probe is
    written with the axis names and never with axis positions.

    `instance` is the index along a `group:<k>` leading dimension and
    `index` the index along the flat connectivity axis; section 30
    names neither, and the report says so."""
    idx = tuple(i for i in (row, instance, draw, node, component, index)
                if i is not None)
    array = np.asarray(array)
    value = array[idx]
    p = {"slot": slot}
    if row is not None:
        p["row"] = int(row)
    if instance is not None:
        p["instance"] = int(instance)
    if draw is not None:
        p["draw"] = int(draw)
    if node is not None:
        p["node"] = int(node)
    if component is not None:
        p["component"] = int(component)
    if index is not None:
        p["index"] = int(index)
    if array.dtype.kind == "f":
        p["value"] = fnum(value)
    else:
        p["value"] = str(int(value))
    return p


def tagged(value):
    """A dictionary value in the tagged form of section 30."""
    if value is None:
        return {"t": "null"}
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S", "O"):
            flat = [v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in value.ravel()]
            return {"t": "strings", "shape": list(value.shape),
                    "data": flat}
        if value.dtype == np.int8:
            return {"t": "array", "dtype": "bool",
                    "shape": list(value.shape),
                    "data": [bool(v) for v in value.ravel()]}
        if value.dtype == np.int32:
            return {"t": "array", "dtype": "int32",
                    "shape": list(value.shape),
                    "data": [int(v) for v in value.ravel()]}
        if value.dtype == np.int64:
            return {"t": "array", "dtype": "int64",
                    "shape": list(value.shape),
                    "data": [int(v) for v in value.ravel()]}
        return {"t": "array", "dtype": "float64",
                "shape": list(value.shape),
                "data": [fnum(v) for v in value.ravel()]}
    if isinstance(value, dict):
        return {"t": "dict",
                "v": dict((k, tagged(value[k])) for k in sorted(value))}
    if isinstance(value, list):
        return {"t": "strings", "shape": [len(value)], "data": value}
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "i64", "v": value}
    if isinstance(value, float):
        return {"t": "f64", "v": fnum(value)}
    if isinstance(value, str):
        return {"t": "str", "v": value}
    raise TypeError("not representable: %r" % (value,))


def expect(description, errors=(), warnings=(), support_ids=None,
           probes=(), codec=None, evaluation=()):
    return {
        "description": description,
        "validator": {"errors": sorted(errors),
                      "warnings": sorted(warnings)},
        "support_ids": dict(support_ids or {}),
        "probes": list(probes),
        "codec": dict(codec or {}),
        "evaluation": list(evaluation),
    }


def canonical(obj):
    """Canonical JSON as section 30 defines it."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False) + "\n"


# ------------------------------------------------------ shared values

N_NODES = 6
CELL_TYPES = np.array([9, 9], dtype="<u1")
CELL_OFFSETS = np.array([0, 4, 8], dtype="<i8")
CELL_CONNECTIVITY = np.array([0, 1, 4, 3, 1, 2, 5, 4], dtype="<i8")
MESH_SID = support_id(N_NODES, CELL_TYPES, CELL_OFFSETS,
                      CELL_CONNECTIVITY)

BASE_COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
MEMBER_COORDS = np.stack([BASE_COORDS,
                          BASE_COORDS * np.array([1.5, 1.0])])
PRESSURE_2 = np.array([[[101.0], [102.0], [103.0],
                        [104.0], [105.0], [106.0]],
                       [[201.0], [202.0], [203.0],
                        [204.0], [205.0], [206.0]]])


def mesh_cells(sup, n_nodes, types, offsets, conn, sid=None):
    """A mesh support's four scales, three cell datasets and four
    attributes, in the order the examples use.  The order is part of
    the golden bytes and must not be changed."""
    node = scale(sup, "node", n_nodes)
    cell = scale(sup, "cell", len(types))
    cell_plus_one = scale(sup, "cell_plus_one", len(offsets))
    index = scale(sup, "index", len(conn))
    dataset(sup, "cell_types", types, "<u1", [cell])
    dataset(sup, "cell_offsets", offsets, "<i8", [cell_plus_one])
    dataset(sup, "cell_connectivity", conn, "<i8", [index])
    sattr(sup, "kind", "mesh")
    iattr(sup, "n_nodes", n_nodes)
    iattr(sup, "n_cells", len(types))
    sattr(sup, "support_id",
          sid if sid is not None
          else support_id(n_nodes, types, offsets, conn))
    return node, cell


def write_mesh_cells(sup, o):
    """The option-driven wrapper the base files use."""
    g = o.get
    return mesh_cells(sup, g("n_nodes", N_NODES),
                      g("cell_types", CELL_TYPES),
                      g("cell_offsets", CELL_OFFSETS),
                      g("cell_connectivity", CELL_CONNECTIVITY),
                      g("support_id", None))


# ------------------------------------------------------ the mesh base

def mesh_base(f, o):
    """The file of docs/example.md file one, with every deviation the
    error and warning cases need behind a named option.  With no
    options at all it is that file exactly, which is what makes the
    byte comparison against docs/examples a test of this generator.

    The creation order below is the corpus's order and must not be
    reordered: HDF5 keeps the links of a small group in creation
    order and the golden bytes depend on it.
    """
    g = o.get
    n_rows = g("n_rows", 2)
    sattr(f, "created", g("created", CREATED))
    sattr(f, "format", g("format", "mestra/0"))
    if g("writer", WRITER) is not None:
        sattr(f, "writer", g("writer", WRITER))
    battr(f, "aligned", g("aligned", True))
    if g("gen_group", "member") is not None:
        sattr(f, "generalisation_group", g("gen_group", "member"))
    if g("unknown_root_attr", False):
        sattr(f, "comment", "an attribute no version 0 reader knows")

    plain = g("plain_scales", ())
    endless = g("unlimited_scales", ())

    def sc(name, length, unlimited=False):
        maker = scale_plain if name in plain else scale
        return maker(f, name, length, unlimited or name in endless)

    row = sc("row", n_rows, unlimited=True)
    component_1 = sc("component_1", 1)
    component_2 = sc("component_2", 2)
    member_cats = g("member_cats", ["wing_a", "wing_b"])
    group_member = sc("group_member", g("n_group", 2))
    category_member = sc("category_member", len(member_cats))
    category_region = sc("category_region", 2)
    extra_scales = {}
    for name, length in g("extra_scales", []):
        extra_scales[name] = scale(f, name, length)

    categories = f.create_group("categories")
    strings(categories, "member", member_cats,
            category_member, size=g("category_size", None))
    strings(categories, "region", ["inlet", "outlet"], category_region,
            raw=g("region_raw", None))
    for name, values in g("extra_categories", []):
        strings(categories, name, values, extra_scales["category_" + name])

    keys = f.create_group("keys")
    mach = dataset(keys, g("mach_name", "mach"),
                   g("mach_values", [0.40, 0.80]), "<f8", [row],
                   n_rows=n_rows)
    if not g("mach_no_role", False):
        sattr(mach, "role", "condition")
    if g("mach_units", "1") is not None:
        sattr(mach, "units", g("mach_units", "1"))
    bounds = g("mach_bounds", (0.1, 0.9))
    if bounds is not None:
        fattr(mach, "lower", bounds[0])
        fattr(mach, "upper", bounds[1])

    member = dataset(keys, "member", g("member_values", [0, 1]),
                     g("member_dtype", "<i4"), [row], n_rows=n_rows)
    sattr(member, "role", "group")
    sattr(member, "category", "member")

    for kname, krole, kvalues, kdtype, kextra in g("extra_keys", []):
        k = dataset(keys, kname, kvalues, kdtype, [row], n_rows=n_rows)
        sattr(k, "role", krole)
        for aname, avalue in kextra:
            sattr(k, aname, avalue)

    scalars = f.create_group("scalars")
    if g("cl_as_group", False):
        cl = scalars.create_group("cl")
        sattr(cl, "units", "1")
        sattr(cl, "source", g("cl_source", "data"))
        if g("cl_output", None) is not None:
            sattr(cl, "output", g("cl_output"))
    else:
        cl_values = g("cl_values", [0.25, 0.55])
        cl = dataset(scalars, "cl", cl_values, g("cl_dtype", "<f8"), [row],
                     n_rows=n_rows,
                     contiguous=g("cl_contiguous", False),
                     gzip=g("cl_gzip", None),
                     shuffle=g("cl_shuffle", False))
        if g("cl_units_vlen", False):
            vattr(cl, "units", "1")
        elif g("cl_units", "1") is not None:
            sattr(cl, "units", g("cl_units", "1"))
        sattr(cl, "source", g("cl_source", "data"))
        for aname, avalue in g("cl_extra", []):
            sattr(cl, aname, avalue)

    sup = f.create_group("supports").create_group("s0")
    node, cell = write_mesh_cells(sup, o)

    coords = g("coords", MEMBER_COORDS)
    c = dataset(sup, "coordinates", coords, "<f8",
                [group_member, node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", g("coords_varies", "group:member"))
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    node_arrays = sup.create_group("node_arrays")
    if g("pressure_as_group", False):
        p = node_arrays.create_group("pressure")
    else:
        pdata = g("pressure", PRESSURE_2)
        p = dataset(node_arrays, "pressure", pdata,
                    g("pressure_dtype", "<f8"),
                    [row, node,
                     None if g("pressure_no_component_scale", False)
                     else component_1],
                    n_rows=n_rows, chunks=g("pressure_chunks", None),
                    fletcher32=g("pressure_fletcher32", False),
                    gzip=g("pressure_gzip", None),
                    shuffle=g("pressure_shuffle", False))
    sattr(p, "role", g("pressure_role", "field"))
    sattr(p, "varies", g("pressure_varies", "row"))
    if g("pressure_units", "Pa") is not None:
        sattr(p, "units", g("pressure_units", "Pa"))
    iattr(p, "components", g("pressure_components", 1))
    sattr(p, "source", g("pressure_source", "data"))
    for aname, avalue in g("pressure_extra", []):
        sattr(p, aname, avalue)

    band = g("band", None)
    if band is not None:
        pb = dataset(node_arrays, "pressure_band", band["values"], "<f8",
                     [row, node, component_1], n_rows=n_rows)
        sattr(pb, "role", "field")
        sattr(pb, "varies", "row")
        sattr(pb, "units", "Pa")
        iattr(pb, "components", 1)
        sattr(pb, "source", "data")
        sattr(pb, "statistic", "band")
        sattr(pb, "of", "pressure")
        if band.get("level") is not None:
            fattr(pb, "level", band["level"])
        if band.get("method") is not None:
            sattr(pb, "method", band["method"])

    cell_arrays = sup.create_group("cell_arrays")
    r = dataset(cell_arrays, "region", [[0], [1]], "<i4",
                [cell, component_1])
    sattr(r, "role", "label")
    sattr(r, "varies", "none")
    iattr(r, "components", 1)
    sattr(r, "source", "data")
    sattr(r, "category", "region")

    if g("weight", False):
        w = dataset(cell_arrays, "measure", [[0.5], [0.5]], "<f8",
                    [cell, component_1])
        sattr(w, "role", "weight")
        sattr(w, "varies", "none")
        iattr(w, "components", 1)
        sattr(w, "source", "data")
        if g("weight_recomputed", True):
            battr(w, "recomputed", True)
        elif g("weight_recomputed_false", False):
            battr(w, "recomputed", False)

    if g("row_support_values", None) is not None:
        dataset(f, "row_support", g("row_support_values"), "<i4",
                [row], n_rows=n_rows)

    if g("private_gen", False):
        private = f.create_group("private")
        sattr(private, "generalisation_group", "member")

    if g("unknown_root_group", False):
        f.create_group("extras")

    return {"row": row, "component_1": component_1,
            "component_2": component_2, "node": node, "cell": cell,
            "support": sup}


# ----------------------------------------------------- the affine base

AFFINE_KEYS = ["mach", "alpha"]
AFFINE = {
    "cl": {
        "A": np.array([[2.0, 0.1]]),
        "b": np.array([0.05]),
        "shape": np.array([], dtype="<i8"),
    },
    "pressure": {
        "A": np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.5],
                       [4.0, 0.5], [5.0, 1.0], [6.0, 1.0]]),
        "b": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "shape": np.array([6, 1], dtype="<i8"),
    },
}


def codec_array(group, name, data, zero_d=False):
    """A dictionary dataset with its `mestra_<name>_d<i>` scales.
    `zero_d` writes it as a zero-dimensional dataset, which section
    25 forbids; only the E32 case asks for that."""
    data = np.asarray(data)
    if zero_d:
        return group.create_dataset(name, shape=(), dtype=data.dtype,
                                    data=data, track_times=False)
    empty = 0 in data.shape
    d = group.create_dataset(
        name, shape=data.shape, dtype=data.dtype.newbyteorder("<"),
        data=data, track_times=False,
        maxshape=(tuple(None if n == 0 else n for n in data.shape)
                  if empty else None),
        chunks=(1,) * data.ndim if empty else None)
    for axis, length in enumerate(data.shape):
        s = scale(group, "mestra_%s_d%d" % (name, axis), length,
                  unlimited=(length == 0))
        d.dims[axis].attach_scale(s)
    return d


def affine_dict(keys=None, outputs=None):
    keys = AFFINE_KEYS if keys is None else keys
    outputs = AFFINE if outputs is None else outputs
    return {"keys": list(keys),
            "outputs": dict((k, dict(outputs[k])) for k in outputs)}


def evaluate_affine(entry, keys, values):
    """Section 27: y = A x + b, reshaped to `shape` in C order.

    Accumulated one key at a time in the declared order, with b added
    last, as section 27 says and as every implementation does. Not a
    matrix product: `A @ x` goes to whatever BLAS numpy was built
    against, which on most machines fuses each multiply and add into
    one rounding, and the expected values then differ in the last
    place from the ones the rule gives. Two numpy operations cannot
    be fused, so this is the rule exactly.
    """
    x = np.array([values[k] for k in keys], dtype="<f8")
    a = np.asarray(entry["A"], dtype="<f8")
    y = np.zeros(a.shape[0], dtype="<f8")
    for at in range(a.shape[1]):
        y += a[:, at] * x[at]
    y += np.asarray(entry["b"], dtype="<f8")
    return y.reshape([int(n) for n in entry["shape"]])


def affine_base(f, o):
    """The file of docs/example.md file two, with the deviations the
    callable error cases need behind named options.  With no options
    it is that file exactly."""
    g = o.get
    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", g("writer", WRITER))
    battr(f, "aligned", True)

    row = scale(f, "row", 0, unlimited=True)
    component_2 = scale(f, "component_2", 2)

    keys = f.create_group("keys")
    alpha = dataset(keys, "alpha", np.zeros(0), "<f8", [row], n_rows=0)
    sattr(alpha, "role", "condition")
    sattr(alpha, "units", "degree")
    fattr(alpha, "lower", -2.0)
    fattr(alpha, "upper", 10.0)

    mach = dataset(keys, "mach", np.zeros(0), "<f8", [row], n_rows=0)
    sattr(mach, "role", "condition")
    sattr(mach, "units", "1")
    fattr(mach, "lower", 0.1)
    fattr(mach, "upper", 0.9)

    scalars = f.create_group("scalars")
    cl = scalars.create_group("cl")
    sattr(cl, "units", "1")
    sattr(cl, "source", g("cl_source", "callable:m1"))
    sattr(cl, "output", "cl")
    if g("band", False):
        clb = scalars.create_group("cl_band")
        sattr(clb, "units", "1")
        sattr(clb, "source", "callable:m1")
        sattr(clb, "output", "cl")
        sattr(clb, "statistic", "band")
        sattr(clb, "of", "cl")

    sup = f.create_group("supports").create_group("s0")
    node, _cell = write_mesh_cells(sup, o)

    c = dataset(sup, "coordinates", BASE_COORDS, "<f8",
                [node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    node_arrays = sup.create_group("node_arrays")
    p = node_arrays.create_group("pressure")
    sattr(p, "role", "field")
    sattr(p, "varies", g("pressure_varies", "row"))
    sattr(p, "units", "Pa")
    iattr(p, "components", 1)
    sattr(p, "source", g("pressure_source", "callable:m1"))
    sattr(p, "output", "pressure")
    for aname, avalue in g("pressure_extra", []):
        sattr(p, aname, avalue)
    if g("band", False):
        pb = node_arrays.create_group("pressure_band")
        sattr(pb, "role", "field")
        sattr(pb, "varies", "row")
        sattr(pb, "units", "Pa")
        iattr(pb, "components", 1)
        sattr(pb, "source", "callable:m1")
        sattr(pb, "output", "pressure")
        sattr(pb, "statistic", "band")
        sattr(pb, "of", "pressure")

    m1 = f.create_group("callables").create_group("m1")
    if g("type", "affine") is not None:
        sattr(m1, "type", g("type", "affine"))
    sattr(m1, "repr", g("repr", "affine(mach, alpha -> cl, pressure)"))
    scale_keys = scale(m1, "mestra_keys_d0", len(AFFINE_KEYS))
    strings(m1, "keys", AFFINE_KEYS, scale_keys)
    if g("zero_d_key", False):
        codec_array(m1, "tolerance", np.float64(1e-9), zero_d=True)

    outputs = m1.create_group("outputs")
    for slot in sorted(AFFINE):
        gslot = outputs.create_group(slot)
        for name in ("A", "b", "shape"):
            codec_array(gslot, name, AFFINE[slot][name])
        if g("band", False):
            # Section 25: a number or a string is an attribute on the
            # enclosing group; the band itself is a dataset.
            codec_array(gslot, "uncertainty",
                        AFFINE_BAND[slot]["uncertainty"])
            fattr(gslot, "level", AFFINE_BAND[slot]["level"])
            sattr(gslot, "method", AFFINE_BAND[slot]["method"])
    return sup


AFFINE_AT = [{"mach": 0.5, "alpha": 4.0}]

#: The band of section 27's worked example: a constant per output.
AFFINE_BAND = {
    "cl": {"uncertainty": np.array([0.02]), "level": 0.95,
           "method": "constant band"},
    "pressure": {"uncertainty": np.array([0.05, 0.10, 0.15, 0.20, 0.25,
                                          0.30]),
                 "level": 0.68, "method": "constant band"},
}


def with_band(outputs, bands):
    """The affine outputs with the band keys merged in, as the codec
    round trip must return them."""
    out = {}
    for name in outputs:
        entry = dict(outputs[name])
        if name in bands:
            entry.update(bands[name])
        out[name] = entry
    return out


def affine_evaluation(cid, keys, outputs, slots, table, bands=None):
    """A worked evaluation of section 27 as an expected.json entry.
    `slots` maps the callable's output name to the slot path and
    `table` is the keys table, one dictionary per table row. The row
    index in each probe is the index into that table, not the index
    of any row stored in the file. `bands` maps an output name to the
    path of the band slot it also fills, whose value is the output's
    constant `uncertainty` in every row."""
    probes = []

    def add(path, r, y):
        if y.ndim == 0:
            probes.append({"slot": path, "row": r, "value": fnum(y)})
        else:
            for n in range(y.shape[0]):
                for c in range(y.shape[1]):
                    probes.append({"slot": path, "row": r, "node": n,
                                   "component": c,
                                   "value": fnum(y[n, c])})

    for output in sorted(slots):
        for r, at in enumerate(table):
            add(slots[output], r, evaluate_affine(outputs[output], keys, at))
    for output in sorted(bands or {}):
        shape = [int(n) for n in outputs[output]["shape"]]
        band = np.asarray(outputs[output]["uncertainty"]).reshape(shape)
        for r in range(len(table)):
            add(bands[output], r, band)
    return {"callable": cid,
            "keys": dict((k, [fnum(at[k]) for at in table])
                         for k in keys),
            "probes": probes}


# ------------------------------------------------------- the two files

def case_mesh_two_rows(f):
    mesh_base(f, {"writer": EXAMPLE_WRITER})
    return expect(
        "File one of docs/example.md, regenerated by this generator "
        "and byte-compared against the committed copy. Two rows of "
        "one parametric family on one mesh support.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/node_arrays/pressure", PRESSURE_2,
                  row=1, node=3, component=0),
            probe("/supports/s0/node_arrays/pressure", PRESSURE_2,
                  row=0, node=5, component=0),
            probe("/supports/s0/coordinates", MEMBER_COORDS,
                  instance=1, node=2, component=0),
            probe("/supports/s0/coordinates", MEMBER_COORDS,
                  instance=0, node=4, component=1),
            probe("/keys/mach", np.array([0.40, 0.80]), row=1),
            probe("/scalars/cl", np.array([0.25, 0.55]), row=0),
            probe("/supports/s0/cell_connectivity", CELL_CONNECTIVITY,
                  index=5),
            probe("/supports/s0/cell_arrays/region",
                  np.array([[0], [1]]), node=1, component=0),
        ])


def case_affine_zero_rows(f):
    affine_base(f, {"writer": EXAMPLE_WRITER})
    return expect(
        "File two of docs/example.md, regenerated by this generator "
        "and byte-compared against the committed copy. No rows; one "
        "affine callable serves a scalar slot and a node-array slot.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/coordinates", BASE_COORDS, node=4,
                  component=1),
            probe("/callables/m1/outputs/pressure/A",
                  AFFINE["pressure"]["A"], node=3, component=1),
        ],
        codec={"m1": tagged(affine_dict())},
        evaluation=[affine_evaluation(
            "m1", AFFINE_KEYS, AFFINE,
            {"cl": "/scalars/cl",
             "pressure": "/supports/s0/node_arrays/pressure"},
            AFFINE_AT)])


def case_dictionary_null(f):
    """The null sentinel of section 18, at the top of a dictionary and
    one level down. No other case carries it."""
    affine_base(f, {"type": "example"})
    m1 = f["callables/m1"]
    rattr(m1, "fallback", b"\0null")
    rattr(m1["outputs"], "retired", b"\0null")
    d = affine_dict()
    d["fallback"] = None
    d["outputs"]["retired"] = None
    return expect(
        "A dictionary holding the null sentinel at its top level and "
        "inside a nested dictionary. Both are values, not strings with a "
        "NUL in them, and both survive a rewrite.",
        support_ids={"s0": MESH_SID},
        codec={"m1": tagged(d)})


def case_dictionary_empty_arrays(f):
    """Empty arrays of more than one dimension inside a dictionary.
    Section 25 makes each zero-length axis unlimited, and E43 allows
    that axis and no other, so the axis of length 2 beside it is
    fixed."""
    affine_base(f, {"type": "example"})
    m1 = f["callables/m1"]
    empties = {"grid": np.zeros((2, 0), dtype="<f8"),
               "pairs": np.zeros((0, 3), dtype="<i8"),
               "flags": np.zeros((0,), dtype="<i4")}
    for name in sorted(empties):
        codec_array(m1, name, empties[name])
    d = affine_dict()
    d.update(empties)
    return expect(
        "A dictionary holding empty arrays of shape (2, 0), (0, 3) and "
        "(0,). Only the zero-length axes are unlimited, and a rewrite "
        "keeps the dtype, the shape and the maximum shape of each.",
        support_ids={"s0": MESH_SID},
        codec={"m1": tagged(d)})


#: A constant: an affine callable with no keys at all (section 27).
AFFINE_CONSTANT = {
    "cl": {"A": np.zeros((1, 0)), "b": np.array([0.05]),
           "shape": np.array([], dtype="<i8")},
}


def case_affine_no_keys(f):
    """An affine callable of no keys, so A has a zero-length axis and
    `keys` is an empty string dataset: section 27 makes it a string
    dataset and section 25 says how an empty one is written."""
    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    row = scale(f, "row", 0, unlimited=True)
    mach = dataset(f.create_group("keys"), "mach", np.zeros(0), "<f8",
                   [row], n_rows=0)
    sattr(mach, "role", "condition")
    sattr(mach, "units", "1")
    fattr(mach, "lower", 0.1)
    fattr(mach, "upper", 0.9)
    cl = f.create_group("scalars").create_group("cl")
    sattr(cl, "units", "1")
    sattr(cl, "source", "callable:m1")
    sattr(cl, "output", "cl")
    m1 = f.create_group("callables").create_group("m1")
    sattr(m1, "type", "affine")
    keys = m1.create_dataset("keys", shape=(0,), dtype=sdtype(1),
                             maxshape=(None,), chunks=(1,),
                             track_times=False)
    keys.dims[0].attach_scale(scale(m1, "mestra_keys_d0", 0,
                                    unlimited=True))
    out = m1.create_group("outputs").create_group("cl")
    for name in ("A", "b", "shape"):
        codec_array(out, name, AFFINE_CONSTANT["cl"][name])
    worked = affine_evaluation("m1", [], AFFINE_CONSTANT,
                               {"cl": "/scalars/cl"}, AFFINE_AT)
    # The file declares mach and the callable reads none of its keys,
    # so the table carries mach, which section 26 says the callable
    # ignores, and the table's one row is what gives cl one row.
    worked["keys"] = {"mach": [fnum(at["mach"]) for at in AFFINE_AT]}
    return expect(
        "An affine callable with no keys: a constant. keys is an empty "
        "fixed-length string dataset and A has shape (1, 0). Evaluated "
        "on a one-row table of the file's key mach, which it ignores, "
        "cl is b.",
        codec={"m1": tagged(affine_dict([], AFFINE_CONSTANT))},
        evaluation=[worked])


# ------------------------------------------------- the five mappings

def pressures(n):
    """n rows of a one-component node field, all values distinct."""
    return np.array([[[100.0 * (i + 1) + j + 1] for j in range(6)]
                     for i in range(n)])


def case_family_static(f):
    """Mapping 1: a parametric family, static, aligned."""
    n_rows = 3
    coords = np.stack([BASE_COORDS * np.array([s, 1.0])
                       for s in (1.0, 1.25, 1.5)])
    cad_face_id = np.array([[12], [12], [17], [17], [23], [23]])
    topo_group = np.array([[0], [1]])
    total_length = np.array([1.0, 1.2, 1.4])
    half_angle = np.array([10.0, 12.0, 14.0])
    nose_radius = np.array([0.10, 0.12, 0.14])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    sattr(f, "generalisation_group", "member")

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)
    group_member = scale(f, "group_member", 3)
    category_member = scale(f, "category_member", 3)
    category_status = scale(f, "category_status", 3)
    category_topo_group = scale(f, "category_topo_group", 2)

    categories = f.create_group("categories")
    strings(categories, "member", ["m000", "m001", "m002"],
            category_member)
    strings(categories, "status",
            ["converged", "failed", "partial"], category_status)
    strings(categories, "topo_group", ["nose", "flank"],
            category_topo_group)

    keys = f.create_group("keys")
    k = dataset(keys, "half_angle", half_angle, "<f8", [row],
                n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "degree")
    fattr(k, "lower", 10.0)
    fattr(k, "upper", 14.0)
    k = dataset(keys, "member", [0, 1, 2], "<i4", [row], n_rows=n_rows)
    sattr(k, "role", "group")
    sattr(k, "category", "member")
    k = dataset(keys, "nose_radius", nose_radius, "<f8", [row],
                n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "m")
    k = dataset(keys, "status", [0, 0, 0], "<i4", [row], n_rows=n_rows)
    sattr(k, "role", "status")
    sattr(k, "category", "status")
    k = dataset(keys, "total_length", total_length, "<f8", [row],
                n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "m")

    sup = f.create_group("supports").create_group("s0")
    node, cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                            CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", coords, "<f8",
                [group_member, node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "group:member")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    na = sup.create_group("node_arrays")
    a = dataset(na, "cad_face_id", cad_face_id, "<i4",
                [node, component_1])
    sattr(a, "role", "label")
    sattr(a, "varies", "none")
    iattr(a, "components", 1)
    sattr(a, "source", "data")

    ca = sup.create_group("cell_arrays")
    a = dataset(ca, "topo_group", topo_group, "<i4",
                [cell, component_1])
    sattr(a, "role", "label")
    sattr(a, "varies", "none")
    iattr(a, "components", 1)
    sattr(a, "source", "data")
    sattr(a, "category", "topo_group")

    return expect(
        "Mapping 1 at toy size: a parametric family of three members "
        "on one shared mesh, with group-varying coordinates, a "
        "per-node label with no category table and a per-cell label "
        "with one.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/coordinates", coords, instance=2,
                  node=1, component=0),
            probe("/supports/s0/coordinates", coords, instance=1,
                  node=4, component=1),
            probe("/supports/s0/node_arrays/cad_face_id", cad_face_id,
                  node=4, component=0),
            probe("/supports/s0/cell_arrays/topo_group", topo_group,
                  node=1, component=0),
            probe("/keys/total_length", total_length, row=2),
            probe("/keys/half_angle", half_angle, row=1),
        ])


def case_cascade_varying_geometry(f):
    """Mapping 2: varying geometry on one shared connectivity."""
    n_rows = 4
    nan = float("nan")
    coords = np.stack([BASE_COORDS * np.array([1.0 + 0.1 * i, 1.0])
                       for i in range(n_rows)])
    mach = np.array([[[0.30 + 0.01 * (10 * i + j)] for j in range(6)]
                     for i in range(n_rows)])
    mach[3, :, 0] = nan
    nut = np.array([[[1e-5 * (10 * i + j + 1)] for j in range(6)]
                    for i in range(n_rows)])
    angle_in = np.array([30.0, 32.0, 34.0, 36.0])
    mach_out = np.array([0.70, 0.75, 0.80, 0.85])
    q = np.array([1.5, 1.6, 1.7, nan])
    power = np.array([2.5, 2.6, 2.7, nan])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    sattr(f, "generalisation_group", "case")

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)
    group_case = scale(f, "group_case", 4)
    category_case = scale(f, "category_case", 4)
    category_split = scale(f, "category_split", 3)
    category_status = scale(f, "category_status", 3)

    categories = f.create_group("categories")
    strings(categories, "case", ["c000", "c001", "c002", "c003"],
            category_case)
    strings(categories, "split", ["train", "validation", "test"],
            category_split)
    strings(categories, "status",
            ["converged", "failed", "partial"], category_status)

    keys = f.create_group("keys")
    k = dataset(keys, "angle_in", angle_in, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "degree")
    fattr(k, "lower", 30.0)
    fattr(k, "upper", 36.0)
    k = dataset(keys, "case", [0, 1, 2, 3], "<i4", [row], n_rows=n_rows)
    sattr(k, "role", "group")
    sattr(k, "category", "case")
    k = row_strings(keys, "id", ["s000", "s001", "s002", "s003"], row,
                    n_rows)
    sattr(k, "role", "id")
    k = dataset(keys, "mach_out", mach_out, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")
    fattr(k, "lower", 0.70)
    fattr(k, "upper", 0.85)
    k = dataset(keys, "split", [0, 0, 1, 2], "<i4", [row], n_rows=n_rows)
    sattr(k, "role", "split")
    sattr(k, "category", "split")
    k = dataset(keys, "status", [0, 0, 0, 2], "<i4", [row],
                n_rows=n_rows)
    sattr(k, "role", "status")
    sattr(k, "category", "status")

    scalars = f.create_group("scalars")
    s = dataset(scalars, "power", power, "<f8", [row], n_rows=n_rows)
    sattr(s, "units", "W")
    sattr(s, "source", "data")
    s = dataset(scalars, "q", q, "<f8", [row], n_rows=n_rows)
    sattr(s, "units", "W m-2")
    sattr(s, "source", "data")

    sup = f.create_group("supports").create_group("s0")
    node, cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                            CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", coords, "<f8",
                [row, node, component_2], n_rows=n_rows)
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "row")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    na = sup.create_group("node_arrays")
    a = dataset(na, "mach", mach, "<f8", [row, node, component_1],
                n_rows=n_rows)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "1")
    iattr(a, "components", 1)
    sattr(a, "source", "data")
    a = dataset(na, "nut", nut, "<f8", [row, node, component_1],
                n_rows=n_rows)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "m2 s-1")
    iattr(a, "components", 1)
    sattr(a, "source", "data")

    return expect(
        "Mapping 2 at toy size: four rows of varying geometry on one "
        "shared connectivity, two fields, two scalars, a split key "
        "and a status key whose partial row holds NaN.",
        warnings=["W02", "W03"],
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/node_arrays/mach", mach, row=1, node=4,
                  component=0),
            probe("/supports/s0/node_arrays/mach", mach, row=3, node=0,
                  component=0),
            probe("/supports/s0/node_arrays/nut", nut, row=2, node=5,
                  component=0),
            probe("/supports/s0/coordinates", coords, row=3, node=2,
                  component=0),
            probe("/scalars/q", q, row=3),
            probe("/scalars/power", power, row=1),
            probe("/keys/angle_in", angle_in, row=2),
        ])


def case_scalars_only(f):
    """Mapping 3: keys and scalars, no support at all."""
    n_rows = 6
    p1 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    p2 = np.array([0.5, 0.5, 0.6, 0.6, 0.7, 0.7])
    incidence = np.array([2.0, 8.0, 2.0, 8.0, 2.0, 8.0])
    cl = np.array([0.21, 0.82, 0.24, 0.88, 0.27, 0.93])
    cd = np.array([0.011, 0.031, 0.012, 0.034, 0.013, 0.037])
    cm = np.array([-0.05, -0.09, -0.06, -0.10, -0.07, -0.11])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    sattr(f, "generalisation_group", "geometry")

    row = scale(f, "row", n_rows, unlimited=True)
    group_geometry = scale(f, "group_geometry", 3)
    category_geometry = scale(f, "category_geometry", 3)
    category_split = scale(f, "category_split", 2)

    categories = f.create_group("categories")
    strings(categories, "geometry", ["g000", "g001", "g002"],
            category_geometry)
    strings(categories, "split", ["train", "test"], category_split)

    keys = f.create_group("keys")
    k = dataset(keys, "geometry", [0, 0, 1, 1, 2, 2], "<i4", [row],
                n_rows=n_rows)
    sattr(k, "role", "group")
    sattr(k, "category", "geometry")
    k = dataset(keys, "incidence", incidence, "<f8", [row],
                n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "degree")
    fattr(k, "lower", 2.0)
    fattr(k, "upper", 8.0)
    k = dataset(keys, "p1", p1, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "1")
    k = dataset(keys, "p2", p2, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "1")
    k = dataset(keys, "split", [0, 0, 0, 0, 1, 1], "<i4", [row],
                n_rows=n_rows)
    sattr(k, "role", "split")
    sattr(k, "category", "split")

    scalars = f.create_group("scalars")
    for name, values in (("cd", cd), ("cl", cl), ("cm", cm)):
        s = dataset(scalars, name, values, "<f8", [row], n_rows=n_rows)
        sattr(s, "units", "1")
        sattr(s, "source", "data")

    return expect(
        "Mapping 3 at toy size: three geometries by two incidences, "
        "three scalars, no support and no array. The split moves "
        "whole geometries, so it does not leak the unit of "
        "generalisation.",
        probes=[
            probe("/scalars/cl", cl, row=3),
            probe("/scalars/cd", cd, row=5),
            probe("/scalars/cm", cm, row=0),
            probe("/keys/incidence", incidence, row=4),
        ])


def case_transient_fixed_mesh(f):
    """Mapping 4: time inside a parametric family, fixed mesh."""
    n_rows = 5
    run = [0, 0, 0, 1, 1]
    t = np.array([0.0, 0.1, 0.3, 0.0, 0.25])
    diffusivity = np.array([0.01, 0.01, 0.01, 0.02, 0.02])
    amplitude = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    u = np.array([[[300.0 + 10.0 * i + j] for j in range(6)]
                  for i in range(n_rows)])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    sattr(f, "generalisation_group", "run")

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)
    group_run = scale(f, "group_run", 2)
    category_run = scale(f, "category_run", 2)

    categories = f.create_group("categories")
    strings(categories, "run", ["r000", "r001"], category_run)

    keys = f.create_group("keys")
    k = dataset(keys, "amplitude", amplitude, "<f8", [row],
                n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "1")
    k = dataset(keys, "diffusivity", diffusivity, "<f8", [row],
                n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "m2 s-1")
    fattr(k, "lower", 0.01)
    fattr(k, "upper", 0.02)
    k = dataset(keys, "run", run, "<i4", [row], n_rows=n_rows)
    sattr(k, "role", "group")
    sattr(k, "category", "run")
    k = dataset(keys, "t", t, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "time")
    sattr(k, "units", "s")
    sattr(k, "trajectory_group", "run")

    sup = f.create_group("supports").create_group("s0")
    node, cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                            CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", BASE_COORDS, "<f8",
                [node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    na = sup.create_group("node_arrays")
    a = dataset(na, "u", u, "<f8", [row, node, component_1],
                n_rows=n_rows)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "K")
    iattr(a, "components", 1)
    sattr(a, "source", "data")

    return expect(
        "Mapping 4 at toy size: two runs of three and two steps on a "
        "fixed mesh, a time key with a trajectory group, and "
        "irregular step counts and step sizes.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/node_arrays/u", u, row=3, node=2,
                  component=0),
            probe("/supports/s0/node_arrays/u", u, row=1, node=5,
                  component=0),
            probe("/keys/t", t, row=4),
            probe("/supports/s0/coordinates", BASE_COORDS, node=3,
                  component=1),
        ])


AXIS_COORDS = np.array([[0.00], [0.25], [0.50], [0.75], [1.00]])
AXIS_SID = support_id(5, axis_coordinates=AXIS_COORDS)


def case_axis_signature(f):
    """Mapping 5: a one-dimensional field on an axis support."""
    n_rows = 4
    area1 = np.array([1.0, 1.0, 2.0, 2.0])
    mach = np.array([1.4, 1.6, 1.4, 1.6])
    altitude = np.array([14000.0, 14000.0, 16000.0, 16000.0])
    overpressure = np.array([[[10.0 * (i + 1) + j] for j in range(5)]
                             for i in range(n_rows)])
    loudness = np.array([82.0, 84.0, 86.0, 88.0])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    sattr(f, "generalisation_group", "design")

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    group_design = scale(f, "group_design", 2)
    category_design = scale(f, "category_design", 2)

    categories = f.create_group("categories")
    strings(categories, "design", ["d000", "d001"], category_design)

    keys = f.create_group("keys")
    k = dataset(keys, "altitude", altitude, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "m")
    fattr(k, "lower", 14000.0)
    fattr(k, "upper", 16000.0)
    k = dataset(keys, "area1", area1, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "m2")
    k = dataset(keys, "design", [0, 0, 1, 1], "<i4", [row],
                n_rows=n_rows)
    sattr(k, "role", "group")
    sattr(k, "category", "design")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")
    fattr(k, "lower", 1.4)
    fattr(k, "upper", 1.6)

    scalars = f.create_group("scalars")
    s = dataset(scalars, "loudness", loudness, "<f8", [row],
                n_rows=n_rows)
    sattr(s, "units", "1")
    sattr(s, "source", "data")

    sup = f.create_group("supports").create_group("s0")
    node = scale(sup, "node", 5)
    sattr(sup, "kind", "axis")
    iattr(sup, "n_nodes", 5)
    iattr(sup, "n_cells", 0)
    sattr(sup, "support_id", AXIS_SID)
    c = dataset(sup, "coordinates", AXIS_COORDS, "<f8",
                [node, component_1])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "s")
    iattr(c, "components", 1)
    sattr(c, "source", "data")

    na = sup.create_group("node_arrays")
    a = dataset(na, "overpressure", overpressure, "<f8",
                [row, node, component_1], n_rows=n_rows)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "Pa")
    iattr(a, "components", 1)
    sattr(a, "source", "data")

    return expect(
        "Mapping 5 at toy size: a ground signature over a five-sample "
        "time axis. The support is an axis, its coordinate is part of "
        "its identity, and the field over it is not a trajectory.",
        support_ids={"s0": AXIS_SID},
        probes=[
            probe("/supports/s0/node_arrays/overpressure", overpressure,
                  row=3, node=1, component=0),
            probe("/supports/s0/node_arrays/overpressure", overpressure,
                  row=1, node=4, component=0),
            probe("/supports/s0/coordinates", AXIS_COORDS, node=3,
                  component=0),
            probe("/scalars/loudness", loudness, row=2),
        ])


# ------------------------------------------------ more than one support

S1_NODES = 4
S1_TYPES = np.array([9], dtype="<u1")
S1_OFFSETS = np.array([0, 4], dtype="<i8")
S1_CONN = np.array([0, 1, 3, 2], dtype="<i8")
S1_COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
S1_SID = support_id(S1_NODES, S1_TYPES, S1_OFFSETS, S1_CONN)


def two_support_base(f, o):
    """Two mesh supports of different size.  Only arrays that do not
    vary along the row live on the supports, because section 22 does
    not say what a row-varying array on support A should hold for a
    row that sits on support B."""
    g = o.get
    n_rows = g("n_rows", 3)
    mach = np.array([0.40 + 0.10 * i for i in range(n_rows)])
    cl = np.array([0.25 + 0.10 * i for i in range(n_rows)])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", g("aligned", False))

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")

    scalars = f.create_group("scalars")
    s = dataset(scalars, "cl", cl, "<f8", [row], n_rows=n_rows)
    sattr(s, "units", "1")
    sattr(s, "source", "data")

    if g("row_support", None) is not None:
        dataset(f, "row_support", g("row_support"), "<i4", [row],
                n_rows=n_rows)

    supports = f.create_group("supports")
    sup0 = supports.create_group("s0")
    node0, cell0 = mesh_cells(sup0, N_NODES, CELL_TYPES, CELL_OFFSETS,
                              CELL_CONNECTIVITY)
    c = dataset(sup0, "coordinates", BASE_COORDS, "<f8",
                [node0, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")
    a = dataset(sup0.create_group("cell_arrays"), "region", [[0], [1]],
                "<i4", [cell0, component_1])
    sattr(a, "role", "label")
    sattr(a, "varies", "none")
    iattr(a, "components", 1)
    sattr(a, "source", "data")

    sup1 = supports.create_group("s1")
    node1, cell1 = mesh_cells(sup1, S1_NODES, S1_TYPES, S1_OFFSETS,
                              S1_CONN)
    c = dataset(sup1, "coordinates", S1_COORDS, "<f8",
                [node1, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")
    a = dataset(sup1.create_group("cell_arrays"), "region", [[0]],
                "<i4", [cell1, component_1])
    sattr(a, "role", "label")
    sattr(a, "varies", "none")
    iattr(a, "components", 1)
    sattr(a, "source", "data")
    return {"mach": mach, "cl": cl}


# ---------------------------------------------------- an axis support

E_AXIS_COORDS = np.array([[0.0], [0.25], [0.50], [0.75]])
E_AXIS_SID = support_id(4, axis_coordinates=E_AXIS_COORDS)


def axis_base(f, o):
    """A small axis support, with the two deviations the E35 and E38
    cases need behind named options."""
    g = o.get
    n_rows = g("n_rows", 2)
    mach = np.array([0.40 + 0.10 * i for i in range(n_rows)])
    field = np.array([[[10.0 * (i + 1) + j] for j in range(4)]
                      for i in range(n_rows)])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")

    sup = f.create_group("supports").create_group("s0")
    node = scale(sup, "node", 4)
    cell = scale(sup, "cell", 1) if g("stray_cells", False) else None
    sattr(sup, "kind", "axis")
    iattr(sup, "n_nodes", 4)
    iattr(sup, "n_cells", 0)
    sattr(sup, "support_id", E_AXIS_SID)
    if g("coords_varies", "none") == "none":
        c = dataset(sup, "coordinates", E_AXIS_COORDS, "<f8",
                    [node, component_1])
        sattr(c, "varies", "none")
    else:
        c = dataset(sup, "coordinates",
                    E_AXIS_COORDS.reshape(1, 4, 1), "<f8",
                    [row, node, component_1], n_rows=n_rows)
        sattr(c, "varies", "row")
    sattr(c, "role", "coordinates")
    sattr(c, "units", "s")
    iattr(c, "components", 1)
    sattr(c, "source", "data")
    if cell is not None:
        dataset(sup, "cell_types", np.array([1], dtype="<u1"), "<u1",
                [cell])

    a = dataset(sup.create_group("node_arrays"), "overpressure", field,
                "<f8", [row, node, component_1], n_rows=n_rows)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "Pa")
    iattr(a, "components", 1)
    sattr(a, "source", "data")
    return field


# ------------------------------------------------------ feature cases

def case_affine_with_rows(f):
    """Section 22: rows and callable slots in one file."""
    n_rows = 2
    mach = np.array([0.10, 0.90])
    alpha = np.array([-2.0, 10.0])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", n_rows, unlimited=True)
    component_2 = scale(f, "component_2", 2)

    keys = f.create_group("keys")
    k = dataset(keys, "alpha", alpha, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "degree")
    fattr(k, "lower", -2.0)
    fattr(k, "upper", 10.0)
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")
    fattr(k, "lower", 0.1)
    fattr(k, "upper", 0.9)

    cl = f.create_group("scalars").create_group("cl")
    sattr(cl, "units", "1")
    sattr(cl, "source", "callable:m1")
    sattr(cl, "output", "cl")

    sup = f.create_group("supports").create_group("s0")
    node, _cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                             CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", BASE_COORDS, "<f8",
                [node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    p = sup.create_group("node_arrays").create_group("pressure")
    sattr(p, "role", "field")
    sattr(p, "varies", "row")
    sattr(p, "units", "Pa")
    iattr(p, "components", 1)
    sattr(p, "source", "callable:m1")
    sattr(p, "output", "pressure")

    m1 = f.create_group("callables").create_group("m1")
    sattr(m1, "type", "affine")
    sattr(m1, "repr", "affine(mach, alpha -> cl, pressure)")
    scale_keys = scale(m1, "mestra_keys_d0", len(AFFINE_KEYS))
    strings(m1, "keys", AFFINE_KEYS, scale_keys)
    outputs = m1.create_group("outputs")
    for slot in sorted(AFFINE):
        gslot = outputs.create_group(slot)
        for name in ("A", "b", "shape"):
            codec_array(gslot, name, AFFINE[slot][name])

    return expect(
        "Two rows of the training design together with slots served "
        "by a callable. What decides whether a slot holds data is its "
        "source and never the row count.",
        support_ids={"s0": MESH_SID},
        probes=[probe("/keys/mach", mach, row=0),
                probe("/keys/alpha", alpha, row=1)],
        codec={"m1": tagged(affine_dict())},
        evaluation=[affine_evaluation(
            "m1", AFFINE_KEYS, AFFINE,
            {"cl": "/scalars/cl",
             "pressure": "/supports/s0/node_arrays/pressure"},
            [{"mach": float(mach[i]), "alpha": float(alpha[i])}
             for i in range(n_rows)])])


TWO_SLOT_KEYS = ["mach"]
TWO_SLOT = {
    "heat_flux": {
        "A": np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]]),
        "b": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "shape": np.array([6, 1], dtype="<i8"),
    },
    "pressure": {
        "A": np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        "b": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "shape": np.array([6, 1], dtype="<i8"),
    },
}
TWO_SLOT_AT = [{"mach": 0.5}, {"mach": 0.8}]


def case_callable_two_slots(f):
    """One callable serving two node-array slots through `output`."""
    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", 0, unlimited=True)
    component_2 = scale(f, "component_2", 2)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", np.zeros(0), "<f8", [row], n_rows=0)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")
    fattr(k, "lower", 0.1)
    fattr(k, "upper", 0.9)

    f.create_group("scalars")
    sup = f.create_group("supports").create_group("s0")
    node, _cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                             CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", BASE_COORDS, "<f8",
                [node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    na = sup.create_group("node_arrays")
    for slot, units in (("heat_flux", "W m-2"), ("pressure", "Pa")):
        p = na.create_group(slot)
        sattr(p, "role", "field")
        sattr(p, "varies", "row")
        sattr(p, "units", units)
        iattr(p, "components", 1)
        sattr(p, "source", "callable:m2")
        sattr(p, "output", slot)

    m2 = f.create_group("callables").create_group("m2")
    sattr(m2, "type", "affine")
    sattr(m2, "repr", "affine(mach -> heat_flux, pressure)")
    scale_keys = scale(m2, "mestra_keys_d0", len(TWO_SLOT_KEYS))
    strings(m2, "keys", TWO_SLOT_KEYS, scale_keys)
    outputs = m2.create_group("outputs")
    for slot in sorted(TWO_SLOT):
        gslot = outputs.create_group(slot)
        for name in ("A", "b", "shape"):
            codec_array(gslot, name, TWO_SLOT[slot][name])

    return expect(
        "No rows; one affine callable fills two node-array slots on "
        "one support, each naming its own output.",
        support_ids={"s0": MESH_SID},
        probes=[probe("/callables/m2/outputs/heat_flux/b",
                      TWO_SLOT["heat_flux"]["b"], node=4)],
        codec={"m2": tagged(affine_dict(TWO_SLOT_KEYS, TWO_SLOT))},
        evaluation=[affine_evaluation(
            "m2", TWO_SLOT_KEYS, TWO_SLOT,
            dict((s, "/supports/s0/node_arrays/" + s)
                 for s in TWO_SLOT),
            TWO_SLOT_AT)])


COORD_MODEL_KEYS = ["mach"]
#: A geometry that stretches along x with mach, x = x0 (1 + mach), and
#: a pressure on it; the coordinates are the model's output like any
#: other, so a family of meshes can be a zero-row file (section 10).
COORD_MODEL = {
    "coordinates": {
        "A": np.array([[0.0], [0.0], [1.0], [0.0], [2.0], [0.0],
                       [0.0], [0.0], [1.0], [0.0], [2.0], [0.0]]),
        "b": BASE_COORDS.ravel(),
        "shape": np.array([6, 2], dtype="<i8"),
    },
    "pressure": {
        "A": np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        "b": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "shape": np.array([6, 1], dtype="<i8"),
    },
}
COORD_MODEL_AT = [{"mach": 0.5}, {"mach": 0.8}]


def case_callable_coordinates(f):
    """A support whose coordinates a callable serves: a model of the
    geometry itself, beside a field on it."""
    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    scale(f, "row", 0, unlimited=True)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", np.zeros(0), "<f8",
                [f["row"]], n_rows=0)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")
    fattr(k, "lower", 0.1)
    fattr(k, "upper", 0.9)

    f.create_group("scalars")
    sup = f.create_group("supports").create_group("s0")
    mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS, CELL_CONNECTIVITY)
    # Section 19: a slot a callable serves is a group with the slot's
    # attributes and no data, and a support's coordinates are a slot.
    c = sup.create_group("coordinates")
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "row")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "callable:m3")
    sattr(c, "output", "coordinates")

    na = sup.create_group("node_arrays")
    p = na.create_group("pressure")
    sattr(p, "role", "field")
    sattr(p, "varies", "row")
    sattr(p, "units", "Pa")
    iattr(p, "components", 1)
    sattr(p, "source", "callable:m3")
    sattr(p, "output", "pressure")

    m3 = f.create_group("callables").create_group("m3")
    sattr(m3, "type", "affine")
    sattr(m3, "repr", "affine(mach -> coordinates, pressure)")
    scale_keys = scale(m3, "mestra_keys_d0", len(COORD_MODEL_KEYS))
    strings(m3, "keys", COORD_MODEL_KEYS, scale_keys)
    outputs = m3.create_group("outputs")
    for slot in sorted(COORD_MODEL):
        gslot = outputs.create_group(slot)
        for name in ("A", "b", "shape"):
            codec_array(gslot, name, COORD_MODEL[slot][name])

    return expect(
        "No rows; one affine callable serves the support's coordinates "
        "and a node-array slot on them. A geometry is an output like "
        "any other: the mesh support carries its cells and its node "
        "count, and its coordinates are a group with the slot's "
        "attributes and no data until the file is evaluated.",
        support_ids={"s0": MESH_SID},
        probes=[probe("/callables/m3/outputs/coordinates/b",
                      COORD_MODEL["coordinates"]["b"], node=7)],
        codec={"m3": tagged(affine_dict(COORD_MODEL_KEYS, COORD_MODEL))},
        evaluation=[affine_evaluation(
            "m3", COORD_MODEL_KEYS, COORD_MODEL,
            {"coordinates": "/supports/s0/coordinates",
             "pressure": "/supports/s0/node_arrays/pressure"},
            COORD_MODEL_AT)])


def case_two_supports_unaligned(f):
    """Two supports, so the file is not aligned."""
    d = two_support_base(f, {"n_rows": 3, "aligned": False,
                             "row_support": [0, 0, 1]})
    return expect(
        "Two mesh supports of different size with aligned = false and "
        "a /row_support column. Index-aligned operations are not "
        "available on this file.",
        warnings=["W05"],
        support_ids={"s0": MESH_SID, "s1": S1_SID},
        probes=[
            probe("/row_support", np.array([0, 0, 1]), row=2),
            probe("/supports/s1/coordinates", S1_COORDS, node=3,
                  component=1),
            probe("/supports/s0/coordinates", BASE_COORDS, node=2,
                  component=0),
            probe("/scalars/cl", d["cl"], row=1),
        ])


def case_draws_and_summaries(f):
    """A draw dimension with the summaries derived from it."""
    n_rows, n_draws = 2, 3
    draws = np.array([[[[100.0 * (r + 1) + 10.0 * (d + 1) + n]
                        for n in range(6)]
                       for d in range(n_draws)]
                      for r in range(n_rows)])
    mean = draws.mean(axis=1)
    std = draws.std(axis=1, ddof=0)
    q90 = np.quantile(draws, 0.9, axis=1)
    mach = np.array([0.40, 0.80])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)
    draw_3 = scale(f, "draw_3", n_draws)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")
    fattr(k, "lower", 0.40)
    fattr(k, "upper", 0.80)

    sup = f.create_group("supports").create_group("s0")
    node, _cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                             CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", BASE_COORDS, "<f8",
                [node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    na = sup.create_group("node_arrays")
    a = dataset(na, "pressure", draws, "<f8",
                [row, draw_3, node, component_1], n_rows=n_rows)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "Pa")
    iattr(a, "components", 1)
    sattr(a, "source", "data")
    sattr(a, "statistic", "draw")
    for name, values, stat in (("pressure_mean", mean, "mean"),
                               ("pressure_q90", q90, "quantile"),
                               ("pressure_std", std, "std")):
        a = dataset(na, name, values, "<f8", [row, node, component_1],
                    n_rows=n_rows)
        sattr(a, "role", "field")
        sattr(a, "varies", "row")
        sattr(a, "units", "Pa")
        iattr(a, "components", 1)
        sattr(a, "source", "data")
        sattr(a, "statistic", stat)
        sattr(a, "of", "pressure")
        if stat == "quantile":
            fattr(a, "quantile", 0.9)

    return expect(
        "Three joint draws of one field, with the mean, the standard "
        "deviation and the 0.9 quantile stored beside them as "
        "statistics of it. The draw slot is the base quantity, so it "
        "carries no `of`.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/node_arrays/pressure", draws, row=1,
                  draw=2, node=4, component=0),
            probe("/supports/s0/node_arrays/pressure", draws, row=0,
                  draw=1, node=5, component=0),
            probe("/supports/s0/node_arrays/pressure_mean", mean,
                  row=1, node=3, component=0),
            probe("/supports/s0/node_arrays/pressure_std", std, row=1,
                  node=3, component=0),
            probe("/supports/s0/node_arrays/pressure_q90", q90, row=0,
                  node=2, component=0),
        ])


def case_labels_tables(f):
    """Labels with and without a category table."""
    n_rows = 2
    mach = np.array([0.40, 0.80])
    cad_face_id = np.array([[12], [12], [17], [17], [23], [23]])
    region = np.array([[0], [1]])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)
    category_region = scale(f, "category_region", 2)

    categories = f.create_group("categories")
    strings(categories, "region", ["inlet", "outlet"], category_region)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")

    sup = f.create_group("supports").create_group("s0")
    node, cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                            CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", BASE_COORDS, "<f8",
                [node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    na = sup.create_group("node_arrays")
    a = dataset(na, "cad_face_id", cad_face_id, "<i8",
                [node, component_1])
    sattr(a, "role", "label")
    sattr(a, "varies", "none")
    iattr(a, "components", 1)
    sattr(a, "source", "data")

    ca = sup.create_group("cell_arrays")
    a = dataset(ca, "region", region, "<i4", [cell, component_1])
    sattr(a, "role", "label")
    sattr(a, "varies", "none")
    iattr(a, "components", 1)
    sattr(a, "source", "data")
    sattr(a, "category", "region")

    return expect(
        "Two labels on one support: a per-cell label that names a "
        "category table and a per-node label whose values are their "
        "own categories and do not start at zero.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/node_arrays/cad_face_id", cad_face_id,
                  node=5, component=0),
            probe("/supports/s0/cell_arrays/region", region, node=1,
                  component=0),
        ])


def case_family_with_time(f):
    """Group-varying coordinates and a time key: rows are members
    times steps."""
    n_rows = 6
    member = [0, 0, 0, 1, 1, 1]
    t = np.array([0.0, 0.5, 1.5, 0.0, 0.5, 1.5])
    span = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5])
    u = np.array([[[300.0 + 10.0 * i + j] for j in range(6)]
                  for i in range(n_rows)])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    sattr(f, "generalisation_group", "member")

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)
    group_member = scale(f, "group_member", 2)
    category_member = scale(f, "category_member", 2)

    categories = f.create_group("categories")
    strings(categories, "member", ["wing_a", "wing_b"],
            category_member)

    keys = f.create_group("keys")
    k = dataset(keys, "member", member, "<i4", [row], n_rows=n_rows)
    sattr(k, "role", "group")
    sattr(k, "category", "member")
    k = dataset(keys, "span", span, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "design")
    sattr(k, "units", "m")
    k = dataset(keys, "t", t, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "time")
    sattr(k, "units", "s")
    sattr(k, "trajectory_group", "member")

    sup = f.create_group("supports").create_group("s0")
    node, _cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                             CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", MEMBER_COORDS, "<f8",
                [group_member, node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "group:member")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    a = dataset(sup.create_group("node_arrays"), "u", u, "<f8",
                [row, node, component_1], n_rows=n_rows)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "K")
    iattr(a, "components", 1)
    sattr(a, "source", "data")

    return expect(
        "Two members by three steps: the geometry varies along the "
        "family and the field varies along the row, so the two "
        "leading dimensions differ within one file.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/coordinates", MEMBER_COORDS,
                  instance=1, node=2, component=0),
            probe("/supports/s0/node_arrays/u", u, row=4, node=1,
                  component=0),
            probe("/keys/t", t, row=5),
        ])


def case_derived_displacement(f):
    """A derived array with derived_from, recipe and reference."""
    n_rows = 2
    mach = np.array([0.40, 0.80])
    displacement = MEMBER_COORDS - MEMBER_COORDS[0]

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    sattr(f, "generalisation_group", "member")

    row = scale(f, "row", n_rows, unlimited=True)
    component_2 = scale(f, "component_2", 2)
    group_member = scale(f, "group_member", 2)
    category_member = scale(f, "category_member", 2)

    categories = f.create_group("categories")
    strings(categories, "member", ["wing_a", "wing_b"],
            category_member)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")
    k = dataset(keys, "member", [0, 1], "<i4", [row], n_rows=n_rows)
    sattr(k, "role", "group")
    sattr(k, "category", "member")

    sup = f.create_group("supports").create_group("s0")
    node, _cell = mesh_cells(sup, N_NODES, CELL_TYPES, CELL_OFFSETS,
                             CELL_CONNECTIVITY)
    c = dataset(sup, "coordinates", MEMBER_COORDS, "<f8",
                [group_member, node, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "group:member")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")

    a = dataset(sup.create_group("node_arrays"), "displacement",
                displacement, "<f8",
                [group_member, node, component_2])
    sattr(a, "role", "derived")
    sattr(a, "varies", "group:member")
    sattr(a, "units", "m")
    iattr(a, "components", 2)
    sattr(a, "source", "data")
    sattr(a, "derived_from", "coordinates")
    sattr(a, "recipe", "minus reference")
    sattr(a, "reference", "group:member=wing_a")

    return expect(
        "A persisted displacement as a derived array over "
        "group-varying coordinates, naming what it was computed "
        "from, the operation, and the instance it is measured "
        "against.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/node_arrays/displacement",
                  displacement, instance=1, node=2, component=0),
            probe("/supports/s0/node_arrays/displacement",
                  displacement, instance=1, node=4, component=1),
            probe("/supports/s0/coordinates", MEMBER_COORDS,
                  instance=1, node=2, component=0),
        ])


NONE_SID = support_id(0)


def case_support_kind_none(f):
    """A support of kind none, whose digest section 24 works out."""
    n_rows = 2
    mach = np.array([0.40, 0.80])
    cl = np.array([0.25, 0.55])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", n_rows, unlimited=True)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")

    scalars = f.create_group("scalars")
    s = dataset(scalars, "cl", cl, "<f8", [row], n_rows=n_rows)
    sattr(s, "units", "1")
    sattr(s, "source", "data")

    sup = f.create_group("supports").create_group("s0")
    sattr(sup, "kind", "none")
    iattr(sup, "n_nodes", 0)
    iattr(sup, "n_cells", 0)
    sattr(sup, "support_id", NONE_SID)

    return expect(
        "A support of kind none: no nodes, no cells, no coordinates "
        "and no node dimension, carrying the worked digest of "
        "section 24.",
        support_ids={"s0": NONE_SID},
        probes=[probe("/scalars/cl", cl, row=1),
                probe("/keys/mach", mach, row=0)])


def case_two_supports_row_varying(f, served=False):
    """Section 22: a row-varying array in an unaligned file holds one
    entry per row referencing its support, in the file's row order,
    over the support-local `row` dimension of section 21.

    With `served`, the field on s1 is a slot a callable serves, which
    holds no data. s1 still carries an array with varies = row, so it
    still carries its own `row` scale (section 21)."""
    n_rows = 3
    row_support = [0, 1, 0]
    mach = np.array([0.40, 0.50, 0.60])
    cl = np.array([0.25, 0.35, 0.45])
    # s0 carries file rows 0 and 2, s1 carries file row 1. The values
    # are keyed to the file row, so an implementation that mapped the
    # leading index to the file row number reads the wrong number.
    p0 = np.array([[[100.0 + j] for j in range(6)],
                   [[300.0 + j] for j in range(6)]])
    p1 = np.array([[[200.0 + j] for j in range(4)]])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", False)

    row = scale(f, "row", n_rows, unlimited=True)
    component_1 = scale(f, "component_1", 1)
    component_2 = scale(f, "component_2", 2)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")

    scalars = f.create_group("scalars")
    sc = dataset(scalars, "cl", cl, "<f8", [row], n_rows=n_rows)
    sattr(sc, "units", "1")
    sattr(sc, "source", "data")

    dataset(f, "row_support", row_support, "<i4", [row], n_rows=n_rows)

    supports = f.create_group("supports")
    sup0 = supports.create_group("s0")
    node0, _c0 = mesh_cells(sup0, N_NODES, CELL_TYPES, CELL_OFFSETS,
                            CELL_CONNECTIVITY)
    row0 = scale(sup0, "row", 2, unlimited=True)
    c = dataset(sup0, "coordinates", BASE_COORDS, "<f8",
                [node0, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")
    a = dataset(sup0.create_group("node_arrays"), "pressure", p0,
                "<f8", [row0, node0, component_1], n_rows=2)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "Pa")
    iattr(a, "components", 1)
    sattr(a, "source", "data")

    sup1 = supports.create_group("s1")
    node1, _c1 = mesh_cells(sup1, S1_NODES, S1_TYPES, S1_OFFSETS,
                            S1_CONN)
    row1 = scale(sup1, "row", 1, unlimited=True)
    c = dataset(sup1, "coordinates", S1_COORDS, "<f8",
                [node1, component_2])
    sattr(c, "role", "coordinates")
    sattr(c, "varies", "none")
    sattr(c, "units", "m")
    iattr(c, "components", 2)
    sattr(c, "source", "data")
    if served:
        a = sup1.create_group("node_arrays").create_group("pressure")
    else:
        a = dataset(sup1.create_group("node_arrays"), "pressure", p1,
                    "<f8", [row1, node1, component_1], n_rows=1)
    sattr(a, "role", "field")
    sattr(a, "varies", "row")
    sattr(a, "units", "Pa")
    iattr(a, "components", 1)
    sattr(a, "source", "callable:m1" if served else "data")
    if served:
        sattr(a, "output", "pressure")
        m1 = f.create_group("callables").create_group("m1")
        sattr(m1, "type", "affine")
        strings(m1, "keys", ["mach"], scale(m1, "mestra_keys_d0", 1))
        out = m1.create_group("outputs").create_group("pressure")
        codec_array(out, "A", SERVED_S1["A"])
        codec_array(out, "b", SERVED_S1["b"])
        codec_array(out, "shape", SERVED_S1["shape"])
        return expect(
            "Two supports, three rows. s0 carries a stored row-varying "
            "field; the field on s1 is served by a callable and holds no "
            "data, and s1 still carries a support-local row scale, "
            "because it carries an array with varies = row.",
            warnings=["W05"],
            support_ids={"s0": MESH_SID, "s1": S1_SID},
            probes=[probe("/supports/s0/node_arrays/pressure", p0,
                          row=1, node=2, component=0)])

    return expect(
        "Three rows over two supports with a row-varying field on "
        "each. File rows 0 and 2 are on s0 and file row 1 is on s1, "
        "so the leading index of each field is a position within its "
        "own support's rows and not a file row number.",
        warnings=["W05"],
        support_ids={"s0": MESH_SID, "s1": S1_SID},
        probes=[
            probe("/row_support", np.array(row_support), row=1),
            probe("/row_support", np.array(row_support), row=2),
            # s0 entry 1 is file row 2. An implementation that read
            # it as file row 1 would return 200 and something.
            probe("/supports/s0/node_arrays/pressure", p0, row=1,
                  node=2, component=0),
            probe("/supports/s0/node_arrays/pressure", p0, row=0,
                  node=4, component=0),
            # s1 has one entry, which is file row 1.
            probe("/supports/s1/node_arrays/pressure", p1, row=0,
                  node=3, component=0),
            probe("/scalars/cl", cl, row=2),
        ])


#: The affine output that serves s1's field in the served variant:
#: one key, four nodes, one component.
SERVED_S1 = {
    "A": np.array([[1.0], [2.0], [3.0], [4.0]]),
    "b": np.array([0.0, 0.5, 1.0, 1.5]),
    "shape": np.array([4, 1], dtype="<i8"),
}


def case_two_supports_one_served(f):
    return case_two_supports_row_varying(f, served=True)


def case_notes_and_private(f):
    """A valid case that carries /notes and /private."""
    n_rows = 2
    mach = np.array([0.40, 0.80])
    cl = np.array([0.25, 0.55])

    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", n_rows, unlimited=True)

    keys = f.create_group("keys")
    k = dataset(keys, "mach", mach, "<f8", [row], n_rows=n_rows)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")

    scalars = f.create_group("scalars")
    sc = dataset(scalars, "cl", cl, "<f8", [row], n_rows=n_rows)
    sattr(sc, "units", "1")
    sattr(sc, "source", "data")

    # /notes is free-form, and its attributes still obey section 18.
    notes = f.create_group("notes")
    sattr(notes, "solver", "an open solver, version 3")
    sattr(notes, "licence", "CC-BY-4.0")
    iattr(notes, "iterations", 480)
    fattr(notes, "residual", 1e-9)
    battr(notes, "converged", True)

    # /private is opaque. The byte-level rules of sections 18 to 25
    # are not checked inside it (section 14), so it deliberately
    # holds things that would be errors in the public part: a float32
    # dataset, a dimension scale of its own, and a nested group.
    private = f.create_group("private")
    pscale = scale(private, "sample", 4)
    pd = private.create_dataset("residuals", data=np.float32(
        [1.0, 0.5, 0.25, 0.125]), dtype="<f4", track_times=False)
    pd.dims[0].attach_scale(pscale)
    history = private.create_group("history")
    sattr(history, "note", "whatever the producer keeps here")

    return expect(
        "A valid file carrying both optional groups: /notes with "
        "free-form attributes, and /private holding a float32 "
        "dataset, a dimension scale of its own and a nested group, "
        "none of which the byte-level rules reach.",
        probes=[probe("/scalars/cl", cl, row=1),
                probe("/keys/mach", mach, row=0)])


def private_as_found(f):
    """A /private whose attributes and datasets are in encodings the
    public part does not allow, which a copy keeps as they were."""
    private = f.create_group("private")
    string_attr_as(private, "origin", "solver log", h5py.h5t.CSET_ASCII,
                   h5py.h5t.STR_NULLTERM)
    private.attrs.create("revision", np.int32(7))
    pd = private.create_dataset("residuals", data=np.array([1.0, 0.5]),
                                dtype=">f8", track_times=False)
    # A dimension scale of its own, so that a generic netCDF-4 reader
    # walking /private still finds a named dimension (section 14).
    pd.dims[0].attach_scale(scale(private, "sample", 2))
    pd.attrs.create("scale", np.float64(2.0))
    string_attr_as(pd, "units", "1", h5py.h5t.CSET_ASCII,
                   h5py.h5t.STR_SPACEPAD)


WIDE_KEYS = 100
WIDE_SCALARS = 4100


def wide_key(i):
    return np.array([1.0 * i, 1.0 * i + 0.5])


def wide_scalar(j):
    return np.array([2.0 * j, 2.0 * j + 0.25])


def case_wide_keys(f):
    """4200 row-dimensioned datasets, which is past the 4085
    attachments one dimension scale could carry before the creation
    property rule of section 21.

    Like the two deep hostile files this one is generated on demand
    and not committed: it is about seventeen megabytes of object
    headers. Its expected.json is committed, so an implementation
    knows it exists and knows to generate it first."""
    n_rows = 2
    if f is None:
        return wide_expectation()
    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)

    row = scale(f, "row", n_rows, unlimited=True)

    keys = f.create_group("keys")
    for i in range(WIDE_KEYS):
        k = dataset(keys, "k%04d" % i, wide_key(i), "<f8", [row],
                    n_rows=n_rows)
        sattr(k, "role", "design")
        sattr(k, "units", "1")

    scalars = f.create_group("scalars")
    for j in range(WIDE_SCALARS):
        sc = dataset(scalars, "s%04d" % j, wide_scalar(j), "<f8",
                     [row], n_rows=n_rows)
        sattr(sc, "units", "1")
        sattr(sc, "source", "data")

    return wide_expectation()


def wide_expectation():
    return expect(
        "Two rows and %d row-dimensioned datasets, %d keys and %d "
        "scalars, all attached to one `row` scale. Before the "
        "creation property rule of section 21 the 4086th attachment "
        "failed, so this file could not be written at all. It is "
        "generated on demand and not committed."
        % (WIDE_KEYS + WIDE_SCALARS, WIDE_KEYS, WIDE_SCALARS),
        probes=[
            probe("/keys/k0000", wide_key(0), row=0),
            probe("/keys/k0099", wide_key(99), row=1),
            probe("/scalars/s0000", wide_scalar(0), row=0),
            probe("/scalars/s2100", wide_scalar(2100), row=1),
            probe("/scalars/s4099", wide_scalar(4099), row=1),
        ])


# --------------------------------------- one file per rule identifier

PRESSURE_3 = pressures(3)
PRESSURE_4 = pressures(4)
MEMBER_COORDS_3 = np.stack([BASE_COORDS * np.array([s, 1.0])
                            for s in (1.0, 1.25, 1.5)])

E21_TYPES = np.array([9, 99], dtype="<u1")
E22_OFFSETS = np.array([0, 4, 9], dtype="<i8")
E22_CONN = np.array([0, 1, 4, 3, 1, 2, 5, 4, 3], dtype="<i8")
E23_TYPES = np.array([9, 7], dtype="<u1")
E23_OFFSETS = np.array([0, 4, 7], dtype="<i8")
E24_CONN = np.array([0, 1, 4, 3, 1, 2, 6, 4], dtype="<i8")

WRONG_SID = "0" * 64


def mk(builder, options, description, errors=(), warnings=(),
       support_ids=None, probes=(), codec=None, evaluation=()):
    """A case that is one base file with one thing changed."""
    def build(f):
        builder(f, options)
        return expect(description, errors, warnings, support_ids,
                      probes, codec, evaluation)
    return build


def then(builder, change):
    """A base file with something done to it after it is built, for
    a deviation no option of the base describes."""
    def build(f, o):
        out = builder(f, o)
        change(f)
        return out
    return build


def support_kind_absent(f):
    """The support does not say what kind it is."""
    del f["supports/s0"].attrs["kind"]


def support_kind_grid(f):
    """The support says it is a "grid", a kind section 6 does not have."""
    support = f["supports/s0"]
    del support.attrs["kind"]
    sattr(support, "kind", "grid")


def label_table_elsewhere(f):
    """The region label names a table, "zone", the file does not have."""
    region = f["supports/s0/cell_arrays/region"]
    del region.attrs["category"]
    sattr(region, "category", "zone")


def big_endian_attr(obj, name, value):
    """An integer attribute stored big-endian, which section 18 does
    not allow: it names H5T_STD_I64LE."""
    del obj.attrs[name]
    obj.attrs.create(name, np.array(value, dtype=">i8"))


def scale_deleted(f):
    """The coordinates' component scale deleted after it was attached,
    so that the axis names an object that is no longer there."""
    del f["component_2"]


def dictionary_scale_of_another(f):
    """A dictionary dataset whose axis carries another dataset's
    scale: section 21 names it mestra_<dataset>_d<i>."""
    m1 = f["callables/m1"]
    d = m1.create_dataset("weights", data=np.array([0.5, 0.25]),
                          track_times=False)
    d.dims[0].attach_scale(scale(m1, "mestra_other_d0", 2))


def table_on_another_scale(f):
    """A second category table attached to the first one's scale:
    section 21 names a table's dimension category_<table>."""
    strings(f["categories"], "zone", ["fore", "aft"],
            f["category_region"])


def dataset_without_scale(f):
    """A dataset this version does not know inside a support group,
    with no dimension scale on its axis."""
    f["supports/s0"].create_dataset("extra", data=np.arange(3.0),
                                    track_times=False)


def string_attr_as(obj, name, value, cset, pad):
    """A string attribute in an encoding section 18 does not allow:
    the character set or the padding other than UTF-8 and NUL."""
    raw = value.encode("utf-8")
    t = h5py.h5t.C_S1.copy()
    t.set_size(len(raw))
    t.set_cset(cset)
    t.set_strpad(pad)
    if name in obj.attrs:
        del obj.attrs[name]
    a = h5py.h5a.create(obj.id, name.encode(), t,
                        h5py.h5s.create(h5py.h5s.SCALAR))
    a.write(np.array(raw, dtype="S%d" % len(raw)), mtype=t)


def strings_not_utf8_nulpad(f):
    string_attr_as(f["keys/mach"], "units", "1", h5py.h5t.CSET_UTF8,
                   h5py.h5t.STR_SPACEPAD)
    string_attr_as(f["scalars/cl"], "units", "1", h5py.h5t.CSET_ASCII,
                   h5py.h5t.STR_NULLPAD)


def soft_link_in(where):
    """A soft link, which a reader never follows (E40), placed in a
    public group that holds no other link this base would check."""
    def change(f):
        g = f[where] if where in f else f.create_group(where)
        g["ghost"] = h5py.SoftLink("/nothing")
    return change


def shared_draws(f):
    """Three draws of a field that varies along nothing: the draw axis
    leads, (draw, node, component)."""
    draw_3 = scale(f, "draw_3", 3)
    sup = f["supports/s0"]
    values = np.array([[[0.1 * k + 0.01 * n] for n in range(N_NODES)]
                       for k in range(3)])
    d = dataset(sup["node_arrays"], "pressure_draws", values, "<f8",
                [draw_3, sup["node"], f["component_1"]])
    sattr(d, "role", "field")
    sattr(d, "varies", "none")
    sattr(d, "units", "Pa")
    iattr(d, "components", 1)
    sattr(d, "source", "data")
    sattr(d, "statistic", "draw")


def dictionary_array_of(dtype):
    """A dictionary dataset of a dtype section 25 does not allow: a
    numeric array there is little-endian int8, int32, int64 or
    float64."""
    def change(f):
        m1 = f["callables/m1"]
        d = m1.create_dataset("weights", data=np.array([0.5, 0.25]),
                              dtype=dtype, track_times=False)
        d.dims[0].attach_scale(scale(m1, "mestra_weights_d0", 2))
    return change


def unknown_group_with_contents(f):
    """A root group this version does not know, holding what the
    byte-level rules would refuse in the public part: an unlimited
    dimension that is not `row`."""
    g = f.create_group("extras")
    events = scale(g, "events", 3, unlimited=True)
    d = g.create_dataset("log", data=np.arange(3.0), maxshape=(None,),
                         chunks=(3,), track_times=False)
    d.dims[0].attach_scale(events)


big_endian_dictionary_array = dictionary_array_of(">f8")


#: A stored band beside the two-row pressure field: (row, node,
#: component), one entry per node, the same in both rows.
PRESSURE_BAND = np.array([[[0.5 + 0.1 * n] for n in range(6)]
                          for _ in range(2)])
PRESSURE_BAND_NEGATIVE = PRESSURE_BAND.copy()
PRESSURE_BAND_NEGATIVE[1, 2, 0] = -0.7


CASES = {
    # the two files of docs/example.md
    "mesh_two_rows": case_mesh_two_rows,
    "affine_no_keys": case_affine_no_keys,
    "affine_zero_rows": case_affine_zero_rows,
    "affine_own_repr": mk(
        affine_base, {"repr": "pressure model fitted on the March runs"},
        "The affine callable of docs/example.md with a repr line of its "
        "producer's own rather than the one an implementation would "
        "make up. A rewrite keeps the line it read.",
        support_ids={"s0": MESH_SID},
        codec={"m1": tagged(affine_dict())}),
    "dictionary_empty_arrays": case_dictionary_empty_arrays,
    "dictionary_null": case_dictionary_null,
    # the five mappings
    "family_static": case_family_static,
    "cascade_varying_geometry": case_cascade_varying_geometry,
    "scalars_only": case_scalars_only,
    "transient_fixed_mesh": case_transient_fixed_mesh,
    "axis_signature": case_axis_signature,
    # the rest of the model
    "affine_with_rows": case_affine_with_rows,
    "callable_two_slots": case_callable_two_slots,
    "callable_coordinates": case_callable_coordinates,
    "two_supports_unaligned": case_two_supports_unaligned,
    "draws_and_summaries": case_draws_and_summaries,
    "labels_tables": case_labels_tables,
    "family_with_time": case_family_with_time,
    "derived_displacement": case_derived_displacement,
    "support_kind_none": case_support_kind_none,
    "two_supports_row_varying": case_two_supports_row_varying,
    "two_supports_one_served": case_two_supports_one_served,
    "notes_and_private": case_notes_and_private,
    "wide_keys": case_wide_keys,
    "band_stored": mk(
        mesh_base, {"band": {"values": PRESSURE_BAND, "level": 0.95,
                             "method": "half-width of a 95 % interval "
                                       "from the residuals of the "
                                       "solver's mesh study"}},
        "A stored field with its band beside it: a band slot is the "
        "half-width of the interval around the base slot it names "
        "with `of`, at the coverage its `level` states, made as its "
        "`method` says.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/node_arrays/pressure", PRESSURE_2,
                  row=1, node=3, component=0),
            probe("/supports/s0/node_arrays/pressure_band",
                  PRESSURE_BAND, row=1, node=3, component=0),
            probe("/supports/s0/node_arrays/pressure_band",
                  PRESSURE_BAND, row=0, node=5, component=0),
        ]),
    "affine_band": mk(
        affine_base, {"band": True},
        "The affine callable of section 27 with a constant band on "
        "both outputs. Two band slots take the `uncertainty` of the "
        "output they name and, once evaluated, carry its `level` and "
        "`method`.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/callables/m1/outputs/pressure/uncertainty",
                  AFFINE_BAND["pressure"]["uncertainty"], index=2),
        ],
        codec={"m1": tagged(affine_dict(
            outputs=with_band(AFFINE, AFFINE_BAND)))},
        evaluation=[affine_evaluation(
            "m1", AFFINE_KEYS, with_band(AFFINE, AFFINE_BAND),
            {"cl": "/scalars/cl",
             "pressure": "/supports/s0/node_arrays/pressure"},
            AFFINE_AT,
            bands={"cl": "/scalars/cl_band",
                   "pressure": "/supports/s0/node_arrays/pressure_band"})]),
    "compressed_field": mk(
        mesh_base, {"pressure_gzip": 4, "pressure_shuffle": True,
                    "cl_gzip": 4},
        "A field compressed with shuffle and gzip at level 4, and a "
        "scalar with gzip alone. Section 23 allows both, and a "
        "writer that drops a filter on a round trip fails the "
        "structural comparison against this file.",
        support_ids={"s0": MESH_SID},
        probes=[
            probe("/supports/s0/node_arrays/pressure", PRESSURE_2,
                  row=1, node=3, component=0),
            probe("/scalars/cl", np.array([0.25, 0.55]), row=1),
        ]),

    # ---------------------------------------------------------- errors
    "err_e01": mk(
        mesh_base, {"format": "mestra/1"},
        "The root format attribute names a major version a version 0 "
        "reader must refuse outright.",
        errors=["E01"], support_ids={"s0": MESH_SID}),
    "err_e02": mk(
        mesh_base, {"mach_no_role": True},
        "A key with no role attribute.",
        errors=["E02"], support_ids={"s0": MESH_SID}),
    "err_e02_kind": mk(
        then(mesh_base, support_kind_grid), {},
        "A support whose kind is \"grid\", which is not one of the "
        "three kinds of section 6. It has no digest (section 24).",
        errors=["E02"], support_ids={}),
    "err_e02_statistic": mk(
        mesh_base, {"pressure_extra": [("statistic", "median"),
                                       ("of", "pressure")]},
        "A node array whose statistic is \"median\", which is not one "
        "of the six statistics of section 9.",
        errors=["E02"], support_ids={"s0": MESH_SID}),
    "err_e02_statistic_without_of": mk(
        mesh_base, {"cl_extra": [("statistic", "median")]},
        "A scalar whose statistic is not a word of section 9 and that "
        "names no base quantity. E12 decides what a statistic needs "
        "from its word, so it has nothing to say here.",
        errors=["E02"], support_ids={"s0": MESH_SID}),
    "err_e03": mk(
        mesh_base, {"extra_keys": [
            ("t1", "time", [0.0, 1.0], "<f8",
             [("units", "s"), ("trajectory_group", "member")]),
            ("t2", "time", [0.0, 2.0], "<f8",
             [("units", "s"), ("trajectory_group", "member")])]},
        "Two keys with the role time, where the role allows at most "
        "one.",
        errors=["E03"], support_ids={"s0": MESH_SID}),
    "err_e04": mk(
        mesh_base, {"pressure_varies": "none"},
        "A node array whose varies says none while its stored shape "
        "still carries the leading row dimension.",
        errors=["E04"], support_ids={"s0": MESH_SID}),
    "err_e04_unknown_group": mk(
        mesh_base, {"coords_varies": "group:nosuch"},
        "An array whose varies names a group key the file does not "
        "declare.",
        errors=["E04"], support_ids={"s0": MESH_SID}),
    "err_e04_unknown_varies": mk(
        mesh_base, {"pressure_varies": "sideways"},
        "A field whose varies is none of the three words section 5 "
        "names. Its shape has as many axes as a row-varying one, so only "
        "the word itself is wrong.",
        errors=["E04"], support_ids={"s0": MESH_SID}),
    "err_e04_served_varies": mk(
        affine_base, {"pressure_varies": "sideways"},
        "A slot a callable serves whose varies is none of the three words "
        "section 5 names. It holds no data, so only the word can be "
        "checked, and it is wrong.",
        errors=["E04"], support_ids={"s0": MESH_SID}),
    "err_e04_empty_varies": mk(
        mesh_base, {"pressure_varies": ""},
        "A field whose varies is the empty string, which is a value and "
        "not one of the three section 5 names.",
        errors=["E04"], support_ids={"s0": MESH_SID}),
    "err_e05": mk(
        mesh_base, {"pressure": PRESSURE_2[:, :5, :]},
        "A node array of five nodes on a support of six.",
        errors=["E05"], support_ids={"s0": MESH_SID}),
    "err_e06": mk(
        two_support_base, {"n_rows": 3, "aligned": False,
                           "row_support": [0, 1, 2]},
        "A row referencing support 2 where the file declares two. W05 "
        "is unavoidable here because the rule needs more than one "
        "support to be reachable at all.",
        errors=["E06"], warnings=["W05"],
        support_ids={"s0": MESH_SID, "s1": S1_SID}),
    "err_e08_negative_nodes": mk(
        lambda f, o: (case_support_kind_none(f),
                      f["supports/s0"].attrs.__setitem__(
                          "n_nodes", np.int64(-1))),
        {},
        "The support of kind none with n_nodes set to -1 and its digest "
        "left as it was. Section 24 hashes n_nodes as an int64, so the "
        "stored digest no longer matches.",
        errors=["E08"],
        support_ids={"s0": support_id(-1)}),
    "err_e08": mk(
        mesh_base, {"support_id": WRONG_SID},
        "A support_id that does not match the stored arrays; the "
        "digest given here is the one an implementation must "
        "compute.",
        errors=["E08"], support_ids={"s0": MESH_SID}),
    "err_e09": mk(
        mesh_base, {"n_rows": 3, "mach_values": [0.4, 0.5, 0.8],
                    "member_values": [0, 0, 1],
                    "cl_values": [0.25, 0.35, 0.55],
                    "pressure": PRESSURE_3,
                    "extra_keys": [
                        ("t", "time", [0.0, 0.0, 1.0], "<f8",
                         [("units", "s"),
                          ("trajectory_group", "member")])]},
        "Time repeats within one trajectory instead of strictly "
        "increasing.",
        errors=["E09"], support_ids={"s0": MESH_SID}),
    "err_e10": mk(
        mesh_base, {"extra_scales": [("category_regime", 2)],
                    "extra_categories": [("regime", ["low", "high"])],
                    "extra_keys": [
                        ("regime", "categorical", [0, 2], "<i4",
                         [("category", "regime")])]},
        "A categorical key whose value 2 is outside a category table "
        "of two entries.",
        errors=["E10"], support_ids={"s0": MESH_SID}),
    "err_e11": mk(
        mesh_base, {"pressure_units": None},
        "A field with no units attribute.",
        errors=["E11"], support_ids={"s0": MESH_SID}),
    "err_e12": mk(
        mesh_base, {"pressure_extra": [("statistic", "mean")]},
        "A slot that declares a statistic and does not say what it is "
        "a statistic of.",
        errors=["E12"], support_ids={"s0": MESH_SID}),
    "err_e12_band": mk(
        mesh_base, {"band": {"values": PRESSURE_BAND, "level": None,
                             "method": "constant band"}},
        "A stored band with no level, so the coverage it claims is "
        "unknown.",
        errors=["E12"], support_ids={"s0": MESH_SID}),
    "err_e12_callable": mk(
        affine_base, {"pressure_extra": [("statistic", "draw")]},
        "A slot served by a callable that declares the statistic "
        "draw, which a callable does not produce: it returns a mean "
        "and at most a band.",
        errors=["E12"], support_ids={"s0": MESH_SID},
        codec={"m1": tagged(affine_dict())}),
    "err_e13": mk(
        mesh_base, {"pressure_role": "derived"},
        "A derived array with neither derived_from nor recipe.",
        errors=["E13"], support_ids={"s0": MESH_SID}),
    "err_e14": mk(
        affine_base, {"pressure_source": "callable:m9"},
        "A slot whose source names a callable id the file does not "
        "hold.",
        errors=["E14"], support_ids={"s0": MESH_SID},
        codec={"m1": tagged(affine_dict())}),
    "err_e15": mk(
        affine_base, {"type": None},
        "A callable group with no type attribute.",
        errors=["E15"], support_ids={"s0": MESH_SID},
        codec={"m1": tagged(affine_dict())}),
    "err_e16": mk(
        mesh_base, {"cl_values": [0.25, 0.55, 0.75]},
        "A stored scalar of three elements in a file of two rows.",
        errors=["E16"], support_ids={"s0": MESH_SID}),
    "err_e16_row_support": mk(
        two_support_base, {"n_rows": 3, "row_support": [0, 1, 0, 1]},
        "A /row_support of four values in a file of three rows.",
        errors=["E16"], warnings=["W05"],
        support_ids={"s0": MESH_SID, "s1": S1_SID}),
    "err_e16_support_rows": mk(
        lambda f, o: (case_two_supports_row_varying(f),
                      f["row_support"].__setitem__(slice(None), [0, 1, 1])),
        {},
        "The row-varying file of two supports with its third row moved "
        "from s0 to s1, so s0 holds two entries for the one row now on it "
        "and s1 one for two. Its fields still line up with their "
        "support-local row scales; it is /row_support that disagrees.",
        errors=["E16"], warnings=["W05"],
        support_ids={"s0": MESH_SID, "s1": S1_SID}),
    "err_e17": mk(
        mesh_base, {"writer": None},
        "The root writer attribute is missing.",
        errors=["E17"], support_ids={"s0": MESH_SID}),
    "err_e18": mk(
        mesh_base, {"gen_group": None, "private_gen": True},
        "The file declares a group key and names the unit of "
        "generalisation only under /private, where a reader is told "
        "not to look. A validator sees the missing root attribute as "
        "E39; E18 is what a writer review adds to it.",
        errors=["E18", "E39"], support_ids={"s0": MESH_SID}),
    "err_e18_role": mk(
        mesh_base, {"extra_keys": [("flow", "regime", [0.0, 1.0], "<f8",
                                    [("units", "1")])],
                    "private_gen": True},
        "A key whose role is a word section 3 does not have, in a file "
        "that also carries /private. E18 goes beside any E02, not only "
        "beside a role that is absent.",
        errors=["E02", "E18"], support_ids={"s0": MESH_SID}),
    "err_e19": mk(
        mesh_base, {"cl_units_vlen": True},
        "A units attribute stored as a variable-length string, which "
        "section 18 forbids anywhere in the file.",
        errors=["E19"], support_ids={"s0": MESH_SID}),
    "err_e19_callable_type": mk(
        then(affine_base, lambda f: string_attr_as(
            f["callables/m1"], "type", "affine", h5py.h5t.CSET_ASCII,
            h5py.h5t.STR_NULLPAD)), {},
        "A callable whose type attribute has the ASCII character set. "
        "type is an attribute section 18 encodes like any other string.",
        errors=["E19"], support_ids={"s0": MESH_SID}),
    "err_e19_vlen_role": mk(
        then(mesh_base, lambda f: (
            f["keys/mach"].attrs.__delitem__("role"),
            vattr(f["keys/mach"], "role", "condition"))), {},
        "A key whose role is stored as a variable-length string. That is "
        "E19; the role is there, so it is not E02 as well.",
        errors=["E19"], support_ids={"s0": MESH_SID}),
    "err_e19_big_endian": mk(
        then(mesh_base, lambda f: big_endian_attr(
            f["supports/s0"], "n_cells", 2)),
        {}, "A support's n_cells stored as a big-endian integer. Section "
        "18 names little-endian int64, so the value is refused whatever "
        "it decodes to.",
        errors=["E19"], support_ids={"s0": MESH_SID}),
    "err_e19_notes_ascii": mk(
        then(mesh_base, lambda f: string_attr_as(
            f.create_group("notes"), "solver", "fun3d",
            h5py.h5t.CSET_ASCII, h5py.h5t.STR_NULLPAD)), {},
        "A note stored as an ASCII string. /notes is free-form in what "
        "it says and not in how it stores a string (section 11).",
        errors=["E19"], support_ids={"s0": MESH_SID}),
    "err_e19_notes_vlen": mk(
        then(mesh_base, lambda f: vattr(f.create_group("notes"),
                                        "solver", "fun3d")), {},
        "A note stored as a variable-length string, which section 18 "
        "forbids anywhere in the file.",
        errors=["E19"], support_ids={"s0": MESH_SID}),
    "err_e19_string_encoding": mk(
        then(mesh_base, strings_not_utf8_nulpad), {},
        "Two units attributes that are fixed-length strings but not the "
        "encoding section 18 names: one space padded, one with the ASCII "
        "character set.",
        errors=["E19"], support_ids={"s0": MESH_SID}),
    "err_e20_big_endian": mk(
        mesh_base, {"cl_dtype": ">f8"},
        "A scalar stored as big-endian float64. Section 19 stores every "
        "numeric dataset little-endian.",
        errors=["E20"], support_ids={"s0": MESH_SID}),
    "err_e20_unsigned_key": mk(
        mesh_base, {"member_dtype": "<u4"},
        "A group key stored as uint32. Section 19 gives category ids "
        "int32 or int64, which are signed.",
        errors=["E20"], support_ids={"s0": MESH_SID}),
    "err_e20_big_endian_key": mk(
        mesh_base, {"member_dtype": ">i4"},
        "A group key stored as big-endian int32.",
        errors=["E20"], support_ids={"s0": MESH_SID}),
    "err_e20_numeric_table": mk(
        then(mesh_base, lambda f: f["categories"].create_dataset(
            "zone", data=np.array([0.0, 1.0]), track_times=False)), {},
        "A category table of float64 values with no dimension scale. Its "
        "dtype is E20, and it is still a public dataset, so the missing "
        "scale is E25.",
        errors=["E20", "E25"], support_ids={"s0": MESH_SID}),
    "err_e20": mk(
        mesh_base, {"pressure_dtype": "<f4"},
        "A field stored as float32, which is not allowed anywhere.",
        errors=["E20"], support_ids={"s0": MESH_SID}),
    "err_e21": mk(
        mesh_base, {"cell_types": E21_TYPES},
        "A cell type code that is not in the table of section 20.",
        errors=["E21"],
        support_ids={"s0": support_id(N_NODES, E21_TYPES, CELL_OFFSETS,
                                      CELL_CONNECTIVITY)}),
    "err_e22": mk(
        mesh_base, {"cell_offsets": E22_OFFSETS,
                    "cell_connectivity": E22_CONN},
        "A quadrilateral whose offsets give it five nodes.",
        errors=["E22"],
        support_ids={"s0": support_id(N_NODES, CELL_TYPES, E22_OFFSETS,
                                      E22_CONN)}),
    "err_e23": mk(
        mesh_base, {"cell_types": E23_TYPES,
                    "cell_offsets": E23_OFFSETS},
        "Cell offsets whose last value is not the length of the "
        "connectivity.",
        errors=["E23"],
        support_ids={"s0": support_id(N_NODES, E23_TYPES, E23_OFFSETS,
                                      CELL_CONNECTIVITY)}),
    "err_e23_counts": mk(
        mesh_base, {"cell_offsets": np.array([0, 4, 9], dtype="<i8")},
        "Cell offsets whose last value is not the length of the "
        "connectivity, which also gives the second cell five nodes. A "
        "node count comes from the offsets, so offsets that are E23 "
        "decide no E22.",
        errors=["E23"],
        support_ids={"s0": support_id(N_NODES, CELL_TYPES,
                                      np.array([0, 4, 9]),
                                      CELL_CONNECTIVITY)}),
    "err_e24": mk(
        mesh_base, {"cell_connectivity": E24_CONN},
        "A connectivity value equal to the node count, one past the "
        "last node.",
        errors=["E24"],
        support_ids={"s0": support_id(N_NODES, CELL_TYPES, CELL_OFFSETS,
                                      E24_CONN)}),
    "err_e25": mk(
        mesh_base, {"pressure_no_component_scale": True},
        "The component axis of a field carries no dimension scale.",
        errors=["E25"], support_ids={"s0": MESH_SID}),
    "err_e25_category_table": mk(
        then(mesh_base, table_on_another_scale), {},
        "A category table attached to another table's dimension scale.",
        errors=["E25"], support_ids={"s0": MESH_SID}),
    "err_e25_dangling": mk(
        then(mesh_base, scale_deleted), {},
        "The coordinates' component scale deleted after it was attached, "
        "so the axis carries a reference to nothing.",
        errors=["E25"], support_ids={"s0": MESH_SID}),
    "err_e25_dictionary": mk(
        then(affine_base, dictionary_scale_of_another), {"type": "example"},
        "A dictionary dataset whose axis carries a scale named for "
        "another dataset.",
        errors=["E25"], support_ids={"s0": MESH_SID}),
    "err_e25_slot_group_dataset": mk(
        then(affine_base, lambda f: f["scalars/cl"].create_dataset(
            "extra", data=np.arange(2.0), track_times=False)), {},
        "A dataset with no dimension scale inside a slot a callable "
        "serves. It is a public dataset, so the byte-level rules hold.",
        errors=["E25"], support_ids={"s0": MESH_SID}),
    "err_e25_unknown_dataset": mk(
        then(mesh_base, dataset_without_scale), {},
        "A dataset this version does not know, inside a support group, "
        "with no dimension scale. It is a public object, so the "
        "byte-level rules are checked on it. W11 is about an attribute "
        "or a group, so a dataset draws none.",
        errors=["E25"], support_ids={"s0": MESH_SID}),
    "err_e25_notes_dataset": mk(
        then(mesh_base, lambda f: f.create_group("notes").create_dataset(
            "history", data=np.arange(3.0), track_times=False)), {},
        "A dataset with no dimension scale inside /notes. /notes is "
        "public, so the byte-level rules are checked on it.",
        errors=["E25"], support_ids={"s0": MESH_SID}),
    "err_e26": mk(
        mesh_base, {"region_raw": [b"in\x00et", b"outlet"]},
        "A category table entry with a NUL byte in the middle of the "
        "string rather than in its trailing padding.",
        errors=["E26"], support_ids={"s0": MESH_SID}),
    "err_e26_notes": mk(
        then(mesh_base, lambda f: rattr(f.create_group("notes"),
                                        "solver", b"fun\x003d")), {},
        "A note holding a NUL byte before its padding, which no string "
        "in the file may hold (section 18).",
        errors=["E26"], support_ids={"s0": MESH_SID}),
    "err_e27": mk(
        mesh_base, {"cl_contiguous": True},
        "A row-dimensioned dataset stored contiguously instead of "
        "chunked.",
        errors=["E27"], support_ids={"s0": MESH_SID}),
    "err_e28": mk(
        mesh_base, {"row_support_values": [0, 0]},
        "A /row_support dataset in a file that declares one support "
        "and sets aligned = true.",
        errors=["E28"], support_ids={"s0": MESH_SID}),
    "err_e29": mk(
        mesh_base, {"pressure_fletcher32": True},
        "A dataset carrying the fletcher32 filter, which is not one "
        "of the two portable filters.",
        errors=["E29"], support_ids={"s0": MESH_SID}),
    "err_e30": mk(
        mesh_base, {"cl_as_group": True},
        "A slot whose source is data stored as a group.",
        errors=["E30"], support_ids={"s0": MESH_SID}),
    "err_e31": mk(
        mesh_base, {"pressure_components": 2},
        "A field declaring two components over a component dimension "
        "of length one.",
        errors=["E31"], support_ids={"s0": MESH_SID}),
    "err_e31_missing": mk(
        then(mesh_base, lambda f: f["supports/s0/node_arrays/pressure"]
             .attrs.__delitem__("components")), {},
        "A field with no components attribute, which section 14 makes "
        "E31 and not E39.",
        errors=["E31"], support_ids={"s0": MESH_SID}),
    "err_e31_negative": mk(
        mesh_base, {"pressure_components": -1},
        "A field declaring -1 components over a component dimension of "
        "length one.",
        errors=["E31"], support_ids={"s0": MESH_SID}),
    "err_e32": mk(
        affine_base, {"type": "example", "zero_d_key": True},
        "A callable dictionary holding a zero-dimensional dataset, "
        "which section 25 says must be written as an attribute "
        "instead.",
        errors=["E32"], support_ids={"s0": MESH_SID}),
    "err_e32_big_endian": mk(
        then(affine_base, big_endian_dictionary_array),
        {"type": "example"},
        "A callable dictionary holding a big-endian float64 array, "
        "which section 25 does not allow: a numeric array there is "
        "little-endian.",
        errors=["E32"], support_ids={"s0": MESH_SID}),
    "err_e32_float32": mk(
        then(affine_base, dictionary_array_of("<f4")),
        {"type": "example"},
        "A callable dictionary holding a float32 array. Section 19 sends "
        "a dictionary dataset to section 25, so it is E32 and not E20, "
        "and E32 does not stop a read.",
        errors=["E32"], support_ids={"s0": MESH_SID}),
    "err_e33": mk(
        mesh_base, {"mach_name": "mestra_mach"},
        "A producer-chosen name beginning with the reserved prefix.",
        errors=["E33"], support_ids={"s0": MESH_SID}),
    "err_e33_newline": mk(
        mesh_base, {"mach_name": "mach\n"},
        "A key whose name ends in a newline, which is not a character a "
        "netCDF-4 name may hold.",
        errors=["E33"], support_ids={"s0": MESH_SID}),
    "err_e33_leading_dot": mk(
        mesh_base, {"mach_name": ".mach"},
        "A key whose name begins with \".\". Section 18 allows \".\" "
        "inside a name and not as its first character.",
        errors=["E33"], support_ids={"s0": MESH_SID}),
    "err_e33_leading_hyphen": mk(
        mesh_base, {"mach_name": "-mach"},
        "A key whose name begins with a hyphen, which section 18 allows "
        "inside a name and not as its first character.",
        errors=["E33"], support_ids={"s0": MESH_SID}),
    "err_e33_dictionary": mk(
        then(affine_base, lambda f: sattr(f["callables/m1/outputs/cl"],
                                          "fitted@march", "yes")),
        {"type": "example"},
        "A dictionary key that is not a legal netCDF-4 name: section 17 "
        "holds every key to the rule of section 18.",
        errors=["E33"], support_ids={"s0": MESH_SID}),
    "err_e34": mk(
        mesh_base, {"n_group": 3, "coords": MEMBER_COORDS_3},
        "A group-varying array with three instances over a group key "
        "of two categories.",
        errors=["E34"], support_ids={"s0": MESH_SID}),
    "err_e35": mk(
        axis_base, {"n_rows": 1, "coords_varies": "row"},
        "An axis support whose coordinates vary along the row, where "
        "the axis coordinate is part of the support's identity.",
        errors=["E35"], support_ids={"s0": E_AXIS_SID}),
    "err_e36": mk(
        mesh_base, {"cl_source": "model"},
        "A source that is neither data nor callable:<id>.",
        errors=["E36"], support_ids={"s0": MESH_SID}),
    "err_e36_empty_id": mk(
        mesh_base, {"cl_as_group": True, "cl_source": "callable:"},
        "A slot stored as a group whose source is \"callable:\" with no "
        "id. It names no callable, so it is E36 and not E14, and the "
        "output a served slot names is not asked of it.",
        errors=["E36"], support_ids={"s0": MESH_SID}),
    "err_e36_empty_id_dataset": mk(
        mesh_base, {"cl_source": "callable:"},
        "The slot of err_e36_empty_id stored as a dataset. Its source "
        "says neither data nor a callable, so E30 has nothing to compare "
        "the storage with.",
        errors=["E36"], support_ids={"s0": MESH_SID}),
    "err_e36_group": mk(
        affine_base, {"cl_source": "row=99"},
        "A slot stored as a group whose source is neither data nor "
        "callable:<id>. That is E36; it is not E30, which is about a "
        "source that says data, and it does not stop a read.",
        errors=["E36"], support_ids={"s0": MESH_SID}),
    "err_e37": mk(
        two_support_base, {"n_rows": 0, "aligned": True},
        "Two supports declared with aligned = true. W05 comes with "
        "it and cannot be avoided.",
        errors=["E37"], warnings=["W05"],
        support_ids={"s0": MESH_SID, "s1": S1_SID}),
    "err_e37_false": mk(
        mesh_base, {"aligned": False, "row_support_values": [0, 0]},
        "One support declared with aligned = false, the other "
        "direction of the same rule. The /row_support column is "
        "present, which is what a file claiming to be unaligned must "
        "carry, so E28 is not in question.",
        errors=["E37"], support_ids={"s0": MESH_SID}),
    "err_e38": mk(
        axis_base, {"stray_cells": True},
        "An axis support carrying a cell_types dataset and a cell "
        "dimension.",
        errors=["E38"], support_ids={"s0": E_AXIS_SID}),
    "draws_shared": mk(
        then(mesh_base, shared_draws), {},
        "Three draws of a field that varies along nothing, so the draw "
        "axis leads: (draw, node, component). A rewrite keeps that "
        "shape.",
        support_ids={"s0": MESH_SID},
        probes=[probe("/supports/s0/node_arrays/pressure_draws",
                      np.array([[[0.1 * k + 0.01 * n]
                                 for n in range(N_NODES)]
                                for k in range(3)]),
                      draw=2, node=4, component=0)]),
    "private_as_found": mk(
        then(mesh_base, private_as_found), {},
        "A /private holding an ASCII null-terminated string attribute, "
        "an int32 attribute and a big-endian dataset with attributes of "
        "its own. None of it is checked, and a rewrite carries every "
        "attribute and dataset with the type it was found with.",
        support_ids={"s0": MESH_SID}),
    "zero_rows_unaligned": mk(
        two_support_base, {"n_rows": 0, "aligned": False,
                           "row_support": np.zeros(0, dtype="<i4")},
        "Two supports and no rows. The file is unaligned, so it carries "
        "/row_support, which is empty rather than absent, and no row "
        "references either support.",
        warnings=["W05", "W15"],
        support_ids={"s0": MESH_SID, "s1": S1_SID}),
    "zero_rows_stored": mk(
        mesh_base, {"n_rows": 0, "mach_values": np.zeros(0),
                    "member_values": np.zeros(0, dtype="<i4"),
                    "cl_values": np.zeros(0),
                    "pressure": np.zeros((0, 6, 1))},
        "No rows, and stored row-varying slots of shape (0, ...): the "
        "keys, the scalar and a (row, node, component) field with a "
        "zero-length row axis. Its shape is (0, 6, 1) and stays that "
        "through a rewrite. Neither group category is used.",
        warnings=["W07"], support_ids={"s0": MESH_SID}),
    "err_e39_category_key": mk(
        mesh_base, {"extra_keys": [
            ("regime", "categorical", [0, 1], "<i4",
             [("category", "regime")])]},
        "A categorical key whose category names a table the file does "
        "not have. There is no table for a value to be outside, so it "
        "is not E10.",
        errors=["E39"], support_ids={"s0": MESH_SID}),
    "err_e39_category_label": mk(
        then(mesh_base, label_table_elsewhere), {},
        "A label whose category names a table the file does not have. "
        "A label may have no table at all, but one it names must exist.",
        errors=["E39"], support_ids={"s0": MESH_SID}),
    "err_e39_category_zero_rows": mk(
        mesh_base, {"n_rows": 0, "mach_values": [], "member_values": [],
                    "cl_values": [], "pressure": np.zeros((0, 6, 1)),
                    "extra_keys": [
                        ("regime", "categorical", [], "<i4",
                         [("category", "regime")])]},
        "The missing table of err_e39_category_key in a file with no "
        "rows: the rule is about the attribute and not the values. "
        "W07 is the group table no row uses.",
        errors=["E39"], warnings=["W07"], support_ids={"s0": MESH_SID}),
    "err_e39_kind": mk(
        lambda f, o: (case_support_kind_none(f),
                      f["supports/s0"].attrs.__delitem__("kind")),
        {},
        "The support of kind none without its kind. That is E39, and no "
        "rule that depends on the kind applies to a support that does not "
        "say it: it is not taken for a mesh missing its cells and its "
        "coordinates, and it has no digest (section 24).",
        errors=["E39"], support_ids={}),
    "err_e39_kind_mesh": mk(
        then(mesh_base, support_kind_absent), {},
        "A mesh support without its kind. It has no digest, though the "
        "one it would have as a mesh is stored beside it, so E08 is not "
        "decided for it either.",
        errors=["E39"], support_ids={}),
    "err_e39_varies": mk(
        then(mesh_base, lambda f: f["supports/s0/node_arrays/pressure"]
             .attrs.__delitem__("varies")), {},
        "A row-varying field without its varies. That is E39, and there "
        "is then nothing to hold its leading dimension against, so it is "
        "not E04 as well.",
        errors=["E39"], support_ids={"s0": MESH_SID}),
    "err_e40_dictionary": mk(
        then(affine_base, soft_link_in("callables/m1/outputs")), {},
        "A soft link inside a callable's dictionary. The dictionary is "
        "public, and a reader follows no link that is not hard.",
        errors=["E40"], support_ids={"s0": MESH_SID}),
    "err_e40_notes": mk(
        then(mesh_base, soft_link_in("notes")), {},
        "A soft link inside /notes, which is public and otherwise holds "
        "only attributes.",
        errors=["E40"], support_ids={"s0": MESH_SID}),
    "err_e40_root": mk(
        then(mesh_base, soft_link_in("/")), {},
        "A soft link at the root of the file.",
        errors=["E40"], support_ids={"s0": MESH_SID}),
    "err_e40_slot_group": mk(
        then(affine_base, soft_link_in("scalars/cl")), {},
        "A soft link inside a slot a callable serves, which is a group "
        "carrying the slot's attributes and nothing else.",
        errors=["E40"], support_ids={"s0": MESH_SID}),
    "err_e40_support": mk(
        then(mesh_base, soft_link_in("supports/s0")), {},
        "A soft link inside a support group, beside its cell arrays.",
        errors=["E40"], support_ids={"s0": MESH_SID}),
    "err_e41_support_dataset": mk(
        then(mesh_base, lambda f: f["supports"].create_dataset(
            "s1", data=np.arange(3.0), track_times=False)), {},
        "A dataset beside the one support under /supports. It is not a "
        "support, so the file still declares one and is aligned; it is "
        "an object a reader cannot read as one (E41), and a public "
        "dataset with no dimension scale (E25).",
        errors=["E25", "E41"], support_ids={"s0": MESH_SID}),
    "err_e42": mk(
        mesh_base, {"plain_scales": ("component_1",)},
        "A dimension scale created with the library's default "
        "property list, which is what every writer did before "
        "section 21 fixed it. Such a scale takes at most 4085 "
        "attachments.",
        errors=["E42"], support_ids={"s0": MESH_SID}),
    "err_e43": mk(
        mesh_base, {"unlimited_scales": ("component_1",)},
        "A dimension other than `row` left unlimited. The dimension "
        "that showed this in practice was `draw`, whose name carries "
        "its own length and stops being true when it grows.",
        errors=["E43"], support_ids={"s0": MESH_SID}),
    "err_e39": mk(
        mesh_base, {"mach_units": None},
        "A condition key with no units, which section 19 requires "
        "and no other rule catches: E11 covers a field and a scalar, "
        "not a key.",
        errors=["E39"], support_ids={"s0": MESH_SID}),

    # ------------------------------------------- units of section 32
    # Each of these is grammatical and must not draw W10. Before the
    # grammar was written out, at least one implementation refused
    # each of them.
    "units_dot_product": mk(
        mesh_base, {"mach_units": "m.s-1"},
        "Units written as a product by \".\", with an integer power "
        "after the second name.",
        support_ids={"s0": MESH_SID}),
    "units_dot_powers": mk(
        mesh_base, {"mach_units": "kg.m.s-2"},
        "Three names in a product by \".\", the last with a "
        "negative power.",
        support_ids={"s0": MESH_SID}),
    "units_fractional_power": mk(
        mesh_base, {"mach_units": "s^0.5"},
        "A power after \"^\" that is not an integer, the one place "
        "section 32 is wider than UDUNITS-2.",
        support_ids={"s0": MESH_SID}),
    "units_star_power": mk(
        mesh_base, {"mach_units": "m**2"},
        "A power written with \"**\".",
        support_ids={"s0": MESH_SID}),
    "units_time_since": mk(
        mesh_base, {"extra_keys": [
            ("t", "time", [0.0, 1.0], "<f8",
             [("units", "days since 2000-01-01"),
              ("trajectory_group", "member")])]},
        "A time key whose units carry a shift to a date, as CF time "
        "coordinates do.",
        support_ids={"s0": MESH_SID}),
    "units_unknown_name": mk(
        mesh_base, {"mach_units": "ppm"},
        "A name no unit table in this repository holds. The verdict "
        "is the grammar's and not a table's, so it parses.",
        support_ids={"s0": MESH_SID}),

    # -------------------------------------------------------- warnings
    "warn_w01": mk(
        mesh_base, {"n_rows": 4, "mach_values": [0.4, 0.5, 0.7, 0.8],
                    "member_values": [0, 0, 1, 1],
                    "cl_values": [0.25, 0.35, 0.45, 0.55],
                    "pressure": PRESSURE_4,
                    "extra_scales": [("category_split", 2)],
                    "extra_categories": [("split", ["train", "test"])],
                    "extra_keys": [
                        ("split", "split", [0, 1, 0, 1], "<i4",
                         [("category", "split")])]},
        "A split that puts both rows of each member on both sides, so "
        "it is not a generalisation test.",
        warnings=["W01"], support_ids={"s0": MESH_SID}),
    "warn_w02": mk(
        mesh_base, {"extra_scales": [("category_status", 2)],
                    "extra_categories": [
                        ("status", ["converged", "failed"])],
                    "extra_keys": [
                        ("status", "status", [0, 1], "<i4",
                         [("category", "status")])]},
        "One row whose status is not converged.",
        warnings=["W02"], support_ids={"s0": MESH_SID}),
    "warn_w03": mk(
        mesh_base, {"cl_values": [0.25, float("nan")]},
        "A scalar holding NaN, which is how this format spells "
        "missing floating-point data.",
        warnings=["W03"], support_ids={"s0": MESH_SID},
        probes=[probe("/scalars/cl", np.array([0.25, float("nan")]),
                      row=1)]),
    "warn_w04": mk(
        mesh_base, {"mach_values": [0.1, 1.2]},
        "A key value above its declared upper bound, with the "
        "declared lower bound met exactly so that stale bounds are "
        "not also in question.",
        warnings=["W04"], support_ids={"s0": MESH_SID}),
    "warn_w05": mk(
        two_support_base, {"n_rows": 3, "aligned": False,
                           "row_support": [0, 0, 1]},
        "More than one support, so index-aligned operations are not "
        "available.",
        warnings=["W05"],
        support_ids={"s0": MESH_SID, "s1": S1_SID}),
    "warn_w06": mk(
        mesh_base, {"weight": True, "weight_recomputed": False},
        "A weight array that does not say it was recomputed from the "
        "connectivity.",
        warnings=["W06"], support_ids={"s0": MESH_SID}),
    "warn_w06_false": mk(
        mesh_base, {"weight": True, "weight_recomputed": False,
                    "weight_recomputed_false": True},
        "A weight array whose recomputed attribute is present and false. "
        "It is not marked as recomputed, so W06 holds as it does when the "
        "attribute is absent.",
        warnings=["W06"], support_ids={"s0": MESH_SID}),
    "warn_w07": mk(
        mesh_base, {"member_cats": ["wing_a", "wing_b", "wing_c"],
                    "n_group": 3, "coords": MEMBER_COORDS_3},
        "A group key whose category table has a third entry no row "
        "uses.",
        warnings=["W07"], support_ids={"s0": MESH_SID}),
    "warn_w08": mk(
        mesh_base, {"mach_bounds": (-1000.0, 1000.0)},
        "Declared bounds thousands of times wider than the "
        "observed range, which is what stale bounds look like.",
        warnings=["W08"], support_ids={"s0": MESH_SID}),
    "warn_w10": mk(
        mesh_base, {"mach_units": "kg/(m s"},
        "A units string with an unbalanced parenthesis, which no "
        "UDUNITS parser accepts.",
        warnings=["W10"], support_ids={"s0": MESH_SID}),
    "warn_w10_dangling_per": mk(
        mesh_base, {"mach_units": "m per"},
        "A units string ending in \"per\", which section 32 reads as "
        "an operator after a power with no power after it.",
        warnings=["W10"], support_ids={"s0": MESH_SID}),
    "warn_w11": mk(
        mesh_base, {"unknown_root_attr": True,
                    "unknown_root_group": True},
        "A root attribute and a root group no version 0 reader knows, "
        "both of which must be ignored and reported.",
        warnings=["W11"], support_ids={"s0": MESH_SID}),
    "warn_w11_contents": mk(
        then(mesh_base, unknown_group_with_contents), {},
        "A root group this version does not know, holding an unlimited "
        "dimension that is not row. The group is W11 and is otherwise "
        "left alone (section 14), so what it holds is not E43.",
        warnings=["W11"], support_ids={"s0": MESH_SID}),
    "warn_w11_support_contents": mk(
        then(mesh_base, lambda f: f["supports/s0"].create_group("extras")
             .create_dataset("quality", data=np.arange(6.0),
                             track_times=False)
             .dims[0].attach_scale(f["supports/s0/node"])), {},
        "A group this version does not know inside a support, holding a "
        "dataset on the support's node dimension. It is W11, nothing in "
        "it is checked, and a rewrite carries it whole.",
        warnings=["W11"], support_ids={"s0": MESH_SID}),
    "warn_w12": mk(
        mesh_base, {"pressure_chunks": (1, 6, 1)},
        "A chunk of one row where the default of section 23 is two.",
        warnings=["W12"], support_ids={"s0": MESH_SID}),
    "warn_w13": mk(
        mesh_base, {"category_size": 10},
        "A category table sized ten bytes where six would do.",
        warnings=["W13"], support_ids={"s0": MESH_SID}),
    "warn_w14": mk(
        mesh_base, {"created": "19/09/2026"},
        "A created attribute that is not an ISO 8601 UTC timestamp.",
        warnings=["W14"], support_ids={"s0": MESH_SID}),
    "warn_w16": mk(
        mesh_base, {"band": {"values": PRESSURE_BAND_NEGATIVE,
                             "level": 0.95, "method": "constant band"}},
        "A band holding a negative value, which no half-width is.",
        warnings=["W16"], support_ids={"s0": MESH_SID}),
    "warn_w15": mk(
        two_support_base, {"n_rows": 3, "aligned": False,
                           "row_support": [0, 0, 0]},
        "A declared support that no row references. W05 comes with "
        "it, since the rule needs two supports to be reachable.",
        warnings=["W05", "W15"],
        support_ids={"s0": MESH_SID, "s1": S1_SID}),
}




# ------------------------------------------------- the hostile subset

# Section 30. These files are not specimens of the format. Each one is
# malformed in a way a reader has to survive rather than describe, so
# the contract is only that the required ids appear, that more are
# allowed, and that the run finishes cleanly.

def hostile_expect(description, required_errors):
    return {
        "description": description,
        "required_errors": sorted(required_errors),
        "allow_extra": True,
        "timeout_seconds": 10,
    }


def hmk(mutate, description, required_errors, options=None):
    """A hostile case that is the ordinary mesh file with one thing
    done to it that no writer would ever do."""
    def build(f):
        mesh_base(f, options or {})
        mutate(f)
        return hostile_expect(description, required_errors)
    return build


def raw_filter_dataset(group, name, data, scales, filt):
    """A dataset created through the low-level API, so that a filter
    pipeline h5py's own interface will not produce can be recorded on
    it. `filt` is (filter id, flags, client data values)."""
    fid, flags, cd = filt
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk(data.shape)
    dcpl.set_obj_track_times(False)
    dcpl.set_filter(fid, flags, tuple(cd))
    maxshape = (h5py.h5s.UNLIMITED,) + data.shape[1:]
    space = h5py.h5s.create_simple(data.shape, maxshape)
    tid = h5py.h5t.py_create(np.dtype("<f8"), logical=True)
    dsid = h5py.h5d.create(group.id, name.encode("utf-8"), tid, space,
                           dcpl=dcpl)
    d = h5py.Dataset(dsid)
    d[...] = data
    for axis, sc in enumerate(scales):
        d.dims[axis].attach_scale(sc)
    return d


def replace_pressure_with_filter(f, filt):
    sup = f["supports/s0"]
    na = sup["node_arrays"]
    del na["pressure"]
    d = raw_filter_dataset(na, "pressure", PRESSURE_2,
                           [f["row"], sup["node"], f["component_1"]],
                           filt)
    sattr(d, "role", "field")
    sattr(d, "varies", "row")
    sattr(d, "units", "Pa")
    iattr(d, "components", 1)
    sattr(d, "source", "data")


def m_attr_array_root(f):
    del f.attrs["aligned"]
    f.attrs.create("aligned", np.array([1, 1, 1], dtype="<i8"))


def m_attr_array_key(f):
    k = f["keys/mach"]
    del k.attrs["lower"]
    k.attrs.create("lower", np.array([0.1, 0.2, 0.3], dtype="<f8"))


def m_attr_vlen_array_slot(f):
    p = f["supports/s0/node_arrays/pressure"]
    del p.attrs["units"]
    p.attrs.create("units", ["Pa", "kPa"],
                   dtype=h5py.string_dtype(encoding="utf-8"))


def m_filter_unknown_id(f):
    replace_pressure_with_filter(f, (39999, h5py.h5z.FLAG_OPTIONAL,
                                     (1,)))


def m_filter_many_client_data(f):
    replace_pressure_with_filter(
        f, (39998, h5py.h5z.FLAG_OPTIONAL, tuple(range(12))))


def deep_chain(group, depth):
    """A chain of `depth` groups, each one inside the last. They are
    created through the low-level API because h5py's create_group
    cannot turn object time tracking off, and a group that records
    the time it was made is not byte reproducible."""
    gcpl = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
    gcpl.set_obj_track_times(False)
    gid = group.id
    for _ in range(depth):
        gid = h5py.h5g.create(gid, b"g", gcpl=gcpl)


def deep_base(f, home):
    """A small, otherwise ordinary file of two rows with no support,
    carrying one chain of thirty thousand groups.

    The file keeps the default HDF5 group layout, which costs about a
    kilobyte a group and makes the file 31 MB. That is deliberate.
    The newer layout costs a seventh of that, but it writes four
    timestamps into the root object header, and a file that records
    when it was written is not byte reproducible. The 31 MB is almost
    entirely repetition: git stores it in under 900 kB."""
    sattr(f, "created", CREATED)
    sattr(f, "format", "mestra/0")
    sattr(f, "writer", WRITER)
    battr(f, "aligned", True)
    row = scale(f, "row", 2, unlimited=True)
    keys = f.create_group("keys")
    k = dataset(keys, "mach", [0.40, 0.80], "<f8", [row], n_rows=2)
    sattr(k, "role", "condition")
    sattr(k, "units", "1")
    scalars = f.create_group("scalars")
    sc = dataset(scalars, "cl", [0.25, 0.55], "<f8", [row], n_rows=2)
    sattr(sc, "units", "1")
    sattr(sc, "source", "data")
    if home == "keys":
        deep_chain(keys, 30000)
    else:
        c0 = f.create_group("callables").create_group("c0")
        sattr(c0, "type", "example")
        deep_chain(c0, 30000)


def hdeep(home, description):
    def build(f):
        if f is not None:
            deep_base(f, home)
        return hostile_expect(description, ["E41"])
    return build


LINK_HOMES = ("keys", "scalars", "supports", "callables")


def each_home(f, make):
    for home in LINK_HOMES:
        g = f[home] if home in f else f.create_group(home)
        g["ghost"] = make(home)


def m_link_soft_dangling(f):
    each_home(f, lambda home: h5py.SoftLink("/%s/nothing" % home))


def m_link_soft_cyclic(f):
    each_home(f, lambda home: h5py.SoftLink("/%s/ghost" % home))


def m_link_external(f):
    each_home(f, lambda home: h5py.ExternalLink("elsewhere.mes",
                                                "/%s/mach" % home))


def m_wrong_object_kinds(f):
    f["keys"].create_group("bogus")
    f["supports"].create_dataset("bogus", data=np.zeros(2),
                                 dtype="<f8", track_times=False)


def m_huge_unwritten_dataset(f):
    d = f["scalars"].create_dataset(
        "huge", shape=(10 ** 12,), dtype="<f8", maxshape=(None,),
        chunks=(1024,), track_times=False)
    d.dims[0].attach_scale(f["row"])
    sattr(d, "units", "1")
    sattr(d, "source", "data")


def m_string_invalid_utf8(f):
    del f["categories"]["region"]
    strings(f["categories"], "region", None, f["category_region"],
            size=6, raw=[b"\xff\xfe", b""])


def m_scale_attached_twice(f):
    f["supports/s0/node_arrays/pressure"].dims[2].attach_scale(
        f["component_2"])


def m_scale_no_name_attr(f):
    del f["component_1"].attrs["NAME"]


HOSTILE = {
    "attr_array_root": hmk(
        m_attr_array_root,
        "The root `aligned` attribute is an int64 array of three "
        "rather than the int8 scalar section 18 requires.",
        ["E19"]),
    "attr_array_key": hmk(
        m_attr_array_key,
        "A key's `lower` bound is a float64 array of three rather "
        "than the float64 scalar section 18 requires.",
        ["E19"]),
    "attr_vlen_array_slot": hmk(
        m_attr_vlen_array_slot,
        "A slot's `units` is an array of two variable-length "
        "strings, which section 18 forbids twice over.",
        ["E19"]),
    "filter_unknown_id": hmk(
        m_filter_unknown_id,
        "A field carrying filter 39999, which no HDF5 build has. It "
        "is marked optional, so the bytes are readable and only the "
        "pipeline is wrong.",
        ["E29"]),
    "filter_many_client_data": hmk(
        m_filter_many_client_data,
        "A field carrying an unknown filter with twelve client data "
        "values, more than the eight some filter interfaces have "
        "room for.",
        ["E29"]),
    "deep_groups_callables": hdeep(
        "callables",
        "Thirty thousand groups nested under /callables/c0, which is "
        "deeper than any reader should recurse. The rest of the file "
        "is an ordinary two-row file with no support."),
    "deep_groups_keys": hdeep(
        "keys",
        "Thirty thousand groups nested under /keys, where a reader "
        "walks looking for key columns. The rest of the file is an "
        "ordinary two-row file with no support."),
    "link_soft_dangling": hmk(
        m_link_soft_dangling,
        "A soft link to a name that does not exist, under each of "
        "/keys, /scalars, /supports and /callables.",
        ["E40"]),
    "link_soft_cyclic": hmk(
        m_link_soft_cyclic,
        "A soft link pointing at itself, under each of /keys, "
        "/scalars, /supports and /callables.",
        ["E40"]),
    "link_external": hmk(
        m_link_external,
        "An external link into a file that is not there, under each "
        "of /keys, /scalars, /supports and /callables. A reader that "
        "followed it would read a file nobody named.",
        ["E40"]),
    "wrong_object_kinds": hmk(
        m_wrong_object_kinds,
        "A member of /keys that is a group and a member of /supports "
        "that is a dataset, where the format requires the opposite "
        "of each.",
        ["E41"]),
    "huge_unwritten_dataset": hmk(
        m_huge_unwritten_dataset,
        "A scalar slot declaring 10^12 rows in a file of two, "
        "chunked and never written. Validating it must not allocate; "
        "only an eager read is E41.",
        ["E16"]),
    "string_invalid_utf8": hmk(
        m_string_invalid_utf8,
        "A category table whose first entry is not valid UTF-8 and "
        "whose second is empty.",
        ["E26"]),
    "scale_attached_twice": hmk(
        m_scale_attached_twice,
        "A second dimension scale attached to a component axis that "
        "already has one.",
        ["E25"]),
    "scale_no_name_attr": hmk(
        m_scale_no_name_attr,
        "A dimension scale with CLASS but no NAME attribute, which "
        "is half of what makes a scale.",
        ["E25"]),
}

# No case needs a library version bound; see deep_base for why the
# two deep files keep the default group layout.
HOSTILE_LIBVER = {}

# wide_keys is about seventeen megabytes and is not committed
# either; --wide produces it, as does --on-demand. See case_wide_keys.
WIDE = {"wide_keys"}

# The two deep files are 31 MB each and are not committed. They are
# generated on demand, with --hostile-deep, and their chain hangs
# below the group named here. Their expected.json is committed like
# every other one, so an implementation knows they exist and knows to
# generate them before running the subset.
DEEP = {
    "deep_groups_callables": "callables/c0",
    "deep_groups_keys": "keys",
}


# ------------------------------------------------------------- output

def write_case(parent, name, builder, libver=None, data=True):
    """Write one case. With data false only expected.json is written,
    which is how the two deep hostile files stay out of the tree."""
    d = os.path.join(parent, name)
    if not os.path.isdir(d):
        os.makedirs(d)
    if data:
        kw = {} if libver is None else {"libver": libver}
        f = h5py.File(os.path.join(d, "case.mes"), "w", **kw)
        try:
            exp = builder(f)
        finally:
            f.close()
    else:
        exp = builder(None)
    with open(os.path.join(d, "expected.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(canonical(exp))
    return exp


def write_on_demand(out):
    """Write the three files that are not committed, and nothing else.

    Their expected.json is committed like every other one and is not
    rewritten here, nor is the manifest, nor any committed case.mes:
    a checkout that runs this stays clean apart from the three files
    the suites need, and the suites then run against the committed
    corpus rather than against whatever this machine's libhdf5 and
    numpy would have written.
    """
    for name in sorted(WIDE):
        d = os.path.join(out, "cases", name)
        if not os.path.isdir(d):
            os.makedirs(d)
        with h5py.File(os.path.join(d, "case.mes"), "w") as f:
            CASES[name](f)
        print("cases/%s/case.mes written" % name)
    for name in sorted(DEEP):
        d = os.path.join(out, "hostile", name)
        if not os.path.isdir(d):
            os.makedirs(d)
        libver = HOSTILE_LIBVER.get(name)
        kw = {} if libver is None else {"libver": libver}
        with h5py.File(os.path.join(d, "case.mes"), "w", **kw) as f:
            HOSTILE[name](f)
        print("hostile/%s/case.mes written" % name)
    return 0


def main(argv):
    deep = "--hostile-deep" in argv
    wide = "--wide" in argv
    rest = [a for a in argv[1:] if not a.startswith("--")]
    out = (rest[0] if rest
           else os.path.dirname(os.path.abspath(__file__)))
    if "--on-demand" in argv:
        return write_on_demand(out)
    entries = []
    for name in sorted(CASES):
        exp = write_case(os.path.join(out, "cases"), name, CASES[name],
                         data=(wide or name not in WIDE))
        entries.append({"name": name,
                        "description": exp["description"]})
    hostile = []
    for name in sorted(HOSTILE):
        exp = write_case(os.path.join(out, "hostile"), name,
                         HOSTILE[name], HOSTILE_LIBVER.get(name),
                         data=(deep or name not in DEEP))
        hostile.append({"name": name,
                        "description": exp["description"]})
    with open(os.path.join(out, "manifest.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(canonical({"corpus": 0, "cases": entries,
                            "hostile": hostile}))
    missing = ([] if wide else ["--wide"]) + \
        ([] if deep else ["--hostile-deep"])
    print("%d cases and %d hostile files written under %s%s"
          % (len(entries), len(hostile), out,
             "" if not missing else
             " (%s not written)" % " and ".join(missing)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
