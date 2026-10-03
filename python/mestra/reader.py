"""Reading a mestra file.

`read(path)` returns a `Dataset` having read attributes and
dataspaces only: the row count, the keys with their roles and
bounds, the supports with their ids, and every slot with its
attributes. Values are read when they are asked for, and a row range
of one slot is read without touching any other slot, which is what
section 29 requires.

Pass `lazy=False` to read every value at once and close the file.

A file is untrusted input. This reader follows no link, walks no
deeper than `limits.MAX_DEPTH`, materialises no more than
`limits.MAX_READ_ELEMENTS` in one call, and turns what it cannot
read into a finding on `dataset.problems` rather than an exception
from inside the library. What it could not copy is named in
`dataset.lossy`, and writing such a dataset is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import h5py

from . import h5safe, limits, opaque
from .callables import callable_from_dict
from .codec import decode_dict
from .encoding import (
    is_fixed_string,
    read_attr,
    string_length,
    support_digest,
)
from .errors import Finding, MestraError
from .model import (
    SUPPORT_KINDS,
    ArraySlot,
    CategoryTable,
    Dataset,
    FileSource,
    Key,
    MemorySource,
    ScalarSlot,
    Storage,
    Support,
)
from .names import logical_dimension

__all__ = ["read", "support_ids", "REFUSED"]

#: The rules under which this reader cannot vouch for what it would
#: return, so `read` refuses a file that breaks one of them rather
#: than hand back values it does not trust. Everything else - a role
#: it does not know, missing units, a cardinality - is the file
#: describing itself badly, and the values still mean what they say.
#: It is the set section 2 of docs/api-conventions.md fixes.
#:
#: One set decides every entry point: the metadata open, an eager
#: read, `mestra info`
#: and any operation that reads a slot all go through `_refuse`
#: below and all name the same rule for the same file. A file this
#: reader opens is a file it will read, and a file it refuses to
#: read is one it refuses to open; there is no file that opens and
#: then fails on the first slot, and none that is refused at the
#: door and would have read cleanly. `test_entry_points.py` holds
#: that over every case of the corpus and every hostile file.
#:
#: The one difference section 29 allows is the element cap, and it
#: goes the other way: "a lazy read and a row-range read are not
#: subject to that last limit ... so the same file may be readable
#: one way and E41 the other". A slot too large to materialise is
#: E41 from an eager read of it and readable a row range at a time.
REFUSED = ("E01", "E16", "E19", "E25", "E26", "E29", "E30", "E40",
           "E41")

_ROOT_ATTRS = ("format", "writer", "created", "aligned",
               "generalisation_group")
_ROOT_GROUPS = ("keys", "scalars", "categories", "supports",
                "callables", "notes", "private")
_KEY_ATTRS = ("role", "units", "lower", "upper", "category",
              "trajectory_group", "parent")
_SLOT_ATTRS = ("role", "varies", "units", "components", "source",
               "output", "statistic", "of", "quantile", "level", "method",
               "category",
               "recomputed", "derived_from", "recipe", "reference")
_SUPPORT_ATTRS = ("kind", "n_nodes", "n_cells", "support_id")
_SUPPORT_MEMBERS = ("node", "cell", "cell_plus_one", "index", "row",
                    "cell_types", "cell_offsets", "cell_connectivity",
                    "coordinates", "node_arrays", "cell_arrays")


def read(path: str, lazy: bool = True, strict: bool = True) -> Dataset:
    """Open a mestra file and return the dataset it holds.

    With `lazy` true, which is the default, no array is read until
    it is asked for and the file stays open; close it with
    `dataset.close()` or use the dataset as a context manager.

    Refusals are `MestraError` naming the rule: E01 for a file that
    is not mestra/0 or not HDF5 at all, E40 for a link this reader
    does not follow, E41 for an object it could not read or an eager
    read past the element limit, and the other rules of `REFUSED`
    for a file whose storage it cannot vouch for. Pass
    `strict=False` to take whatever could be read anyway, with the
    findings on `dataset.problems`.
    """
    handle = h5safe.open_file(str(path))
    if strict:
        _refuse(handle, str(path))
    try:
        dataset = _read(handle, lazy)
    except MestraError:
        handle.close()
        raise
    except Exception as exc:
        handle.close()
        raise MestraError(
            "E41", "this file could not be read: %s"
            % h5safe._brief(exc), str(path)) from None
    dataset.path = str(path)
    if lazy:
        dataset._file = handle
    else:
        # An eager read reads everything, the callables' dictionaries
        # with it, and it must do so before the file is closed.
        dataset.callables.read_all()
        handle.close()
    return dataset


def _refuse(handle: h5py.File, path: str) -> None:
    """Refuse a file this reader cannot vouch for (section 29).

    Section 7 of docs/api-conventions.md fixes what this pass may
    look at: "attributes, dataspaces, link types, and dimension-scale
    structure, and ... a category table in full", and "never a slot's
    data and never a dataset inside a callable's dictionary". The
    nine rules of `REFUSED` are decided from exactly that, which is
    why an open and a read name the same rule for the same file, and
    why opening a file costs a check and not a read. The element cap
    stands beside it for the tables, which are small by construction
    but are still somebody else's number.
    """
    from .validator import scan
    report = scan(handle, limit=limits.MAX_OPEN_ELEMENTS,
                  tables_only=True)
    bad = [f for f in report.errors if f.rule in REFUSED]
    if bad:
        handle.close()
        raise MestraError(
            bad[0].rule, "%s. This file breaks %s, so this reader "
            "cannot vouch for what it would return; read it with "
            "strict=False to take what there is"
            % (bad[0].message, ", ".join(sorted({f.rule for f in bad}))),
            bad[0].where or path)


def support_ids(path: str) -> dict[str, str]:
    """The digest of section 24 for every support in a file.

    This is the cross-file identity check of section 8: same support
    is an id comparison and not an array comparison. It reads a
    support's own arrays and nothing else, and it does not look at
    the file's major version, because the digest is defined over
    bytes and not over what they mean.

    A support that does not declare one of the kinds of section 6
    has no digest (section 24) and is left out.
    """
    out: dict[str, str] = {}
    with h5safe.open_file(str(path)) as f:
        root = _by_name(f)
        supports = root.get("supports")
        if supports is None or not isinstance(supports.obj, h5py.Group):
            return out
        for member in h5safe.members(supports.obj):
            if not member.usable or not isinstance(member.obj,
                                                   h5py.Group):
                continue
            digest = _digest_of(member.obj)
            if digest is not None:
                out[member.name] = digest
    return out


def _digest_of(group: h5py.Group) -> str | None:
    """One support's digest, reading only what section 24 hashes, or
    None for a support whose kind does not say what that is."""
    kind = _attr(group, "kind")
    if kind not in SUPPORT_KINDS:
        return None
    inside = _by_name(group)
    cells: list[Any] = [None, None, None]
    if kind == "mesh":
        for at, which in enumerate(("cell_types", "cell_offsets",
                                    "cell_connectivity")):
            found = inside.get(which)
            if found is not None and isinstance(found.obj, h5py.Dataset):
                cells[at] = h5safe.read_values(found.obj,
                                               found.obj.name)
    axis = None
    if kind == "axis":
        found = inside.get("coordinates")
        if found is not None and isinstance(found.obj, h5py.Dataset):
            axis = h5safe.read_values(found.obj, found.obj.name)
    n_nodes = _attr(group, "n_nodes")
    return support_digest(int(n_nodes) if isinstance(n_nodes, int) else 0,
                          cells[0], cells[1], cells[2], axis)


def _read(f: h5py.File, lazy: bool) -> Dataset:
    ds = Dataset(writer="", created="")
    root = _by_name(f)
    _read_root(f, ds, root)
    ds.n_rows = _row_count(f, root)
    ds.stored_row_count = ds.n_rows
    _read_categories(f, ds, root, lazy)
    _read_keys(f, ds, root, lazy)
    _read_scalars(f, ds, root, lazy)
    found = root.get("row_support")
    if found is not None and isinstance(found.obj, h5py.Dataset):
        ds._row_support = _source(f, "/row_support", found.obj, lazy)
    # One scale index for the whole open, built from the handle this
    # reader holds. Section 21 asks for a map from each scale's
    # address to its link name, built during the reader's own
    # bounded walk; building one per dataset instead is what made an
    # open cost the square of the number of datasets.
    _read_supports(f, ds, root, lazy, h5safe.scale_index(f))
    _read_callables(f, ds, root)
    notes = root.get("notes")
    if notes is not None and isinstance(notes.obj, h5py.Group):
        ds.present.add("notes")
        for name in h5safe.attr_names(notes.obj):
            ds.notes[name] = read_attr(notes.obj, name)
    _read_unknown_groups(ds, root, lazy, h5safe.scale_index(f))
    return ds


# ------------------------------------------------------- safe traversal

def _by_name(group: h5py.Group) -> dict[str, h5safe.Member]:
    """Every member of a group, by name, following no link."""
    return {m.name: m for m in h5safe.members(group)}


def _problem(ds: Dataset, where: str, message: str,
             lossy: bool = False, rule: str = "E41") -> None:
    ds.problems.append(Finding(rule, where, message))
    if lossy:
        ds.lossy.append(where)


def _attr(obj: Any, name: str) -> Any:
    if name in h5safe.attr_names(obj):
        return read_attr(obj, name)
    return None


# ----------------------------------------------------------------- root

def _read_root(f: h5py.File, ds: Dataset,
               root: dict[str, h5safe.Member]) -> None:
    names = h5safe.attr_names(f)
    if "format" not in names:
        raise MestraError("E01", "the root attribute format is missing",
                          "/")
    ds.format = read_attr(f, "format")
    major = _major(ds.format)
    if major != 0:
        raise MestraError(
            "E01", "this reader accepts mestra/0 and the file says %r"
            % (ds.format,), "/")
    ds.writer = read_attr(f, "writer") if "writer" in names else ""
    ds.created = read_attr(f, "created") if "created" in names else ""
    if "aligned" in names:
        ds.stored_aligned = bool(read_attr(f, "aligned"))
    if "generalisation_group" in names:
        ds.generalisation_group = read_attr(f, "generalisation_group")
    for name in names:
        if name not in _ROOT_ATTRS:
            ds.extra[name] = read_attr(f, name)
    for name, member in root.items():
        if not member.usable and member.problem:
            _problem(ds, "/" + name, member.problem,
                     lossy=member.kind == h5safe.EXTERNAL,
                     rule=member.rule)


def _major(text: Any) -> int | None:
    """The major version of a `format` string, or None."""
    if not isinstance(text, str) or not text.startswith("mestra/"):
        return None
    tail = text[len("mestra/"):]
    return int(tail) if tail.isdigit() and len(tail) < 10 else None


def _row_count(f: h5py.File, root: dict[str, h5safe.Member]) -> int:
    """The row count, from the `row` scale (section 21)."""
    found = root.get("row")
    if found is not None and isinstance(found.obj, h5py.Dataset):
        return int(found.obj.shape[0]) if found.obj.shape else 0
    for group in ("keys", "scalars"):
        holder = root.get(group)
        if holder is None or not isinstance(holder.obj, h5py.Group):
            continue
        for member in h5safe.members(holder.obj):
            if isinstance(member.obj, h5py.Dataset) and member.obj.shape:
                return int(member.obj.shape[0])
    return 0


# ----------------------------------------------------------- the pieces

def _source(f: h5py.File, path: str, dset: h5py.Dataset,
            lazy: bool) -> Any:
    """A source for one dataset, lazy or read at once."""
    decode = is_fixed_string(dset.dtype)
    source = FileSource(f, path, dset.shape, dset.dtype, decode)
    if lazy:
        return source
    return MemorySource(source.read())


def _storage(dset: h5py.Dataset) -> Storage:
    try:
        compression = dset.compression
        options = dset.compression_opts
        shuffle = bool(dset.shuffle)
        chunks = dset.chunks
    except Exception:                                   # pragma: no cover
        return Storage()
    return Storage(chunks=chunks,
                   gzip=options if compression == "gzip" else None,
                   shuffle=shuffle, contiguous=chunks is None)


def _dims(dset: h5py.Dataset, fallback: tuple[str, ...],
          scales: Mapping[int, str]) -> tuple[str, ...]:
    """The logical dimension names, taken from the scales.

    A reader takes a dimension's name from the scale's link name and
    never from its NAME attribute (section 21). When an axis carries
    no scale or more than one, which is a broken file (E25), the
    names the slot's own attributes imply are used instead.
    """
    names = h5safe.scale_names(dset, scales)
    if not names or any(len(axis) != 1 for axis in names):
        return fallback
    return tuple(logical_dimension(axis[0]) for axis in names)


def _extra(obj: Any, known: tuple[str, ...]) -> dict[str, Any]:
    return {name: read_attr(obj, name) for name in h5safe.attr_names(obj)
            if name not in known}


def _group_of(ds: Dataset, root: dict[str, h5safe.Member],
              name: str) -> h5py.Group | None:
    """One of the root groups, when it is a group and opened."""
    found = root.get(name)
    if found is None:
        return None
    if not found.usable:
        return None
    if not isinstance(found.obj, h5py.Group):
        _problem(ds, "/" + name, "this is a dataset where the format "
                                 "has a group, so it is not read")
        return None
    ds.present.add(name)
    return found.obj


def _read_categories(f: h5py.File, ds: Dataset,
                     root: dict[str, h5safe.Member], lazy: bool) -> None:
    group = _group_of(ds, root, "categories")
    if group is None:
        return
    for member in h5safe.members(group):
        where = "/categories/" + member.name
        if not member.usable:
            _problem(ds, where, member.problem or "unreadable",
                     lossy=member.kind == h5safe.EXTERNAL,
                     rule=member.rule)
            continue
        if not isinstance(member.obj, h5py.Dataset):
            _problem(ds, where, "a category table is a dataset, and "
                                "this is a group")
            continue
        try:
            values = h5safe.read_values(member.obj, where)
        except MestraError as exc:
            _problem(ds, where, exc.message, lossy=True)
            continue
        entries = [h5safe.decode_bytes(v) if isinstance(v, bytes)
                   else str(v) for v in values]
        ds.categories[member.name] = CategoryTable(
            entries, itemsize=string_length(member.obj.dtype))


def _read_keys(f: h5py.File, ds: Dataset,
               root: dict[str, h5safe.Member], lazy: bool) -> None:
    group = _group_of(ds, root, "keys")
    if group is None:
        return
    for member in h5safe.members(group):
        where = "/keys/" + member.name
        if not member.usable:
            _problem(ds, where, member.problem or "unreadable",
                     lossy=member.kind == h5safe.EXTERNAL,
                     rule=member.rule)
            continue
        dset = member.obj
        if not isinstance(dset, h5py.Dataset):
            _problem(ds, where, "a key is a dataset, and this is a "
                                "group")
            continue
        attrs = h5safe.attr_names(dset)
        ds.keys[member.name] = Key(
            member.name,
            read_attr(dset, "role") if "role" in attrs else "",
            units=read_attr(dset, "units") if "units" in attrs else None,
            lower=read_attr(dset, "lower") if "lower" in attrs else None,
            upper=read_attr(dset, "upper") if "upper" in attrs else None,
            category=(read_attr(dset, "category")
                      if "category" in attrs else None),
            trajectory_group=(read_attr(dset, "trajectory_group")
                              if "trajectory_group" in attrs else None),
            parent=read_attr(dset, "parent") if "parent" in attrs
            else None,
            data=_source(f, where, dset, lazy),
            itemsize=string_length(dset.dtype),
            extra=_extra(dset, _KEY_ATTRS))


def _slot_attrs(obj: Any) -> dict[str, Any]:
    attrs = h5safe.attr_names(obj)
    out = {name: read_attr(obj, name) for name in attrs
           if name in _SLOT_ATTRS}
    out["extra"] = _extra(obj, _SLOT_ATTRS)
    return out


def _read_scalars(f: h5py.File, ds: Dataset,
                  root: dict[str, h5safe.Member], lazy: bool) -> None:
    group = _group_of(ds, root, "scalars")
    if group is None:
        return
    for member in h5safe.members(group):
        where = "/scalars/" + member.name
        if not member.usable:
            _problem(ds, where, member.problem or "unreadable",
                     lossy=member.kind == h5safe.EXTERNAL,
                     rule=member.rule)
            continue
        got = _slot_attrs(member.obj)
        slot = ScalarSlot(
            member.name, units=got.get("units"),
            source=got.get("source", "data"), output=got.get("output"),
            statistic=got.get("statistic"), of=got.get("of"),
            quantile=got.get("quantile"), level=got.get("level"),
            method=got.get("method"), extra=got["extra"])
        if isinstance(member.obj, h5py.Dataset):
            slot.data = _source(f, where, member.obj, lazy)
            slot.storage = _storage(member.obj)
        ds.scalars[member.name] = slot


def _read_supports(f: h5py.File, ds: Dataset,
                   root: dict[str, h5safe.Member], lazy: bool,
                   scales: Mapping[int, str]) -> None:
    group = _group_of(ds, root, "supports")
    if group is None:
        return
    for member in h5safe.members(group):
        where = "/supports/" + member.name
        if not member.usable:
            _problem(ds, where, member.problem or "unreadable",
                     lossy=member.kind == h5safe.EXTERNAL,
                     rule=member.rule)
            continue
        if not isinstance(member.obj, h5py.Group):
            _problem(ds, where, "a support is a group, and this is a "
                                "dataset")
            continue
        ds.supports[member.name] = _read_support(
            f, ds, member.name, member.obj, lazy, scales)


def _read_support(f: h5py.File, ds: Dataset, name: str,
                  group: h5py.Group, lazy: bool,
                  scales: Mapping[int, str]) -> Support:
    attrs = h5safe.attr_names(group)
    n_nodes = read_attr(group, "n_nodes") if "n_nodes" in attrs else 0
    n_cells = read_attr(group, "n_cells") if "n_cells" in attrs else 0
    support = Support(
        name,
        read_attr(group, "kind") if "kind" in attrs else None,
        n_nodes=n_nodes if isinstance(n_nodes, int) else 0,
        n_cells=n_cells if isinstance(n_cells, int) else 0,
        stored_support_id=(read_attr(group, "support_id")
                           if "support_id" in attrs else None),
        extra=_extra(group, _SUPPORT_ATTRS))
    support.dataset = ds
    base = "/supports/" + name
    inside = _by_name(group)
    for which in ("cell_types", "cell_offsets", "cell_connectivity"):
        found = inside.get(which)
        if found is None:
            continue
        if not found.usable or not isinstance(found.obj, h5py.Dataset):
            _problem(ds, "%s/%s" % (base, which),
                     found.problem or "this is not a dataset",
                     rule=found.rule)
            continue
        support._cells[which] = _source(f, base + "/" + which, found.obj,
                                        lazy)
    found = inside.get("coordinates")
    if found is not None:
        if found.usable:
            support.coordinates = _read_array(
                f, support, "coordinates", found.obj,
                base + "/coordinates", "node", lazy, scales)
        else:
            _problem(ds, base + "/coordinates",
                     found.problem or "unreadable",
                     lossy=found.kind == h5safe.EXTERNAL,
                     rule=found.rule)
    for which, location, into in (
            ("node_arrays", "node", support.node_arrays),
            ("cell_arrays", "cell", support.cell_arrays)):
        holder = inside.get(which)
        if holder is None:
            continue
        if not holder.usable or not isinstance(holder.obj, h5py.Group):
            _problem(ds, "%s/%s" % (base, which),
                     holder.problem or "this is not a group",
                     rule=holder.rule)
            continue
        support.present.add(which)
        for member in h5safe.members(holder.obj):
            where = "%s/%s/%s" % (base, which, member.name)
            if not member.usable:
                _problem(ds, where, member.problem or "unreadable",
                         lossy=member.kind == h5safe.EXTERNAL)
                continue
            into[member.name] = _read_array(
                f, support, member.name, member.obj, where, location,
                lazy, scales)
    for member in h5safe.members(group):
        if member.name in _SUPPORT_MEMBERS:
            continue
        where = "%s/%s" % (base, member.name)
        if not member.usable:
            _problem(ds, where, member.problem or "unreadable",
                     lossy=member.kind == h5safe.EXTERNAL,
                     rule=member.rule)
            continue
        if isinstance(member.obj, h5py.Group):
            support.opaque[member.name] = opaque.capture(
                member.obj, where, 0, ds.problems, ds.lossy, lazy,
                scales)
    return support


def _read_array(f: h5py.File, support: Support, name: str, member: Any,
                path: str, location: str, lazy: bool,
                scales: Mapping[int, str]) -> ArraySlot:
    got = _slot_attrs(member)
    components = got.get("components", 1)
    slot = ArraySlot(
        name, got.get("role", ""), varies=got.get("varies", "none"),
        components=components if isinstance(components, int) else 1,
        location=location,
        units=got.get("units"), category=got.get("category"),
        recomputed=got.get("recomputed"),
        derived_from=got.get("derived_from"), recipe=got.get("recipe"),
        reference=got.get("reference"), support=support,
        source=got.get("source", "data"), output=got.get("output"),
        statistic=got.get("statistic"), of=got.get("of"),
        quantile=got.get("quantile"), level=got.get("level"),
        method=got.get("method"), extra=got["extra"])
    if isinstance(member, h5py.Dataset):
        slot.data = _source(f, path, member, lazy)
        slot.storage = _storage(member)
        slot.dims = _dims(member, slot.dims, scales)
    return slot


def _read_callables(f: h5py.File, ds: Dataset,
                    root: dict[str, h5safe.Member]) -> None:
    group = _group_of(ds, root, "callables")
    if group is None:
        return
    for member in h5safe.members(group):
        where = "/callables/" + member.name
        if not member.usable:
            _problem(ds, where, member.problem or "unreadable",
                     lossy=member.kind == h5safe.EXTERNAL,
                     rule=member.rule)
            continue
        if not isinstance(member.obj, h5py.Group):
            _problem(ds, where, "a callable is a group, and this is a "
                                "dataset")
            continue
        attrs = h5safe.attr_names(member.obj)
        kind = read_attr(member.obj, "type") if "type" in attrs else ""
        line = read_attr(member.obj, "repr") if "repr" in attrs else None
        # The id, the type and the repr line are a link name and two
        # attributes, so an open has them. The dictionary is datasets
        # and section 7 of docs/api-conventions.md says an open never
        # reads one, so it waits until something asks for the
        # callable.
        ds.callables.defer(member.name, _build_callable(
            ds, where, member.obj,
            kind if isinstance(kind, str) else "",
            line if isinstance(line, str) else None))


def _build_callable(ds: Dataset, where: str, group: h5py.Group,
                    kind: str, line: str | None) -> Any:
    """What `Dataset.callables` runs the first time it is asked for
    this callable: decode the dictionary and hand it to its type."""
    def build() -> Any:
        trouble: list[Finding] = []
        body = decode_dict(group, problems=trouble)
        if trouble:
            ds.problems.extend(trouble)
            ds.lossy.append(where)
        try:
            return callable_from_dict(kind, body, line)
        except MestraError as exc:
            _problem(ds, where, "this callable's dictionary does not "
                                "fit its type: %s" % exc.message)
            return callable_from_dict("", body, None)
    return build


def _read_unknown_groups(ds: Dataset, root: dict[str, h5safe.Member],
                         lazy: bool,
                         scales: Mapping[int, str]) -> None:
    """Keep /private and any group section 28 lets a version add.

    A reader must not interpret /private and must not require it
    (section 29); it is copied whole so that rewriting a file does
    not throw away the producer's own records. What it could not
    copy, because of a link, a depth or a size, is a finding and a
    refusal to rewrite rather than a silent loss.

    A lazy open copies the shape of what is there and leaves the
    values in the file, because section 29 says an open reads no
    array and a producer's own records are arrays like any other.
    `scales` is this pass's scale index, which is how an axis
    attached to a scale outside the copied group is named.
    """
    for name, member in root.items():
        if name == "row_support":
            continue
        if name in _ROOT_GROUPS and name != "private":
            continue
        if not member.usable:
            continue                  # already reported by _read_root
        if not isinstance(member.obj, h5py.Group):
            continue                  # a dimension scale at root
        ds.opaque[name] = opaque.capture(member.obj, "/" + name, 0,
                                         ds.problems, ds.lossy, lazy,
                                         scales)
