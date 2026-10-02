"""The objects a mestra file holds, and the calls that build one.

A `Dataset` is the whole file: keys, scalars, category tables,
supports with their arrays, callables, and the metadata of section
11. Every array carries its dimension names, so a value is found by
name and never by axis position.

Building one from arrays takes a handful of calls:

    ds = mestra.Dataset(writer="my tool 1")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    ds.add_category_table("member", ["wing_a", "wing_b"])
    ds.add_key("member", [0, 1], role="group", category="member")
    ds.set_generalisation_group("member")
    ds.add_scalar("cl", [0.25, 0.55], units="1")
    s = ds.add_support("s0", coordinates=xy, cells=(types, offsets,
                                                    conn))
    s.add_node_array("pressure", p, units="Pa",
                     dims=("row", "node"))

The dimensions, the bounds, the component axis and the support id are
filled in. Every builder refuses at build time, naming the rule of
section 14 and the argument to change, anything the validator would
refuse in the file.
"""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from . import h5safe
from .encoding import decode_string, support_digest
from .errors import Finding, MestraError
from .names import is_legal_name, is_reserved

__all__ = [
    "NamedArray",
    "fit_dims",
    "Storage",
    "CategoryTable",
    "Key",
    "ScalarSlot",
    "ArraySlot",
    "Support",
    "Callables",
    "Dataset",
    "KEY_ROLES",
    "ARRAY_ROLES",
    "STATISTICS",
    "STATUS_WORDS",
    "SUPPORT_KINDS",
    "FORMAT",
]

FORMAT = "mestra/0"

#: Section 3. The cardinality of each key role, None for 0..n.
KEY_ROLES: dict[str, int | None] = {
    "design": None, "condition": None, "time": 1, "categorical": None,
    "group": None, "split": 1, "id": 1, "status": 1,
}

#: Section 3. The cardinality of each array role, per support.
ARRAY_ROLES: dict[str, int | None] = {
    "coordinates": 1, "field": None, "label": None, "weight": 1,
    "normal": 1, "derived": None,
}

#: Section 9.
STATISTICS = ("value", "mean", "band", "std", "quantile", "draw")

#: Section 6.
SUPPORT_KINDS = ("mesh", "axis", "none")

#: Section 19: which keys are stored as integers.
_INTEGER_KEY_ROLES = ("categorical", "group", "split", "status")

#: Roles whose key carries units, and roles whose key names a
#: category table instead (section 3).
_UNIT_KEY_ROLES = ("design", "condition", "time")
_TABLE_KEY_ROLES = ("categorical", "group", "split", "status")

#: The status words section 3 recommends. A producer may add its own,
#: and only `converged` means the row is fit for modelling.
STATUS_WORDS = ("converged", "failed", "partial")

#: How many elements `NamedArray` prints rather than summarises.
_REPR_ELEMENTS = 12


# --------------------------------------------------------------- arrays

def fit_dims(dims: Sequence[str], ndim: int) -> tuple[str, ...]:
    """Names for `ndim` axes, whatever the slot declared.

    A file may declare a slot with dimensions its stored array does
    not have; that is E04 or E25 and the validator's business, but a
    reader still has to hand back an array with one name per axis.
    The names it keeps are the leading ones, and any axis left over
    is called "?" rather than guessed at.
    """
    names = tuple(dims)
    if len(names) == ndim:
        return names
    if len(names) > ndim:
        return names[:ndim]
    return names + ("?",) * (ndim - len(names))


class NamedArray:
    """A numpy array together with the name of each of its axes.

    `values` is the array as stored, `dims` the logical dimension
    names of section 21 in the same order. Index by name with
    `at()`, or reorder with `transpose()`; a reader in another
    language may hand back another order, and the names are what
    both agree on.
    """

    __slots__ = ("values", "dims")

    def __init__(self, values: Any, dims: Sequence[str]) -> None:
        self.values = np.asarray(values)
        self.dims = tuple(dims)
        if len(self.dims) != self.values.ndim:
            raise ValueError(
                "%d dimension names for an array of %d axes"
                % (len(self.dims), self.values.ndim))

    # -- the array itself

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        if dtype is None:
            return self.values
        return self.values.astype(dtype, copy=False)

    def __getitem__(self, item: Any) -> Any:
        return self.values[item]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def ndim(self) -> int:
        return self.values.ndim

    def axis(self, name: str) -> int:
        """The position of the axis called `name`.

        "instance" is accepted for the leading axis of an array that
        varies along a group, which is what the corpus calls it.
        """
        for at, dim in enumerate(self.dims):
            if dim == name:
                return at
            if name == "instance" and dim.startswith("group:"):
                return at
            if name == "group" and dim.startswith("group:"):
                return at
        raise KeyError(
            "no axis called %r; this array has %s"
            % (name, ", ".join(self.dims) or "no axes"))

    def at(self, **where: int) -> Any:
        """The value at the named indices, for example

            array.at(row=1, node=3, component=0)

        Axes not named come back whole.
        """
        index: list[Any] = [slice(None)] * self.values.ndim
        for name, position in where.items():
            index[self.axis(name)] = position
        return self.values[tuple(index)]

    def transpose(self, *dims: str) -> NamedArray:
        """The same values with the axes in the order named."""
        order = [self.axis(name) for name in dims]
        if len(order) != self.values.ndim:
            raise ValueError("name every axis to transpose")
        return NamedArray(self.values.transpose(order), dims)

    def __repr__(self) -> str:
        """The values themselves when there are few, as numpy does.

        A one-element scalar column is the commonest thing to print,
        and a shape is not what the caller wanted to see. Above
        `_REPR_ELEMENTS` the shape is all that would fit.
        """
        if self.values.size <= _REPR_ELEMENTS:
            body = np.array2string(self.values, separator=", ",
                                   threshold=_REPR_ELEMENTS)
        else:
            body = "x".join(str(n) for n in self.values.shape) or "scalar"
        return "NamedArray(%s, dims=%s)" % (body, ", ".join(self.dims))


class Storage:
    """How a dataset is laid out, when the file says something other
    than the default of section 23.

    A reader fills this in so that a file written back out keeps the
    layout it had; a writer that finds it empty uses the default.
    """

    __slots__ = ("chunks", "gzip", "shuffle", "contiguous")

    def __init__(self, chunks: Sequence[int] | None = None,
                 gzip: int | None = None, shuffle: bool = False,
                 contiguous: bool = False) -> None:
        self.chunks = tuple(int(c) for c in chunks) if chunks else None
        self.gzip = gzip
        self.shuffle = shuffle
        self.contiguous = contiguous

    def __repr__(self) -> str:
        return "Storage(chunks=%r, gzip=%r, shuffle=%r)" % (
            self.chunks, self.gzip, self.shuffle)


class _Source:
    """Where a slot's values come from: memory, or the file."""

    def read(self, rows: slice | None = None) -> np.ndarray:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def dtype(self) -> np.dtype:
        raise NotImplementedError


class MemorySource(_Source):
    """Values already in memory."""

    __slots__ = ("array",)

    def __init__(self, array: Any) -> None:
        self.array = np.asarray(array)

    def read(self, rows: slice | None = None) -> np.ndarray:
        if rows is None:
            return self.array
        return self.array[rows]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype


class FileSource(_Source):
    """Values still in the file, read when they are asked for.

    Reading a row range touches that range of that one dataset and
    nothing else, which is what section 29 requires of a reader.
    `reads` counts the reads that have happened, so that a test can
    show that opening a file reads no array at all.
    """

    __slots__ = ("_file", "path", "_shape", "_dtype", "_decode", "reads")

    def __init__(self, file: Any, path: str, shape: Sequence[int],
                 dtype: np.dtype, decode: bool = False) -> None:
        self._file = file
        self.path = path
        self._shape = tuple(int(n) for n in shape)
        self._dtype = dtype
        self._decode = decode
        self.reads = 0

    def read(self, rows: slice | None = None) -> np.ndarray:
        """The values, or one row range of them.

        A read that would materialise more elements than
        `limits.MAX_READ_ELEMENTS` is refused rather than attempted,
        and so is a dataset the library itself will not convert.
        """
        self.reads += 1
        if not self._file:
            raise MestraError(
                "E41", "the file has been closed; read it with "
                "lazy=False to keep the values", self.path)
        values = h5safe.read_values(self._file[self.path], self.path,
                                    rows)
        if self._decode:
            return _decode_strings(values)
        return values

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype


def _wrap(values: Any, dtype: str) -> _Source | None:
    """Hold values that are already in memory, or nothing."""
    if values is None:
        return None
    if isinstance(values, _Source):
        return values
    return MemorySource(np.asarray(values, dtype=dtype))


def _decode_strings(values: Any) -> np.ndarray:
    """Fixed-length bytes as read from HDF5 into text (section 18).

    Leniently: a string that is not valid UTF-8 comes back with the
    bad bytes replaced, and the validator reports it (E26).
    """
    flat = [decode_string(v) if isinstance(v, bytes) else str(v)
            for v in np.asarray(values).reshape(-1)]
    return np.array(flat, dtype=np.str_).reshape(np.asarray(values).shape)


# ------------------------------------------------------ category tables

class CategoryTable(Sequence[str]):
    """The named categories of a key or a label.

    A category id is the position of its entry: the first entry is
    id 0 (section 21).
    """

    __slots__ = ("entries", "itemsize")

    def __init__(self, entries: Sequence[str],
                 itemsize: int | None = None) -> None:
        self.entries = [str(e) for e in entries]
        #: The stored byte size, when a file declared a larger one.
        self.itemsize = itemsize

    def __getitem__(self, at: Any) -> Any:
        return self.entries[at]

    def __len__(self) -> int:
        return len(self.entries)

    def id_of(self, name: str) -> int:
        """The category id of an entry, by name."""
        try:
            return self.entries.index(name)
        except ValueError:
            raise KeyError(
                "no category %r; this table holds %s"
                % (name, ", ".join(self.entries))) from None

    def __repr__(self) -> str:
        return "CategoryTable(%r)" % (self.entries,)


# ----------------------------------------------------------------- keys

class Key:
    """One key column: a per-row value with a role (section 3)."""

    def __init__(self, name: str, role: str, *, units: str | None = None,
                 lower: float | None = None, upper: float | None = None,
                 category: str | None = None,
                 trajectory_group: str | None = None,
                 parent: str | None = None,
                 data: _Source | None = None,
                 itemsize: int | None = None,
                 extra: Mapping[str, Any] | None = None) -> None:
        self.name = name
        self.role = role
        self.units = units
        self.lower = lower
        self.upper = upper
        self.category = category
        self.trajectory_group = trajectory_group
        self.parent = parent
        self.data = data
        #: The stored byte size of a string id column.
        self.itemsize = itemsize
        #: Attributes this reader does not know, kept for a rewrite.
        self.extra = dict(extra or {})

    @property
    def dims(self) -> tuple[str, ...]:
        """A key column has one dimension, `row` (section 19)."""
        return ("row",)

    @property
    def dtype(self) -> np.dtype:
        """The dtype the column is stored in (section 19)."""
        return self.data.dtype if self.data is not None else np.dtype("f8")

    @property
    def values(self) -> NamedArray:
        """The whole column, with its dimension name."""
        return self.read()

    def read(self, rows: slice | None = None) -> NamedArray:
        """The column, or one row range of it."""
        if self.data is None:
            return NamedArray(np.zeros(0), ("row",))
        values = self.data.read(rows)
        return NamedArray(values, fit_dims(("row",),
                                           np.asarray(values).ndim))

    def __repr__(self) -> str:
        return "Key(%r, role=%r, units=%r)" % (
            self.name, self.role, self.units)


# ---------------------------------------------------------------- slots

class Slot:
    """A quantity that holds stored data or names a callable."""

    def __init__(self, name: str, *, units: str | None = None,
                 source: str = "data", output: str | None = None,
                 statistic: str | None = None, of: str | None = None,
                 quantile: float | None = None,
                 level: float | None = None, method: str | None = None,
                 data: _Source | None = None,
                 storage: Storage | None = None,
                 extra: Mapping[str, Any] | None = None) -> None:
        self.name = name
        self.units = units
        self.source = source
        self.output = output
        self.statistic = statistic
        self.of = of
        self.quantile = quantile
        #: The coverage a band claims and how it was made (section 9);
        #: None on anything that is not a band.
        self.level = level
        self.method = method
        self.data = data
        self.storage = storage or Storage()
        self.extra = dict(extra or {})

    @property
    def is_callable(self) -> bool:
        """True when a callable serves this slot rather than data."""
        return self.source.startswith("callable:")

    @property
    def callable_id(self) -> str | None:
        """The id of the callable that serves the slot, or None."""
        if self.is_callable:
            return self.source[len("callable:"):]
        return None


class ScalarSlot(Slot):
    """A per-row quantity of interest, with units (section 2)."""

    role = "scalar"

    @property
    def dims(self) -> tuple[str, ...]:
        """A scalar has one dimension, `row` (section 19)."""
        return ("row",)

    def read(self, rows: slice | None = None) -> NamedArray:
        """The column, or one row range of it."""
        if self.data is None:
            raise MestraError(
                "E30", "this slot is served by %s and holds no data"
                % self.source, self.name)
        values = self.data.read(rows)
        return NamedArray(values, fit_dims(("row",),
                                           np.asarray(values).ndim))

    @property
    def values(self) -> NamedArray:
        """The whole column, with its dimension name."""
        return self.read()

    def __repr__(self) -> str:
        return "ScalarSlot(%r, units=%r, source=%r)" % (
            self.name, self.units, self.source)


class ArraySlot(Slot):
    """A field-like quantity on a support (sections 3 and 5)."""

    def __init__(self, name: str, role: str, *, varies: str = "none",
                 components: int = 1, location: str = "node",
                 units: str | None = None, category: str | None = None,
                 recomputed: bool | None = None,
                 derived_from: str | None = None,
                 recipe: str | None = None, reference: str | None = None,
                 dims: Sequence[str] | None = None,
                 support: Support | None = None, **rest: Any) -> None:
        super().__init__(name, units=units, **rest)
        self.role = role
        self.varies = varies
        self.components = int(components)
        self.location = location
        self.category = category
        self.recomputed = recomputed
        self.derived_from = derived_from
        self.recipe = recipe
        self.reference = reference
        self.support = support
        self._dims = tuple(dims) if dims is not None else None

    @property
    def dims(self) -> tuple[str, ...]:
        """The logical name of each axis, leading axis first.

        (row | group:<k> | nothing, [draw], node | cell, component),
        which is the order section 19 stores them in.
        """
        if self._dims is not None:
            return self._dims
        return array_dims(self.varies, self.statistic, self.location)

    @dims.setter
    def dims(self, value: Sequence[str] | None) -> None:
        self._dims = tuple(value) if value is not None else None

    def read(self, rows: slice | None = None) -> NamedArray:
        """The array, or one range of its leading dimension.

        `rows` applies to the leading dimension only, and in an
        unaligned file that dimension counts the rows on this
        support and not the file's rows (section 22).
        """
        if self.data is None:
            raise MestraError(
                "E30", "this slot is served by %s and holds no data"
                % self.source, self.name)
        values = self.data.read(rows)
        return NamedArray(values, fit_dims(self.dims,
                                           np.asarray(values).ndim))

    @property
    def values(self) -> NamedArray:
        """The whole array, with the name of each of its axes."""
        return self.read()

    def __repr__(self) -> str:
        return ("ArraySlot(%r, role=%r, varies=%r, units=%r, "
                "source=%r)" % (self.name, self.role, self.varies,
                                self.units, self.source))


def array_dims(varies: str, statistic: str | None,
               location: str) -> tuple[str, ...]:
    """The dimension names an array slot has, from its attributes."""
    dims: list[str] = []
    if varies != "none":
        dims.append(varies)
    if statistic == "draw":
        dims.append("draw")
    dims.append(location)
    dims.append("component")
    return tuple(dims)


# ------------------------------------------------------------- supports

class _Arrays(dict):
    """The arrays at one location, with an answer for a name that is
    not there.

    A plain `KeyError` for `node_arrays["coordinates"]` tells a
    caller nothing, and coordinates are the one array that is an
    attribute of the support rather than a member of this mapping.
    """

    def __init__(self, location: str) -> None:
        super().__init__()
        self._location = location

    def __missing__(self, name: str) -> Any:
        if name == "coordinates":
            raise KeyError(
                "the coordinates are support.coordinates, not "
                "support.%s_arrays['coordinates']" % self._location)
        raise KeyError(
            "no %s array called %r; this support has %s"
            % (self._location, name,
               ", ".join(sorted(self)) or "none"))


class Support:
    """The structure a field lives on: a mesh, an axis, or none.

    It carries `kind` (None when a file it was read from did not say
    it), `n_nodes`, `n_cells`, the three cell arrays,
    `support_id`, and the arrays on it: `coordinates`, which a
    support has exactly one of and is part of what the support is,
    and `node_arrays` and `cell_arrays`, which hold every other one.
    Its builders are `add_node_array`, `add_cell_array`,
    `add_callable_slot` and, for coordinates a callable serves,
    `set_callable_coordinates`.
    """

    def __init__(self, name: str, kind: str | None = "mesh", *,
                 n_nodes: int = 0, n_cells: int = 0,
                 cell_types: Any = None, cell_offsets: Any = None,
                 cell_connectivity: Any = None,
                 stored_support_id: str | None = None,
                 extra: Mapping[str, Any] | None = None) -> None:
        self.name = name
        self.kind = kind
        self.n_nodes = int(n_nodes)
        self.n_cells = int(n_cells)
        self._cells: dict[str, _Source | None] = {
            "cell_types": _wrap(cell_types, "<u1"),
            "cell_offsets": _wrap(cell_offsets, "<i8"),
            "cell_connectivity": _wrap(cell_connectivity, "<i8"),
        }
        self.coordinates: ArraySlot | None = None
        self.node_arrays: dict[str, ArraySlot] = _Arrays("node")
        self.cell_arrays: dict[str, ArraySlot] = _Arrays("cell")
        #: What the file said, kept so that the validator can check
        #: it. Changing a cell array clears it, so that a support
        #: whose arrays have changed writes a fresh digest.
        self.stored_support_id = stored_support_id
        self.extra = dict(extra or {})
        self.dataset: Dataset | None = None
        #: Optional groups the support carried, empty ones included.
        self.present: set[str] = set()
        #: Groups under the support this reader does not know.
        self.opaque: dict[str, Any] = {}

    # -- the cells

    @property
    def cell_types(self) -> np.ndarray | None:
        """VTK cell type codes, one per cell (section 20)."""
        return self._read_cells("cell_types")

    @cell_types.setter
    def cell_types(self, values: Any) -> None:
        self._cells["cell_types"] = _wrap(values, "<u1")
        self.stored_support_id = None

    @property
    def cell_offsets(self) -> np.ndarray | None:
        """Where each cell starts in the connectivity, n_cells + 1."""
        return self._read_cells("cell_offsets")

    @cell_offsets.setter
    def cell_offsets(self, values: Any) -> None:
        self._cells["cell_offsets"] = _wrap(values, "<i8")
        self.stored_support_id = None

    @property
    def cell_connectivity(self) -> np.ndarray | None:
        """The nodes of every cell, cell by cell (section 20)."""
        return self._read_cells("cell_connectivity")

    @cell_connectivity.setter
    def cell_connectivity(self, values: Any) -> None:
        self._cells["cell_connectivity"] = _wrap(values, "<i8")
        self.stored_support_id = None

    def _read_cells(self, which: str) -> np.ndarray | None:
        source = self._cells.get(which)
        return None if source is None else source.read()

    def cells_of(self, cell: int) -> np.ndarray:
        """The nodes of one cell, in VTK order for its type."""
        offsets = self.cell_offsets
        connectivity = self.cell_connectivity
        if offsets is None or connectivity is None:
            raise MestraError(
                "E38", "a support of kind %s carries no cells"
                % self.kind, self.name)
        return connectivity[offsets[cell]:offsets[cell + 1]]

    # -- identity

    @property
    def support_id(self) -> str:
        """The content hash of section 24.

        What the file stored when it stored one, and otherwise the
        digest computed from this support's own arrays.
        """
        if self.stored_support_id is not None:
            return self.stored_support_id
        return self.computed_support_id()

    def computed_support_id(self) -> str:
        """The digest of section 24, computed from the arrays.

        Coordinates of a mesh support are not hashed, because they
        may vary between rows while the support does not; the
        coordinates of an axis support are part of its identity.

        The declared kind decides which arrays are hashed, so a
        support that declares none, or one section 6 does not name,
        has no digest and this refuses it (E39, E02).
        """
        if self.kind is None:
            raise MestraError(
                "E39", "support %s does not say its kind, and the kind "
                "decides which of its arrays section 24 hashes, so it has "
                "no support_id" % self.name, self.name)
        if self.kind not in SUPPORT_KINDS:
            raise MestraError(
                "E02", "support %s is of kind %r, which section 6 does not "
                "have, so section 24 says nothing about which of its "
                "arrays to hash and it has no support_id"
                % (self.name, self.kind), self.name)
        axis_coordinates = None
        if (self.kind == "axis" and self.coordinates is not None
                and self.coordinates.data is not None):
            axis_coordinates = self.coordinates.data.read()
        if self.kind != "mesh":
            # A support of kind axis or none has no cell arrays, so
            # they contribute no bytes at all for it (section 24).
            return support_digest(self.n_nodes,
                                  axis_coordinates=axis_coordinates)
        return support_digest(self.n_nodes, self.cell_types,
                              self.cell_offsets, self.cell_connectivity,
                              axis_coordinates)

    # -- arrays

    def arrays(self) -> dict[str, ArraySlot]:
        """Every array on this support, coordinates included."""
        out: dict[str, ArraySlot] = {}
        if self.coordinates is not None:
            out["coordinates"] = self.coordinates
        for name, slot in self.node_arrays.items():
            out["node_arrays/" + name] = slot
        for name, slot in self.cell_arrays.items():
            out["cell_arrays/" + name] = slot
        return out

    def add_node_array(self, name: str, values: Any = None, *,
                       units: str | None = None,
                       dims: Sequence[str] | None = None,
                       role: str = "field", varies: str | None = None,
                       components: int | None = None,
                       category: str | None = None,
                       categories: Sequence[str] | None = None,
                       statistic: str | None = None,
                       of: str | None = None,
                       quantile: float | None = None,
                       level: float | None = None,
                       method: str | None = None,
                       recomputed: bool | None = None,
                       derived_from: str | None = None,
                       recipe: str | None = None,
                       reference: str | None = None,
                       callable_id: str | None = None,
                       output: str | None = None,
                       dtype: Any = None, **rest: Any) -> ArraySlot:
        """Add an array over this support's nodes.

        `dims` names the axes of the array you are passing, in your
        own axis order: "row" or "group:<k>", "draw", "node" and
        "component". The builder reads `varies` and `components` off
        it and stores the array in the order section 19 requires.
        Name every axis your array has and no more: the component
        axis is the one you may leave out, and it is added for you
        with length one.

        `role` is one of section 3: field, label, weight, normal or
        derived (`coordinates` is the support's own array and is not
        added this way). A `varies` that disagrees with `dims` is
        refused at build time with E04. Without `dims`, `varies` is
        worked out from the shape by finding the axis whose length
        is the support's, and a shape that fits two readings is
        refused rather than guessed.
        """
        return self._add(
            self.node_arrays, "node", name, values, units=units,
            dims=dims, role=role, varies=varies, components=components,
            category=category, categories=categories,
            statistic=statistic, of=of, quantile=quantile, level=level,
            method=method, recomputed=recomputed, derived_from=derived_from,
            recipe=recipe, reference=reference,
            callable_id=callable_id, output=output, dtype=dtype, **rest)

    def add_cell_array(self, name: str, values: Any = None, *,
                       units: str | None = None,
                       dims: Sequence[str] | None = None,
                       role: str = "field", varies: str | None = None,
                       components: int | None = None,
                       category: str | None = None,
                       categories: Sequence[str] | None = None,
                       statistic: str | None = None,
                       of: str | None = None,
                       quantile: float | None = None,
                       level: float | None = None,
                       method: str | None = None,
                       recomputed: bool | None = None,
                       derived_from: str | None = None,
                       recipe: str | None = None,
                       reference: str | None = None,
                       callable_id: str | None = None,
                       output: str | None = None,
                       dtype: Any = None, **rest: Any) -> ArraySlot:
        """Add an array over this support's cells. As
        `add_node_array`, with `cell` in place of `node` in `dims`."""
        return self._add(
            self.cell_arrays, "cell", name, values, units=units,
            dims=dims, role=role, varies=varies, components=components,
            category=category, categories=categories,
            statistic=statistic, of=of, quantile=quantile, level=level,
            method=method, recomputed=recomputed, derived_from=derived_from,
            recipe=recipe, reference=reference,
            callable_id=callable_id, output=output, dtype=dtype, **rest)

    def add_callable_slot(self, name: str, *, location: str = "node",
                          units: str | None = None,
                          dims: Sequence[str] | None = None,
                          callable: Any = None,
                          output: str | None = None,
                          role: str = "field",
                          varies: str | None = None,
                          components: int | None = None,
                          **rest: Any) -> ArraySlot:
        """Add an array slot a callable serves rather than data.

        `callable` is the id you gave `Dataset.add_callable`, or the
        object itself; `output` names which of that callable's
        outputs fills this slot, and defaults to the slot's name. A
        callable slot stores no values, so it declares its shape:
        `dims` or `components`.
        """
        if location not in ("node", "cell"):
            raise MestraError(
                "E30", "location is node or cell, and this is %r; pass "
                "location=\"node\" or location=\"cell\"" % location,
                name)
        into = self.node_arrays if location == "node" else self.cell_arrays
        return self._add(
            into, location, name, None, units=units, dims=dims,
            role=role, varies=varies, components=components,
            callable_id=_callable_id(self.dataset, callable, name),
            output=output, **rest)

    def set_callable_coordinates(self, *, units: str,
                                 callable: Any = None,
                                 output: str = "coordinates",
                                 components: int | None = None,
                                 varies: str = "row") -> ArraySlot:
        """Coordinates a callable serves rather than data.

        The one coordinates array of a mesh support, with the values
        left out and the callable and its output added, the way
        `add_callable_slot` is `add_node_array` without its values: a
        model of the geometry itself. Add the support with `n_nodes=`
        and no `coordinates=`, then call this. `callable` is the id
        you gave `Dataset.add_callable`, or the object itself; the
        slot stores no values, so it declares its `components`. An
        axis support's coordinates are part of its identity (section
        24) and are always stored.
        """
        name = "coordinates"
        if self.kind != "mesh":
            raise MestraError(
                "E03", "a callable may serve the coordinates of a mesh "
                "support; this support is of kind %s, whose coordinates "
                "are stored" % self.kind, name)
        if self.coordinates is not None:
            raise MestraError(
                "E03", "this support already has its one coordinates "
                "array", name)
        if components is None or int(components) < 1:
            raise MestraError(
                "E31", "a callable slot stores no values, so it declares "
                "its width; pass components=", name)
        if not units:
            raise MestraError(
                "E39", "coordinates carry units; pass units=", name)
        callable_id = _callable_id(self.dataset, callable, name)
        if self.dataset is None or \
                callable_id not in self.dataset.callables:
            raise MestraError(
                "E14", "no callable called %r has been added; call "
                "add_callable first" % callable_id, name)
        _check_varies(self.dataset, varies, None, name)
        self.coordinates = ArraySlot(
            name, "coordinates", varies=varies, components=int(components),
            location="node", units=units, support=self,
            source="callable:" + callable_id, output=output)
        return self.coordinates

    def _add(self, into: dict[str, ArraySlot], location: str, name: str,
             values: Any, *, units: str | None,
             dims: Sequence[str] | None = None, role: str = "field",
             varies: str | None = None,
             categories: Sequence[str] | None = None,
             category: str | None = None, callable_id: str | None = None,
             output: str | None = None, components: int | None = None,
             statistic: str | None = None, dtype: Any = None,
             **rest: Any) -> ArraySlot:
        _check_name(name)
        if role not in ARRAY_ROLES:
            raise MestraError(
                "E02", "%r is not an array role; pass role= one of %s"
                % (role, ", ".join(sorted(ARRAY_ROLES))), name)
        if name in into:
            raise MestraError(
                "E33", "this support already has a %s array called %r; "
                "give this one another name" % (location, name), name)
        if categories is not None:
            if self.dataset is None:
                raise MestraError(
                    "E10", "add the support to a dataset before naming "
                    "categories", name)
            category = category or name
            self.dataset.add_category_table(category, categories)
        _check_statistic(statistic, rest.get("of"), rest.get("quantile"),
                         name, rest.get("level"), rest.get("method"),
                         served=callable_id is not None)
        if callable_id is not None:
            if self.dataset is None or \
                    callable_id not in self.dataset.callables:
                raise MestraError(
                    "E14", "this dataset holds no callable called %r; "
                    "call add_callable(%r, ...) first"
                    % (callable_id, callable_id), name)
            if dims is not None:
                varies = _split_dims(dims, location, varies, statistic,
                                     name)[1]
            if components is None:
                raise MestraError(
                    "E31", "a slot served by a callable stores no "
                    "values, so it declares its shape; pass "
                    "components=", name)
            varies = varies or "row"
            _check_varies(self.dataset, varies, None, name)
            slot = ArraySlot(
                name, role, varies=varies,
                components=components, location=location, units=units,
                category=category, support=self,
                source="callable:" + callable_id,
                output=output or name, statistic=statistic, **rest)
            _check_array_attrs(slot, name)
            into[name] = slot
            return slot
        array = _as_array(values, role, dtype, name)
        if dims is not None:
            array, varies, components = _array_from_dims(
                array, dims, location, self, varies, statistic,
                components, name)
        else:
            if varies is None:
                varies = _guess_varies(array, self, location, statistic,
                                       name)
            array = _shape_array(array, location, varies, statistic,
                                 name)
            if components is None:
                components = int(array.shape[-1])
        _check_extent(array, location, varies, statistic, self, name)
        _check_varies(self.dataset, varies, array, name)
        slot = ArraySlot(name, role, varies=varies, components=components,
                         location=location, units=units,
                         category=category, support=self,
                         statistic=statistic,
                         data=MemorySource(array), **rest)
        _check_array_attrs(slot, name)
        _check_categories(self.dataset, slot, array, name)
        # The row count is checked before the slot goes in, so that a
        # refused call leaves the support as it found it.
        if self.dataset is not None and varies == "row":
            self._note_rows(int(array.shape[0]), name)
        into[name] = slot
        return slot

    def _note_rows(self, count: int, name: str) -> None:
        """E16: the leading extent against the rows it must have.

        Section 22: in an unaligned file a row-varying array on a
        support holds one entry per row referencing *that support*,
        and not one per row of the file.
        """
        dataset = self.dataset
        if dataset is None:
            return
        if dataset.has_row_support and len(dataset.supports) > 1:
            mine = len(dataset.rows_on(self))
            if count != mine:
                raise MestraError(
                    "E16", "values holds %d entries where %d rows of "
                    "this file are on support %s; a row-varying array "
                    "on a support in an unaligned file holds one entry "
                    "per row that references it (section 22)"
                    % (count, mine, self.name), name)
            return
        dataset._note_rows(count, name)

    def __repr__(self) -> str:
        return "Support(%r, kind=%r, n_nodes=%d, n_cells=%d)" % (
            self.name, self.kind, self.n_nodes, self.n_cells)


# ------------------------------------------------------------ callables

class Callables(dict):
    """The file's callables by id, each read when it is asked for.

    Section 7 of docs/api-conventions.md fixes what a metadata open
    may read: "attributes, dataspaces, link types, and
    dimension-scale structure, and ... a category table in full",
    and "never a dataset inside a callable's dictionary; those wait
    for the read". A callable's id is a link name and its `type` and
    `repr` are attributes, so an open has every id without reading
    anything; the dictionary is decoded the first time something
    asks for the callable.

    Everything that needs only the ids -- `len`, `in`, `sorted`,
    truthiness -- therefore costs nothing, and `[id]`, `items()` and
    `values()` read what they must. Reading needs the file still
    open, as any other value a lazy read left behind does; an eager
    read asks for them before it closes the file.

    A dataset a builder makes has nothing pending and behaves as the
    plain dictionary it is.
    """

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._pending: dict[str, Any] = {}

    def defer(self, name: str, build: Any) -> None:
        """Record a callable to be built the first time it is
        asked for, and its id now."""
        self._pending[name] = build
        dict.__setitem__(self, name, None)

    def __getitem__(self, name: Any) -> Any:
        build = self._pending.pop(name, None)
        if build is not None:
            dict.__setitem__(self, name, build())
        return dict.__getitem__(self, name)

    def __setitem__(self, name: Any, value: Any) -> None:
        self._pending.pop(name, None)
        dict.__setitem__(self, name, value)

    def __delitem__(self, name: Any) -> None:
        self._pending.pop(name, None)
        dict.__delitem__(self, name)

    def get(self, name: Any, default: Any = None) -> Any:
        if name not in self:
            return default
        return self[name]      # noqa: SIM401 - self.get would recurse

    def pop(self, name: Any, *default: Any) -> Any:
        if name in self:
            value = self[name]
            dict.__delitem__(self, name)
            return value
        if default:
            return default[0]
        raise KeyError(name)

    def values(self) -> Any:
        self.read_all()
        return dict.values(self)

    def items(self) -> Any:
        self.read_all()
        return dict.items(self)

    def copy(self) -> dict[str, Any]:
        self.read_all()
        return dict(dict.items(self))

    def __repr__(self) -> str:
        self.read_all()
        return dict.__repr__(self)

    # An unread callable is held as None under its id, so that the
    # ids cost nothing to list. Defining these two in Python is what
    # keeps that private: it makes `dict(callables)` and
    # `{**callables}` go through `keys()` and `__getitem__` instead
    # of copying the underlying dictionary, which would hand out the
    # placeholders.

    def __iter__(self) -> Any:
        return dict.__iter__(self)

    def keys(self) -> Any:
        return dict.keys(self)

    def read_all(self) -> None:
        """Read every dictionary still in the file. An eager read
        calls this while the file is open."""
        for name in list(self._pending):
            self[name]


# -------------------------------------------------------------- dataset

class Dataset:
    """One mestra file: its rows, its supports, and its metadata.

    It carries `keys`, `scalars`, `categories`, `supports`,
    `callables`, `row_support`, `notes`, `n_rows` and `aligned`, and
    it is also the builder. Eleven calls reach a complete file, in
    the order section 1 of `docs/api-conventions.md` fixes:

        add_key(name, values, role, units)
        add_scalar(name, values, units)
        add_category_table(name, entries)
        set_generalisation_group(name)
        add_support(name, kind, coordinates, cells, units)
        support.add_node_array(name, values, units, dims)
        support.add_cell_array(name, values, units, dims)
        add_callable(id, callable)
        support.add_callable_slot(name, units, callable, output)
        add_callable_slot(name, units, callable, output)
        set_row_support(values)

    Everything after `values` is a keyword argument, which is
    Python's spelling of the name-value pairs the other three
    languages take, and the support is the receiver of the two array
    builders, which is Python's spelling of their support-first
    argument order. The dimensions, the bounds, the component axis,
    the support id and the alignment flag are filled in.

    Every builder refuses at build time, naming the rule of section
    14 and the argument to change, anything the validator would
    refuse in the file; `mestra.write` validates again before it
    writes. `to_xarray()` is an optional adapter.
    """

    def __init__(self, *, writer: str = "mestra python 0",
                 created: str | None = None,
                 format: str = FORMAT,
                 generalisation_group: str | None = None,
                 notes: Mapping[str, Any] | None = None) -> None:
        self.format = format
        self.writer = writer
        self.created = created or _now()
        self.generalisation_group = generalisation_group
        self.keys: dict[str, Key] = {}
        self.scalars: dict[str, ScalarSlot] = {}
        self.categories: dict[str, CategoryTable] = {}
        self.supports: dict[str, Support] = {}
        self.callables: Callables = Callables()
        self._row_support: _Source | None = None
        self.notes: dict[str, Any] = dict(notes or {})
        #: Root attributes this reader does not know (W11).
        self.extra: dict[str, Any] = {}
        #: Groups this reader does not know, and /private, kept whole.
        self.opaque: dict[str, Any] = {}
        #: Optional groups the file carried, so that an empty one
        #: survives a rewrite.
        self.present: set[str] = set()
        #: What the reader met and could not classify as a rule of
        #: section 14: a link it will not follow, a member of the
        #: wrong kind, something the library would not read. The
        #: validator reports these too.
        self.problems: list[Finding] = []
        #: Paths this reader could not copy into memory, so that a
        #: rewrite would lose them. `write` refuses while any
        #: remain.
        self.lossy: list[str] = []
        #: What the file said, kept so that the validator can check it.
        self.stored_aligned: bool | None = None
        self.stored_row_count: int | None = None
        self._n_rows: int | None = None
        self._file: Any = None
        self.path: str | None = None

    # -- which support each row is on

    @property
    def row_support(self) -> np.ndarray | None:
        """The support of each row, or None in an aligned file."""
        if self._row_support is None:
            return None
        return self._row_support.read()

    @row_support.setter
    def row_support(self, values: Any) -> None:
        self._row_support = (None if values is None
                             else _wrap(values, "<i4"))

    @property
    def has_row_support(self) -> bool:
        """True when the file carries a /row_support column."""
        return self._row_support is not None

    # -- the two structural facts

    @property
    def aligned(self) -> bool:
        """True when the file declares at most one support (section 8).

        Alignment is structural: it is not a separate claim, and
        index-aligned operations are valid only when it holds.
        """
        return len(self.supports) <= 1

    @property
    def n_rows(self) -> int:
        """The number of rows the file holds."""
        if self._n_rows is not None:
            return self._n_rows
        for key in self.keys.values():
            if key.data is not None:
                return int(key.data.shape[0])
        for scalar in self.scalars.values():
            if scalar.data is not None:
                return int(scalar.data.shape[0])
        return 0

    @n_rows.setter
    def n_rows(self, value: int) -> None:
        self._n_rows = int(value)

    def _note_rows(self, count: int, where: str) -> None:
        if self._n_rows is None:
            self._n_rows = count
        elif self._n_rows != count:
            raise MestraError(
                "E16", "values holds %d rows where the dataset has %d; "
                "every key, scalar and row-varying array has one entry "
                "per row" % (count, self._n_rows), where)

    # -- keys

    def key_names(self) -> list[str]:
        """The file's key order: the names sorted by their UTF-8
        bytes (section 26)."""
        return sorted(self.keys, key=lambda n: n.encode("utf-8"))

    def keys_of_role(self, role: str) -> list[Key]:
        """Every key with this role, in the file's key order."""
        return [self.keys[n] for n in self.key_names()
                if self.keys[n].role == role]

    def add_key(self, name: str, values: Any, *, role: str,
                units: str | None = None,
                category: str | None = None,
                lower: float | None = None,
                upper: float | None = None,
                categories: Sequence[str] | None = None,
                trajectory_group: str | None = None,
                parent: str | None = None,
                generalisation: bool = False,
                bounds: str | None = "observed",
                dtype: Any = None) -> Key:
        """Add a key column: `add_key(name, values, role, units)`.

        `role` is one of section 3: design, condition, time,
        categorical, group, split, id or status. A design, condition
        or time key carries `units`; a categorical, group, split, id
        or status key carries `category`, naming a table
        `add_category_table` has already written, instead, and its
        values are the positions of that table's entries, counting
        from 0.

        `trajectory_group=` on a time key names the group key whose
        categories are the trajectories (section 7); without it a
        transient file is a pile of rows. `parent=` on a group key
        names the group it nests inside.

        Bounds: when you give neither, the builder records the
        observed finite minimum and maximum, so that the same arrays
        give the same file in every language and W04 and W08 are
        decidable. Pass `lower` and `upper` to declare a wider domain
        of validity, or `bounds=None` to leave them out.

        `categories=[...]` is sugar for `add_category_table` on a
        table of this key's own name, and nothing more.
        """
        _check_name(name)
        if role not in KEY_ROLES:
            raise MestraError(
                "E02", "%r is not a key role; pass role= one of %s"
                % (role, ", ".join(sorted(KEY_ROLES))), name)
        if name in self.keys:
            raise MestraError(
                "E33", "this dataset already has a key called %r; give "
                "this one another name" % name, name)
        limit = KEY_ROLES[role]
        if limit is not None and len(self.keys_of_role(role)) >= limit:
            raise MestraError(
                "E03", "a file has at most %d key with the role %s, and "
                "this one already has %s; pass another role="
                % (limit, role,
                   ", ".join(k.name for k in self.keys_of_role(role))),
                name)
        if role in _INTEGER_KEY_ROLES:
            given = _numbers(values, None, name)
            if given.dtype.kind not in "iub":
                raise MestraError(
                    "E20", "a %s key stores category ids, which are "
                    "integers; these values are %s. Pass the ids, and "
                    "their names as categories=" % (role, given.dtype),
                    name)
            array = given.astype(dtype or "<i4")
        elif role == "id":
            array = _numbers(values, None, name)
            if array.dtype.kind in "US":
                array = array.astype(np.str_)
            elif array.dtype.kind in "iub":
                array = array.astype(dtype or "<i8")
            else:
                raise MestraError(
                    "E20", "an id key stores row identifiers, which are "
                    "integers or text; these values are %s. Pass the "
                    "identifiers as they are, not as floats"
                    % array.dtype, name)
        else:
            array = _numbers(values, dtype or "<f8", name)
        if array.ndim != 1:
            raise MestraError(
                "E04", "a key column has one dimension, row; these "
                "values have %d. Pass one value per row" % array.ndim,
                name)
        if categories is not None:
            category = category or name
            self.add_category_table(category, categories)
        _check_key_units(role, units, name)
        _check_key_table(self, role, category, array, name)
        _check_bounds(lower, upper, name)
        if role in _UNIT_KEY_ROLES and bounds:
            lower, upper = _observed_bounds(array, lower, upper)
        key = Key(name, role, units=units, lower=lower, upper=upper,
                  category=category, trajectory_group=trajectory_group,
                  parent=parent, data=MemorySource(array))
        # Before the key goes in, so that a refused call leaves the
        # dataset as it found it.
        self._note_rows(int(array.shape[0]), name)
        self.keys[name] = key
        if generalisation:
            self.set_generalisation_group(name)
        return key

    def set_generalisation_group(self, name: str | None) -> None:
        """Name the key that is the unit of generalisation.

        Section 7: exactly one group key is the unit a split must
        keep whole. `None` clears it. `add_key(...,
        generalisation=True)` is sugar for this call.
        """
        if name is None:
            self.generalisation_group = None
            return
        key = self.keys.get(name)
        if key is None:
            raise MestraError(
                "E03", "this dataset declares no key called %r; add the "
                "group key first" % name, "/generalisation_group")
        if key.role != "group":
            raise MestraError(
                "E03", "the unit of generalisation is a key of role "
                "group, and %r has the role %s" % (name, key.role),
                "/keys/" + name)
        self.generalisation_group = name

    # -- scalars

    def add_scalar(self, name: str, values: Any = None, *,
                   units: str | None = None,
                   callable_id: str | None = None,
                   output: str | None = None,
                   statistic: str | None = None, of: str | None = None,
                   quantile: float | None = None,
                   level: float | None = None, method: str | None = None,
                   **rest: Any) -> ScalarSlot:
        """Add a per-row quantity of interest:
        `add_scalar(name, values, units)`.

        Pass `values` for stored data, or use `add_callable_slot` for
        a slot a callable serves. A band (`statistic="band"`) names
        its base with `of` and states its `level` and `method`.
        """
        _check_name(name)
        if name in self.scalars:
            raise MestraError(
                "E33", "this dataset already has a scalar called %r; "
                "give this one another name" % name, name)
        if not units:
            raise MestraError(
                "E11", "a scalar carries units; pass units= (\"1\" for "
                "a dimensionless one)", name)
        if not isinstance(units, str):
            raise MestraError(
                "E19", "units is a string, and this is %s; pass the "
                "unit as text" % type(units).__name__, name)
        _check_statistic(statistic, of, quantile, name, level, method,
                         served=callable_id is not None)
        if callable_id is not None:
            if callable_id not in self.callables:
                raise MestraError(
                    "E14", "this dataset holds no callable called %r; "
                    "call add_callable(%r, ...) first"
                    % (callable_id, callable_id), name)
            slot = ScalarSlot(name, units=units,
                              source="callable:" + callable_id,
                              output=output or name, statistic=statistic,
                              of=of, quantile=quantile, level=level,
                              method=method, **rest)
            self.scalars[name] = slot
            return slot
        if values is None:
            raise MestraError(
                "E30", "a scalar holding data needs values; pass "
                "values=, or add_callable_slot for a slot a callable "
                "serves", name)
        array = _numbers(values, "<f8", name)
        if array.ndim != 1:
            raise MestraError(
                "E04", "a scalar has one dimension, row; these values "
                "have %d. Pass one value per row" % array.ndim, name)
        slot = ScalarSlot(name, units=units, data=MemorySource(array),
                          statistic=statistic, of=of, quantile=quantile,
                          level=level, method=method, **rest)
        self._note_rows(int(array.shape[0]), name)
        self.scalars[name] = slot
        return slot

    def add_callable_slot(self, name: str, *, units: str | None = None,
                          callable: Any = None,
                          output: str | None = None,
                          **rest: Any) -> ScalarSlot:
        """Add a scalar slot a callable serves rather than data.

        `callable` is the id you gave `add_callable`, or the object
        itself; `output` names which of that callable's outputs fills
        this slot, and defaults to the slot's name.
        `Support.add_callable_slot` is the same call for an array.
        """
        return self.add_scalar(
            name, units=units,
            callable_id=_callable_id(self, callable, name),
            output=output, **rest)

    # -- categories

    def add_category_table(self, name: str,
                           entries: Sequence[str]) -> CategoryTable:
        """Add or replace a category table.

        Call it before the key or the label that names it with
        `category`. A category id is the position of its entry, so
        the first entry is id 0 (section 21).
        """
        _check_name(name)
        table = CategoryTable(entries)
        self.categories[name] = table
        return table

    # -- supports

    def add_support(self, name: str | None = None, *,
                    kind: str | None = None, coordinates: Any = None,
                    cells: tuple[Any, Any, Any] | None = None,
                    units: str = "m", varies: str = "none",
                    n_nodes: int | None = None) -> Support:
        """Add a support, from its coordinates and its cells.

        `cells` is (cell_types, cell_offsets, cell_connectivity) as
        section 6 stores them; `kind="axis"` and no cells is a
        support of nodes along one coordinate, and `kind="none"` is
        no support at all, for a file of scalars. The kind, the node
        and cell counts and the support id are worked out from what
        is given.
        """
        if name is None:
            name = "s%d" % len(self.supports)
        _check_name(name)
        if name in self.supports:
            raise MestraError(
                "E33", "this dataset already has a support called %r; "
                "give this one another name" % name, name)
        coords = None if coordinates is None else _shape_array(
            _numbers(coordinates, "<f8", name + "/coordinates"), "node",
            varies, None, "coordinates")
        if kind is None:
            if cells is not None:
                kind = "mesh"
            elif coords is not None:
                kind = "axis"
            else:
                kind = "none"
        if kind not in SUPPORT_KINDS:
            raise MestraError(
                "E02", "a support is mesh, axis or none; pass kind= one "
                "of those, not %r" % kind, name)
        if coords is not None and not units:
            raise MestraError(
                "E39", "coordinates carry units; pass units=", name)
        if kind == "axis" and varies != "none":
            raise MestraError(
                "E35", "the coordinates of an axis support have varies "
                "= none, because the axis is part of the support's "
                "identity; pass varies=\"none\", or make the quantity "
                "that differs between rows a field on the axis", name)
        # A mesh may take its coordinates from a callable instead,
        # through `Support.set_callable_coordinates`; it then needs
        # its node count said out loud, and `write` holds it to E03
        # if the coordinates never arrive.
        if kind == "axis" and coords is None:
            raise MestraError(
                "E03", "an axis support has exactly one coordinates "
                "array, and it is stored; pass coordinates=", name)
        if kind == "mesh" and coords is None and n_nodes is None:
            raise MestraError(
                "E03", "a mesh support has exactly one coordinates "
                "array; pass coordinates=, or n_nodes= and then "
                "set_callable_coordinates for coordinates a callable "
                "serves", name)
        _check_varies(self, varies, coords, name)
        if kind == "mesh" and cells is None:
            raise MestraError(
                "E38", "a mesh support carries cell_types, cell_offsets "
                "and cell_connectivity; pass cells=(types, offsets, "
                "connectivity), or kind=\"axis\" for nodes along one "
                "coordinate", name)
        if kind != "mesh" and cells is not None:
            raise MestraError(
                "E38", "a support of kind %s carries no cells; drop "
                "cells=, or leave kind out to get a mesh" % kind, name)
        types = offsets = conn = None
        if cells is not None:
            types, offsets, conn = cells
        if n_nodes is None:
            n_nodes = 0 if coords is None else int(coords.shape[-2])
        support = Support(
            name, kind, n_nodes=n_nodes,
            n_cells=0 if types is None else int(len(types)),
            cell_types=types, cell_offsets=offsets,
            cell_connectivity=conn)
        support.dataset = self
        self.supports[name] = support
        if coords is not None:
            support.coordinates = ArraySlot(
                "coordinates", "coordinates", varies=varies,
                components=int(coords.shape[-1]), location="node",
                units=units, support=support,
                data=MemorySource(coords))
            if varies == "row":
                self._note_rows(int(coords.shape[0]), "coordinates")
        return support

    def set_row_support(self, values: Any) -> None:
        """Say which support each row is on (section 22).

        The value is the zero-based position of the row's support in
        the file's support order, which is the support names sorted
        by their UTF-8 bytes.
        """
        array = np.asarray(values, dtype="<i4")
        if array.ndim != 1:
            raise MestraError(
                "E04", "/row_support has one dimension, row",
                "/row_support")
        self._note_rows(int(array.shape[0]), "/row_support")
        self._row_support = MemorySource(array)

    def support_names(self) -> list[str]:
        """The file's support order: names sorted by UTF-8 bytes."""
        return sorted(self.supports, key=lambda n: n.encode("utf-8"))

    def support_of_row(self, row: int) -> Support | None:
        """The support a row sits on, or None when there is none."""
        names = self.support_names()
        if not names:
            return None
        if not self.has_row_support:
            return self.supports[names[0]]
        at = int(np.asarray(self.row_support)[row])
        if not 0 <= at < len(names):
            raise MestraError(
                "E06", "row %d references support %d, and the file "
                "declares %d" % (row, at, len(names)), "/row_support")
        return self.supports[names[at]]

    def rows_on(self, support: str | Support) -> np.ndarray:
        """The file rows that reference a support, in file order.

        A row-varying array on that support holds one entry per row
        in exactly this order (section 22).
        """
        name = support if isinstance(support, str) else support.name
        names = self.support_names()
        if name not in self.supports:
            raise KeyError("no support called %r" % name)
        if not self.has_row_support:
            return np.arange(self.n_rows)
        want = names.index(name)
        return np.flatnonzero(np.asarray(self.row_support) == want)

    # -- callables

    def add_callable(self, identifier: str, callable: Any) -> Any:
        """Store a callable under an id that slots may reference.

        `add_callable(id, callable)`, and then `add_callable_slot`
        for each slot it serves.
        """
        _check_name(identifier)
        self.callables[identifier] = callable
        return callable

    # -- everything at once

    def slots(self) -> dict[str, Slot]:
        """Every slot in the file, by its HDF5 path."""
        out: dict[str, Slot] = {}
        for name, scalar in self.scalars.items():
            out["/scalars/" + name] = scalar
        for sname, support in self.supports.items():
            for where, slot in support.arrays().items():
                out["/supports/%s/%s" % (sname, where)] = slot
        return out

    # -- adapters

    def to_xarray(self) -> Any:
        """This dataset as an xarray Dataset, dimensions and all.

        An optional adapter: it needs xarray, which is not a
        dependency of this package. Dimension names are the names on
        disk of section 21, with a support's own `node`, `cell` and
        `row` prefixed by the support name when the file declares
        more than one support, since their lengths differ.
        """
        import xarray as xr

        variables: dict[str, Any] = {}
        attributes = {"format": self.format, "writer": self.writer,
                      "created": self.created,
                      "aligned": int(self.aligned)}
        if self.generalisation_group:
            attributes["generalisation_group"] = self.generalisation_group
        for name in self.key_names():
            key = self.keys[name]
            if key.data is None:
                continue
            variables[name] = (("row",), key.data.read(),
                               _key_attributes(key))
        for name, scalar in sorted(self.scalars.items()):
            if scalar.data is None:
                continue
            variables[name] = (("row",), scalar.data.read(),
                               _slot_attributes(scalar))
        many = len(self.supports) > 1
        for sname in self.support_names():
            support = self.supports[sname]
            for where, slot in support.arrays().items():
                if slot.data is None:
                    continue
                values = slot.data.read()
                dims = tuple(
                    _xarray_dimension(dim, sname, many,
                                      values.shape[at])
                    for at, dim in enumerate(slot.dims))
                short = where.rsplit("/", 1)[-1]
                label = "%s_%s" % (sname, short) if many else short
                variables[label] = (dims, values, _slot_attributes(slot))
        return xr.Dataset(variables, attrs=attributes)

    # -- the file behind a lazy dataset

    def close(self) -> None:
        """Close the file a lazy dataset is reading from."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Dataset:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return ("Dataset(rows=%d, keys=%d, scalars=%d, supports=%d, "
                "callables=%d, aligned=%s)"
                % (self.n_rows, len(self.keys), len(self.scalars),
                   len(self.supports), len(self.callables), self.aligned))


# ------------------------------------------------------------- helpers

def _key_attributes(key: Key) -> dict[str, Any]:
    out: dict[str, Any] = {"role": key.role}
    for name in ("units", "category", "trajectory_group", "parent"):
        value = getattr(key, name)
        if value is not None:
            out[name] = value
    for name in ("lower", "upper"):
        value = getattr(key, name)
        if value is not None:
            out[name] = float(value)
    return out


def _slot_attributes(slot: Slot) -> dict[str, Any]:
    out: dict[str, Any] = {"source": slot.source}
    for name in ("role", "varies", "units", "statistic", "of",
                 "category", "recipe", "derived_from", "reference"):
        value = getattr(slot, name, None)
        if value is not None:
            out[name] = value
    if slot.quantile is not None:
        out["quantile"] = float(slot.quantile)
    if slot.level is not None:
        out["level"] = float(slot.level)
    if slot.method is not None:
        out["method"] = slot.method
    return out


def _xarray_dimension(dim: str, support: str, many: bool,
                      length: int) -> str:
    """The dimension name to give xarray (section 21 on disk)."""
    from .names import disk_dimension
    if dim in ("node", "cell", "row") and many:
        return "%s_%s" % (support, dim)
    return disk_dimension(dim, length)


def _now() -> str:
    """An ISO 8601 timestamp in UTC (section 11)."""
    now = _datetime.datetime.now(_datetime.timezone.utc)
    return now.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_name(name: str) -> None:
    if not is_legal_name(name):
        raise MestraError(
            "E33", "%r is not a legal netCDF-4 name: letters, digits, "
            "underscore, hyphen, . and + only" % name, name)
    if is_reserved(name):
        raise MestraError(
            "E33", "names beginning with mestra_ are reserved for the "
            "container", name)


def _as_array(values: Any, role: str, dtype: Any, name: str
              ) -> np.ndarray:
    if values is None:
        raise MestraError(
            "E30", "a slot holding data needs values; pass values=, or "
            "add_callable_slot for a slot a callable serves", name)
    if role == "label":
        array = _numbers(values, dtype or "<i4", name)
        if array.dtype.kind not in "iu":
            raise MestraError(
                "E20", "a label is int32 or int64, and these values are "
                "%s; pass integer category ids" % array.dtype, name)
        return array
    array = _numbers(values, dtype or "<f8", name)
    if array.dtype != np.dtype("<f8"):
        raise MestraError(
            "E20", "a %s array is float64, and these values are %s"
            % (role, array.dtype), name)
    return array


def _numbers(values: Any, dtype: Any, name: str) -> np.ndarray:
    """`values` as one rectangular array, or E20 naming the argument.

    numpy's own refusals are about an inhomogeneous shape or a
    string it cannot convert, and a caller building a file should be
    told which argument and what to pass.
    """
    try:
        return np.asarray(values, dtype=dtype)
    except (ValueError, TypeError) as exc:
        want = ("numbers" if dtype is None
                else "values of dtype %s" % np.dtype(dtype))
        raise MestraError(
            "E20", "values must be a rectangular array of %s, and these "
            "are not (%s); pass one value per row and the same shape "
            "in every row" % (want, exc), name) from None


def _check_bounds(lower: Any, upper: Any, name: str) -> None:
    """E19: a bound the caller declares is finite (section 18)."""
    for label, value in (("lower", lower), ("upper", upper)):
        if value is None:
            continue
        try:
            finite = np.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise MestraError(
                "E19", "%s is a finite float64 bound, and %r is not; "
                "pass a number, or leave it out for the observed one"
                % (label, value), name)


def _observed_bounds(array: np.ndarray, lower: float | None,
                     upper: float | None
                     ) -> tuple[float | None, float | None]:
    """The bounds to record: what the caller gave, and the observed
    finite range for what they did not (section 1 of the
    conventions)."""
    if lower is not None and upper is not None:
        return lower, upper
    if array.dtype.kind not in "fiu" or not array.size:
        return lower, upper
    finite = array[np.isfinite(array)] if array.dtype.kind == "f" \
        else array
    if not finite.size:
        return lower, upper
    if lower is None:
        lower = float(np.min(finite))
    if upper is None:
        upper = float(np.max(finite))
    return lower, upper


def _check_key_units(role: str, units: str | None, name: str) -> None:
    """E39: which key roles carry units, and which do not."""
    if units is not None and not isinstance(units, str):
        raise MestraError(
            "E19", "units is a string, and this is %s; pass the unit "
            "as text, \"1\" for a dimensionless one"
            % type(units).__name__, name)
    if role in _UNIT_KEY_ROLES and not units:
        raise MestraError(
            "E39", "a %s key carries units; pass units= (\"1\" for a "
            "dimensionless one)" % role, name)
    if role not in _UNIT_KEY_ROLES and units:
        raise MestraError(
            "E39", "a %s key carries no units, because its values are "
            "%s; drop units="
            % (role, "row identifiers" if role == "id"
               else "category ids"), name)


def _check_key_table(dataset: Dataset, role: str, category: str | None,
                     array: np.ndarray, name: str) -> None:
    """E39 and E10: the category table a key role requires."""
    if role not in _TABLE_KEY_ROLES:
        return
    if not category:
        raise MestraError(
            "E39", "a %s key names its category table; call "
            "add_category_table(name, entries) and pass category=name, "
            "or pass categories=[...]" % role, name)
    table = dataset.categories.get(category)
    if table is None:
        raise MestraError(
            "E39", "this dataset has no category table called %r; call "
            "add_category_table(%r, entries) first" % (category,
                                                       category), name)
    if array.size and ((array < 0) | (array >= len(table))).any():
        raise MestraError(
            "E10", "a value is outside the category table %r, which has "
            "%d entries; the id of an entry is its position, counting "
            "from 0" % (category, len(table)), name)


def _check_statistic(statistic: str | None, of: str | None,
                     quantile: float | None, name: str,
                     level: float | None = None,
                     method: str | None = None,
                     served: bool = False) -> None:
    """E02 and E12: a statistic with what it needs (section 9), and
    on a slot a callable serves, one a callable produces (section
    10)."""
    if statistic != "band" and (level is not None or method is not None):
        raise MestraError(
            "E12", "level and method belong to a band; pass "
            "statistic=\"band\" with of= naming the base quantity",
            name)
    if statistic is None:
        return
    if statistic not in STATISTICS:
        raise MestraError(
            "E02", "%r is not a statistic of section 9; pass statistic= "
            "one of %s" % (statistic, ", ".join(STATISTICS)), name)
    if statistic == "quantile" and quantile is None:
        raise MestraError(
            "E12", "a quantile statistic carries its quantile; pass "
            "quantile=", name)
    if statistic not in ("value", "draw") and not of:
        raise MestraError(
            "E12", "a %s names the quantity it is a statistic of; pass "
            "of=" % statistic, name)
    if served and statistic in ("std", "quantile", "draw"):
        raise MestraError(
            "E12", "a callable returns a mean and at most a band "
            "(section 10), so a slot it serves is value, mean or band; "
            "store a %s as data" % statistic, name)
    if statistic == "band" and not served:
        if level is None:
            raise MestraError(
                "E12", "a band states the coverage it claims; pass "
                "level= in (0, 1), for example 0.95", name)
        if not (0.0 < float(level) < 1.0):
            raise MestraError(
                "E12", "level is the coverage a band claims, a fraction "
                "in (0, 1), and %r is not; a 1.96-sigma Gaussian band "
                "is 0.95" % (level,), name)
        if not method:
            raise MestraError(
                "E12", "a band says how it was made; pass method= with "
                "one sentence", name)


def _callable_id(dataset: Dataset | None, obj: Any, name: str) -> str:
    """The id a callable is stored under, from the id or the object."""
    if obj is None:
        raise MestraError(
            "E14", "a callable slot names the callable that serves it; "
            "pass callable=<id>", name)
    if isinstance(obj, str):
        return obj
    if dataset is not None:
        for identifier, held in dataset.callables.items():
            if held is obj:
                return identifier
    raise MestraError(
        "E14", "this callable is not in the dataset; call "
        "add_callable(<id>, callable) first, and pass callable=<id>",
        name)


def _check_array_attrs(slot: ArraySlot, name: str) -> None:
    """E11, E13 and E39: what an array role requires (section 3)."""
    if slot.role == "field" and not slot.units:
        raise MestraError(
            "E11", "a field carries units; pass units= (\"1\" for a "
            "dimensionless one)", name)
    if slot.role in ("coordinates", "derived") and not slot.units:
        raise MestraError(
            "E39", "a %s array carries units; pass units=" % slot.role,
            name)
    if slot.role == "derived" and not (slot.derived_from and slot.recipe):
        raise MestraError(
            "E13", "a derived array carries derived_from and recipe; "
            "pass both, naming the arrays it came from and the "
            "operation", name)


def _check_categories(dataset: Dataset | None, slot: ArraySlot,
                      array: np.ndarray, name: str) -> None:
    """E10: a label's values inside the table it names.

    Section 3: a label's category table is optional, and when it is
    absent the values are their own categories.
    """
    if slot.category is None:
        return
    table = None if dataset is None else dataset.categories.get(
        slot.category)
    if table is None:
        raise MestraError(
            "E39", "this dataset has no category table called %r; call "
            "add_category_table(%r, entries) first" % (slot.category,
                                                       slot.category),
            name)
    if array.dtype.kind in "iu" and array.size and \
            ((array < 0) | (array >= len(table))).any():
        raise MestraError(
            "E10", "a value is outside the category table %r, which has "
            "%d entries; the id of an entry is its position, counting "
            "from 0" % (slot.category, len(table)), name)


def _check_varies(dataset: Dataset | None, varies: str,
                  array: np.ndarray | None, name: str) -> None:
    """E04 and E34: a group-varying array against its group key."""
    if not varies.startswith("group:"):
        if varies not in ("none", "row"):
            raise MestraError(
                "E04", "varies is none, row or group:<k>, and this is "
                "%r" % varies, name)
        return
    group = varies[len("group:"):]
    key = None if dataset is None else dataset.keys.get(group)
    if key is None:
        raise MestraError(
            "E04", "varies is %r and this dataset declares no key "
            "called %r; add the group key before the arrays that vary "
            "along it" % (varies, group), name)
    if key.role != "group":
        raise MestraError(
            "E04", "varies is %r and %r has the role %s, not group"
            % (varies, group, key.role), name)
    table = None if dataset is None else dataset.categories.get(
        key.category or "")
    if array is not None and table is not None and \
            int(array.shape[0]) != len(table):
        raise MestraError(
            "E34", "this holds %d instances where the group %r has %d "
            "categories; a group-varying array holds one instance per "
            "category" % (int(array.shape[0]), group, len(table)), name)


def _check_extent(array: np.ndarray, location: str, varies: str,
                  statistic: str | None, support: Support,
                  name: str) -> None:
    """E05: the node or cell axis against the support."""
    width = support.n_nodes if location == "node" else support.n_cells
    at = len(array_dims(varies, statistic, location)) - 2
    if int(array.shape[at]) != width:
        raise MestraError(
            "E05", "the %s axis of this array is %d long where the "
            "support has %d %ss; check values, or name the axes with "
            "dims=" % (location, int(array.shape[at]), width, location),
            name)


def _split_dims(dims: Sequence[str], location: str, varies: str | None,
                statistic: str | None, name: str
                ) -> tuple[tuple[str, ...], str]:
    """The caller's axis names, checked, and the `varies` they mean.

    `dims` names the axes of the array the caller is passing, in the
    caller's own order: "row" or "group:<k>", "draw", the location,
    and "component". A name the array does not have is simply not
    listed; the builder adds a component axis of length one.
    """
    names = tuple(str(d) for d in dims)
    if len(set(names)) != len(names):
        raise MestraError(
            "E04", "dims names an axis twice (%s); name each axis of "
            "your array once" % ", ".join(names), name)
    other = "cell" if location == "node" else "node"
    for one in names:
        if one in ("row", "draw", location, "component") or \
                one.startswith("group:"):
            continue
        if one == other:
            raise MestraError(
                "E04", "dims names a %s axis on an array over the %ss; "
                "add_%s_array is the call for that" % (other, location,
                                                       other), name)
        if one in ("instance", "group"):
            raise MestraError(
                "E04", "dims says %r; name the group key it varies "
                "along, as in \"group:member\"" % one, name)
        raise MestraError(
            "E04", "dims may name row or group:<k>, draw, %s and "
            "component, and this names %r" % (location, one), name)
    leading = [n for n in names if n == "row" or n.startswith("group:")]
    if len(leading) > 1:
        raise MestraError(
            "E04", "dims names %d leading axes (%s); an array varies "
            "along one of them, or along none"
            % (len(leading), ", ".join(leading)), name)
    found = leading[0] if leading else "none"
    if varies is not None and varies != found:
        raise MestraError(
            "E04", "varies says %r and dims says %r; change one of "
            "them" % (varies, found), name)
    if statistic == "draw" and "draw" not in names:
        raise MestraError(
            "E04", "the statistic is draw, so the array has a draw "
            "axis; name it in dims", name)
    if statistic != "draw" and "draw" in names:
        raise MestraError(
            "E04", "dims names a draw axis, so pass statistic=\"draw\"",
            name)
    if location not in names:
        raise MestraError(
            "E04", "dims names no %s axis; every array on a support "
            "has one" % location, name)
    return names, found


def _array_from_dims(array: np.ndarray, dims: Sequence[str],
                     location: str, support: Support,
                     varies: str | None, statistic: str | None,
                     components: int | None, name: str
                     ) -> tuple[np.ndarray, str, int]:
    """The array in the order section 19 stores it, from the order
    the caller has it in."""
    names, found = _split_dims(dims, location, varies, statistic, name)
    if len(names) != array.ndim:
        raise MestraError(
            "E04", "dims names %d axes and the array has %d; name every "
            "axis of the array you are passing"
            % (len(names), array.ndim), name)
    wanted = array_dims(found, statistic, location)
    array = array.transpose([names.index(d) for d in wanted
                             if d in names])
    if "component" not in names:
        # Section 19: the component axis is always present, with
        # length one for a single-component quantity.
        array = array.reshape(array.shape + (1,))
    if components is not None and components != int(array.shape[-1]):
        raise MestraError(
            "E31", "components says %d and the component axis of this "
            "array is %d long; change one of them"
            % (components, int(array.shape[-1])), name)
    return array, found, int(array.shape[-1])


def _guess_varies(array: np.ndarray, support: Support, location: str,
                  statistic: str | None, name: str) -> str:
    """Take `varies` from the shape when the caller did not say.

    The node or cell axis is the one whose length is the support's,
    which is what tells (node, component) apart from (row, node).
    An array that varies along a group is never guessed: say
    `varies="group:<k>"` for that.
    """
    width = support.n_nodes if location == "node" else support.n_cells
    draws = 1 if statistic == "draw" else 0
    if array.ndim == 1 + draws:
        if array.shape[-1] == width:
            return "none"
    elif array.ndim == 3 + draws:
        if array.shape[-2] == width:
            return "row"
    elif array.ndim == 2 + draws:
        fits_none = array.shape[-2] == width
        fits_row = array.shape[-1] == width
        if fits_none and not fits_row:
            return "none"
        if fits_row and not fits_none:
            return "row"
        if fits_none and fits_row:
            raise MestraError(
                "E04", "a %s array of shape %s on a support of %d %ss "
                "is either (%s, component) or (row, %s); say which "
                "with varies" % (location, array.shape, width,
                                 location, location, location), name)
    raise MestraError(
        "E05", "an array of shape %s does not fit a support of %d "
        "%ss; the %s axis is the one of that length, and dims= names "
        "the axes you have" % (array.shape, width, location, location),
        name)


def _shape_array(array: np.ndarray, location: str, varies: str,
                 statistic: str | None, name: str) -> np.ndarray:
    """Give an array the component axis section 19 always requires."""
    wanted = len(array_dims(varies, statistic, location))
    if array.ndim == wanted:
        return array
    if array.ndim == wanted - 1:
        # The component axis is always present, with length 1 for a
        # single-component quantity (section 19).
        return array.reshape(array.shape + (1,))
    raise MestraError(
        "E04", "an array that varies along %s on %ss has %d axes; this "
        "one has %d. Name the axes you have with dims="
        % (varies, location, wanted, array.ndim), name)
