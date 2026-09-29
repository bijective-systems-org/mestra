"""The callable protocol, its registry, and the `affine` reference.

A callable is four things and nothing more (SPEC.md section 10):
`__call__` takes a keys table and returns one `Prediction` per output
it serves, `to_dict` represents it as a nested dictionary, `from_dict`
is the inverse dispatched on a `type` string, and `__repr__` is an
optional one-line description. Everything else it knows is inside its
own dictionary and is its own business.

A `Prediction` is a mean and, when the model has one, a band: the
half-width `uncertainty` of the interval around the mean at coverage
`level`, made as `method` says. How the band was computed is the
model's business; what it claims is on the record.

Adding a type is a subclass and one registration call:

    class Lookup(mestra.Callable):
        type = "lookup"
        def __call__(self, keys): ...
        def to_dict(self): ...
        @classmethod
        def from_dict(cls, d): ...

    mestra.register_callable(Lookup)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .errors import MestraError

__all__ = [
    "Prediction",
    "Callable",
    "Affine",
    "OpaqueCallable",
    "register_callable",
    "callable_types",
    "callable_from_dict",
    "keys_table",
    "table_length",
]


# ------------------------------------------------------- the keys table

def keys_table(table: Any, names: Sequence[str] | None = None
               ) -> dict[str, np.ndarray]:
    """Normalise a keys table to a mapping of name to column.

    Section 26 gives Python two spellings: a mapping from key name to
    a one-dimensional array, every array the same length, or a
    two-dimensional array of shape (rows, keys) together with the key
    names in the file's key order. Both are accepted here.
    """
    if isinstance(table, Mapping):
        out = {str(name): _column(name, values)
               for name, values in table.items()}
    else:
        values = np.asarray(table)
        if values.ndim != 2:
            raise MestraError(
                "", "a keys table is a mapping of name to column, or a "
                "two-dimensional array with the key names (section 26)")
        if names is None or len(names) != values.shape[1]:
            raise MestraError(
                "", "a two-dimensional keys table needs one name per "
                "column, in the file's key order; pass names="
                "(section 26)")
        out = {str(name): values[:, at]
               for at, name in enumerate(names)}
    lengths = {len(column) for column in out.values()}
    if len(lengths) > 1:
        raise MestraError(
            "", "every column of a keys table has the same length; "
            "these have %s"
            % ", ".join(str(n) for n in sorted(lengths)))
    return out


def _column(name: Any, values: Any) -> np.ndarray:
    column = np.asarray(values)
    if column.ndim == 0:
        column = column.reshape(1)
    if column.ndim != 1:
        raise MestraError(
            "", "the column of a keys table has one dimension, row",
            str(name))
    return column


def table_length(table: Mapping[str, np.ndarray]) -> int:
    """The number of rows in a normalised keys table."""
    for column in table.values():
        return int(len(column))
    return 0


# -------------------------------------------------------- the record

@dataclass(frozen=True)
class Prediction:
    """What a callable returns for one output (section 10).

    `mean` is the point prediction, shaped as the slot is stored:
    (row, node | cell, component) for an array, (row) for a scalar.
    `uncertainty` is optional: a band, the half-width of the interval
    around `mean` at coverage `level`, of the same shape, made as
    `method` says. The three go together, and a band that is missing
    its level or its method is refused here rather than in a file.

        Prediction(mean)
        Prediction(mean, band, level=0.95, method="...")
    """

    mean: np.ndarray
    uncertainty: np.ndarray | None = None
    level: float | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype="<f8")
        object.__setattr__(self, "mean", mean)
        if self.uncertainty is None:
            if self.level is not None or self.method is not None:
                raise MestraError(
                    "section 10", "level and method go with an "
                    "uncertainty; a prediction without one has neither")
            return
        band = np.asarray(self.uncertainty, dtype="<f8")
        if band.shape != mean.shape:
            raise MestraError(
                "section 10", "a band has the shape of its mean; the "
                "mean is %s and the band %s"
                % (mean.shape, band.shape))
        if self.level is None or not (0.0 < float(self.level) < 1.0):
            raise MestraError(
                "section 10", "a band states the coverage it claims as "
                "level in (0, 1), and this one gives %r" % (self.level,))
        if not self.method:
            raise MestraError(
                "section 10", "a band says how it was made; give method "
                "one sentence")
        if bool((band < 0).any()):
            raise MestraError(
                "section 10", "a band is a half-width and is never "
                "negative")
        object.__setattr__(self, "uncertainty", band)
        object.__setattr__(self, "level", float(self.level))
        object.__setattr__(self, "method", str(self.method))

    @property
    def has_uncertainty(self) -> bool:
        """True when the record carries a band."""
        return self.uncertainty is not None


# ---------------------------------------------------------- the protocol

class Callable(ABC):
    """Keys in, predictions out, plus a dictionary that represents it.

    A subclass sets `type` to the string that names it in a file and
    implements the three methods below.
    """

    #: The public type string, so that a reader knows which tool can
    #: evaluate the callable (section 10).
    type: ClassVar[str] = ""

    @abstractmethod
    def __call__(self, keys: Any) -> dict[str, Prediction]:
        """One prediction per output this callable serves.

        `keys` is a keys table (section 26). The result is a mapping
        from an output name to a `Prediction` whose `mean` is shaped
        as the slot would be stored, (row, node | cell, component)
        for an array and (row) for a scalar, with a band when the
        model has one.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """A nested dictionary of arrays, numbers and strings that
        fully represents this callable (sections 10 and 17)."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Callable:
        """Rebuild a callable from the dictionary `to_dict` made."""


_REGISTRY: dict[str, type[Callable]] = {}


def register_callable(kind: type[Callable]) -> type[Callable]:
    """Register a callable class under its `type` string.

    Usable as a decorator. A second registration of the same type
    replaces the first, so a tool can override a reference type with
    its own.
    """
    if not getattr(kind, "type", ""):
        raise MestraError(
            "E15", "a callable class must set a type string",
            kind.__name__)
    _REGISTRY[kind.type] = kind
    return kind


def callable_types() -> dict[str, type[Callable]]:
    """The registered callable types, by their type string."""
    return dict(_REGISTRY)


def callable_from_dict(kind: str, d: Mapping[str, Any],
                       repr_line: str | None = None) -> Any:
    """Rebuild a callable of type `kind`, or keep it opaque.

    A reader that does not own the type may still copy the
    dictionary and must not interpret it (section 12), which is what
    `OpaqueCallable` is for.
    """
    known = _REGISTRY.get(kind)
    if known is None:
        return OpaqueCallable(kind, d, repr_line)
    return known.from_dict(d)


class OpaqueCallable:
    """A callable whose type this reader does not own.

    It round-trips the dictionary unchanged and refuses to be
    called.
    """

    def __init__(self, kind: str, d: Mapping[str, Any],
                 repr_line: str | None = None) -> None:
        self.type = kind
        self._dict = dict(d)
        self._repr = repr_line

    def __call__(self, keys: Any) -> dict[str, Prediction]:
        raise MestraError(
            "section 10", "this reader does not own the callable type "
            "%r, so it can copy it but not evaluate it" % self.type)

    def to_dict(self) -> dict[str, Any]:
        """The dictionary as it was read, unchanged."""
        return dict(self._dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> OpaqueCallable:
        return cls("", d)

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        return "OpaqueCallable(type=%r)" % self.type


# ------------------------------------------------------------- affine

_AFFINE_KEYS = frozenset(["A", "b", "shape"])
_BAND_KEYS = frozenset(["uncertainty", "level", "method"])


@register_callable
class Affine(Callable):
    """y = A x + b per slot: the one callable type this package
    defines (section 27).

    It exists so that the protocol, the codec and evaluation can be
    conformance tested with no proprietary model. It is
    deterministic. An output may carry a constant band, the same in
    every row, given as `uncertainty`, `level` and `method` together.

        m = Affine(["mach", "alpha"],
                   {"cl": {"A": [[2.0, 0.1]], "b": [0.05],
                           "shape": []}})
        m({"mach": [0.5], "alpha": [4.0]})["cl"].mean
    """

    type: ClassVar[str] = "affine"

    def __init__(self, keys: Sequence[str],
                 outputs: Mapping[str, Mapping[str, Any]],
                 repr_line: str | None = None) -> None:
        self.keys = [str(k) for k in keys]
        self.outputs: dict[str, dict[str, np.ndarray]] = {}
        for name, entry in outputs.items():
            extra = set(entry) - _AFFINE_KEYS - _BAND_KEYS
            if extra:
                raise MestraError(
                    "section 27", "an affine output holds A, b and "
                    "shape, and a band as uncertainty, level and "
                    "method, and nothing else; this one also holds %s"
                    % ", ".join(sorted(extra)), str(name))
            missing = _AFFINE_KEYS - set(entry)
            if missing:
                raise MestraError(
                    "section 27", "an affine output holds A, b and "
                    "shape; this one has no %s"
                    % ", ".join(sorted(missing)), str(name))
            band_keys = set(entry) & _BAND_KEYS
            if band_keys and band_keys != _BAND_KEYS:
                raise MestraError(
                    "section 27", "a band on an affine output is all "
                    "three of uncertainty, level and method; this one "
                    "has %s" % ", ".join(sorted(band_keys)), str(name))
            shape = np.asarray(entry["shape"], dtype="<i8").reshape(-1)
            matrix = np.asarray(entry["A"], dtype="<f8")
            offset = np.asarray(entry["b"], dtype="<f8").reshape(-1)
            flat = int(np.prod(shape)) if shape.size else 1
            if matrix.shape != (flat, len(self.keys)):
                raise MestraError(
                    "section 27", "A is (%d, %d) for this output and "
                    "the file gives %s"
                    % (flat, len(self.keys), matrix.shape), str(name))
            if offset.shape != (flat,):
                raise MestraError(
                    "section 27", "b is (%d,) for this output and the "
                    "file gives %s" % (flat, offset.shape), str(name))
            made: dict[str, Any] = {"A": matrix, "b": offset,
                                    "shape": shape}
            if band_keys:
                band = np.asarray(entry["uncertainty"],
                                  dtype="<f8").reshape(-1)
                if band.shape != (flat,):
                    raise MestraError(
                        "section 27", "uncertainty is (%d,) for this "
                        "output and the file gives %s"
                        % (flat, band.shape), str(name))
                level = float(entry["level"])
                if not (0.0 < level < 1.0):
                    raise MestraError(
                        "section 27", "level is a coverage in (0, 1), "
                        "and this output gives %r" % (level,), str(name))
                method = str(entry["method"])
                if not method:
                    raise MestraError(
                        "section 27", "method is one sentence, not the "
                        "empty string", str(name))
                made.update(uncertainty=band, level=level, method=method)
            self.outputs[str(name)] = made
        self._repr = repr_line

    # -- the protocol

    def __call__(self, keys: Any) -> dict[str, Prediction]:
        """Evaluate every output on a keys table (section 27)."""
        table = keys_table(keys)
        missing = [k for k in self.keys if k not in table]
        if missing:
            raise MestraError(
                "", "the keys table has no column %s; this callable "
                "reads %s" % (", ".join(missing),
                              ", ".join(self.keys)))
        rows = table_length(table)
        columns = np.empty((rows, len(self.keys)), dtype="<f8")
        for at, name in enumerate(self.keys):
            columns[:, at] = np.asarray(table[name], dtype="<f8")
        return {name: self._predict(entry, columns)
                for name, entry in self.outputs.items()}

    def to_dict(self) -> dict[str, Any]:
        """The dictionary of section 27, and nothing else."""
        return {
            "keys": list(self.keys),
            "outputs": {name: {key: entry[key]
                               for key in ("A", "b", "shape",
                                           "uncertainty", "level",
                                           "method") if key in entry}
                        for name, entry in self.outputs.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Affine:
        extra = set(d) - {"keys", "outputs"}
        if extra:
            raise MestraError(
                "section 27", "an affine dictionary holds keys and "
                "outputs and nothing else; this one also holds %s"
                % ", ".join(sorted(extra)))
        missing = {"keys", "outputs"} - set(d)
        if missing:
            raise MestraError(
                "section 27", "an affine dictionary holds keys and "
                "outputs; this one has no %s"
                % ", ".join(sorted(missing)))
        keys = [str(k) for k in d["keys"]]
        return cls(keys, d["outputs"])

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        return "affine(%s -> %s)" % (", ".join(self.keys),
                                     ", ".join(sorted(self.outputs)))

    # -- the arithmetic, in the order section 27 fixes

    @classmethod
    def _predict(cls, entry: Mapping[str, Any],
                 columns: np.ndarray) -> Prediction:
        """The mean, and the constant band repeated over the rows."""
        mean = cls._evaluate(entry, columns)
        if "uncertainty" not in entry:
            return Prediction(mean)
        band = np.asarray(entry["uncertainty"]).reshape(mean.shape[1:])
        return Prediction(
            mean, np.broadcast_to(band, mean.shape).copy(),
            level=entry["level"], method=entry["method"])

    @staticmethod
    def _evaluate(entry: Mapping[str, Any],
                  columns: np.ndarray) -> np.ndarray:
        """Y = X A' + b, accumulated in the declared key order.

        The dot product is accumulated over the keys in the declared
        order and b is added last, with no fused multiply-add,
        because the corpus compares float64 results bit for bit and
        the other orders differ in the last place.
        """
        matrix = entry["A"]
        rows = columns.shape[0]
        out = np.zeros((rows, matrix.shape[0]), dtype="<f8")
        for at in range(matrix.shape[1]):
            out += columns[:, at:at + 1] * matrix[:, at]
        out += entry["b"]
        shape = tuple(int(n) for n in entry["shape"])
        return out.reshape((rows,) + shape)

    # -- building one

    @classmethod
    def from_arrays(cls, keys: Sequence[str],
                    outputs: Mapping[str, tuple[Any, Any, Sequence[int]]]
                    ) -> Affine:
        """Build from (A, b, shape) triples, one per output."""
        return cls(keys, {name: {"A": a, "b": b, "shape": shape}
                          for name, (a, b, shape) in outputs.items()})
