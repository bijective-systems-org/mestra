"""The validator of SPEC.md section 14.

`validate(path)` opens a file and reports every rule it breaks, by
identifier. `validate(dataset)` checks what an in-memory dataset can
be checked for, which is everything except the byte-level rules of
sections 18 to 25.

Identifiers are stable: within a major version a rule is never
renumbered and a retired rule's identifier is never reused. E07 and
W09 were retired on 2026-09-20 and this validator never emits them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from . import h5safe, limits
from .encoding import (
    NULL_SENTINEL,
    SCALE_ATTR_ORDER,
    default_chunk_rows,
    is_fixed_string,
    read_attr,
    string_length,
    support_digest,
)
from .errors import Finding, MestraError, TooLarge
from .model import (
    ARRAY_ROLES,
    KEY_ROLES,
    STATISTICS,
    STATUS_WORDS,
    Dataset,
)
from .names import MACHINERY, RESERVED_PREFIX, is_legal_name
from .units import is_parseable

__all__ = ["Report", "validate"]

#: Section 20: the allowed cell type codes and the nodes each takes.
CELL_TYPES: dict[int, int] = {
    1: 1, 3: 2, 5: 3, 7: -1, 9: 4, 10: 4, 12: 8, 13: 6, 14: 5,
    21: 3, 22: 6, 23: 8, 24: 10, 25: 20, 26: 15, 27: 13,
}

_ROOT_ATTRS = ("format", "writer", "created", "aligned",
               "generalisation_group")
_ROOT_GROUPS = ("keys", "scalars", "categories", "supports",
                "callables", "notes", "private")
_KEY_ATTRS = ("role", "units", "lower", "upper", "category",
              "trajectory_group", "parent")
_SCALAR_ATTRS = ("units", "source", "output", "statistic", "of",
                 "quantile", "level", "method")
_ARRAY_ATTRS = ("role", "varies", "units", "components", "source",
                "output", "statistic", "of", "quantile", "level", "method",
                "category",
                "recomputed", "derived_from", "recipe", "reference")
_SUPPORT_ATTRS = ("kind", "n_nodes", "n_cells", "support_id")
_SUPPORT_MEMBERS = ("node", "cell", "cell_plus_one", "index", "row",
                    "cell_types", "cell_offsets", "cell_connectivity",
                    "coordinates", "node_arrays", "cell_arrays")

#: Attributes whose encoding section 18 fixes, by the kind it fixes.
_ATTR_KIND = {
    "format": "string", "writer": "string", "created": "string",
    "generalisation_group": "string", "role": "string",
    "units": "string", "category": "string", "source": "string",
    "output": "string", "statistic": "string", "of": "string",
    "varies": "string", "kind": "string", "support_id": "string",
    "derived_from": "string", "recipe": "string", "reference": "string",
    "trajectory_group": "string", "parent": "string", "type": "string",
    "repr": "string", "method": "string",
    "n_nodes": "integer", "n_cells": "integer", "components": "integer",
    "lower": "float", "upper": "float", "quantile": "float",
    "level": "float",
    "aligned": "boolean", "recomputed": "boolean",
}

#: The findings that say public information is missing, which is what
#: a validator can see of E18.
_MISSING_PUBLIC = ("E02", "E11", "E13", "E15", "E17", "E31", "E39")

#: W08: bounds wider than the observed range by more than this.
_STALE_BOUNDS_FACTOR = 4.0

#: The two portable filters of section 23, by their HDF5 identifier.
_FILTER_DEFLATE = 1
_FILTER_SHUFFLE = 2


@dataclass
class Report:
    """What the validator found, by rule identifier.

    `errors` and `warnings` carry the rules of section 14. A link
    this reader will not follow is E40 and an object it could not
    read is E41; both are errors like any other, and `unclassified`
    is the pair of them, for a caller that wants to tell what the
    file says from what the reader could not do with it.
    """

    errors: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the file is accepted: no error."""
        return not self.errors

    @property
    def unclassified(self) -> list[Finding]:
        """The findings about the reader rather than the format:
        E40, a link it does not follow, and E41, an object it could
        not read (section 14)."""
        return [f for f in self.errors if f.rule in ("E40", "E41")]

    @property
    def error_ids(self) -> list[str]:
        """The error identifiers, sorted and without duplicates."""
        return sorted({f.rule for f in self.errors})

    @property
    def warning_ids(self) -> list[str]:
        """The warning identifiers, sorted and without duplicates."""
        return sorted({f.rule for f in self.warnings})

    @property
    def findings(self) -> list[Finding]:
        """Everything found, errors first."""
        return self.errors + self.warnings

    def __str__(self) -> str:
        lines = [str(f) for f in self.findings]
        if not lines:
            return "no error and no warning"
        return "\n".join(lines)


def scan(f: h5py.File, limit: int | None = None,
         tables_only: bool = False) -> Report:
    """Validate an open file, reading no more than `limit` elements
    of any one dataset.

    With `tables_only`, the pass reads a category table and nothing
    else, which is what section 7 of docs/api-conventions.md allows
    a metadata open: "attributes, dataspaces, link types, and
    dimension-scale structure, and ... a category table in full",
    and "never a slot's data and never a dataset inside a callable's
    dictionary". `read` uses it that way to decide whether it can
    vouch for what it would return; `validate` reads what a rule
    asks for.
    """
    checker = _FileValidator(f)
    if limit is not None:
        checker.limit = limit
    checker.tables_only = tables_only
    return checker.run()


def validate(target: Any) -> Report:
    """Validate a file by path, or a dataset already in memory.

    A file this reader cannot open at all comes back as a report
    saying so (E01) rather than as an exception, so that a caller
    can validate a directory of strangers in one pass.

    A dataset in memory is checked for everything except the
    byte-level rules, which are about a file and not about a
    dataset. Its key and scalar columns are read, so that W01, W02,
    W03 and W04 say the same thing about a dataset as they do about
    the file; the arrays on a support are not, because a dataset
    opened lazily should not have every field pulled into memory by
    a check.
    """
    if isinstance(target, Dataset):
        return _validate_dataset(target)
    try:
        f = h5safe.open_file(str(target))
    except MestraError as exc:
        return Report(errors=[Finding(exc.rule, exc.where, exc.message)])
    try:
        return _FileValidator(f).run()
    finally:
        f.close()


# --------------------------------------------------------------- a file

class _FileValidator:
    """Every rule of section 14 that a file can break."""

    def __init__(self, f: h5py.File) -> None:
        self.f = f
        self.report = Report()
        self._scales: dict[int, str] | None = None
        self._listed: dict[int, list[Any]] = {}
        #: True for a metadata open: read a category table and
        #: nothing else (section 7 of docs/api-conventions.md).
        self.tables_only = False
        self.visited = 0
        self.limit = limits.MAX_CHECK_ELEMENTS
        self.too_large: list[str] = []
        self.n_rows = 0
        self.support_names: list[str] = []
        self.rows_on: dict[str, np.ndarray] = {}
        self.row_support: np.ndarray | None = None
        self.categories: dict[str, list[str]] = {}
        self.keys: dict[str, h5py.Dataset] = {}
        self.group_keys: list[str] = []

    @property
    def scales(self) -> dict[int, str]:
        """Section 21's map from a scale's address to its link name.

        Built once, on first use, from the handle this pass holds,
        and handed to every `scale_names` call the pass makes.
        Building one per dataset instead is what made an open cost
        the square of the number of row-dimensioned datasets, which
        is the cost section 29 puts a requirement on.
        """
        if self._scales is None:
            self._scales = h5safe.scale_index(self.f)
        return self._scales

    # -- collecting findings

    def error(self, rule: str, where: str, message: str) -> None:
        _once(self.report.errors, rule, where, message)

    def warn(self, rule: str, where: str, message: str) -> None:
        _once(self.report.warnings, rule, where, message)

    def note(self, where: str, message: str, rule: str = "E41") -> None:
        """An object this reader could not read (E41), or a link it
        will not follow (E40)."""
        self.error(rule, where, message)

    def guarded(self, where: str, step: Any, *args: Any) -> Any:
        """Run one step, and keep going when it fails.

        A file may hold an object the library will not read. That is
        the file's problem and this reader's finding; it is not a
        reason to abandon everything after it.
        """
        try:
            return step(*args)
        except MestraError as exc:
            self.note(exc.where or where, exc.message,
                      exc.rule if exc.rule in ("E40", "E41") else "E41")
        except RecursionError:
            self.note(where, "this object nests deeper than this "
                             "reader will follow")
        except Exception as exc:
            self.note(where, "this object could not be checked: %s"
                             % h5safe._brief(exc))
        return None

    def run(self) -> Report:
        if not self.guarded("/", self._version):
            return self.report
        for where, step in (("/", self._collect), ("/", self._root),
                            ("/categories", self._categories),
                            ("/keys", self._keys),
                            ("/scalars", self._scalars),
                            ("/supports", self._supports),
                            ("/", self._alignment),
                            ("/callables", self._callables),
                            ("/", self._bytes),
                            ("/private", self._private)):
            self.guarded(where, step)
        return self.report

    # -- reading a value, or saying why not

    def values(self, dset: h5py.Dataset, where: str) -> Any:
        """A dataset's values, or None and a finding.

        Everything a rule needs to look inside a dataset comes
        through here, so that a hostile size or a type the library
        will not convert costs one finding and not the pass, and so
        that a pass that may not read a dataset has one place to say
        so.
        """
        if self.tables_only and not where.startswith("/categories/"):
            # A metadata open. The rule that wanted these values is
            # left unchecked, which is not a finding: the nine
            # structural rules of section 2 of the conventions do not
            # need them, and everything else waits for the read.
            return None
        try:
            return np.asarray(h5safe.read_values(dset, where,
                                                 limit=self.limit))
        except TooLarge:
            # Section 14 reserves the size case for an eager read, so
            # the rules that need these values go unchecked rather
            # than this becoming E41.
            self.too_large.append(where)
        except MestraError as exc:
            self.note(where, exc.message)
        except Exception as exc:
            self.note(where, "this dataset could not be read: %s"
                             % h5safe._brief(exc))
        return None

    def members(self, group: h5py.Group, where: str) -> list[Any]:
        """The members of a group, reporting the ones it will not
        follow.

        Read once per group and kept for the pass. Several rules ask
        a group for its members, and opening every member again for
        each of them is a constant factor on the cost of a file with
        many of them. A finding is recorded once per rule per object
        however many times a rule looks, so the report is the same.
        """
        at = h5safe.address(group)
        listed = self._listed.get(at) if at is not None else None
        if listed is None:
            listed = h5safe.members(group)
            if at is not None:
                self._listed[at] = listed
        out = []
        for member in listed:
            if member.usable:
                out.append(member)
            else:
                self.note("%s/%s" % (where.rstrip("/"), member.name),
                          member.problem or "this member cannot be "
                                            "read", member.rule)
        return out

    def child(self, group: h5py.Group, where: str,
              name: str) -> Any:
        """One member by name, or None."""
        for member in self.members(group, where):
            if member.name == name:
                return member.obj
        return None

    # -- section 28: the version decides whether to read at all

    def _version(self) -> bool:
        if "format" not in self.f.attrs:
            self.error("E01", "/", "the root attribute format is "
                                   "missing")
            self.error("E17", "/", "format is required")
            return False
        text = read_attr(self.f, "format")
        major = None
        if isinstance(text, str) and text.startswith("mestra/"):
            tail = text[len("mestra/"):]
            major = int(tail) if tail.isdigit() else None
        if major != 0:
            self.error("E01", "/", "this reader accepts mestra/0 and "
                                   "the file says %r" % (text,))
            return False
        return True

    # -- what the rest of the checks need

    def _collect(self) -> None:
        self.root = {m.name: m for m in self.members(self.f, "/")}
        found = self.root.get("row")
        if found is not None and isinstance(found.obj, h5py.Dataset):
            shape = found.obj.shape
            self.n_rows = int(shape[0]) if shape else 0
        group = self._root_group("categories")
        if group is not None:
            for member in self.members(group, "/categories"):
                if isinstance(member.obj, h5py.Dataset):
                    self.categories[member.name] = self._strings_in(
                        member.obj, "/categories/" + member.name)
        group = self._root_group("keys")
        if group is not None:
            for member in self.members(group, "/keys"):
                if isinstance(member.obj, h5py.Dataset):
                    self.keys[member.name] = member.obj
                    if _attr(member.obj, "role") == "group":
                        self.group_keys.append(member.name)
                else:
                    self.note("/keys/" + member.name,
                              "a key is a dataset, and this is a group")
        group = self._root_group("supports")
        if group is not None:
            self.support_names = sorted(
                (m.name for m in self.members(group, "/supports")
                 if isinstance(m.obj, h5py.Group)),
                key=lambda n: n.encode("utf-8"))
        found = self.root.get("row_support")
        column = (found.obj if found is not None
                  and isinstance(found.obj, h5py.Dataset) else None)
        if column is not None:
            self.row_support = self.values(column, "/row_support")
        for at, name in enumerate(self.support_names):
            if column is None:
                # An aligned file: every row is on the one support.
                self.rows_on[name] = np.arange(self.n_rows)
            elif self.row_support is not None:
                self.rows_on[name] = np.flatnonzero(
                    self.row_support == at)
            # Otherwise the file carries the column and this pass did
            # not read it, which is a metadata open. `rows_on` stays
            # empty and the row count of a row-varying array comes
            # from the support's own `row` scale instead.

    def _root_group(self, name: str) -> Any:
        """One of the root groups, when it is a group and opened."""
        found = getattr(self, "root", {}).get(name)
        if found is None or not found.usable:
            return None
        if not isinstance(found.obj, h5py.Group):
            return None
        return found.obj

    def _strings_in(self, dset: h5py.Dataset, where: str) -> list[str]:
        """A fixed-length string dataset as text, leniently."""
        values = self.values(dset, where)
        if values is None:
            return []
        return [h5safe.decode_bytes(v) if isinstance(v, bytes)
                else str(v) for v in values.reshape(-1)]

    # -- the root group

    def _root(self) -> None:
        attrs = _names(self.f)
        for name in ("format", "writer", "created"):
            if name not in attrs:
                self.error("E17", "/", "the root attribute %s is "
                                       "missing" % name)
        if "aligned" not in attrs:
            self.error("E39", "/", "the root attribute aligned is "
                                   "required")
        if self.group_keys and "generalisation_group" not in attrs:
            self.error(
                "E39", "/", "a file that declares a group key names "
                "the unit of generalisation in the root attribute "
                "generalisation_group")
        if "created" in attrs and not _is_iso_utc(
                read_attr(self.f, "created")):
            self.warn("W14", "/", "created is not an ISO 8601 UTC "
                                  "timestamp")
        for name in attrs:
            if name not in _ROOT_ATTRS:
                self.warn("W11", "/", "this reader does not know the "
                                      "attribute %s, and ignores it"
                          % name)
        for name, member in self.root.items():
            if name in _ROOT_GROUPS or name == "row_support":
                continue
            if member.usable and isinstance(member.obj, h5py.Dataset):
                continue                    # a dimension scale at root
            if not member.usable:
                continue                    # reported where it is met
            self.warn("W11", "/" + name, "this reader does not know "
                                         "this group, and ignores it")

    def _private(self) -> None:
        """E18, from the public objects a validator can see missing.

        Section 29 forbids a validator to interpret /private, so all
        it can say is that public information is missing from a file
        that carries one.
        """
        if "private" not in getattr(self, "root", {}):
            return
        missing = [f for f in self.report.errors
                   if f.rule in _MISSING_PUBLIC]
        if missing:
            self.error(
                "E18", "/private", "public information is missing "
                "from this file (%s) and the file carries a private "
                "group; a writer must not put public information only "
                "there" % ", ".join(sorted({f.rule for f in missing})))

    # -- category tables

    def _categories(self) -> None:
        group = self._root_group("categories")
        if group is None:
            return
        for member in self.members(group, "/categories"):
            name, dset = member.name, member.obj
            where = "/categories/" + name
            if not isinstance(dset, h5py.Dataset):
                self.note(where, "a category table is a dataset, and "
                                 "this is a group")
                continue
            if not is_fixed_string(dset.dtype):
                self.error("E20", where, "a category table is a "
                                         "fixed-length UTF-8 string "
                                         "dataset")
                continue
            self._strings_of(dset, where, "E26")

    def _strings_of(self, dset: h5py.Dataset, where: str,
                    rule: str) -> None:
        """E26: valid UTF-8, and no NUL outside the padding."""
        size = string_length(dset.dtype) or 1
        longest = 0
        values = self.values(dset, where)
        if values is None:
            return
        for raw in values.reshape(-1):
            if not isinstance(raw, bytes):
                continue
            body = raw.rstrip(b"\x00")
            longest = max(longest, len(body))
            if b"\x00" in body:
                self.error(rule, where, "a fixed-length string holds a "
                                        "NUL byte outside its trailing "
                                        "padding")
            try:
                body.decode("utf-8")
            except UnicodeDecodeError:
                self.error(rule, where, "a fixed-length string is not "
                                        "valid UTF-8")
        if size > max(1, longest):
            self.warn("W13", where, "this string dataset is %d bytes "
                                    "wide where %d would do"
                      % (size, max(1, longest)))

    # -- keys

    def _keys(self) -> None:
        roles: dict[str, list[str]] = {}
        for name in sorted(self.keys):
            dset = self.keys[name]
            where = "/keys/" + name
            self._name(name, where)
            attrs = _names(dset)
            role = _attr(dset, "role")
            if role is None:
                self.error("E02", where, "a key carries a role")
            elif role not in KEY_ROLES:
                self.error("E02", where, "%r is not a key role of "
                                         "section 3" % role)
            else:
                roles.setdefault(role, []).append(name)
            self._unknown_attrs(dset, where, _KEY_ATTRS)
            self._units(dset, where)
            if (role in ("design", "condition", "time")
                    and "units" not in attrs):
                self.error("E39", where, "a %s key carries units"
                           % role)
            if (role in ("categorical", "group", "split", "status")
                    and "category" not in attrs):
                self.error("E39", where, "a %s key names its category "
                                         "table" % role)
            if (role == "time" and self.group_keys
                    and "trajectory_group" not in attrs):
                self.error("E39", where, "the time key names its "
                                         "trajectory_group")
            self._key_dtype(dset, where, role)
            self._rows(dset, where, self.n_rows, one_dimensional=True)
            self._categories_of(dset, where, role)
            self._bounds(dset, where, role)
        self._cardinality(roles)
        self._time(roles)
        self._status(roles)
        self._split(roles)

    def _key_dtype(self, dset: h5py.Dataset, where: str,
                   role: str | None) -> None:
        dtype = dset.dtype
        if role in ("design", "condition", "time"):
            _ = self._float64(dset, where, "a %s key" % role)
        elif role in ("categorical", "group", "split", "status"):
            if dtype not in _SIGNED_LE:
                self.error("E20", where, "a %s key is little-endian "
                                         "int32 or int64, and this is %s"
                           % (role, dtype))
        elif role == "id" and not is_fixed_string(dtype) and \
                dtype != np.dtype("<i8"):
            self.error("E20", where, "an id key is little-endian int64 "
                                     "or a fixed-length UTF-8 string")

    def _categories_of(self, dset: h5py.Dataset, where: str,
                       role: str | None) -> None:
        """E10 and W07: values against the category table."""
        if role not in ("categorical", "group", "split", "status"):
            return
        table_name = _attr(dset, "category")
        if table_name is None:
            return
        table = self.categories.get(table_name)
        if table is None:
            self.error("E10", where, "there is no category table "
                                     "called %r" % table_name)
            return
        values = self.values(dset, where)
        if values is None:
            return
        if values.size and values.dtype.kind in "iu":
            outside = values[(values < 0) | (values >= len(table))]
            if outside.size:
                self.error("E10", where, "the value %d is outside a "
                                         "category table of %d entries"
                           % (int(outside[0]), len(table)))
        if role == "group":
            used = {int(v) for v in np.asarray(values).reshape(-1)}
            unused = [t for at, t in enumerate(table) if at not in used]
            if unused:
                self.warn("W07", where, "the category table has an "
                                        "entry no row uses: %s"
                          % ", ".join(unused))

    def _bounds(self, dset: h5py.Dataset, where: str,
                role: str | None) -> None:
        """W04 and W08: the declared bounds against the values."""
        attrs = _names(dset)
        lower = read_attr(dset, "lower") if "lower" in attrs else None
        upper = read_attr(dset, "upper") if "upper" in attrs else None
        if lower is None and upper is None:
            return
        if not isinstance(lower, (int, float)) or not isinstance(
                upper, (int, float)):
            lower = lower if isinstance(lower, (int, float)) else None
            upper = upper if isinstance(upper, (int, float)) else None
        if lower is None and upper is None:
            return
        values = self.values(dset, where)
        if values is None:
            return
        if values.dtype.kind not in "fiu" or not values.size:
            return
        finite = values[np.isfinite(values)] if values.dtype.kind == "f" \
            else values
        if not finite.size:
            return
        said = w04(values, lower, upper)
        if said:
            self.warn("W04", where, said)
            return
        if lower is None or upper is None:
            return
        observed = float(np.max(finite)) - float(np.min(finite))
        declared = float(upper) - float(lower)
        if observed > 0 and declared > _STALE_BOUNDS_FACTOR * observed:
            self.warn("W08", where, "the declared bounds are %.1f "
                                    "times wider than the observed "
                                    "range, which is what stale bounds "
                                    "look like" % (declared / observed))

    def _cardinality(self, roles: dict[str, list[str]]) -> None:
        """E03: a role's cardinality violated."""
        for role, limit in KEY_ROLES.items():
            if limit is None:
                continue
            found = roles.get(role, [])
            if len(found) > limit:
                self.error("E03", "/keys", "a file has at most %d key "
                                           "with the role %s, and this "
                                           "one has %d: %s"
                           % (limit, role, len(found),
                              ", ".join(found)))

    def _time(self, roles: dict[str, list[str]]) -> None:
        """E09: time strictly increasing within each trajectory."""
        for name in roles.get("time", []):
            dset = self.keys[name]
            values = self.values(dset, "/keys/" + name)
            if values is None or values.dtype.kind not in "fiu":
                continue
            group = _attr(dset, "trajectory_group")
            labels: Any = None
            table = None
            if group and group in self.keys:
                labels = self.values(self.keys[group], "/keys/" + group)
                table = self.categories.get(
                    _attr(self.keys[group], "category") or "")
            if labels is None:
                labels = np.zeros(len(values), dtype="i8")
            if len(labels) != len(values):
                continue
            bad = [one for one in np.unique(labels)
                   if values[labels == one].size > 1
                   and not np.all(np.diff(values[labels == one]) > 0)]
            if bad:
                named = [_named(table, one) for one in bad]
                self.error(
                    "E09", "/keys/" + name, "time is not strictly "
                    "increasing within %d trajector%s (%s%s)"
                    % (len(named), "y" if len(named) == 1 else "ies",
                       ", ".join(named[:3]),
                       " and %d more" % (len(named) - 3)
                       if len(named) > 3 else ""))

    def _status(self, roles: dict[str, list[str]]) -> None:
        """W02: rows whose status is not converged."""
        for name in roles.get("status", []):
            dset = self.keys[name]
            table = self.categories.get(_attr(dset, "category") or "")
            if not table:
                continue
            values = self.values(dset, "/keys/" + name)
            if values is None or values.dtype.kind not in "iu":
                continue
            said = w02(values, table)
            if said:
                self.warn("W02", "/keys/" + name, said)

    def _split(self, roles: dict[str, list[str]]) -> None:
        """W01: a split that leaks the unit of generalisation."""
        if "generalisation_group" not in _names(self.f):
            return
        unit = read_attr(self.f, "generalisation_group")
        if unit not in self.keys:
            return
        units = self.values(self.keys[unit], "/keys/" + unit)
        if units is None:
            return
        table = self.categories.get(_attr(self.keys[unit],
                                          "category") or "")
        for name in roles.get("split", []):
            split = self.values(self.keys[name], "/keys/" + name)
            if split is None or len(split) != len(units):
                continue
            said = w01(units, split, str(unit), table)
            if said:
                self.warn("W01", "/keys/" + name, said)

    # -- scalars

    def _scalars(self) -> None:
        group = self._root_group("scalars")
        if group is None:
            return
        for found in self.members(group, "/scalars"):
            name, member = found.name, found.obj
            where = "/scalars/" + name
            self._name(name, where)
            self._unknown_attrs(member, where, _SCALAR_ATTRS)
            attrs = _names(member)
            if "units" not in attrs:
                self.error("E11", where, "a scalar carries units")
            self._units(member, where)
            self._source(member, where)
            self._statistic(member, where)
            if isinstance(member, h5py.Dataset):
                self._float64(member, where, "a scalar")
                self._rows(member, where, self.n_rows,
                           one_dimensional=True)
                self._finite(member, where)
                self._band(member, where)
            elif "components" in attrs:
                self.warn("W11", where, "components is not used on a "
                                        "scalar")

    # -- supports

    def _supports(self) -> None:
        group = self._root_group("supports")
        if group is None:
            return
        for found in self.members(group, "/supports"):
            if not isinstance(found.obj, h5py.Group):
                self.note("/supports/" + found.name,
                          "a support is a group, and this is a dataset")
                continue
            self.guarded("/supports/" + found.name, self._support,
                         found.name, found.obj)

    def _support(self, name: str, group: h5py.Group) -> None:
        where = "/supports/" + name
        self._name(name, where)
        self._unknown_attrs(group, where, _SUPPORT_ATTRS)
        attrs = _names(group)
        for required in _SUPPORT_ATTRS:
            if required not in attrs:
                self.error("E39", where, "a support carries %s"
                           % required)
        kind = _attr(group, "kind") or "mesh"
        n_nodes = _attr(group, "n_nodes")
        n_nodes = int(n_nodes) if isinstance(n_nodes, int) else 0
        n_cells = _attr(group, "n_cells")
        n_cells = int(n_cells) if isinstance(n_cells, int) else 0
        inside = {m.name: m.obj for m in self.members(group, where)}
        cells = {which: inside[which]
                 for which in ("cell_types", "cell_offsets",
                               "cell_connectivity")
                 if isinstance(inside.get(which), h5py.Dataset)}
        if kind == "mesh" and len(cells) != 3:
            self.error("E38", where, "a mesh support carries "
                                     "cell_types, cell_offsets and "
                                     "cell_connectivity")
        if kind in ("axis", "none") and (cells or "cell" in inside):
            self.error("E38", where, "a support of kind %s carries no "
                                     "cell datasets and no cell "
                                     "dimension" % kind)
        self.guarded(where, self._cells, where, cells, n_nodes,
                     n_cells if kind == "mesh" else None)
        self.guarded(where, self._support_id, where, group, kind,
                     n_nodes, cells, inside)

        arrays = []
        if "coordinates" in inside:
            arrays.append(("coordinates", inside["coordinates"], "node"))
        for which, location in (("node_arrays", "node"),
                                ("cell_arrays", "cell")):
            holder = inside.get(which)
            if holder is None:
                continue
            if not isinstance(holder, h5py.Group):
                self.note("%s/%s" % (where, which),
                          "this is a dataset where the format has a "
                          "group of slots")
                continue
            for member in self.members(holder, "%s/%s" % (where, which)):
                arrays.append(("%s/%s" % (which, member.name),
                               member.obj, location))
        if kind in ("mesh", "axis") and "coordinates" not in inside:
            self.error("E03", where, "a %s support has exactly one "
                                     "coordinates array" % kind)
        roles: dict[tuple[str, str], list[str]] = {}
        for slot_name, member, location in arrays:
            role = self.guarded(
                "%s/%s" % (where, slot_name), self._array, where, name,
                slot_name, member, location, kind, n_nodes, n_cells)
            if role is not None:
                roles.setdefault((role, location), []).append(slot_name)
        # Section 3 counts a role at each location: "weight 0..1 per
        # location", and a support may have both node and cell
        # weights. Coordinates are on the nodes, so for them the two
        # readings are the same count.
        for (role, location), found in sorted(roles.items()):
            limit = ARRAY_ROLES.get(role)
            if limit is not None and len(found) > limit:
                self.error("E03", where, "a support has at most %d "
                                         "array with the role %s at the "
                                         "%ss, and this has %d: %s"
                           % (limit, role, location, len(found),
                              ", ".join(sorted(found))))
        for member_name, member in inside.items():
            if member_name in _SUPPORT_MEMBERS:
                continue
            if isinstance(member, h5py.Group):
                self.warn("W11", "%s/%s" % (where, member_name),
                          "this reader does not know this group, and "
                          "ignores it")

    def _cells(self, where: str, cells: dict[str, h5py.Dataset],
               n_nodes: int, n_cells: int | None) -> None:
        """E05 and E21 to E24: the cell arrays against section 20.

        `n_cells` is the count a mesh declares, and None on any other
        kind, whose cell datasets are E38 alone.
        """
        if len(cells) != 3:
            return
        types = self.values(cells["cell_types"], where + "/cell_types")
        offsets = self.values(cells["cell_offsets"],
                              where + "/cell_offsets")
        connectivity = self.values(cells["cell_connectivity"],
                                   where + "/cell_connectivity")
        if types is None or offsets is None or connectivity is None:
            return
        if (types.dtype.kind not in "iu" or offsets.dtype.kind not in "iu"
                or connectivity.dtype.kind not in "iu"):
            return
        if n_cells is not None and len(types) != n_cells:
            # The cell count a mesh declares is the length of its
            # cell_types; a cell array is checked against the
            # declaration (E05), so the declaration is checked here
            # against the one dataset that defines it, or a mesh with
            # no cell arrays would carry any n_cells at all.
            self.error("E05", where + "/cell_types", "this support "
                                                     "declares %d cells "
                                                     "and cell_types "
                                                     "holds %d"
                       % (n_cells, len(types)))
        if cells["cell_types"].dtype != np.dtype("u1"):
            self.error("E20", where + "/cell_types", "cell_types is "
                                                     "uint8")
        for which in ("cell_offsets", "cell_connectivity"):
            if cells[which].dtype != np.dtype("<i8"):
                self.error("E20", where + "/" + which,
                           "%s is little-endian int64" % which)
        for code in np.unique(types):
            if int(code) not in CELL_TYPES:
                self.error("E21", where + "/cell_types", "%d is not a "
                                                         "cell type "
                                                         "code of "
                                                         "section 20"
                           % int(code))
        bad_offsets = (len(offsets) != len(types) + 1
                       or (offsets.size and offsets[0] != 0)
                       or np.any(np.diff(offsets) < 0)
                       or (offsets.size
                           and offsets[-1] != len(connectivity)))
        if bad_offsets:
            self.error("E23", where + "/cell_offsets",
                       "cell_offsets starts at 0, does not decrease, "
                       "and ends at the length of cell_connectivity")
        else:
            for at, code in enumerate(types):
                width = int(offsets[at + 1] - offsets[at])
                wanted = CELL_TYPES.get(int(code))
                if wanted is None:
                    continue
                if wanted == -1:
                    if width < 3:
                        self.error("E22", where, "cell %d is a polygon "
                                                 "of %d nodes"
                                   % (at, width))
                elif width != wanted:
                    self.error("E22", where, "cell %d is type %d and "
                                             "takes %d nodes, and the "
                                             "offsets give it %d"
                               % (at, int(code), wanted, width))
        if connectivity.size:
            outside = connectivity[(connectivity < 0)
                                   | (connectivity >= n_nodes)]
            if outside.size:
                self.error("E24", where + "/cell_connectivity",
                           "the value %d is outside [0, %d)"
                           % (int(outside[0]), n_nodes))

    def _support_id(self, where: str, group: h5py.Group, kind: str,
                    n_nodes: int, cells: dict[str, Any],
                    inside: dict[str, Any]) -> None:
        """E08: the digest against the stored arrays (section 24)."""
        if "support_id" not in _names(group):
            return
        axis_coordinates = None
        if kind == "axis" and isinstance(inside.get("coordinates"),
                                         h5py.Dataset):
            # Section 24: the digest of an axis support
            # is over the coordinates as they are stored, so that a
            # file whose axis coordinates wrongly vary breaks E35 and
            # nothing else.
            axis_coordinates = self.values(inside["coordinates"],
                                           where + "/coordinates")
            if axis_coordinates is None:
                return
        if kind != "mesh":
            # A support of kind axis or none has no cell arrays, so
            # steps 2 to 4 contribute no bytes at all for it.
            cells = {}
        parts: list[Any] = []
        for which in ("cell_types", "cell_offsets", "cell_connectivity"):
            if which not in cells:
                parts.append(None)
                continue
            values = self.values(cells[which], "%s/%s" % (where, which))
            if values is None:
                return
            parts.append(values)
        digest = support_digest(n_nodes, parts[0], parts[1], parts[2],
                                axis_coordinates)
        stored = _attr(group, "support_id")
        if stored != digest:
            self.error("E08", where, "the stored support_id does not "
                                     "match the arrays; they hash to "
                                     "%s" % digest)

    def _array(self, support_where: str, support_name: str, name: str,
               member: Any, location: str, kind: str, n_nodes: int,
               n_cells: int) -> str | None:
        where = "%s/%s" % (support_where, name)
        self._name(name.rsplit("/", 1)[-1], where)
        self._unknown_attrs(member, where, _ARRAY_ATTRS)
        attrs = _names(member)
        role = _attr(member, "role")
        if role is None:
            self.error("E02", where, "an array carries a role")
        elif role not in ARRAY_ROLES:
            self.error("E02", where, "%r is not an array role of "
                                     "section 3" % role)
        varies = _attr(member, "varies")
        if varies is None:
            self.error("E39", where, "an array carries varies")
            varies = "none"
        if role == "field" and "units" not in attrs:
            self.error("E11", where, "a field carries units")
        if role in ("coordinates", "derived") and "units" not in attrs:
            self.error("E39", where, "a %s array carries units" % role)
        self._units(member, where)
        self._source(member, where)
        self._statistic(member, where)
        if role == "derived" and ("derived_from" not in attrs
                                  or "recipe" not in attrs):
            self.error("E13", where, "a derived array carries "
                                     "derived_from and recipe")
        if role in ("weight", "normal") and not _marked_recomputed(member):
            # Present and false is no more a mark than absent: the
            # array does not say it was recomputed (section 14).
            self.warn("W06", where, "%s arrays are recomputed from the "
                                    "connectivity, never imported; say "
                                    "so with recomputed" % role)
        if name == "coordinates" and kind == "axis" and varies != "none":
            self.error("E35", where, "the coordinates of an axis "
                                     "support are part of its identity "
                                     "and have varies = none")
        components = _attr(member, "components")
        if components is None:
            self.error("E31", where, "an array slot declares its "
                                     "components")
        if not (varies in ("none", "row") or varies.startswith("group:")):
            # Section 5 names three: none, row and group:<k>. No
            # leading dimension agrees with any other word.
            self.error("E04", where, "varies is none, row or group:<k>, "
                                     "and this is %r" % varies)
            return role

        if not isinstance(member, h5py.Dataset):
            return role
        self._array_dtype(member, where, role)
        dims = _logical(member, self.scales)
        wanted = 1 + (1 if varies != "none" else 0) \
            + (1 if _attr(member, "statistic") == "draw" else 0) + 1
        if member.ndim != wanted:
            self.error("E04", where, "an array with varies = %s has %d "
                                     "axes, and this one has %d"
                       % (varies, wanted, member.ndim))
        elif varies != "none":
            leading = dims[0] if dims else ""
            if varies == "row" and leading != "row":
                self.error("E04", where, "varies = row, and the "
                                         "leading dimension is %s"
                           % leading)
            if varies.startswith("group:") and leading != varies:
                self.error("E04", where, "varies = %s, and the leading "
                                         "dimension is %s"
                           % (varies, leading))
        width = n_nodes if location == "node" else n_cells
        at = member.ndim - 2
        if at >= 0 and member.shape[at] != width:
            self.error("E05", where, "this support has %d %ss and the "
                                     "array has %d"
                       % (width, location, member.shape[at]))
        if (components is not None and member.ndim
                and member.shape[-1] != int(components)):
            self.error("E31", where, "components is %d and the "
                                     "component dimension is %d long"
                       % (int(components), member.shape[-1]))
        if varies == "row":
            self._rows(member, where,
                       self._rows_of(member, support_name))
        if varies.startswith("group:"):
            key = varies[len("group:"):]
            table = None
            if key in self.keys:
                table = self.categories.get(
                    _attr(self.keys[key], "category") or "")
            else:
                self.error("E04", where, "varies names the group key "
                                         "%r, which the file does not "
                                         "declare" % key)
            if table is not None and member.ndim and \
                    member.shape[0] != len(table):
                self.error("E34", where, "the group key %s has %d "
                                         "categories and this array "
                                         "has %d instances"
                           % (key, len(table), member.shape[0]))
        if role == "label":
            self._label(member, where)
        if role in ("field", "derived"):
            self._finite(member, where)
            self._band(member, where)
        return role

    def _array_dtype(self, dset: h5py.Dataset, where: str,
                     role: str | None) -> None:
        if role == "label":
            if dset.dtype not in _SIGNED_LE:
                self.error("E20", where, "a label is little-endian int32 "
                                         "or int64, and this is %s"
                           % dset.dtype)
            return
        if role in ("coordinates", "field", "derived", "weight",
                    "normal"):
            self._float64(dset, where, "a %s array" % role)

    def _label(self, dset: h5py.Dataset, where: str) -> None:
        """E10: a label whose table does not hold one of its values."""
        table_name = _attr(dset, "category")
        if table_name is None:
            return                  # its values are its own categories
        table = self.categories.get(table_name)
        if table is None:
            self.error("E10", where, "there is no category table "
                                     "called %r" % table_name)
            return
        values = self.values(dset, where)
        if values is None:
            return
        if values.size and values.dtype.kind in "iu":
            outside = values[(values < 0) | (values >= len(table))]
            if outside.size:
                self.error("E10", where, "the value %d is outside a "
                                         "category table of %d entries"
                           % (int(outside[0]), len(table)))

    # -- alignment

    def _alignment(self) -> None:
        """E06, E28, E37, W05 and W15 (sections 8 and 22)."""
        count = len(self.support_names)
        aligned = None
        if "aligned" in _names(self.f):
            aligned = bool(read_attr(self.f, "aligned"))
        has_column = self.root.get("row_support") is not None and \
            isinstance(getattr(self.root.get("row_support"), "obj", None),
                       h5py.Dataset)
        if aligned is not None:
            if aligned and count > 1:
                self.error("E37", "/", "aligned is true and the file "
                                       "declares %d supports" % count)
            if not aligned and count <= 1:
                self.error("E37", "/", "aligned is false and the file "
                                       "declares %d support" % count)
            if aligned and has_column:
                self.error("E28", "/row_support", "an aligned file "
                                                  "carries no "
                                                  "/row_support")
            if not aligned and not has_column:
                self.error("E28", "/", "an unaligned file carries a "
                                       "/row_support for every row")
        if count > 1:
            self.warn("W05", "/supports", "the file declares %d "
                                          "supports, so index-aligned "
                                          "operations are not "
                                          "available" % count)
        column = self.root.get("row_support")
        if has_column and column is not None and isinstance(
                column.obj, h5py.Dataset):
            if column.obj.dtype != np.dtype("<i4"):
                self.error("E20", "/row_support", "/row_support is "
                                                  "little-endian int32")
            self._rows(column.obj, "/row_support", self.n_rows)
        if has_column and self.row_support is not None:
            # The rules below are about the values in the column. A
            # pass that did not read them - a metadata open - checks
            # its dtype and its length above and stops here.
            values = np.asarray(self.row_support)
            outside = values[(values < 0) | (values >= max(count, 1))]
            if outside.size:
                self.error("E06", "/row_support", "a row references "
                                                  "support %d, and the "
                                                  "file declares %d"
                           % (int(outside[0]), count))
            for at, name in enumerate(self.support_names):
                if not np.any(values == at):
                    self.warn("W15", "/supports/" + name, "no row "
                                                          "references "
                                                          "this "
                                                          "support")

    # -- callables

    def _callables(self) -> None:
        group = self._root_group("callables")
        if group is None:
            return
        for found in self.members(group, "/callables"):
            name, obj = found.name, found.obj
            where = "/callables/" + name
            self._name(name, where)
            if not isinstance(obj, h5py.Group):
                self.note(where, "a callable is a group, and this is a "
                                 "dataset")
                continue
            self.guarded(where, self._dictionary_of, obj, where)

    def _dictionary_of(self, group: h5py.Group, where: str) -> None:
        if "type" not in _names(group):
            self.error("E15", where, "a callable carries a type "
                                     "string")
        self._dictionary(group, where, top_level=True)

    def _dictionary(self, group: h5py.Group, where: str,
                    top_level: bool, depth: int = 0) -> None:
        """E32 and E33 over one dictionary group (section 25)."""
        if depth > limits.MAX_DEPTH:
            self.note(where, "this dictionary nests more than %d "
                             "groups deep, and is not checked further"
                      % limits.MAX_DEPTH)
            return
        for found in self.members(group, where):
            name, member = found.name, found.obj
            if name.startswith(RESERVED_PREFIX):
                continue
            path = "%s/%s" % (where, name)
            if top_level and name in ("type", "repr"):
                self.error("E32", path, "type and repr are the "
                                        "container's attributes, so a "
                                        "dictionary may not use them "
                                        "at its top level")
            if not is_legal_name(name):
                self.error("E33", path, "a dictionary key is a legal "
                                        "netCDF-4 name")
            if isinstance(member, h5py.Group):
                self._dictionary(member, path, top_level=False,
                                 depth=depth + 1)
                continue
            if h5safe.is_scale(member):
                continue
            if member.ndim == 0:
                self.error("E32", path, "a zero-dimensional array is "
                                        "written as an attribute, not "
                                        "as a dataset")
                continue
            if is_fixed_string(member.dtype):
                self._strings_of(member, path, "E32")
                continue
            if member.dtype.name not in ("int8", "int32", "int64",
                                         "float64"):
                self.error("E32", path, "%s is not a dtype a dictionary "
                                        "may hold" % member.dtype)
            elif member.dtype.byteorder == ">":
                self.error("E32", path, "a numeric array in a dictionary "
                                        "is stored little-endian, and "
                                        "this one is big-endian")
        for name in _names(group):
            if name.startswith(RESERVED_PREFIX):
                continue
            if top_level and name in ("type", "repr"):
                continue
            if not is_legal_name(name):
                self.error("E33", "%s/%s" % (where, name), "a "
                           "dictionary key is a legal netCDF-4 name")

    # -- the byte-level sweep

    def _bytes(self) -> None:
        """E19, E25, E26, E27, E29, E33 and W12 over every object.

        /private is left alone, because section 12 lets a producer
        keep its own records there in whatever representation it
        chooses and section 29 forbids a reader to interpret them.
        A group this version does not know is left alone too:
        section 14 says it "is reported as W11 and otherwise left
        alone", and section 28 says where one may be, "at the root
        and inside a support". `_unknown_group` below is those two
        places; the W11 itself is made by the root check and by the
        support check.

        A *dataset* this version does not know, inside a group it
        does know, is checked like any other. The text settles
        that in two steps.
        Section 14 exempts two things from the byte-level rules of
        sections 18 to 25 and names them: "`/private` is not
        checked, and neither is any group this version of the format
        does not know". A dataset in a support group is neither, so
        it is one of the "public objects only" the rules are checked
        on, and an unchunked one with a row dimension is E27.
        Section 28 is the second step: what a version may add is
        "new optional attributes" and "new optional groups ... at
        the root and inside a support", and not datasets, so an
        unknown dataset is not a forward-compatibility extension at
        all and there is nothing to make room for. It draws no W11
        either, because W11 is "an attribute or a group this reader
        does not know" and this is neither.
        """
        self._object(self.f, "/")
        stack = [(self.f, "", 0)]
        while stack:
            group, prefix, depth = stack.pop()
            if depth > limits.MAX_DEPTH:
                self.note(prefix or "/", "this group nests more than "
                                         "%d levels deep, and is not "
                                         "checked further"
                          % limits.MAX_DEPTH)
                continue
            for member in self.members(group, prefix or "/"):
                name, obj = member.name, member.obj
                path = "%s/%s" % (prefix, name)
                if path == "/private":
                    continue
                if isinstance(obj, h5py.Group) and _unknown_group(
                        prefix, name):
                    continue
                self.visited += 1
                if self.visited > limits.MAX_OBJECTS:
                    self.note(path, "this file holds more than %d "
                                    "objects, and is not checked "
                                    "further" % limits.MAX_OBJECTS)
                    return
                self.guarded(path, self._object, obj, path)
                if isinstance(obj, h5py.Group):
                    stack.append((obj, path, depth + 1))
                else:
                    self.guarded(path, self._dataset, obj, path)

    def _object(self, obj: Any, path: str) -> None:
        for name in h5safe.attr_names(obj):
            if not is_legal_name(name):
                self.error("E33", path, "the attribute name %r is not "
                                        "a legal netCDF-4 name" % name)
            self.guarded(path, self._attr_encoding, obj, name, path)

    def _attr_encoding(self, obj: Any, name: str, path: str) -> None:
        """E19: the encoding section 18 requires (section 18)."""
        attr = obj.attrs.get_id(name)
        kind = _ATTR_KIND.get(name)
        htype = attr.get_type()
        # From the identifier already in hand: asking h5safe would
        # open the attribute a second time, once per attribute of
        # every object in the file.
        shape = tuple(attr.shape)
        if shape and kind is not None:
            # Section 18 gives every attribute it names a scalar
            # dataspace. An attribute it does not name is W11 and
            # nothing else, whatever its shape: section 28 says an
            # unknown attribute is ignored and reported, and the
            # other three implementations read it that way.
            self.error("E19", path, "the attribute %s has the shape "
                                    "%s, and section 18 gives it a "
                                    "scalar dataspace"
                       % (name, tuple(shape)))
        if isinstance(htype, h5py.h5t.TypeStringID):
            if htype.is_variable_str():
                self.error("E19", path, "the attribute %s is a "
                                        "variable-length string, which "
                                        "section 18 forbids anywhere"
                           % name)
                return
            if htype.get_cset() != h5py.h5t.CSET_UTF8:
                self.error("E19", path, "the attribute %s is a string "
                                        "that is not UTF-8" % name)
            if htype.get_strpad() != h5py.h5t.STR_NULLPAD:
                self.error("E19", path, "the attribute %s is a string "
                                        "that is not NUL padded" % name)
            if kind is not None and kind != "string":
                self.error("E19", path, "the attribute %s is a %s and "
                                        "is stored as a string"
                           % (name, kind))
            try:
                raw = obj.attrs[name]
            except Exception:
                return
            if isinstance(raw, np.ndarray) and raw.size:
                raw = raw.reshape(-1)[0]
            if isinstance(raw, bytes) and raw != NULL_SENTINEL:
                body = raw.rstrip(b"\x00")
                if b"\x00" in body:
                    self.error("E26", path, "the attribute %s holds a "
                                            "NUL byte outside its "
                                            "trailing padding" % name)
                try:
                    body.decode("utf-8")
                except UnicodeDecodeError:
                    self.error("E26", path, "the attribute %s is not "
                                            "valid UTF-8" % name)
            return
        dtype = attr.dtype
        if kind is None:
            # A numeric attribute section 18 does not name: W11 from
            # the object's own walk, and no encoding to hold it to.
            return
        if kind == "string":
            self.error("E19", path, "the attribute %s is a string and "
                                    "is stored as %s" % (name, dtype))
            return
        if kind == "boolean":
            if dtype != np.int8:
                self.error("E19", path, "the boolean attribute %s is "
                                        "int8, and this is %s"
                           % (name, dtype))
            elif int(np.asarray(obj.attrs[name]).reshape(-1)[0]) \
                    not in (0, 1):
                self.error("E19", path, "the boolean attribute %s is 0 "
                                        "or 1" % name)
            return
        if kind == "integer":
            if dtype != np.dtype("int64"):
                self.error("E19", path, "the integer attribute %s is "
                                        "int64, and this is %s"
                           % (name, dtype))
            return
        if kind == "float":
            if dtype != np.dtype("float64"):
                self.error("E19", path, "the float attribute %s is "
                                        "float64, and this is %s"
                           % (name, dtype))
            elif not np.isfinite(np.asarray(obj.attrs[name])).all():
                self.error("E19", path, "the attribute %s declares a "
                                        "bound, a level or a quantile "
                                        "and must be finite" % name)

    def _dataset(self, dset: h5py.Dataset, path: str) -> None:
        if h5safe.is_scale(dset):
            # Section 21: a scale carries CLASS and NAME, and NAME is
            # the sentence with the dimension's length. Half a scale
            # is not one, and a reader that took the name from NAME
            # would have nothing to take.
            if "NAME" not in dset.attrs:
                self.error("E25", path, "this dimension scale carries "
                                        "CLASS and no NAME, which is "
                                        "half of what makes a scale")
            if (path.rsplit("/", 1)[-1] == "row"
                    and dset.maxshape and dset.maxshape[0] is not None):
                self.error("E27", path, "row is an unlimited dimension "
                                        "in every file")
            self._scale_properties(dset, path)
            self._unlimited(dset, path, (path.rsplit("/", 1)[-1],))
            return
        # Once per dataset. Every axis name below comes from this
        # one call: reading them again for each check meant
        # dereferencing every attached scale three more times, which
        # is a constant factor on the cost section 29 bounds.
        attached = h5safe.scale_names(dset, self.scales)
        for axis, names in enumerate(attached):
            if len(names) != 1:
                self.error("E25", path, "axis %d carries %d dimension "
                                        "scales, and every axis "
                                        "carries exactly one"
                           % (axis, len(names)))
        self._scale_names(dset, path, attached)
        self._unlimited(dset, path,
                        tuple(names[0] if len(names) == 1 else None
                              for names in attached))
        self._filters(dset, path)
        dims = _logical_of(attached)
        if dims[:1] == ("row",):
            if dset.chunks is None:
                self.error("E27", path, "a row-dimensioned dataset is "
                                        "chunked, because row is "
                                        "unlimited")
            else:
                self._chunk(dset, path)

    def _scale_properties(self, dset: h5py.Dataset, path: str) -> None:
        """E42: the creation property list of section 21.

        Read back from the file, because this is the one rule in the
        format that a property list and not a byte position decides.
        A scale created without attribute creation order tracked and
        indexed keeps its REFERENCE_LIST in an object header message,
        which may not exceed 64 KiB, so it takes at most 4085
        attachments and the 4086th destroys the attribute it was
        extending on its way to failing. Every reader and every
        validator accepts what that leaves behind, which is why the
        rule is about how the scale was made rather than about what
        the file now says.

        Section 21 asks for object time tracking off in the same
        list. That one keeps the file byte reproducible and section
        14 does not name it here: E42 is "attribute creation order
        not tracked and indexed", and nothing else. The corpus's own
        `check.py` is where the times are checked, on the golden
        files, because byte reproducibility is the corpus's rule and
        not a rule about a file somebody hands this reader.
        """
        order = h5safe.attr_creation_order(dset)
        if order is None or order == SCALE_ATTR_ORDER:
            return
        self.error("E42", path, "this dimension scale was created "
                                "with attribute creation order %d "
                                "where section 21 requires %d "
                                "(tracked and indexed), so it takes "
                                "at most 4085 attachments"
                   % (order, SCALE_ATTR_ORDER))

    def _unlimited(self, dset: h5py.Dataset, path: str,
                   dims: Sequence[str | None]) -> None:
        """E43: an unlimited dimension other than `row`.

        `dims` is the name on disk of the dimension on each axis: for
        a scale its own link name, and for any other dataset the link
        name of the scale attached to that axis. `row` is unlimited
        in every file, file-level and support-local, and no other
        dimension may be: every other name carries its own length,
        directly in `draw_<n>` and `component_<n>` and through the
        table it names in `group_<k>` and `category_<t>`, so a
        dimension that grows makes its own name false.

        The one exception is a zero-length axis of a dataset inside a
        callable's dictionary, which section 25 requires to be
        unlimited because HDF5 has no other legal way to write it.
        """
        maxshape = dset.maxshape or ()
        for axis, limit in enumerate(maxshape):
            if limit is not None:
                continue
            if axis < len(dims) and dims[axis] == "row":
                continue
            if (path.startswith("/callables/")
                    and axis < len(dset.shape) and dset.shape[axis] == 0):
                continue
            self.error("E43", path, "axis %d is unlimited, and %s is "
                                    "not `row`" % (
                                        axis,
                                        "the dimension %s" % dims[axis]
                                        if axis < len(dims)
                                        and dims[axis] else "it"))

    def _filters(self, dset: h5py.Dataset, path: str) -> None:
        """E29: only gzip at levels 1 to 9, and shuffle (23).

        Read from the creation property list rather than from h5py's
        own properties, which report only the filters it knows: an
        identifier no library has, or one with more client-data
        values than a reader expects, would otherwise look like no
        filter at all.
        """
        for code, _flags, values, name in h5safe.filters_of(dset):
            if code == _FILTER_SHUFFLE:
                continue
            if code == _FILTER_DEFLATE:
                level = int(values[0]) if values else 0
                if not 1 <= level <= 9:
                    self.error("E29", path, "gzip is allowed at levels "
                                            "1 to 9, and this is %d"
                               % level)
                if len(values) > 1:
                    self.error("E29", path, "gzip takes one client-data "
                                            "value and this has %d"
                               % len(values))
                continue
            self.error("E29", path, "the filter %d (%s) with %d "
                                    "client-data value(s) is not one of "
                                    "the two portable filters"
                       % (code, name or "unnamed", len(values)))

    def _scale_names(self, dset: h5py.Dataset, path: str,
                     attached: list[tuple[str, ...]]) -> None:
        """E25: the name section 21 requires for each axis."""
        wanted = _wanted_dims(path, dset)
        if wanted is None:
            return
        got = [names[0] if len(names) == 1 else None
               for names in attached]
        for axis, (have, allowed) in enumerate(zip(got, wanted)):
            if have is None or allowed is None:
                continue
            if not allowed(have, dset.shape[axis]):
                self.error("E25", path, "axis %d carries the dimension "
                                        "%s, which is not the name "
                                        "section 21 requires"
                           % (axis, have))

    def _chunk(self, dset: h5py.Dataset, path: str) -> None:
        """W12: a chunk shape that is not the default of section 23."""
        rows = _row_length(dset, self.scales)
        default = (default_chunk_rows(dset.dtype.itemsize,
                                      dset.shape[1:], rows),) \
            + tuple(dset.shape[1:])
        if tuple(dset.chunks) != default:
            self.warn("W12", path, "the chunk is %s where the default "
                                   "of section 23 is %s"
                      % (tuple(dset.chunks), default))

    # -- small shared checks

    def _name(self, name: str, where: str) -> None:
        """E33: a producer-chosen name."""
        if not is_legal_name(name):
            self.error("E33", where, "%r is not a legal netCDF-4 name"
                       % name)
        elif name.startswith(RESERVED_PREFIX):
            self.error("E33", where, "%s is reserved for the container "
                                     "everywhere in the file"
                       % RESERVED_PREFIX)

    def _unknown_attrs(self, obj: Any, where: str,
                       known: Sequence[str]) -> None:
        """W11: an attribute this reader does not know."""
        for name in _names(obj):
            if name not in known:
                self.warn("W11", where, "this reader does not know the "
                                        "attribute %s, and ignores it"
                          % name)

    def _units(self, obj: Any, where: str) -> None:
        """W10: a units string the validator cannot parse."""
        if "units" not in _names(obj):
            return
        text = read_attr(obj, "units")
        if isinstance(text, str) and not is_parseable(text):
            self.warn("W10", where, "the units string %r does not parse"
                      % text)

    def _source(self, obj: Any, where: str) -> None:
        """E14, E30, E36 and E39 over one slot's source."""
        attrs = _names(obj)
        if "source" not in attrs:
            self.error("E39", where, "a slot carries source")
            return
        source = read_attr(obj, "source")
        if source == "data":
            if not isinstance(obj, h5py.Dataset):
                self.error("E30", where, "a slot whose source is data "
                                         "is a dataset, not a group")
            return
        if not isinstance(source, str) or \
                not source.startswith("callable:"):
            self.error("E36", where, "source is data or callable:<id>, "
                                     "and this is %r" % (source,))
            return
        if isinstance(obj, h5py.Dataset):
            self.error("E30", where, "a slot a callable serves is a "
                                     "group, not a dataset")
        identifier = source[len("callable:"):]
        if "callables" not in self.f or identifier not in \
                self.f["callables"]:
            self.error("E14", where, "the file holds no callable "
                                     "called %r" % identifier)
        if "output" not in attrs:
            self.error("E39", where, "a slot a callable serves names "
                                     "which output fills it")

    def _statistic(self, obj: Any, where: str) -> None:
        """E12: a statistic with what it needs (section 9), and on a
        slot a callable serves, one a callable produces (section
        10)."""
        attrs = _names(obj)
        if "statistic" not in attrs:
            return
        statistic = read_attr(obj, "statistic")
        if statistic not in STATISTICS:
            self.error("E02", where, "%r is not a statistic of section "
                                     "9" % statistic)
            return
        served = False
        if "source" in attrs:
            source = read_attr(obj, "source")
            served = isinstance(source, str) and \
                source.startswith("callable:")
        if statistic == "quantile" and "quantile" not in attrs:
            self.error("E12", where, "a quantile statistic carries its "
                                     "quantile")
        if statistic not in ("value", "draw") and "of" not in attrs:
            self.error("E12", where, "a %s names the quantity it is a "
                                     "statistic of" % statistic)
        if served and statistic in ("std", "quantile", "draw"):
            self.error("E12", where, "a callable returns a mean and at "
                                     "most a band (section 10), so a "
                                     "slot it serves is value, mean or "
                                     "band, not %s" % statistic)
        if statistic == "band" and not served:
            if "level" not in attrs:
                self.error("E12", where, "a band states the coverage it "
                                         "claims with level")
            else:
                level = read_attr(obj, "level")
                if isinstance(level, float) and not 0.0 < level < 1.0:
                    self.error("E12", where, "level is a coverage "
                                             "fraction in (0, 1), and "
                                             "this is %r" % level)
            if "method" not in attrs:
                self.error("E12", where, "a band says how it was made "
                                         "with method")

    def _band(self, dset: h5py.Dataset, where: str) -> None:
        """W16: a negative value in a band (section 9)."""
        if "statistic" not in _names(dset) or \
                read_attr(dset, "statistic") != "band":
            return
        if dset.dtype.kind != "f" or not dset.size:
            return
        values = self.values(dset, where)
        if values is None or values.dtype.kind != "f":
            return
        leading = _logical(dset, self.scales)
        said = w16(values, leading[0] if leading else "index")
        if said:
            self.warn("W16", where, said)

    def _float64(self, dset: h5py.Dataset, where: str,
                 what: str) -> bool:
        if dset.dtype != np.dtype("<f8"):
            self.error("E20", where, "%s is little-endian float64, and "
                                     "this is %s" % (what, dset.dtype))
            return False
        return True

    def _rows_of(self, member: h5py.Dataset, support_name: str) -> int:
        """The row count a row-varying array on a support must have.

        Section 14, E16: the file's row count, "except for a
        row-varying array on a support in an unaligned file, where it
        is the number of rows referencing that support". Section 21
        puts that same number in the file as the length of the
        support's own `row` scale, so a pass that did not read
        `/row_support` -- a metadata open, by section 7 of the
        conventions -- takes it from there. The two agree in a
        conforming file; where they do not, only a pass that read the
        column can say so, and that is `validate`.
        """
        rows = self.rows_on.get(support_name)
        if rows is not None:
            return len(rows)
        length = _leading_scale_length(member, self.scales)
        return self.n_rows if length is None else length

    def _rows(self, dset: h5py.Dataset, where: str, rows: int,
              one_dimensional: bool = False) -> None:
        """E16: the leading dimension against the row count."""
        if one_dimensional and dset.ndim != 1:
            self.error("E16", where, "a dataset under /keys or "
                                     "/scalars has exactly one "
                                     "dimension, and this has %d"
                       % dset.ndim)
        if dset.ndim and dset.shape[0] != rows:
            self.error("E16", where, "this holds %d rows where the "
                                     "file has %d"
                       % (dset.shape[0], rows))

    def _finite(self, dset: h5py.Dataset, where: str) -> None:
        """W03: non-finite values in a field or a scalar."""
        if dset.dtype.kind != "f" or not dset.size:
            return
        values = self.values(dset, where)
        if values is None or values.dtype.kind != "f":
            return
        bad = ~np.isfinite(values)
        if not bad.any():
            return
        leading = _logical(dset, self.scales)
        said = w03(values, leading[0] if leading else "index")
        if said:
            self.warn("W03", where, said)


# -------------------------------------------------------------- helpers

def _once(into: list[Finding], rule: str, where: str,
          message: str) -> None:
    """Record a finding, once per rule per object.

    Section 5 of the conventions: one finding per rule per object.
    A rule that can be broken more than once in the same place says
    so in its own message, with the count and the first three; this
    is the net under all of them, so that no rule has to remember.
    """
    for found in into:
        if found.rule == rule and found.where == where:
            return
    into.append(Finding(rule, where, message))


def _count(many: int, singular: str, plural: str) -> str:
    """"a value" or "4 values", so that a message reads."""
    return singular if many == 1 else "%d %s" % (many, plural)


def _at(dimension: str, indices: Any) -> str:
    """Where a per-row rule was broken: the first three, and a count.

    Section 5 of the conventions: a finding that could repeat per
    row reports once, with the count and the first three indices.
    """
    listed = [int(i) for i in list(indices)[:3]]
    text = "at %s %s" % (dimension, ", ".join(str(i) for i in listed))
    if len(indices) > len(listed):
        text += " and %d more" % (len(indices) - len(listed))
    return text


def _named(table: Any, value: Any) -> str:
    """A category id as the file's own table names it (section 21)."""
    at = int(value)
    if table is not None and 0 <= at < len(table):
        return str(table[at])
    return str(at)


# The three rules that could repeat per row, each written once and
# used by both the file validator and the in-memory one, so that the
# same file gets the same sentence whichever way it is checked. Each
# returns the message, or None when the rule is not broken.

def w01(units: np.ndarray, split: np.ndarray, unit: str,
        table: Any) -> str | None:
    """W01: a split that leaks the unit of generalisation."""
    leaked = [one for one in np.unique(units)
              if len(np.unique(split[units == one])) > 1]
    if not leaked:
        return None
    named = [_named(table, one) for one in leaked]
    if len(named) == 1:
        return ("the rows of %s %s are on both sides of the split, so "
                "this is not a generalisation test" % (unit, named[0]))
    return ("the rows of %d units of generalisation (%s%s) are on both "
            "sides of the split, so this is not a generalisation test"
            % (len(named), ", ".join(named[:3]),
               " and %d more" % (len(named) - 3) if len(named) > 3
               else ""))


def w02(values: np.ndarray, table: Any) -> str | None:
    """W02: rows whose status is not converged."""
    bad = [row for row, value in enumerate(values)
           if 0 <= int(value) < len(table)
           and table[int(value)] != "converged"]
    if not bad:
        return None
    return ("%s of %d with a status other than converged (the first is "
            "%r), %s; section 3's words are %s, and a row that is not "
            "converged is left out of modelling unless it is asked for"
            % (_count(len(bad), "one row", "rows"), len(values),
               table[int(values[bad[0]])], _at("row", bad),
               ", ".join(STATUS_WORDS)))


def w03(values: np.ndarray, axis: str) -> str | None:
    """W03: non-finite values in a field, derived array or scalar."""
    if values.dtype.kind != "f" or not values.size:
        return None
    bad = ~np.isfinite(values)
    if not bad.any():
        return None
    rows = sorted({int(at[0]) for at in np.argwhere(bad)})
    return ("%s, which is how this format spells missing "
            "floating-point data, %s"
            % (_count(int(bad.sum()), "a non-finite value",
                      "non-finite values"), _at(axis, rows)))


def w16(values: np.ndarray, axis: str) -> str | None:
    """W16: negative values in a band, which is a half-width."""
    if values.dtype.kind != "f" or not values.size:
        return None
    bad = values < 0
    if not bad.any():
        return None
    rows = sorted({int(at[0]) for at in np.argwhere(bad)})
    return ("%s below zero in a band, which is a half-width and is "
            "never negative, %s"
            % (_count(int(bad.sum()), "a value", "values"),
               _at(axis, rows)))


def w04(values: np.ndarray, lower: Any, upper: Any) -> str | None:
    """W04: key values outside the declared bounds."""
    outside = np.zeros(values.shape, dtype=bool)
    if lower is not None:
        outside |= values < lower
    if upper is not None:
        outside |= values > upper
    if values.dtype.kind == "f":
        outside &= np.isfinite(values)
    if not outside.any():
        return None
    rows = sorted({int(at[0]) for at in np.argwhere(outside)})
    return ("%s outside the declared bounds [%s, %s], %s"
            % (_count(int(outside.sum()), "a value", "values"),
               lower, upper, _at("row", rows)))


def _marked_recomputed(obj: Any) -> bool:
    """True when `recomputed` is there and says true. A malformed
    value is E19's to report, and is not a mark either."""
    try:
        value = _attr(obj, "recomputed")
    except Exception:  # noqa: BLE001
        return False
    return value is True or (isinstance(value, (np.integer, int))
                             and not isinstance(value, bool)
                             and int(value) == 1)


#: Section 19's integer dtypes for category ids and labels: signed and
#: little-endian, which a test of kind and size alone does not see.
_SIGNED_LE = (np.dtype("<i4"), np.dtype("<i8"))


def _names(obj: Any) -> list[str]:
    return [n for n in obj.attrs if n not in MACHINERY]


def _attr(obj: Any, name: str) -> Any:
    if name not in obj.attrs or name in MACHINERY:
        return None
    return read_attr(obj, name)


def _strings(dset: h5py.Dataset) -> list[str]:
    return [v.rstrip(b"\x00").decode("utf-8", errors="replace")
            if isinstance(v, bytes) else str(v) for v in dset[()]]


def _unknown_group(prefix: str, name: str) -> bool:
    """True for a group this version of the format does not know.

    Section 28 says where one may appear: "new optional groups may
    be added at the root and inside a support". Those are the two
    places, so they are the two places this asks about. Section 14
    then says what to do with one: it "is reported as W11 and
    otherwise left alone", and the byte-level rules of sections 18
    to 25 are not checked inside it, exactly as they are not checked
    inside `/private`. A group anywhere else is not an extension
    point and is checked; so is every dataset, wherever it sits,
    because section 28 never adds datasets.
    """
    if not prefix:
        return name not in _ROOT_GROUPS and name != "row_support"
    parts = prefix.strip("/").split("/")
    if len(parts) == 2 and parts[0] == "supports":
        return name not in _SUPPORT_MEMBERS
    return False


def _logical(dset: h5py.Dataset,
             scales: Mapping[int, str]) -> tuple[str, ...]:
    """The logical dimension names of a dataset, from its scales."""
    return _logical_of(h5safe.scale_names(dset, scales))


def _logical_of(attached: list[tuple[str, ...]]) -> tuple[str, ...]:
    """The same, from names a caller has already read once."""
    from .names import logical_dimension
    return tuple(logical_dimension(names[0]) if len(names) == 1 else ""
                 for names in attached)


def _leading_scale_length(dset: h5py.Dataset,
                          scales: Mapping[int, str]) -> int | None:
    """The length of the dimension the leading axis is attached to,
    or None when the axis does not carry exactly one scale."""
    names = h5safe.scale_names(dset, scales)
    if not names or len(names[0]) != 1:
        return None
    try:
        return int(dset.dims[0][0].shape[0])
    except Exception:                                   # pragma: no cover
        return None


def _row_length(dset: h5py.Dataset, scales: Mapping[int, str]) -> int:
    """The length of the row dimension a dataset is written over.

    In an unaligned file a support-local `row` shadows the file's
    one (section 21), and the chunk default follows the dimension
    the dataset actually uses.
    """
    names = h5safe.scale_names(dset, scales)
    if names and len(names[0]) == 1:
        scale = dset.dims[0][0]
        return int(scale.shape[0])
    return int(dset.shape[0])


def _exact(name: str) -> Any:
    return lambda have, length: have == name


def _component(have: str, length: int) -> bool:
    return have == "component_%d" % length


def _leading(have: str, length: int) -> bool:
    return (have == "row" or have.startswith("group_")
            or have == "draw_%d" % length)


def _wanted_dims(path: str, dset: h5py.Dataset) -> Any:
    """What each axis of a dataset may be called (section 21)."""
    parts = path.strip("/").split("/")
    if path == "/row_support" or parts[:1] == ["keys"]:
        return [_exact("row")]
    if parts[:1] == ["scalars"]:
        return [_exact("row")]
    if parts[:1] == ["categories"] and len(parts) == 2:
        return [_exact("category_" + parts[1])]
    if parts[:1] == ["callables"]:
        name = parts[-1]
        return [(lambda have, length, at=at, name=name:
                 have == "%s%s_d%d" % (RESERVED_PREFIX, name, at))
                for at in range(dset.ndim)]
    if parts[:1] != ["supports"] or len(parts) < 3:
        return None
    last = parts[-1]
    if last == "cell_types":
        return [_exact("cell")]
    if last == "cell_offsets":
        return [_exact("cell_plus_one")]
    if last == "cell_connectivity":
        return [_exact("index")]
    location = "cell" if "cell_arrays" in parts else "node"
    if dset.ndim < 2:
        return None
    return ([_leading] * (dset.ndim - 2)
            + [_exact(location), _component])


def _is_iso_utc(text: Any) -> bool:
    """W14: an ISO 8601 timestamp in UTC (section 11)."""
    if not isinstance(text, str):
        return False
    body = text.strip()
    if body.endswith("Z"):
        body = body[:-1]
    elif body.endswith("+00:00"):
        body = body[:-len("+00:00")]
    else:
        return False
    import datetime
    for form in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                 "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            datetime.datetime.strptime(body, form)
            return True
        except ValueError:
            continue
    return False


# ------------------------------------------------------- a dataset only

def _units_text(units: Any, where: str, error: Any, warn: Any) -> None:
    """E19 for a units that is not text at all, W10 for text that
    does not parse; an in-memory dataset can hold either."""
    if units is None or units == "":
        return
    if not isinstance(units, str):
        error("E19", where, "the attribute units is a string, and this "
                            "is %s" % type(units).__name__)
    elif not is_parseable(units):
        warn("W10", where, "the units string %r does not parse" % units)


def _validate_dataset(ds: Dataset) -> Report:
    """The rules an in-memory dataset can break.

    The byte-level rules of sections 18 to 25 are about a file and
    are not checked here; validate the file itself for those.

    What is read: the key columns and the scalar columns, which is
    one value per row each, so that W01, W02, W03 and W04 say the
    same thing here as they do about the file. The arrays on a
    support are not read, because a dataset opened lazily should not
    have every field pulled into memory by a check; W03 on a field
    is a finding about the file, from `validate(path)`.
    """
    report = Report()

    def error(rule: str, where: str, message: str) -> None:
        _once(report.errors, rule, where, message)

    def warn(rule: str, where: str, message: str) -> None:
        _once(report.warnings, rule, where, message)

    # What the reader met while opening the file this dataset came
    # from: a link it would not follow, a member of the wrong kind,
    # something the library would not read. `unclassified` is a view
    # of `errors`, so these belong in `errors`.
    for problem in ds.problems:
        error(problem.rule, problem.where, problem.message)

    if ds.format != "mestra/0":
        error("E01", "/", "this reader accepts mestra/0 and the "
                          "dataset says %r" % ds.format)
        return report
    for name, value in (("writer", ds.writer), ("created", ds.created)):
        if not value:
            error("E17", "/", "the root attribute %s is missing" % name)
    if not _is_iso_utc(ds.created):
        warn("W14", "/", "created is not an ISO 8601 UTC timestamp")
    groups = [k for k in ds.keys.values() if k.role == "group"]
    if groups and not ds.generalisation_group:
        error("E39", "/", "a file that declares a group key names the "
                          "unit of generalisation")

    counted: dict[str, int] = {}
    for name in ds.key_names():
        key = ds.keys[name]
        where = "/keys/" + name
        if key.role not in KEY_ROLES:
            error("E02", where, "%r is not a key role of section 3"
                  % key.role)
            continue
        counted[key.role] = counted.get(key.role, 0) + 1
        if key.role in ("design", "condition", "time") and not key.units:
            error("E39", where, "a %s key carries units" % key.role)
        _units_text(key.units, where, error, warn)
        bounds: list[tuple[str, Any]] = [("lower", key.lower),
                                         ("upper", key.upper)]
        for label, value in bounds:
            if value is not None and not np.isfinite(float(value)):
                error("E19", where, "the attribute %s declares a bound "
                                    "and must be finite, and this is %r"
                      % (label, value))
        if key.role in ("categorical", "group", "split", "status"):
            table = ds.categories.get(key.category or "")
            if table is None:
                error("E39", where, "a %s key names its category table"
                      % key.role)
            elif key.data is not None:
                values = np.asarray(key.data.read())
                if values.size and ((values < 0)
                                    | (values >= len(table))).any():
                    error("E10", where, "a value is outside a category "
                                        "table of %d entries"
                          % len(table))
                elif key.role == "status":
                    said = w02(values, table)
                    if said:
                        warn("W02", where, said)
        if key.role in ("design", "condition", "time") and \
                key.data is not None and \
                (key.lower is not None or key.upper is not None):
            said = w04(np.asarray(key.data.read()), key.lower, key.upper)
            if said:
                warn("W04", where, said)
        if key.data is not None and int(key.data.shape[0]) != ds.n_rows:
            error("E16", where, "this holds %d rows where the dataset "
                                "has %d"
                  % (int(key.data.shape[0]), ds.n_rows))
    for role, limit in KEY_ROLES.items():
        if limit is not None and counted.get(role, 0) > limit:
            error("E03", "/keys", "a file has at most %d key with the "
                                  "role %s" % (limit, role))

    unit = ds.generalisation_group
    if unit and unit in ds.keys and ds.keys[unit].data is not None:
        units = np.asarray(ds.keys[unit].read().values)
        table = ds.categories.get(ds.keys[unit].category or "")
        for key in ds.keys_of_role("split"):
            if key.data is None:
                continue
            split = np.asarray(key.read().values)
            if len(split) != len(units):
                continue
            said = w01(units, split, unit, table)
            if said:
                warn("W01", "/keys/" + key.name, said)

    for name, slot in sorted(ds.scalars.items()):
        where = "/scalars/" + name
        if not slot.units:
            error("E11", where, "a scalar carries units")
        _units_text(slot.units, where, error, warn)
        _check_source(ds, slot, where, error)
        if slot.data is not None:
            values = np.asarray(slot.data.read())
            said = w03(values, "row")
            if said:
                warn("W03", where, said)
            if int(values.shape[0]) != ds.n_rows:
                error("E16", where, "this holds %d rows where the "
                                    "dataset has %d"
                      % (int(values.shape[0]), ds.n_rows))

    for sname in ds.support_names():
        support = ds.supports[sname]
        where = "/supports/" + sname
        if support.kind in ("mesh", "axis") and \
                support.coordinates is None:
            error("E03", where, "a %s support has exactly one "
                                "coordinates array" % support.kind)
        if support.kind == "mesh" and support.cell_types is None:
            error("E38", where, "a mesh support carries cell_types, "
                                "cell_offsets and cell_connectivity")
        if support.stored_support_id is not None and \
                support.stored_support_id != support.computed_support_id():
            error("E08", where, "the stored support_id does not match "
                                "the arrays")
        counted_arrays: dict[tuple[str, str], int] = {}
        for array in support.arrays().values():
            at = (array.role, array.location)
            counted_arrays[at] = counted_arrays.get(at, 0) + 1
        for (role, location), found in sorted(counted_arrays.items()):
            limit = ARRAY_ROLES.get(role)
            if limit is not None and found > limit:
                error("E03", where, "a support has at most %d array "
                                    "with the role %s at the %ss, and "
                                    "this has %d"
                      % (limit, role, location, found))
        for slot_name, array in support.arrays().items():
            slot_where = "%s/%s" % (where, slot_name)
            if array.role not in ARRAY_ROLES:
                error("E02", slot_where, "%r is not an array role of "
                                         "section 3" % array.role)
            if array.role == "field" and not array.units:
                error("E11", slot_where, "a field carries units")
            _units_text(array.units, slot_where, error, warn)
            if array.role == "derived" and not (array.derived_from
                                                and array.recipe):
                error("E13", slot_where, "a derived array carries "
                                         "derived_from and recipe")
            if (array.role in ("weight", "normal")
                    and not array.recomputed):
                warn("W06", slot_where, "%s arrays are recomputed from "
                                        "the connectivity, never "
                                        "imported" % array.role)
            if array.name == "coordinates" and support.kind == "axis" \
                    and array.varies != "none":
                error("E35", slot_where, "the coordinates of an axis "
                                         "support have varies = none")
            _check_source(ds, array, slot_where, error)
    if len(ds.supports) > 1:
        warn("W05", "/supports", "the dataset declares %d supports, so "
                                 "index-aligned operations are not "
                                 "available" % len(ds.supports))
        if not ds.has_row_support:
            error("E28", "/", "an unaligned file carries a "
                              "/row_support for every row")
    elif ds.has_row_support:
        error("E28", "/row_support", "an aligned file carries no "
                                     "/row_support")
    return report


def _check_source(ds: Dataset, slot: Any, where: str,
                  error: Any) -> None:
    if slot.source == "data":
        if slot.data is None:
            error("E30", where, "a slot whose source is data holds "
                                "data")
        return
    if not slot.source.startswith("callable:"):
        error("E36", where, "source is data or callable:<id>, and this "
                            "is %r" % slot.source)
        return
    if (slot.callable_id or "") not in ds.callables:
        error("E14", where, "the dataset holds no callable called %r"
              % slot.callable_id)

