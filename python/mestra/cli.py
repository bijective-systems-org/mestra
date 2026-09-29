"""The command line: `mestra validate FILE` and `mestra info FILE`.

    $ mestra validate run.mes
    0 error(s), 0 warning(s)

    $ mestra validate suspect.mes
    E11 /supports/s0/node_arrays/pressure: a field carries units
    1 error(s), 0 warning(s)

    $ mestra info run.mes
    ... one screen: the keys with their roles, units, bounds,
    category, trajectory group and parent, the supports with their
    kind, counts and id, and every slot with its shape under named
    axes, its units and its source.

Section 5 of `docs/api-conventions.md` fixes both shapes: a finding
is `<id> <path>: <message>` and a run ends with
`<n> error(s), <m> warning(s)`, in every language.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from . import FORMAT, __version__
from .errors import MestraError
from .model import Dataset
from .reader import read
from .validator import validate

__all__ = ["main"]

#: What the command line does not do, and where it is done instead.
#: A user who has met `mestra validate` should not have to find the
#: library by guessing at import names.
LIBRARY = """the library, for everything this command does not do:

  import mestra; help(mestra)
    mestra.Dataset()      add_key, add_scalar, add_category_table,
                          set_generalisation_group, add_support,
                          add_callable, add_callable_slot; then
                          support.add_node_array, add_cell_array
    mestra.write/read     a file, validated first; opened lazily
    mestra.validate       the findings above, as a Report by rule id
    mestra.evaluate       a callable file on a keys table, giving a
                          file of data
    mestra.compute_weights, mestra.support_ids
    mestra.post           prediction, field_statistics, integrate,
                          time_series,
                          grouped_split, split_leaks
    mestra.limits         what this reader refuses to go past, and why
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the exit status."""
    parser = argparse.ArgumentParser(
        prog="mestra", description="Read and check mestra files.",
        epilog=LIBRARY,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version",
                        version="mestra %s (format %s)"
                        % (__version__, FORMAT))
    commands = parser.add_subparsers(dest="command")
    check = commands.add_parser(
        "validate", help="report every rule of section 14 a file "
                         "breaks",
        description="Print every finding as `<id> <path>: <message>`, "
                    "then `<n> error(s), <m> warning(s)`. Exits "
                    "non-zero when a file has an error, so it fits in "
                    "a build; a warning does not. With several files "
                    "each one's findings are introduced by its name, "
                    "because a finding's own path is a path inside a "
                    "file, and the summary counts the run.")
    check.add_argument("file", nargs="+")
    check.add_argument("--quiet", action="store_true",
                       help="print the summary line only")
    show = commands.add_parser(
        "info", help="one screen about a file",
        description="Print, for every key: name, role, units, bounds, "
                    "category, trajectory group and parent; for every "
                    "support: kind, counts and id; for every slot: its "
                    "shape under the name of each axis, its units, its "
                    "source, and for a callable slot the callable's id "
                    "and output. Reading it costs an open and not a "
                    "read.")
    show.add_argument("file", nargs="+")
    args = parser.parse_args(list(argv) if argv is not None
                             else sys.argv[1:])
    if args.command == "validate":
        return _validate(args.file, args.quiet)
    if args.command == "info":
        return _info(args.file)
    parser.print_help()
    return 2


def _validate(paths: Sequence[str], quiet: bool) -> int:
    """Every finding, then the one summary line of section 5.

    With more than one file the findings of each are introduced by
    its name, because a finding's own path is a path inside the
    file; the summary counts the run.
    """
    errors = warnings = 0
    for path in paths:
        try:
            report = validate(path)
        except (OSError, MestraError) as exc:
            errors += 1
            if not quiet:
                if len(paths) > 1:
                    print(path)
                print("E01 %s: this file cannot be read: %s"
                      % (path, exc))
            continue
        errors += len(report.errors)
        warnings += len(report.warnings)
        if not quiet:
            if len(paths) > 1:
                print(path)
            for finding in report.findings:
                print(finding)
    print("%d error(s), %d warning(s)" % (errors, warnings))
    return 1 if errors else 0


def _info(paths: Sequence[str]) -> int:
    status = 0
    for at, path in enumerate(paths):
        if at:
            print("")
        try:
            with read(path) as ds:
                _print_dataset(path, ds)
        except (OSError, MestraError) as exc:
            print("%s: cannot be read: %s" % (path, exc))
            status = 1
    return status


def _print_dataset(path: str, ds: Dataset) -> None:
    print(path)
    for finding in ds.problems:
        print("  %s" % finding)
    print("  %s written by %r on %s" % (ds.format, ds.writer,
                                        ds.created))
    # A file with no support at all is not "aligned with" anything,
    # so it does not say so.
    if not ds.supports:
        structure = "no support"
    else:
        structure = "aligned" if ds.aligned else "not aligned"
    line = "  %d row(s), %s" % (ds.n_rows, structure)
    if ds.generalisation_group:
        line += ", generalisation unit %s" % ds.generalisation_group
    print(line)

    if ds.keys:
        print("  keys")
        for name in ds.key_names():
            key = ds.keys[name]
            print("    %-16s %-12s %s" % (name, key.role,
                                          _key_detail(ds, key)))
    if ds.scalars:
        print("  scalars")
        for name in sorted(ds.scalars):
            slot = ds.scalars[name]
            print("    %-16s %-30s %s" % (name, _dims(slot),
                                          _slot_detail(slot)))
    for sname in ds.support_names():
        support = ds.supports[sname]
        print("  support %s  %s, %d node(s), %d cell(s)"
              % (sname, support.kind, support.n_nodes, support.n_cells))
        print("    id %s" % support.support_id)
        for where, array in support.arrays().items():
            print("    %-16s %-30s %s"
                  % (where.rsplit("/", 1)[-1], _dims(array),
                     _slot_detail(array)))
    if ds.callables:
        print("  callables")
        for name in sorted(ds.callables):
            obj = ds.callables[name]
            print("    %-16s %-12s %s"
                  % (name, getattr(obj, "type", "?"), repr(obj)))
    if ds.notes:
        print("  notes")
        for name in sorted(ds.notes):
            print("    %-16s %s" % (name, ds.notes[name]))


def _key_detail(ds: Dataset, key: Any) -> str:
    out = []
    if key.units:
        out.append("units %s" % key.units)
    if key.lower is not None or key.upper is not None:
        out.append("bounds [%s, %s]" % (key.lower, key.upper))
    if key.category:
        table = ds.categories.get(key.category)
        entries = ", ".join(table[:4]) if table else "?"
        if table and len(table) > 4:
            entries += ", ..."
        out.append("categories %s (%s)" % (key.category, entries))
    if key.trajectory_group:
        out.append("trajectory %s" % key.trajectory_group)
    if key.parent:
        out.append("parent %s" % key.parent)
    return "  ".join(out)


def _dims(slot: Any) -> str:
    """The slot's shape under the name of each of its axes."""
    shape = ""
    if slot.data is not None:
        shape = "x".join(str(n) for n in slot.data.shape)
    return "(%s) %s" % (", ".join(slot.dims), shape)


def _slot_detail(slot: Any) -> str:
    out = []
    role = getattr(slot, "role", "scalar")
    out.append(role)
    if slot.units:
        out.append("units %s" % slot.units)
    if getattr(slot, "category", None):
        out.append("categories %s" % slot.category)
    out.append(slot.source)
    if slot.output:
        out.append("output %s" % slot.output)
    if slot.statistic:
        out.append("statistic %s" % slot.statistic)
    if slot.of:
        out.append("of %s" % slot.of)
    if slot.level is not None:
        out.append("level %s" % slot.level)
    if slot.method:
        out.append("method %r" % slot.method)
    return "  ".join(out)


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main())
