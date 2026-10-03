"""Names: on disk, logical, and what counts as a legal one.

SPEC.md section 21 gives every logical dimension a name on disk and
section 18 says which names a file may use at all.
"""

from __future__ import annotations

import re

__all__ = [
    "MACHINERY",
    "RESERVED_PREFIX",
    "disk_dimension",
    "logical_dimension",
    "is_legal_name",
    "is_reserved",
]

#: Written by the HDF5 dimension scale machinery and by netCDF-C, no
#: part of this format, ignored wherever they appear (section 18).
MACHINERY = frozenset([
    "CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST",
    "DIMENSION_LABELS", "_Netcdf4Dimid", "_Netcdf4Coordinates",
    "_nc3_strict", "_NCProperties",
])

#: Reserved for the container everywhere in the file (section 18).
RESERVED_PREFIX = "mestra_"

_LEGAL = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*$")


def is_legal_name(name: str) -> bool:
    """True when `name` is a legal netCDF-4 name (section 18).

    Not empty, no "/" and no NUL, not beginning or ending with a
    space, built from letters, digits, underscore, hyphen, "." and
    "+", and beginning with a letter, a digit or an underscore.
    """
    if not name:
        return False
    # fullmatch: `$` in a pattern also matches before a final newline,
    # so a name ending in one would pass.
    return bool(_LEGAL.fullmatch(name))


def is_reserved(name: str) -> bool:
    """True when a producer may not choose this name (section 18)."""
    return name.startswith(RESERVED_PREFIX)


def disk_dimension(logical: str, length: int = 0) -> str:
    """The name on disk of a logical dimension (section 21).

    `length` is the component count for "component" and the draw
    count for "draw"; it is ignored for the others.
    """
    if logical == "draw":
        return "draw_%d" % length
    if logical == "component":
        return "component_%d" % length
    if logical.startswith("group:"):
        return "group_" + logical[len("group:"):]
    if logical.startswith("category:"):
        return "category_" + logical[len("category:"):]
    return logical


def logical_dimension(disk: str) -> str:
    """The logical name of a dimension stored under `disk`.

    A reader recovers the logical name by this table and never by
    the axis position (section 21).
    """
    if disk == "row":
        return "row"
    if disk.startswith("group_"):
        return "group:" + disk[len("group_"):]
    if disk.startswith("draw_"):
        return "draw"
    if disk.startswith("component_"):
        return "component"
    if disk.startswith("category_"):
        return "category:" + disk[len("category_"):]
    return disk
