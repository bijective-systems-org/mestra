"""mestra: a container for simulation and surrogate data.

Rows of observations over design parameters, operating conditions
and time; scalar quantities and fields on shared supports; the claim
that rows are index-aligned, stated so a reader can check it.

    import mestra

    ds = mestra.read("run.mes")
    print(ds.n_rows, ds.aligned)
    pressure = ds.supports["s0"].node_arrays["pressure"]
    print(pressure.dims)                      # row, node, component
    print(pressure.read(slice(0, 4)).at(row=1, node=3, component=0))

    report = mestra.validate("run.mes")
    for finding in report.errors + report.warnings:
        print(finding)

Where everything is:

    Dataset          the builders, in the order of the conventions:
                     add_key, add_scalar, add_category_table,
                     set_generalisation_group, add_support,
                     add_callable, add_callable_slot, set_row_support,
                     and on a support add_node_array, add_cell_array,
                     add_callable_slot. `help(mestra.Dataset)`
    write, read      a file, validated before it is written; opened
                     lazily, so an open costs a check and not a read
    validate         a Report of findings by rule id, from a path or
                     from a dataset in memory
    evaluate         a file of callables on a keys table, giving a
                     file of data
    compute_weights  the integration measure of section 3, computed
                     from the connectivity, and support_ids beside it
    Callable         the four-method protocol, with Affine as the one
                     type this package defines, and register_callable
    Prediction       what a callable returns per output: a mean and,
                     when it has one, a band with its level and method
    mestra.post      prediction, field_statistics, integrate,
                     time_series,
                     grouped_split, split_leaks
    mestra.units     parse, is_parseable, same_dimensions
    mestra.limits    what this reader refuses to go past, and why
    mestra validate  the same findings from a shell, and `mestra info`
                     for one screen about a file

This package implements SPEC.md version 0. The specification and the
conformance corpus define the format between them; no implementation
is the reference.
"""

from __future__ import annotations

from .callables import (
    Affine,
    Callable,
    OpaqueCallable,
    Prediction,
    callable_from_dict,
    callable_types,
    keys_table,
    register_callable,
)
from .codec import decode_dict, encode_dict
from .encoding import support_digest
from .errors import Finding, MestraError
from .evaluate import evaluate
from .model import (
    ArraySlot,
    CategoryTable,
    Dataset,
    Key,
    NamedArray,
    ScalarSlot,
    Slot,
    Storage,
    Support,
)
from .post import prediction
from .reader import read, support_ids
from .validator import Report, validate
from .weights import compute_weights
from .writer import write

__version__ = "0.1.3"

#: The format version this package reads and writes.
FORMAT = "mestra/0"

__all__ = [
    "Affine",
    "ArraySlot",
    "Callable",
    "CategoryTable",
    "Dataset",
    "FORMAT",
    "Finding",
    "Key",
    "MestraError",
    "NamedArray",
    "OpaqueCallable",
    "Prediction",
    "Report",
    "ScalarSlot",
    "Slot",
    "Storage",
    "Support",
    "__version__",
    "callable_from_dict",
    "callable_types",
    "compute_weights",
    "decode_dict",
    "encode_dict",
    "evaluate",
    "keys_table",
    "prediction",
    "read",
    "register_callable",
    "support_digest",
    "support_ids",
    "validate",
    "write",
]
