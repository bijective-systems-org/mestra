# mestra

An open container for simulation and surrogate data with declared
structure. One file holds the rows of a study and says what they are,
so that a reader can check its claims instead of assuming them.

What a file holds:

    rows        one observation each, located by keys that carry roles
    scalars     per-row quantities, with units
    supports    a mesh, a one-dimensional axis, or none
    arrays      fields and labels on a support, with what they vary
                along: the row, a group, or nothing
    callables   a slot may name the model that produces it, so a
                fitted model is a file with no rows
    alignment   every row on the same support: stated, and checkable

Install:

    python   pip install mestra   (or pip install -e python/ from a checkout)
    matlab   addpath('<this repository>/matlab')
    c++      cmake -S cpp -B cpp/build -DHDF5_ROOT="$HDF5_ROOT"
             cmake --build cpp/build -j
    julia    julia --project=julia   (or Pkg.develop(path="julia"))

Write a file and read a value back by name:

    import mestra

    ds = mestra.Dataset(writer="my tool 1")
    ds.add_key("mach", [0.4, 0.8], role="condition", units="1")
    s = ds.add_support("s0", coordinates=[0., 1., 2.], units="m")
    s.add_node_array("pressure", [[101., 102., 103.],
                                  [201., 202., 203.]],
                     units="Pa", dims=("row", "node"))
    mestra.write(ds, "run.mes")
    with mestra.read("run.mes") as d:
        p = d.supports["s0"].node_arrays["pressure"]
        print(p.values.at(row=1, node=2, component=0))    # 203.0

The same thing on a mesh, with three members and a label, runnable:
`docs/examples/a-support-and-a-field/`.

`docs/guide.md` is the format in plain words with one example per
concept, `SPEC.md` is the reference, and each of `python/`, `matlab/`,
`cpp/` and `julia/` has a README for that interface.

Status: **specification version 0**, with Python, MATLAB, C++ and Julia
implementations checked against an 81-case conformance corpus and a
hostile-file subset. MATLAB supports ASCII strings only; see the
[capability table](docs/compatibility.md) for implementation limits and
the distinction between reading a model file and executing its callable.
The name is written `mestra`, lower case, everywhere. No implementation
is the reference: the spec and the corpus are.

Licence: code under Apache-2.0 (see LICENSE); specification text under
CC-BY-4.0. Copyright (c) 2026 Bijective Systems.

[Contributing and changing the format](CONTRIBUTING.md) describes the
compatibility rules. [Releasing](docs/releasing.md) describes the checks
required for a release and for updating downstream consumers.
