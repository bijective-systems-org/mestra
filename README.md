# mestra

[![PyPI](https://img.shields.io/pypi/v/mestra)](https://pypi.org/project/mestra/)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)

An open file format for the results of a simulation study and the
surrogate models fitted to them. One file holds every run of a study
and says what each number is, so that a reader can check its claims
instead of assuming them.

A parametric study usually ends as a folder of solver outputs, a
spreadsheet of the inputs and a script that knows how to join them.
What the numbers mean -- their units, which runs failed, which rows
share a geometry, which were held out -- lives in that script, and
every tool that reads the folder has to guess it again. mestra writes
it into the file, where a validator can check it:

- A train/test split that puts one geometry, or one transient run, on
  both sides is reported (W01). A model scored that way is scored on
  cases it has already seen.
- Every field and scalar carries units in the UDUNITS grammar that CF
  uses (E11, W10), and every run can say whether it converged (W02).
- Whether every row is on the same mesh is stated in the file, and
  each mesh carries an id hashed from its connectivity (E08), so two
  files can be checked for sharing a mesh before their rows are
  compared or combined.
- A fitted model is a file of the same shape with no rows: its slots
  name the model that serves them, its keys carry the range it was
  trained over, and its support has the same id as its training data.

It is for engineers who run parametric or transient studies and want
the results to outlive the script that made them, for people who fit
surrogate or reduced-order models to CFD and FEA results, and for
anyone publishing a simulation dataset for machine learning, where the
split and the units should travel with the data.

Nothing is locked in. A mestra file is HDF5, laid out so that it is
also a valid netCDF-4 file, and `h5dump`, `ncdump` and xarray open it
without this package:

    import xarray as xr
    xr.open_datatree("run.mes")     # every group, every named dimension

What mestra adds is the meaning: roles, units, alignment, statistics
and models, stored as attributes any HDF5 reader can see.
[How it relates to VTK, CGNS, netCDF-CF and the machine-learning
dataset formats](docs/comparison.md).

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

Then check it, or anyone else's file, from a shell:

    $ mestra validate run.mes
    0 error(s), 0 warning(s)
    $ mestra info run.mes          # keys, supports and slots, one screen

The same thing on a mesh, with three members and a label, runnable:
[`docs/examples/a-support-and-a-field/`](docs/examples/a-support-and-a-field/).
A split that leaks, and a split by member that does not:
[`docs/examples/groups-and-splits/`](docs/examples/groups-and-splits/).

[`docs/guide.md`](docs/guide.md) is the format in plain words with one
example per concept, [`SPEC.md`](SPEC.md) is the reference, and each of
[`python/`](python/), [`matlab/`](matlab/), [`cpp/`](cpp/) and
[`julia/`](julia/) has a README for that interface.

Status: **specification version 0**, with Python, MATLAB, C++ and Julia
implementations checked against a 130-case conformance corpus and a
hostile-file subset. Within version 0 the meaning of an existing file
does not change; what may be added, and how, is in
[CONTRIBUTING.md](CONTRIBUTING.md). MATLAB supports ASCII strings only;
see the [capability table](docs/compatibility.md) for implementation
limits and the distinction between reading a model file and executing
its callable. The name is written `mestra`, lower case, everywhere. No
implementation is the reference: the spec and the corpus are.

Questions, a file the format cannot represent, or a reader that gets
something wrong: [open an issue](https://github.com/bijective-systems-org/mestra/issues/new/choose).
If you use mestra in published work, [CITATION.cff](CITATION.cff) says
how to cite it.

Licence: code under Apache-2.0 (see LICENSE); specification text under
CC-BY-4.0. Copyright (c) 2026 Bijective Systems.

[Contributing and changing the format](CONTRIBUTING.md) describes the
compatibility rules. [Releasing](docs/releasing.md) describes the checks
required for a release and for updating downstream consumers.
