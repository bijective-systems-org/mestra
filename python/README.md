mestra for Python
=================

Install
-------

    pip install mestra

or, from a checkout of the repository, `pip install -e python/`.
Runtime dependencies are numpy and h5py, and nothing else; netCDF4,
h5netcdf and xarray are optional, used by the cross-reader checks and
by one adapter. The suite needs pytest, which the `dev` extra
installs:

    pip install -e "python/[dev]"
    python -m pytest python/tests

Three golden files are too large to commit. Write them once before
the suite, as `../vectors/README.md` says; a test that wants one
skips without it rather than failing.

    python vectors/generate.py --on-demand


The ten-line example
--------------------

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

It is the same data and the same value in all four languages.
`../docs/examples/` has seven worked examples, one per concept, each
runnable and under thirty lines, and `../docs/guide.md` says what the
concepts mean. Every call has its own documentation where you meet
it: `help(mestra)` and `help(mestra.Dataset)` for the library,
`mestra --help` for the command line.


Axis order and permute by name
------------------------------

Python hands an array back in the order the file stores it (SPEC
section 19): what it varies along first, then the nodes or the cells,
then the components. A reader in a column-major language hands the
same array back in the reverse order, and the dimension names are
what the two agree on, so index by name and permute by name and never
count axes.

    p.dims                                   # row, node, component
    p.values.at(row=1, node=2, component=0)              # 203.0
    p.values.transpose("component", "node", "row")

The names are those of SPEC section 4, an axis `at()` is not given
comes back whole, and the value at row 1, node 2, component 0 of the
example above is 203.0 in every language.


What the reader refuses
-----------------------

`read` refuses a file that breaks a rule it cannot vouch for what it
would return under -- E01, E16, E19, E25, E26, E29, E30, E40, E41,
which are `mestra.reader.REFUSED` -- and raises `MestraError` naming
the first; a semantic fault never stops a read, and
`read(path, strict=False)` hands back whatever could be read anyway.
The findings are on `dataset.problems` either way, `mestra.validate`
repeats them, and what could not be copied is in `dataset.lossy`.
`help(mestra.reader)` and `mestra.limits` carry the rest.

A refusal is what this package can promise. What it cannot promise is
that libhdf5 returns control on a file whose bytes were damaged: a
crafted single-byte change can fault or hang the library before any
code here runs (`../docs/compatibility.md`, "Damaged files";
`tests/test_mutation.py` measures it). Read a file nobody vouches for
from a process you can kill, as `mestra validate` does.


Where to go next
----------------

`../docs/guide.md` is the format in plain words, with one runnable
example per concept. `../SPEC.md` is the reference, read by section.
`../docs/api-conventions.md` is the names, the argument order, the
defaults and the messages all four languages follow, for anyone
comparing two of them.

Licence: Apache-2.0, as the rest of the code in this repository.
