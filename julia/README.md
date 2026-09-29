mestra for Julia
================

Install
-------

    julia --project=julia          # or Pkg.develop(path = "julia")
    julia> using Mestra

HDF5.jl is the only dependency outside the standard library, and the
package is `Mestra` because Julia capitalises package names; the
format is written `mestra`, lower case, everywhere.

    julia --project=julia -e 'using Pkg; Pkg.test()'

runs the suite: the 81-case conformance corpus, the hostile files and
the worked examples. Three corpus files are generated rather than
committed, because of their size; write them once, before the suite
runs and never while it is running:

    python vectors/generate.py --on-demand


The ten-line example
--------------------

    using Mestra

    ds = Mestra.Dataset(writer = "my tool 1")
    Mestra.add_key!(ds, "mach", [0.4, 0.8]; role = :condition, units = "1")
    s = Mestra.add_axis_support!(ds, "s0"; coordinates = [0.0, 1.0, 2.0],
                                 units = "m")
    Mestra.add_node_array!(ds, s, "pressure",
                           [101.0 102 103; 201.0 202 203];
                           units = "Pa", dims = (:row, :node))
    Mestra.write(ds, "run.mes")

    d = Mestra.read("run.mes")
    p = Mestra.values(d, d["pressure"])
    Mestra.at(p; row = 2, node = 3, component = 1)        # 203.0

The same data and the same value in all four languages.
`../docs/examples/` has seven worked examples, one per concept, each
with a `julia.jl` beside its README, and `../docs/guide.md` says what
the concepts mean. Every call has a docstring of its own:
`?Mestra.read`, `?Mestra.add_key!`, and `?Mestra` for the public API.


Axis order and permute by name
------------------------------

The format stores an array as (row, [draw], node | cell, component) in
C order. Julia is column major, so HDF5 hands the same bytes back with
the axes **reversed**: the `pressure` above arrives as (component,
node, row), and a MATLAB reader does the same. What two languages
agree on is not the axis positions but the name of each axis and the
value at one named coordinate, so nothing here takes an axis by
position:

    Mestra.dimnames(p)                   # (:component, :node, :row)
    q = Mestra.permute(p, (:row, :node, :component))
    q[2, 3, 1]                           # 203.0
    Mestra.at(p; row = 2, node = 3, component = 1)       # 203.0

Indices are one based, as everywhere else in Julia, so the corpus's
probe of (row 1, node 2, component 0) is that one. `?Mestra.at` has
the other axis names, `:instance` and `:draw` among them.


What the reader refuses
-----------------------

`Mestra.read` is strict, and refuses a file that breaks one of
`Mestra.STRUCTURAL_RULES` -- E01, E16, E19, E25, E26, E29, E30, E40
and E41 -- reading no array to decide it. `Mestra.read(path; strict =
false)` opens that file anyway and lists what it would not follow or
could not read in `ds.findings`, which is the call for a file you are
inspecting rather than trusting. What a file means rather than what it
is made of never stops a read and is reported by `Mestra.validate`,
printed by `Mestra.report`; `?Mestra.read` has the caps and the link
rules a hostile file meets.


Where to go next
----------------

`../docs/guide.md` is the format in plain words, one worked example a
concept. `../SPEC.md` is the reference, by section.
`../docs/api-conventions.md` is for anyone comparing two languages.
