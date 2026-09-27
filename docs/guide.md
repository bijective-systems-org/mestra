A guide to mestra
=================

This is for an engineer who has simulation results, or a model fitted
to them, and wants one file that says what they are. It explains the
model in plain words and points at one runnable example per idea. It
is not the reference: `../SPEC.md` is, and this guide links into it by
section rather than repeating it.

Read it in order once; it is about twenty minutes with the README.
Every example lives under `examples/`, is under thirty lines, and its
README states the data and the exact output, so you can check a run
against the page. The output blocks are what the Python script
prints; a port in another language prints the same values in that
language's own formatting.

The examples are written in Python because the package installs in one
line. The same calls exist in MATLAB, C++ and Julia under the same
names and in the same order (`api-conventions.md`), so nothing here is
Python-only except the spelling.


1. What a file holds
--------------------

One file is one dataset: a table of rows, plus whatever each row
carries. A row is one observation, which is usually one design at one
operating point at one time. Every column of that table is either a
*key*, saying where the row sits in the space that was sampled, or a
*scalar*, a per-row quantity with units. Fields live on a *support*,
which is the mesh or the axis they are sampled over, and are stored as
*arrays* beside it.

Nothing else is required. A file of keys and scalars with no support
at all is a complete, valid file; so is a file with no rows whose
slots are served by a model.

The container is HDF5, laid out so that a valid file is also a valid
netCDF-4 file (SPEC section 13). The extension is `.mes`, and the root
attribute `format` is what decides, not the extension.


2. Rows, keys and scalars
-------------------------

Every key carries a *role*, and the role is what tools act on
(SPEC section 3):

    design       a geometry or design parameter
    condition    an operating point: Mach, altitude, load
    time         one per file, increasing within a trajectory
    categorical  an integer with a table of category names
    group        which rows belong together; see section 4 below
    split        train, validation, test or holdout
    id           unique per row
    status       whether the row is fit to use

A design, condition or time key carries units; a categorical, group,
split, id or status key carries the name of a category table instead.
The units are strings in the UDUNITS grammar that CF uses: `Pa`,
`m s-1`, `W m-2`, and `1` for a dimensionless quantity.

Two things about keys are worth knowing before you write one.

*Bounds.* A design, condition or time key may declare the range it is
valid over. If you do not pass one, the builder records the observed
finite range, so the same arrays give the same file in every language.
Pass `lower` and `upper` yourself when the domain of validity is wider
than the data you happen to have.

*Status.* The words the validator knows are `converged`, `failed` and
`partial`. A producer may add its own, but only `converged` means the
row is fit for modelling; any other value is reported as W02, which is
a warning and not a refusal. A file whose status column says `ok` will
warn on every row.

A scalar is simpler: a per-row number with units, and nothing else.

    ds.add_key("mach", [0.4, 0.8, 0.4, 0.8, 0.4, 0.8],
               role="condition", units="1")
    ds.add_scalar("cl", [0.21, 0.25, 0.30, 0.36, 0.41, 0.48],
                  units="1")

The example: `examples/rows-and-roles/`. Six rows, two keys, one
scalar, no support.


3. Supports, arrays and labels
------------------------------

A support is the structure a field lives on (SPEC section 6). There
are three kinds:

    mesh   nodes with coordinates and cells with connectivity
    axis   nodes along one coordinate and no cells: frequency bands,
           observer angles, a ground time axis
    none   for a file of scalars

Cells are stored VTK-style in one structure, mixed types allowed:
`cell_types`, `cell_offsets`, `cell_connectivity` (SPEC section 20 has
the type codes). There are no blocks and no element sets. A region is
a *label*: an integer array over nodes or cells with a table of
category names, or, where the integers are already meaningful (CAD
face ids), with no table at all.

An array on a support declares what it *varies along* (SPEC section
5), which is one of three things:

    none         one instance shared by every row
    row          one instance per row
    group:<k>    one instance per category of the group key k

A fixed mesh has coordinates that vary along none; a parametric family
has coordinates that vary along `group:member`; a moving mesh in a
transient has coordinates that vary along the row. Coordinates are
always absolute positions, and a support has exactly one coordinates
array, which is why it is reached as `support.coordinates` and is not
one of `support.node_arrays`.

You do not have to work out `varies` yourself. Name the axes of the
array you are passing, in your own order, and the builder derives the
rest:

    support.add_node_array("pressure", pressure, units="Pa",
                           dims=("row", "node"))

`dims` is the same idea in all four languages. The names are `row`,
`group:<k>`, `draw`, `node` or `cell`, and `component`. Name every
axis your array has and no more; the component axis is the one you may
leave out, and it is added for you with length one.

Two array roles are never imported: `weight` and `normal` are computed
from the connectivity, because a weight that came from somewhere else
is a number nobody can check. Every implementation offers
`compute_weights(support, location)` for that, and `integrate`
computes one on the fly, saying so, when the file has none.

The example: `examples/a-support-and-a-field/`. A two-quad mesh of six
nodes, coordinates varying by member, one field varying by row, one
label over the cells.


4. Groups, trajectories and splits
----------------------------------

A group key partitions the rows: which member of a family, which
trajectory, which run. Groups may nest, so `trajectory` may declare
`parent = member`.

Exactly one group is declared the *unit of generalisation*
(SPEC section 7). It is the thing a model is supposed to generalise
over, and it is a property of the dataset, not of the key:

    ds.set_generalisation_group("member")

It matters because of splits. A split that puts rows of one unit on
both sides is not a generalisation test: the model has already seen
that geometry, and its test score is flattering. A file carrying such
a split is valid, and the validator warns W01 and names the unit that
leaked. The helper `grouped_split` moves whole units, takes a seed
with a documented default, and refuses to run on a file that declares
no unit of generalisation.

Time is a key, not a support coordinate, whenever it is something you
sampled. A time key names the group whose categories are the
trajectories:

    ds.add_key("t", t, role="time", units="s",
               trajectory_group="run")

Within one trajectory time must be strictly increasing, and it is an
error (E09) if it is not. Trajectories may have different lengths and
irregular steps; rows are just rows.

The example: `examples/groups-and-splits/`. Three members, six rows, a
split written the naive way and the leak it causes.


5. Alignment, and why it is the point
-------------------------------------

A file is *aligned* when it declares at most one support, so that
every row is on the same one and node 3 means the same node in every
row. That is what makes per-node comparison across rows, coordinate
ensembles and reduced bases valid, and it is stated in the file as
`aligned` rather than assumed (SPEC section 8). A file with several
supports is valid and not aligned, and says so.

Each support also carries a `support_id`, a hash over its node count
and its connectivity, and over its coordinates as well when it is an
axis (SPEC section 24). Within one file it adds nothing. Across files it
is the whole check: a model file and the training file it was fitted
on agree on one attribute, with no array read and no tolerance.

    print(support.support_id[:16])

Connectivity never varies within a support. Two rows with different
connectivity are on different supports by definition.


6. Callables and evaluation
---------------------------

Any slot may hold stored data or name a *callable* that produces it
(SPEC section 10). A fitted model is therefore an ordinary file with
zero rows: its key columns carry only their bounds, which is the
domain the model is valid over, and its slots carry their units and
their shape and say `source = callable:m1`. A mesh support's
coordinates are a slot too, so a model of the geometry itself is the
same kind of file: the support carries its cells and its node count,
and its coordinates are served, beside the fields on them.

A callable is four things and nothing more: call it with a keys table
and get one prediction per output back, a mean and, when the model
has one, a band; turn it into a dictionary; rebuild it from one; and
optionally describe itself in a line. Everything else it knows,
including its algorithm and its fitted state, lives inside that
dictionary and is nobody else's business. That is what keeps a
proprietary model and an open format compatible: the `type` string is
public, so a reader knows which tool can evaluate it, and the contents
are opaque to every reader that does not own the type.

Evaluating a file on a keys table gives a file with the same slots,
now holding data and no callables at all. Distillation is that
operation on a grid, and what comes out is a plain data file anyone
can read.

    out = mestra.evaluate(ds, {"mach": [0.5], "alpha": [4.0]})

`affine` is the one callable type the format defines, so that the
protocol can be tested in every language with no proprietary model
(SPEC section 27).

The example: `examples/callables-and-evaluation/`. A zero-row file
with two callable slots, evaluated on one keys table.


7. Uncertainty: a band, or draws
--------------------------------

A model's output is stored with the same roles as data, plus a
`statistic` saying what the numbers are: `value`, `mean`, `band`,
`std`, `quantile` or `draw` (SPEC section 9). Anything but `value` and
`draw` names the quantity it summarises with `of`.

A band is the one representation a callable returns: the half-width
of the interval around the mean, at the coverage its `level` states
(0.95 for a 1.96-sigma Gaussian band), made as its `method` says. It
is taken as given, because the error a model knows about is not
always something it sampled. `prediction(ds, "pressure")` returns the
mean with its band whether the slot holds data or a callable serves
it, which is what lets one viewer show both.

Draws are stored data: a whole field at once, joint across the nodes,
so the draws of a field carry an extra `draw` axis between the row
and the node, and the summaries beside them can be recomputed from
the file. How many draws there were, from what seed, in what batches,
is the producer's record and not part of the format.

The example: `examples/uncertainty-as-draws/`. Four draws per row, and
the mean and standard deviation beside them.


8. Metadata, notes, and the public/private line
-----------------------------------------------

A file carries three pieces of metadata and no more: `format`,
`writer` and `created` (SPEC section 11). An optional `notes` group
holds free-form attributes, a solver name or a dataset licence, and no
tool may require anything in it. Lineage, history, sign-off and
validation records are deliberately not part of the format.

Everything described in this guide is public, and an open reader can
read all of it with no dependency on any proprietary tool. Two things
are not: a callable's dictionary, which a reader that does not own its
`type` may copy but must not interpret, and an optional `private`
group, in which a producer keeps its own records in whatever form it
likes. A writer must not put public information in either of them, and
a reader must not require anything from either (SPEC section 12). A
distilled table is stored data and therefore public; the model that
produced it is not.


9. Validating a file
--------------------

Every rule has an identifier that never changes. An error means the
file is rejected; a warning means it is accepted and the reader must
say so. The rules are listed in SPEC section 14, and the same
identifier means the same thing in every language.

    $ mestra validate run.mes
    0 error(s), 0 warning(s)

A finding prints as `<id> <path>: <message>`, one per rule per object,
and the run ends with that summary line. The command exits non-zero
only on an error, so it fits in a build. In a program, `validate`
returns a report you can ask by rule id:

    report = mestra.validate("run.mes")
    report.ok, report.error_ids, report.warning_ids

Three habits save time. First, you will usually meet a rule before the
file exists: every builder refuses at build time, naming the rule and
the argument to change, anything the validator would refuse. Second,
`write` validates before it writes and refuses on any error, because a
file outlives the session that made it. Third, a rule identifier is
always about a file: asking for a slot that is not there is your
mistake, not the file's, and it is raised without one.

The example: `examples/validating/`. One refusal at build time, one
warning that does not stop the file.


10. Reading a file somebody else made
-------------------------------------

Opening a file reads its structure and none of its data, so it is
cheap on a large file. From there the file answers for itself:

    with mestra.read("theirs.mes") as d:
        print(d.n_rows, d.aligned, d.key_names())
        p = d.supports["s0"].node_arrays["pressure"]
        print(p.dims, p.units)
        print(p.values.at(row=1, node=3, component=0))

Find a value by name, never by position: the slot by its name, and the
place in it by the name of each axis. `at()` takes the dimension names
of SPEC section 4, and `instance` is accepted for the leading axis of
an array that varies along a group. This is what makes the file mean
the same thing in a row-major and a column-major language: MATLAB
hands you the same array with its axes in the opposite order, and both
readers answer the same number to the same question. Nothing in your
code should ever count axes.

`slot.read(rows)` takes a row range without reading the rest of the
file. From the command line, `mestra info theirs.mes` prints the same
survey: every key with its role, units and bounds, every support with
its counts and id, every slot with its shape under the name of each
axis.

A reader treats every file as untrusted input: a malformed or hostile
file gets a finding or a refusal that names the rule, never a crash, a
hang, or an allocation that takes the machine with it. Links are never
followed. Each language's README lists the rules its reader refuses to
open a file on. One thing no reader can promise is what the HDF5
library does with damaged bytes before the reader runs: a crafted
change to one byte of a valid file can fault the library itself, so a
file nobody vouches for is best read from a process that can be
killed, as the command line is (`compatibility.md`, "Damaged files").

Because the layout is netCDF-4, other tools can open the file too, and
`h5dump`, `ncdump` and xarray all work. Note that the reverse is not
true: netCDF-C writes its text attributes in a way this format does
not allow, so a conforming file cannot be produced with netCDF-C.

The example: `examples/reading-someone-elses-file/`. A committed file
written with h5py alone, surveyed and then asked for one value.


11. Where to go next
--------------------

    ../SPEC.md            the reference. Sections 1 to 12 are the
                          model, 13 to 17 the container and the
                          rules, 18 to 30 the byte-level detail an
                          implementer needs
    example.md            two valid files listed object by object and
                          value by value, with the values to check a
                          reader against
    mappings.md           five real datasets mapped onto the model,
                          which is the test the model had to pass
    design-notes.md       why it is shaped this way, and the
                          alternatives that were considered
    api-conventions.md    the names, the argument order and the
                          defaults every language follows
    ../vectors/           the conformance corpus every implementation
                          runs, and the hostile subset

Each language has a README beside its code with the ten-line example
in that language and its own axis-order statement. They also carry the
generic post-processing, which needs nothing but the format and so
runs the same on solver output and on a model's predictions: per-field
statistics, integration over a label, a time series at a node, a
grouped split. C++ leaves the last two out on purpose and says why.
