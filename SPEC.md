mestra: specification, version 0
================================

Date: 2026-09-19, revised 2026-09-22. Status: version 0 is what the
0.1 implementations read and write, and the root attribute `format`
names it as "mestra/0". From here a change within version 0 follows
section 28, and a change that would make a version-0 file mean
something else is version 1. The spec text is CC-BY-4.0.

This document defines a data model and its serialisation. It does not
define a mesh format for solvers, a model format for fitted surrogate
state, or a database. Those live elsewhere; this format carries the
data they consume and produce, with enough declared structure that a
reader can check it and generic tools can act on it.


1. Purpose
----------

One container for:

  - parametric data: many simulations over design parameters and
    operating conditions;
  - time-dependent data: trajectories, including inside a parametric
    family;
  - scalar quantities of interest, with or without any field;
  - fields on shared supports: meshes, one-dimensional axes, or none;
  - the outputs of surrogate models, in the same shape as the data
    they were trained on, including uncertainty;

The single most important thing the format states is alignment
(section 8): every row shares one support, so node k means the same
thing in every row. Tools built on the format rely on that; it makes
it explicit and checkable instead of assumed.


2. Concepts
-----------

Row          One observation. A row is one point in the space that was
             sampled: a design, at an operating point, at a time.

Key          A per-row column that locates the row in that space. Every
             key carries a role (section 3).

Scalar       A per-row quantity of interest, with units.

Support      The structure a field lives on. Kinds: `mesh` (nodes with
             coordinates, cells with connectivity), `axis` (nodes along
             one coordinate, no cells; frequency bands, observer
             angles, a ground track), `none` (used only for scalars).

Array        A field-like quantity on a support: a node array or a cell
             array. Every array declares what it varies along (section
             5), its units, its components, and its location.

Label        An integer array with a category table: regions, CAD face
             ids, topology groups, materials. Labels are how regions
             are represented; there are no separate set structures.

Group        A key whose role is `group`: a label per row that says
             which rows belong together (member of a family,
             trajectory, split). One group is declared the unit of
             generalisation (section 7).

Callable     An object that maps keys in to values out. Any array or
             scalar slot may hold stored data or a reference to a
             callable that produces it. Section 10.

Metadata     The format version, the writer, and the time. Section 11.
             Lineage and audit history are not part of the format.

Public and   Everything above is public. A file may carry a private
private      group that readers treat as opaque bytes. Section 12.


3. Roles
--------

Every key and every array carries exactly one role. The validator is
driven by this table.

Key roles (per row):

  design       0..n  continuous; a family's geometry or design
                     parameters
  condition    0..n  continuous; operating point (Mach, load, ...)
  time         0..1  monotonic within each trajectory group; units
                     required
  categorical  0..n  integer with a category table
  group        0..n  integer with a category table; exactly one is the
                     unit of generalisation (section 7); a group may
                     declare a parent group
  split        0..1  categorical with categories from {train,
                     validation, test, holdout}; informational
  id           0..1  unique per row; string or integer
  status       0..1  categorical; the category table is the file's
                     own (converged, failed, partial are recommended;
                     a producer may add its own, such as
                     extrapolated); rows whose status is not converged
                     must be excluded from modelling unless asked for

Array roles (per support):

  coordinates  exactly 1 per mesh or axis support; components equal
               the spatial dimension (axis: 1)
  field        0..n  a physical quantity; units required; components
               1 (scalar), d (vector), d*d (tensor); location node or
               cell
  label        0..n  integer, location node or cell; a category table
               is optional, and when absent the values are their own
               categories (CAD face ids, for example)
  weight       0..1 per location; integration weights (node or cell
               measure); computed from connectivity, never imported
  normal       0..1  unit normals; location node or cell; components d;
               computed from connectivity, never imported
  derived      0..n  a quantity computed from other arrays; must carry
               `derived_from` (names) and `recipe` (a string naming
               the operation)

Scalars carry the role `scalar` implicitly and require units.

Units are strings in the UDUNITS grammar that CF uses ("Pa", "m s-1",
"W m-2", "1" for dimensionless). A units string parses when it matches
the grammar of section 32 (products by a space, ".", "*" or "-"; "/"
and "per" after a factor; "^", "**" or a trailing integer for powers;
shifts by "@", "since", "from", "after" and "ref"; log references),
whatever names it uses. In version 0 a string that does not parse is a
warning, not an error; tools may convert between units they know and
must refuse to combine units they do not.


4. Dimensions and axis order
----------------------------

Every dataset carries named dimensions. The logical names are:

  row        the observation index
  group:<k>  the category index of group key k (for arrays that vary
             along a group)
  draw       an uncertainty draw index (section 9)
  node       nodes of a support
  cell       cells of a support
  component  vector or tensor components
  index      flat connectivity storage
  cell_plus_one  the cell offsets, one longer than `cell`
  key        unused; keys are separate datasets, not one matrix

The logical order of an array is (row | group | none, [draw], node |
cell, component). On disk, arrays are stored in that order in C
(row-major) layout. A reader in a column-major language returns
whatever order is natural for it, but must expose the dimension names
so that permutation is by name, never by position. Two readers in two
languages must agree on the value at (row r, node n, component c).

Section 21 gives the name each of these has on disk and how the
dimension scales are written.


5. Varies-along
---------------

Every array declares `varies`:

  none         one instance shared by every row; leading dimension
               absent
  row          one instance per row; leading dimension `row`
  group:<k>    one instance per category of group key k; leading
               dimension `group:<k>`; the row's category selects the
               instance

Coordinates use the same mechanism, with one exception: the
coordinates of an `axis` support must have `varies = none`, because
they are part of that support's identity (section 20). A fixed mesh
has coordinates with
`varies = none`; a parametric family has `varies = group:member`; a
moving mesh in a transient has `varies = row`. Coordinates are always
absolute positions; a support has exactly one coordinates array and
displacements are never stored as coordinates. A persisted
displacement is a `derived` array with recipe "minus reference",
carrying a `reference` attribute that names the row or the group
category of the coordinates array it is measured against, written as
`row=<i>` or `group:<k>=<category name>`. It is never the previous
row, so nothing accumulates or drifts. Connectivity
does not vary (section 8).


6. Supports
-----------

A file carries a list of supports. Each has:

  kind         mesh | axis | none
  n_nodes      integer (0 for none)
  n_cells      integer (0 for axis and none)
  support_id   content hash for cross-file identity (section 8)

Mesh cells are stored VTK-style, in one structure, mixed types
allowed: `cell_types (cell)`, `cell_offsets (cell + 1)`,
`cell_connectivity (index)`. There are no blocks; regions are labels.
A support with no cells carries none of those three datasets; the
allowed cell type codes and the rules on the offsets are in section
20.

Each row references exactly one support. When the file declares at
most one support it omits the per-row reference and sets
`aligned = true` (sections 8 and 22).

Weights and normals, when present, are arrays with those roles,
recomputed from coordinates and connectivity by the writer or by a
tool; an importer must not copy them from an upstream exporter.


7. Groups, trajectories, and the unit of generalisation
------------------------------------------------------

A group key partitions rows. Common groups: `member` (which geometry
of a family), `trajectory` (which time series), `case` (which run).
Groups may nest: `trajectory` may declare `parent = member`.

Exactly one group is declared the unit of generalisation. A split
that places rows of one unit on both sides is not a generalisation
test; the validator warns, and tools built on the format should
refuse by default.

A trajectory is the set of rows sharing one value of the group that
the time key declares as its `trajectory_group`. Within a trajectory,
time must be strictly increasing. Trajectories may have different
lengths and irregular time steps.


8. The alignment claim and support identity
---------------------------------------------

A file is aligned when it declares at most one support, so that every
row is on the same one. That is a structural fact of the file, not a
separate claim: an aligned file carries `aligned = true` and no
per-row reference, an unaligned file carries `aligned = false` and a
per-row reference for every row, and the validator checks both against
the supports the file declares (section 22). Index-aligned operations
(per-node comparison across rows, coordinate ensembles, reduced bases
over nodes) are valid only on aligned files, and tools must check the
flag, not assume it.

Files with several supports are valid and unaligned. They can hold
benchmark datasets with varying meshes; the validator reports the
number of supports so that a user knows which operations apply.

Connectivity never varies within a support. Two rows with different
connectivity are on different supports by definition.

Each support also carries a `support_id`: a content hash over
(n_nodes, cell_types, cell_offsets, cell_connectivity) and, for axis
supports, the axis coordinates, computed as SHA-256 over the arrays
serialised as little-endian bytes in that order. Coordinates of mesh
supports are excluded because they may vary. The id adds nothing
inside one file; its purpose is across files: a callable declaring the
support it produces on, several callables being composed, a prediction
compared against a truth file, rows appended from a second file. In
every such case "same support" is an id comparison, not an array
comparison, and a mismatch is an error before any data is read.

9. Uncertainty
--------------

Representing uncertainty is optional. A file that carries none is an
ordinary file, and nothing in the format obliges a producer to
quantify anything.

Where it is carried, model outputs are stored as arrays and scalars
with the same roles as data, plus:

  statistic    value | mean | band | std | quantile | draw
  of           the name of the base quantity this is a statistic of
  level        a number in (0, 1) when statistic = band: the coverage
               the band claims
  method       a string when statistic = band: how the band was made
  quantile     a number in (0, 1) when statistic = quantile

The slot's `statistic` and `of` are what say which of these a slot
is; a reader learns it from the attributes and never has to ask the
producer. These six are the statistics the format names. Any other
summary is a derived array carrying `derived_from` and a `recipe`
(section 3), so that every number in a file says where it came from.

The band. A band is the half-width of a symmetric interval around its
base quantity: the producer's claim is that the true value lies
between `mean - band` and `mean + band` for the fraction `level` of
cases. It has the shape and the units of its base quantity and it is
never negative (W16). It is the one representation of uncertainty a
callable returns (section 10), and it is taken as given: the producer
puts into it everything it knows about its own error, sampled or not,
so a reader draws the band it finds and does not recompute one from
anything else in the file. `method` says in words how the band was
made, for instance from the spread of samples pushed through a
nonlinear map with a truncation term added; the numbers behind that
sentence are the producer's own records. `level` is a fraction and
not a multiple of a standard deviation: under a Gaussian assumption a
one-sigma band is 0.6827 and a 1.96-sigma band is 0.95, and a producer
that assumes something else states the coverage its band has.

Draws, standard deviations and quantiles are stored data about stored
data. Draws carry the `draw` dimension. A draw is one whole field or
one whole scalar: draw k of pressure is a complete field over every
node, so whatever dependence the producer has across nodes is
preserved in the file instead of being collapsed to a per-node number
that cannot be put back. How many draws, from which seed, in what
batches, are the producer's business and its records, not the
format's; the file may repeat them as optional metadata, and nothing
in the format depends on them. Where a file carries both draws and a
summary of them, the summary is derived from the draws by an open
routine, so it can be recomputed from the file. Nothing ties draws to
a callable: a producer that samples stores its samples as data, and
what it has learned from them, if it states it, is a band.


10. Callables
-------------

A callable is any object that conforms to four things and nothing
more:

  call        keys in, predictions out: given a keys table (one
              column per key the file declares, any number of rows),
              return one prediction per output it serves. A
              prediction is a record of four things:
                mean         required: the point prediction, shaped
                             as the slot would be stored, that is
                             (row, node | cell, component) for an
                             array and (row) for a scalar
                uncertainty  optional: a band (section 9), the
                             half-width of the interval around
                             `mean`, of the same shape and units and
                             never negative
                level        required with `uncertainty`: the
                             coverage the band claims, in (0, 1)
                method       required with `uncertainty`: how the
                             band was made, in words
  to_dict     a nested dictionary of arrays, numbers, and strings
              that fully represents it
  from_dict   the inverse, dispatched on a `type` string
  repr        optional, a one-line description for printing

Everything else a callable knows (its algorithm, fitted state, its
own records, evaluation parameters such as draw count, seed, and
batch size) is inside its dictionary and is its own business. The
format does not constrain it, and future models are integrated by
conforming to the four things above.

Any array or scalar slot may hold stored data or a reference to a
callable. The slot's attributes (role, units, components, statistic)
describe what the slot is in either case; its location and its support
are given by where it sits in the file and are not attributes (section
19); the content attribute `source` is `data` or `callable:<id>`, and
`output` names which of the callable's outputs fills the slot. Which
part of that output fills it is the slot's `statistic`: a slot whose
statistic is `value` or `mean`, or that has none, takes the output's
`mean`; a slot whose statistic is `band` takes its `uncertainty`, and
evaluation writes the record's `level` and `method` onto the slot. So
a callable that serves `pressure` and `pressure_band`, both with
`output = pressure`, fills both from one record, and `of` links the
band to its base exactly as it does for stored data. A callable serves
nothing else: `std`, `quantile` and `draw` are stored data about
stored data (section 9), and a callable slot with one of them is E12.
A band slot whose output carries no `uncertainty` is an error at
evaluation, since the file would otherwise hold a band with no level.

The record is the whole of a callable's contract. Every output is a
prediction of the same four things whatever the model, so a tool that
shows a stored mean with its band shows a callable's the same way and
never has to know what is inside the callable; how the band was
computed is the callable's own business and its `method` line. A
callable is stored once under `/callables/<id>` and may be referenced
by any number of slots. A slot holding data is a dataset; a slot
served by a callable is a group with the same attributes and no data
(section 19).

A support's coordinates are an array slot like any other and may be
served the same way: the support carries its cells and its node
count, and its coordinates are a group with the slot's attributes
and no data until the file is evaluated. That is how a model of the
geometry itself is stored, beside the fields on it. The coordinates
of an `axis` support are part of its identity (section 24) and are
always stored.

A file with callable slots may have zero rows, and may also have rows
(section 22). With zero rows its key columns
carry only their attributes and bounds, which is the domain the
callable is valid over. Evaluating the file on a keys table yields a
file with the same slots, now holding data. Distillation is that
operation on a grid. Composition is a file whose slots reference
several callables that share a support; the checks are support_id
equality and the intersection of the key bounds.

Storage of a callable is by the codec in section 17: the dictionary
maps to an HDF5 group so that every language round-trips it
identically. The `type` string is public so a reader knows which tool
can evaluate the callable; the dictionary's contents are opaque to
any reader that does not own the type.

11. Metadata
------------

A file carries the minimum a reader needs to open it, and no more:

  format          "mestra/0"; authoritative over the extension
  writer          the tool and version that wrote the file
  created         ISO 8601 timestamp, UTC

An optional `notes` group may hold free-form attributes (a solver
name, a dataset licence, a comment). Nothing in the format depends on
them and no tool may require them. They are public, so section 18
holds for them as for any attribute: a string among them is a
fixed-length UTF-8 string padded with NUL (E19) holding legal text
(E26), and never a variable-length one. A number among them is the
producer's choice, because the format names no number there.

Lineage, history, upstream links, validation records, sign-off, and
evaluation parameters are deliberately not part of the format. They
are the audit trail, they belong to the callable and the tool that
produced it, and they live in the private part or in that tool's own
records.

12. Public and private
----------------------

Everything in sections 2 through 11 is public: an open reader must be
able to read all of it with no dependency on any proprietary tool.

A callable's dictionary is opaque to readers that do not own its
`type`: they may copy it and must not interpret it. A file may also
carry a `private` group for a producer's own records (lineage,
validation, history) in whatever representation it chooses; readers
treat it the same way and must not require anything in it. Writers
must not place any public information only in either place. A
distilled lookup table is stored data and therefore public; the
callable that produced it, its history, and its validation records
are not.


13. Container
-------------

HDF5. The layout is constrained so that a valid file is also a valid
netCDF-4 file: named dimensions as dimension scales, attributes as
strings or numbers, groups for structure, and, in the data and in the
attributes this format defines, no compound types, no object
references, no HDF5 enums and no variable-length types. The HDF5
dimension scale machinery is exempt from that sentence: its
DIMENSION_LIST is a variable-length list of object references and its
REFERENCE_LIST is a compound, and netCDF-4 itself is built on them, so
a file cannot carry named dimensions without them. A reader ignores
those attributes (section 18).

Strings in tables and in attributes
are fixed-length UTF-8 (section 18). Arrays are chunked along `row`,
which is an unlimited dimension in every file, and may be compressed
with gzip and shuffle (section 23). The extension is `.mes`; the root
attribute `format` is authoritative, not the extension.

Layout sketch. Sections 18 to 25 are the normative detail; `?` marks
an optional attribute. Every axis of every dataset also carries a
dimension scale, named as section 21 requires.

    /                          attrs: format, writer, created,
                               aligned, generalisation_group?
    /row                       dimension scale, unlimited
    /component_<n>             dimension scale, one per component
                               count used
    /draw_<n>                  dimension scale, one per draw count
                               used
    /group_<k>                 dimension scale, one per group key
    /category_<t>              dimension scale, one per category table
    /keys/<name>               (row; may be empty)  attrs: role, units,
                               lower?, upper?, category?,
                               trajectory_group?, parent?
    /scalars/<name>            (row)  attrs: units, source, output?,
                               statistic?, of?, level?, method?,
                               quantile?
                               a group with the same attributes and no
                               data when source is a callable
    /categories/<name>         (category_<name>)  fixed-length strings
    /row_support               (row)  int32; present only when the
                               file declares more than one support
    /supports/<s>              attrs: kind, n_nodes, n_cells,
                               support_id
      /node                    dimension scale
      /cell                    dimension scale, absent when n_cells=0
      /cell_plus_one           dimension scale, absent when n_cells=0
      /index                   dimension scale, absent when n_cells=0
      /coordinates             (row|group|-, node, component)
                               attrs: role, varies, units, components,
                               source
      /cell_types              (cell)       uint8
      /cell_offsets            (cell_plus_one)  int64
      /cell_connectivity       (index)      int64
      /node_arrays/<name>      (row|group|-, [draw], node, component)
                               attrs: role, units, varies, components,
                               source,
                               output?, statistic?, of?, level?,
                               method?, quantile?,
                               category?, recomputed?,
                               derived_from?, recipe?, reference?
                               a group with the same attributes and no
                               data when source is a callable
      /cell_arrays/<name>      same, over cell
    /callables/<id>            attrs: type, repr?; the dictionary as a
                               group by the codec of sections 17
                               and 25
    /notes                     optional free-form attributes
    /private                   opaque

A container group, that is `/keys`, `/scalars`, `/categories`,
`/supports`, `/callables` and a support's `node_arrays` and
`cell_arrays`, may be absent when it would be empty, or present and
empty. A reader accepts both and must not require either. A writer
reproducing a file it read keeps whichever of the two it found, so
that a round trip does not silently add or drop a group.


14. Validator
-------------

Every rule has an identifier. The identifiers are stable: within a
major version a rule is never renumbered and a retired rule's
identifier is never reused. The conformance corpus (section 30) names
rules by identifier and nothing else.

Errors (the file is rejected):

  E01  `format` missing or not "mestra/<n>" with n this reader
       accepts
  E02  a key or array without a role, or with a role not in section 3;
       a support whose `kind` is not one of the three words of section
       6; a slot whose `statistic` is not one of the six words of
       section 9. What a statistic needs follows from its word, so a
       slot with a word section 9 does not name is not also judged by
       E12
  E03  a role's cardinality violated (two time keys; no coordinates
       on a mesh or axis support; two coordinates arrays on one
       support; two units of generalisation)
  E04  an array whose leading dimension disagrees with `varies`, or
       a `varies` naming a group key the file does not declare
  E05  an array whose node or cell count disagrees with its support
  E06  a row referencing a support that does not exist
  E07  retired. A file with rows on more than one support and
       `aligned = true` also breaks E28 and E37, both decidable from
       the file, so nothing was left for E07 to catch.
  E08  a `support_id` that does not match the stored arrays
  E09  time not strictly increasing within a trajectory
  E10  a categorical, group, label, or status value outside its
       category table
  E11  units absent from a field or a scalar. Units on a key, on
       coordinates and on a derived array are E39; `weight` and
       `normal` do not require units at all
  E12  a statistic without what it needs: a quantile statistic
       without a quantile; a statistic other than `value` or `draw`
       without `of`; a `band` statistic on a stored slot without
       `level` in (0, 1) or without `method`; or a slot served by a
       callable whose statistic is `std`, `quantile` or `draw`,
       which a callable does not produce (section 10). A slot whose
       statistic is `draw` holds the base quantity itself and does
       not name it again. A band slot served by a callable carries
       `level` and `method` once it is evaluated and not before
  E13  a derived array without `derived_from` and `recipe`
  E14  a slot whose `source` names a callable id that does not exist
  E15  a callable group without `type`
  E16  a length disagreeing with the row count it must have, in a
       slot whose leading dimension is `row`, a key column, or
       `/row_support`. The count is the file's row count, except for
       a row-varying array on a support in an unaligned file, where
       it is the number of rows referencing that support (section
       22). A slot whose `varies` is `none` or `group:<k>` has no row
       dimension and is not subject to this rule; its leading
       dimension is E04 or E34. Also a dataset under `/keys` or
       `/scalars` that does not have exactly one dimension
  E17  `format`, `writer`, or `created` missing
  E18  a required public attribute or object absent, by any of E02,
       E11, E13, E15, E17, E31 or E39, in a file that also carries a
       `/private` group. It is reported beside that rule, at the path
       of each object that rule found something missing from, and
       never by interpreting `/private`, which section 29 forbids a
       reader to interpret. A writer that moved the public thing into the
       private part is what the rule is about; a reader can only see
       the two facts that make it likely, and says so

The rules below come from the byte-level layout of sections 18 to 25.
They are checked on the public objects only. `/private` is not
checked, and neither is any group this version of the format does not
know, which is reported as W11 and otherwise left alone. A producer
that wants its file to open in a generic netCDF-4 tool keeps its
private part within the same constraints anyway, because the tool
will walk it; that is the producer's choice and not this format's
rule.

  E19  an attribute whose HDF5 type is not the one section 18
       requires for it: a variable-length string, a string that is
       not UTF-8 with NUL padding, an integer that is not int64, a
       float that is not float64, a boolean that is not int8 or whose
       value is not 0 or 1
  E20  a dataset whose dtype is not allowed for its role (section 19)
  E21  a cell type code not in the table of section 20
  E22  a cell whose node count disagrees with its cell type
  E23  `cell_offsets` that does not start at 0, is not
       non-decreasing, or whose last value is not the length of
       `cell_connectivity`
  E24  a connectivity value outside [0, n_nodes)
  E25  an axis of a dataset with no dimension scale attached, more
       than one scale attached, or a scale whose name is not one
       section 21 allows at that axis position. Where the required
       name follows from an attribute another rule already checks,
       that rule reports it and E25 does not: the leading axis
       against `varies` is E04, the node or cell count is E05, and
       the component count is E31. A dimension scale dataset is not
       itself subject to this rule and carries no scale on its own
       axis. Once an axis of a dataset draws E25, a reader does not
       know that dataset's dimensions, and whether it goes on to
       report there the rules decided from them (E04, E05, E16, E27,
       E31, E34, E43 and W12) is implementation-defined. The corpus
       does not compare them: no corpus file has a dataset that draws
       E25 and would draw one of those. A rule decided without the
       dimensions, such as the dtype of E20, is reported as usual
  E26  a fixed-length string that is not valid UTF-8, or that holds a
       NUL byte anywhere but in its trailing padding
  E27  `row` not an unlimited dimension, or a row-dimensioned dataset
       that is not chunked
  E28  `/row_support` present when `aligned = true`, or absent when
       `aligned = false`
  E29  a dataset with a filter other than gzip at level 1 to 9 and
       shuffle
  E30  a slot with `source = data` stored as a group, or a slot with
       `source = callable:<id>` stored as a dataset
  E31  an array slot without `components`, or whose `components`
       disagrees with the length of its component dimension
  E32  a callable dictionary holding something section 25 says is not
       representable: a ragged or mixed nested list, a
       zero-dimensional dataset, a disallowed dtype, a string with an
       embedded NUL, or a top-level key `type` or `repr`
  E33  a name that is not a legal netCDF-4 name, or a producer-chosen
       name that begins with `mestra_`
  E34  a group-varying array whose leading dimension length differs
       from the number of categories of its group key
  E35  an `axis` support whose coordinates do not have `varies = none`
  E36  a `source` that is neither `data` nor `callable:<id>`, which
       includes `callable:` with an empty id. Such a slot says neither
       that it holds data nor which callable serves it, so E14, E30
       and the `output` of E39 are not decided for it
  E37  `aligned` disagreeing with the number of supports declared:
       true with more than one, or false with at most one
  E38  a mesh support missing `cell_types`, `cell_offsets` or
       `cell_connectivity`, or an `axis` or `none` support carrying
       any cell dataset or a `cell` dimension. The digest of such a
       support is unaffected (section 24)
  E39  a required attribute of section 19 absent from an object that
       requires it, where no other rule covers it. E02 covers a
       missing role, E11 units on a field or a scalar, E13
       `derived_from` and `recipe`, E15 a callable's `type`, E17
       `format`, `writer` and `created`, E31 `components`, and W06
       `recomputed`. This rule is what covers units on a key of role
       `design`, `condition` or `time`, units on coordinates and
       units on a derived array. It also covers a `category`, on a
       key or on a label, that names no table under `/categories`:
       the attribute refers to nothing, so the values cannot be held
       against a table at all, which is not E10, and the finding does
       not depend on the file having rows
  E40  a link in the public tree that is not a hard link: a soft
       link, whether it resolves, dangles or loops, or an external
       link. A reader never follows one (section 29)
  E41  an object the reader cannot read, reported with its path while
       the pass continues: a malformed header or attribute, a group
       or dictionary nesting deeper than the reader's cap, or, on an
       eager read only, a dataset above the stated maximum element
       count (section 29). A lazy read and a row-range read are not
       subject to the last of these, so a file can be valid for one
       reader and E41 for another; the corpus states which
  E42  a dimension scale whose creation properties are not the ones
       section 21 requires: attribute creation order not tracked and
       indexed. A file written before this rule refuses it, which is
       the point of the rule: such a file cannot carry more than 4085
       attachments to one scale, and the 4086th destroys the scale's
       REFERENCE_LIST on its way to failing
  E43  an unlimited dimension other than `row`, file-level or
       support-local. The one exception is a zero-length axis of a
       dictionary dataset, which section 25 requires to be unlimited

Warnings (the file is accepted; the reader must report):

  W01  a split that places rows of one generalisation unit on both
       sides
  W02  rows with status other than converged
  W03  non-finite values in a field, a derived array or a scalar,
       with the row and name
  W04  a key value outside its declared bounds
  W05  more than one support (index-aligned operations unavailable)
  W06  weights or normals present but not marked as recomputed
  W07  a group key whose category table has an entry no row uses.
       A group value with no entry in the table is E10
  W08  declared bounds wider than the observed range of a key by
       more than a factor of four in width (possibly stale bounds).
       A key value outside its bounds is W04 and not W08. The rule
       does not apply when the observed width is zero, which covers a
       file with no rows, a key with one distinct value, and a key
       with no finite value at all
  W09  retired. A categorical key stored as floating point is an
       error by the dtype table of section 19 (E20)
  W10  a units string that does not match the grammar of section 32
  W11  an attribute or a group this reader does not know, ignored
       under section 28
  W12  a chunk shape that is not the default of section 23, on a
       chunked dataset that carries a row dimension. A contiguous
       dataset, a dimension scale and a dataset inside a callable's
       dictionary never draw it
  W13  a fixed-length string dataset whose size is larger than its
       longest element needs
  W14  `created` that is not an ISO 8601 UTC timestamp
  W15  a support that no row references. It is decidable only in a
       file that carries `/row_support`; in an aligned file every row
       is on the one support and the rule does not apply
  W16  a negative value in a band, with the row and name. A band is a
       half-width (section 9)


15. Conformance
---------------

The spec plus a corpus of golden files define conformance. Each
golden file ships with the expected outcome of every validator rule
and the expected value at named coordinates (row r, node n, component
c), in every language's natural axis order. Every reader and writer,
in every language, runs the corpus in its own test suite. No
implementation is the reference.


16. Resolved questions
----------------------

An appendix rather than a rule. Each entry is a question the rest of
this document once left open, answered two ways, or described in terms
no file could satisfy; it gives what was settled and the one reason
that matters, and names the section that carries the rule. Nothing
here is normative on its own, and nothing in it is still open.

Layout:

  - The file extension is `.mes`.
  - Names, category entries and string ids stay ASCII in a file meant to
    be read everywhere, because one binding in common use cannot carry a
    fixed-length string that is not (section 25).
  - Category tables hold fixed-length UTF-8 strings (section 19).
  - float32 is not allowed anywhere; a producer that computes in single
    precision widens on write (section 19).
  - The component dimension is always present, with length 1 for a
    single-component quantity (sections 4 and 19).
  - `row` is an unlimited dimension in every file, which is what makes a
    zero-row file a legal netCDF-4 file (section 21).
  - Only `row`, file-level or support-local, may be unlimited; every other
    dimension, `draw` included, is fixed (E43). The one exception is the
    zero-length dictionary axis section 25 requires. Every dimension but
    `row` carries its length in its name, so one that grows makes its own
    name false, and implementations did not agree on what a file with an
    appended draw held at the new draw. Draws are produced at evaluation
    and never appended (sections 19 and 21).
  - A dimension scale that is not unlimited is chunked with a chunk equal
    to its length, which is what H5DSset_scale leaves behind (sections 21
    and 23).
  - The row count in the chunk default is the length of the row dimension
    the leading axis is attached to, not the dataset's own extent (section
    23).
  - A reader resolves the link name of an attached dimension scale from a
    map built during its own bounded walk, keyed by object address or
    token, and never by asking the library for the scale object's path:
    that search walks the group hierarchy and overflows the stack on a
    deep file (section 21).
  - A container group may be absent when empty, or present and empty; a
    reader accepts both and a writer keeps what it found (section 13).

Roles and rules:

  - Units are strings in the UDUNITS grammar; a string the validator
    cannot parse is a warning in version 0 and not an error (section 3).
  - Whether a units string parses is decided by a grammar and never by
    a list of names, because no two unit databases agree on a list: the
    four implementations once took 77 strings four ways (sections 3
    and 32).
  - `split` is a key role, and `status` is a key role with an open
    category table. A label's category table is optional (section 3).
  - Bounds live on the key columns. There is no separate domain group, so
    extrapolation warnings are generic rather than a special case
    (sections 3 and 14).
  - Provenance, lineage, history, validation, sign-off and evaluation
    parameters are not part of the format. A file carries `format`,
    `writer` and `created`, plus optional notes; everything else of that
    kind belongs to whatever produced the file (sections 11 and 12).
  - `reference` is an attribute of the derived array and not of the
    coordinates, written `row=<i>` or `group:<k>=<category name>`
    (sections 5 and 19).
  - A mesh support's coordinates may be served by a callable, because
    section 10 makes every array slot servable and nothing in the
    support's identity depends on them; an axis support's coordinates
    are its identity and are stored. The builders spell it as the
    coordinates builder with the values dropped and the callable and its
    output added, like every other callable slot (section 10, conventions
    section 1, `callable_coordinates` in the corpus).
  - E04 also covers a `varies` naming a group key the file does not
    declare (section 14).
  - W08 has a number where it once had a tolerance stated nowhere:
    declared bounds wider than the observed range by more than a factor of
    four in width. A value outside the bounds is W04 and not W08, and W08
    does not apply when the observed width is zero (section 14).
  - W07 is narrowed to a category table entry no row uses, because a group
    value with no entry in the table was already E10 and the two could not
    be told apart (section 14).
  - W09 is retired: a floating-point categorical key is E20 by the dtype
    table, so W09 could never be the only outcome (sections 14 and 19).
  - E07 is retired for the same reason, and E37 is widened to `aligned`
    disagreeing with the number of supports in either direction, which
    also closes a case section 22 requires and no rule caught (section
    14).
  - E18 is a rule for a writer, because section 29 forbids the reader it
    addressed to interpret `/private`, and it is reported beside the rule
    that found the missing public thing in a file that also carries
    `/private`; it was not decidable before. E39 was added for a required
    attribute of section 19 that is simply absent, which nothing covered
    (section 14).
  - A name begins with a letter, a digit or an underscore, as netCDF-C
    requires. Without that, "." was a legal name that HDF5 reads as the
    group itself, and a key named that way could be built in every
    language and written in none (section 18).
  - `callable:` with nothing after it is E36 alone. It names no callable,
    so it is not the E14 of a callable that is missing, and it does not
    say which kind of slot was meant, so E30 and E39 have nothing to
    decide; two implementations had reported those three (section 14).
  - A string attribute under `/notes` is held to the string encoding of
    section 18 (E19, E26). One implementation did so, one only for a
    variable-length string and NUL bytes, and two not at all, so a note
    stored in ASCII opened in two languages and not in the others
    (section 11).
  - E18 is reported at the object the public thing is missing from, once
    for each such object, and not at `/private`: the object is what a
    writer has to fix, and `/private` is the one place a reader may not
    look. Three implementations had said `/private` and one the first
    object it found (section 14).
  - E02 covers a support's `kind` and a slot's `statistic` as it covers a
    role: each is one word from a closed list, and a word outside it says
    nothing a reader can act on. Only one implementation had reported an
    unknown kind (as E39) and only one an unknown statistic (section 14).
  - A support that declares no kind, or a kind section 6 does not have,
    has no digest. The kind decides which arrays are hashed, and
    implementations took such a support for a mesh or hashed its node
    count alone, so one file had two ids (section 24).
  - A `category` that names no table in the file is E39 and not E10. E10
    is a value outside a table that exists; with no table there is
    nothing to be outside, and implementations had said E10, E39, or
    nothing at all when the file had no rows (section 14).
  - E11 covers units on a field and a scalar; E39 covers units on a key,
    on coordinates and on a derived array; `weight` and `normal` need none
    (section 14).
  - E16 is about a row length, on a row-dimensioned slot, a key column or
    `/row_support`, and it also catches a key or scalar dataset that is
    not one-dimensional (section 14).
  - E25 does not fire where the required dimension name follows from an
    attribute another rule already checks, and a dimension scale is not
    subject to it (section 14).
  - After E25 on a dataset, the rules decided from that dataset's
    dimensions are implementation-defined there, and the corpus does not
    compare them. A reader that cannot name an axis can only guess its
    length and its role, and the four guessed differently, one adding
    E04, one E27, one leaving out E31; requiring any one guess would
    make a reader invent a dimension the file does not state (section
    14).
  - W03 covers a derived array as well as a field and a scalar, and W12
    applies only to a chunked dataset carrying a row dimension: a
    contiguous dataset is E27 or nothing, and scales and dictionary
    datasets are outside it. A zero-length axis is chunked with 1
    (sections 14 and 23).
  - W15 is decidable only in a file that carries `/row_support` (section
    14).
  - The byte-level rules of sections 18 to 25 are checked on the public
    objects only (section 14).
  - A reader treats a file as untrusted input: no crash, no hang, no
    unbounded allocation in the reader's own code (the HDF5 library
    beneath it parses the file first and is outside this promise; the
    compatibility notes say what that means); it never follows a link
    that is not a hard link
    (E40); an object it cannot read is E41 and the pass continues;
    recursion is capped and an eager read has a stated maximum element
    count, 2^31 by default (sections 14 and 29).

Callables:

  - A callable is exactly `call`, `to_dict`, `from_dict` and an optional
    `repr`. It fills slots, is stored once by the codec under a public
    `type`, and everything else it knows is inside its own dictionary, so
    that a model the format has never heard of integrates by conforming to
    four things (sections 10 and 17).
  - A slot served by a callable is a group; a slot holding data is a
    dataset (section 19).
  - A file with callable slots may have zero rows and may also carry rows,
    for example the training design. What decides whether a slot holds
    data is its `source` and never the row count (section 22).
  - A zero-dimensional array in a callable's dictionary is stored as an
    attribute, because no rule that preserved it as an array could
    round-trip in MATLAB and C++ as well as in Python (section 17).
  - The affine evaluation accumulates the dot product in the declared key
    order and adds `b` last, with no fused multiply-add. The formula alone
    left the last bit of the worked example open (section 27).
  - The keys table in C++ holds the names beside the columns and is looked
    up by name, because C++ has no run-time member names; Julia uses a
    dictionary from name to column (section 26).

Alignment and supports:

  - Coordinates are positions, one array per support. A displacement is
    carried only as a derived array relative to a declared reference
    (sections 3 and 5).
  - A support hash is an identity for cross-file checks and not an in-file
    claim; alignment itself is structural (sections 8 and 24).
  - The support_id of an axis support whose coordinates wrongly vary is
    computed over the stored bytes as they are, so that such a file breaks
    E35 and nothing else (section 24).
  - E38 covers any cell dataset, and a `cell` dimension on an `axis` or
    `none` support; the declared kind decides which steps of section 24
    contribute bytes, so the digest still matches (sections 14 and 24).
  - A support of kind `none` carries no `node` dimension scale, because a
    zero-length fixed dimension is not legal (sections 20 and 21).
  - In an unaligned file a row-varying array on a support has one entry
    per row referencing that support, in the file's row order, over a
    support-local `row` dimension (sections 14, 21 and 22).

Uncertainty:

  - Representing uncertainty at all is optional; a file that carries none
    is an ordinary file (section 9).
  - A slot's `statistic` and `of` say what the slot means (sections 9
    and 14).
  - A callable returns, per output, a mean and at most one band with its
    level and method, and serves no other statistic; std, quantile and
    draw are stored data about stored data. One record for every model
    is what lets a tool show a stored prediction and a callable's the
    same way (sections 9 and 10).
  - A band is a half-width at a stated coverage and is taken as given,
    because the error a model knows about is not always something it
    sampled: a spread of samples through a truncated map cannot see the
    truncation, and a reader that rebuilt a band from samples would
    understate it (section 9).
  - `band` was added to the statistics before any version-0 file was in
    use, which is why section 28's rule on new statistics is not broken
    by it (sections 9 and 28).
  - A slot whose statistic is `draw` holds the base quantity itself and
    carries no `of` (section 14).
  - The `draw` dimension is present only when the slot holds draws
    (sections 9 and 19).

Scale:

  - A dimension scale is created with attribute creation order tracked and
    indexed, and with object time tracking off (E42). Without it a scale
    takes at most 4085 attachments: the 4086th fails after destroying the
    REFERENCE_LIST it was extending and leaves a file that every reader
    and every validator accepts. This is a ceiling on the container and
    not on any reader, and it is the only entry in this appendix that
    changes the bytes of every golden file (sections 21 and 23).
  - E42 is about attribute creation order alone. Object time tracking off
    stays a writer requirement of section 21 and is not validated, because
    HDF5 stores the flag only in a version 2 object header: read back from
    anything else it reports tracking as on whatever the writer asked, so
    a check of it would fault only the files E42 already rejects (sections
    14 and 21).
  - H5Pset_attr_phase_change asks for the same thing and is silently
    ignored under the default library version bounds, so a writer must not
    rely on it. The bounds stay at the default for every object, so the
    superblock and every non-scale object are unchanged. REFERENCE_LIST is
    informational: an axis is resolved through DIMENSION_LIST and the
    reader's own address map, and a scale whose REFERENCE_LIST is missing
    or short is not an error (section 21).
  - A round trip through netCDF-C does not leave a file unchanged, and
    section 21 no longer claims it does: the layout is the one netCDF-C
    reads as an unlimited dimension, which is a claim about reading. What
    netCDF-C writes differently is listed there, so that a reader knows a
    netCDF-C-written file is not a conforming file (section 21).

Conformance:

  - expected.json has a sixth field, `evaluation`, without which the
    affine callable could not be conformance tested at all (section 30).
  - The probe schema names the leading dimension of a group-varying array
    `instance` and names the axes of a support's own cell datasets;
    neither had a name, and the corpus must probe both. A probe's `value`
    is a decimal string: the "%.17e" form for a float64 slot and the plain
    decimal form for an integer slot (section 30).
  - A probe on a dataset inside a callable's dictionary has no logical
    dimension names; its index fields are applied in a fixed order to the
    axes in file order, and `cell` and `cell_plus_one` join the probe
    field list (section 30).
  - Implementations link different libhdf5 versions and do not agree byte
    for byte, so structural equality is the comparison and byte identity
    is not tested (section 30).
  - The corpus carries a hostile subset under vectors/hostile with a
    looser contract: the required ids must appear, more are allowed, and
    the run must finish cleanly inside ten seconds (section 30).
  - The corpus also carries a case with a compressed field, one with 4200
    row-dimensioned datasets, and one carrying `/notes` and `/private`.
    The first catches a writer that drops gzip from a large file, the
    second is the only case that reaches the attachment ceiling above, and
    the third covers two groups no other valid case exercises (section
    30).
  - `grouped_split` has one portable algorithm, so that a seed names the
    same split in every language: the units ordered by their category
    names as bytes, one splitmix64 draw each, the units sorted by the
    draw, and part sizes by largest remainder with a floor of one unit per
    part (section 31).
  - Three questions about the interfaces are answered by the API
    conventions and not by any rule here: a metadata open may also read
    `/row_support`, because it is a column rather than a slot and E16
    needs it in an unaligned file; an evaluated file does not carry
    `/private`, because the result is a new dataset whose producer
    attaches its own records; and an unknown dataset inside a known group
    is not carried through a rewrite, but is listed as lossy so that
    nothing is dropped in silence. Section 14 gains no rule and section 29
    is unchanged.


17. Dictionary codec
--------------------

`to_dict` yields a nested dictionary whose leaves are numeric arrays,
numbers, booleans, or strings. It maps to an HDF5 group as follows,
so that every language round-trips it identically:

  nested dictionary   a subgroup with the key as its name
  numeric array       a dataset, C order, with a dimension scale on
                      each axis named `mestra_<dataset>_d<i>`
  number or boolean   an attribute on the enclosing group
  string              an attribute on the enclosing group,
                      fixed-length UTF-8
  list of numbers     a one-dimensional dataset
  list of strings     a one-dimensional fixed-length string dataset
  null                an attribute with the reserved string "\0null"
  anything else       not representable; the writer must refuse

The dimension scale names are per dataset because two datasets in one
group may have different lengths on the same axis number, and the
reserved prefix keeps them apart from dictionary keys.

Key names must be valid netCDF-4 names and must not begin with
`mestra_`, which is reserved for the container everywhere in the file.
Attribute order and key order are not
significant. A reader reconstructs the dictionary and hands it to
`from_dict` for the group's `type`. Section 25 fixes the cases this
table leaves open.

18. Attribute encodings
-----------------------

Every attribute this specification names has exactly one encoding. A
writer must use it; a reader must refuse an attribute stored any other
way (E19).

  boolean   H5T_STD_I8LE (int8), scalar dataspace, value 0 for false
            and 1 for true. No other value is legal.
  integer   H5T_STD_I64LE (int64), scalar dataspace.
  float     H5T_IEEE_F64LE (float64), scalar dataspace. NaN and the
            infinities may appear in datasets; an attribute that
            declares a bound, a level or a quantile must be finite.
  string    a fixed-length HDF5 string, scalar dataspace, with
            character set H5T_CSET_UTF8 and padding H5T_STR_NULLPAD.

A writer must not use a variable-length string anywhere in the file.

The size of a string attribute is the number of bytes in the UTF-8
encoding of the value, or 1 when the value is the empty string, since
HDF5 has no zero-size string type. The stored bytes are that encoding,
padded on the right with NUL bytes to the declared size. A reader
strips trailing NUL bytes and then decodes UTF-8. A string value must
not contain a NUL byte; the null sentinel below is the only exception.

A list of strings is never an attribute. Where a list of names is
needed in an attribute (`derived_from`), the value is a single string
holding the names separated by one space; names cannot contain a
space, so this is unambiguous.

Null sentinel. The reserved value of section 17 is a string attribute
whose stored bytes are exactly 0x00 0x6E 0x75 0x6C 0x6C, that is one
NUL byte followed by "null", with size 5. Because no other string may
contain a NUL byte, the sentinel cannot be confused with a value. It
occurs only inside a callable's dictionary.

The netCDF-C string reader alters a value that begins with a NUL byte;
it reports the sentinel as "null" rather than as the five stored
bytes. The sentinel is therefore exact only through an HDF5 reader.
This costs nothing, because the sentinel occurs only inside a
callable's dictionary, which a generic netCDF tool copies and does not
interpret, and because a conforming reader reads the codec through
HDF5 and compares the raw bytes.

Names. Every group, dataset, attribute and dimension name in the file
must be a legal netCDF-4 name: not empty, no "/" and no NUL, not
beginning or ending with a space, and built from letters, digits,
underscore, hyphen, "." and "+", and the first character is a letter,
a digit or an underscore. Names a producer chooses (keys,
scalars, category tables, arrays, support groups, callable ids,
dictionary keys) must not begin with `mestra_`, which this format
reserves everywhere in the file and not only in the codec (E33).

Attributes written by the HDF5 dimension scale machinery and by
netCDF-C are not part of this format and a reader must ignore them
wherever they appear: CLASS, NAME, DIMENSION_LIST, REFERENCE_LIST,
DIMENSION_LABELS, _Netcdf4Dimid, _Netcdf4Coordinates, _nc3_strict and
_NCProperties.


19. Dataset encodings
---------------------

Every numeric dataset is little-endian and in C (row-major) order. The
format has no missing-value convention: a writer must not set an HDF5
fill value and must not write a `_FillValue` attribute, a reader must
ignore one if it finds it, and every element the file declares must be
written. Missing floating-point data is NaN, which the validator
reports (W03). Missing integer data has no representation; use a
status key or a category for it.

Dtypes by role. Any other dtype is an error (E20). float32 is not
allowed anywhere: a producer that computes in single precision widens
on write, so that two readers never disagree about a value.

  coordinates, field, derived, weight, normal   float64
  a scalar under /scalars                       float64
  label                                         int32 or int64
  key with role design, condition, time         float64
  key with role categorical, group, split,
      status                                    int32 or int64
  key with role id                              int64, or a
                                                fixed-length UTF-8
                                                string
  /row_support                                  int32
  cell_types                                    uint8
  cell_offsets                                  int64
  cell_connectivity                             int64
  /categories/<name>                            fixed-length UTF-8
                                                string
  a dataset inside a callable's dictionary      section 25

A categorical, group, split or status key stored as floating point is
an error by this table (E20). There is no warning for it.

A fixed-length string dataset has character set H5T_CSET_UTF8 and
padding H5T_STR_NULLPAD, as in section 18. Its size is the largest
UTF-8 byte length among its elements, or 1 when every element is
empty; shorter elements are padded on the right with NUL bytes. A
reader strips trailing NUL bytes and decodes UTF-8. A size larger than
the longest element needs is legal and draws a warning (W13).

Shapes. The logical order of an array is (row | group:<k> | nothing,
[draw], node | cell, component), stored in that order. Three rules
make the shape of every slot decidable without reading it:

  - the leading dimension is present when `varies` is `row` or
    `group:<k>` and absent when `varies` is `none` (E04);
  - the component dimension is always present, with length 1 for a
    single-component quantity. This holds for every array role,
    including labels and one-component fields. A reader therefore
    never has to guess whether a trailing axis is a component axis;
  - the draw dimension is present only when the slot holds draws,
    that is when `statistic` is `draw`. A slot with `statistic` of
    `value`, `mean`, `band`, `std` or `quantile` has no draw
    dimension.

A dataset under /scalars has exactly one dimension, `row`. A dataset
under /keys has exactly one dimension, `row`. Neither carries a
component dimension.

Zero rows. The `row` dimension is always an HDF5 unlimited dimension
(section 21), so a file with no rows stores each row-dimensioned
dataset with shape (0, ...) and the rest of its extents as usual. A
zero-length extent is legal only for the `row` dimension and for a
zero-length axis of a dictionary dataset (section 25); every other
dimension has length one or more.

`row` is also the only dimension that may be unlimited. Every other
dimension scale, and every axis attached to one, is created with a
maximum extent equal to its length (E43). The one exception is a
zero-length axis of a dictionary dataset, which section 25 requires
to be unlimited because HDF5 has no other legal way to write it. The
reason to forbid rather than allow is that every dimension except
`row` carries its length in its name, directly in `draw_<n>` and
`component_<n>` and through the table it names in `group_<k>` and
`category_<t>`, so a dimension that grows makes its own name false.
Measured on a file whose `draw` dimension had been appended to, two
implementations refused it and two returned the value at the new
draw; and draws are produced at evaluation and never appended, so
nothing is lost by forbidding it.

Slots. A slot whose `source` is `data` is a dataset. A slot whose
`source` is `callable:<id>` is an empty HDF5 group carrying the slot's
attributes and no datasets (E30). A group is used because a callable
slot has no shape and no row count of its own, and this makes the two
kinds of slot distinguishable without reading any data, with rows
present or absent.

Required attributes, by object:

  /                       format (string), writer (string), created
                          (string), aligned (boolean);
                          generalisation_group (string) when the file
                          declares any group key
  /keys/<name>            role (string); units (string) for design,
                          condition and time; category (string) for
                          categorical, group, split and status;
                          optional lower and upper (float);
                          trajectory_group (string) on the time key
                          when the file declares any group key;
                          optional parent (string) on a group key
  /scalars/<name>         units (string), source (string), components
                          is not used; output (string) when source is
                          a callable; optional statistic, of, level,
                          method, quantile
  /categories/<name>      no required attribute
  /row_support            no required attribute
  /supports/<s>           kind (string), n_nodes (integer), n_cells
                          (integer), support_id (string)
  /supports/<s>/coordinates
                          role = "coordinates", varies (string),
                          units (string), components (integer),
                          source (string)
  /supports/<s>/node_arrays/<name> and .../cell_arrays/<name>
                          role (string), varies (string), components
                          (integer), source (string); units (string)
                          for field and derived; category (string)
                          when a label uses a table; recomputed
                          (boolean) on weight and normal;
                          derived_from (string) and recipe (string)
                          on derived, with optional reference, whose
                          value is `row=<i>` or
                          `group:<k>=<category name>` (section 5);
                          output (string) when source is a callable;
                          optional statistic, of, level, method,
                          quantile
  /callables/<id>         type (string), optional repr (string)

`components` is required on every array slot, and for a slot that
holds data it must equal the length of its component dimension (E31).
It is the only way a callable slot can declare its width.

`location` and `support` are not attributes. A slot's location is
`node` when it sits under `node_arrays` and `cell` when it sits under
`cell_arrays`; its support is the support group it sits under; a
scalar has neither. `source` is either the string `data` or the string
`callable:<id>`, with an id that is not empty, and nothing else (E36).


20. Cells and cell types
------------------------

`cell_types` holds VTK cell type codes as uint8. The allowed codes in
version 0, with the number of nodes each takes:

  code  cell                      nodes
     1  vertex                        1
     3  line                          2
     5  triangle                      3
     7  polygon                       3 or more
     9  quadrilateral                 4
    10  tetrahedron                   4
    12  hexahedron                    8
    13  wedge                         6
    14  pyramid                       5
    21  quadratic line                3
    22  quadratic triangle            6
    23  quadratic quadrilateral       8
    24  quadratic tetrahedron        10
    25  quadratic hexahedron         20
    26  quadratic wedge              15
    27  quadratic pyramid            13

Any other code is an error (E21). The order of the nodes within a cell
is the VTK order for that code; this format adds nothing to it.

`cell_offsets` has length n_cells + 1. Its first value is 0, its
values are non-decreasing, and its last value is the length of
`cell_connectivity` (E23). Cell j occupies the half-open range
[cell_offsets[j], cell_offsets[j+1]) of `cell_connectivity`. The
length of that range must equal the node count of cell_types[j], or be
3 or more when cell_types[j] is 7 (E22). Every value in
`cell_connectivity` must be in [0, n_nodes) (E24).

A support of kind `axis` or `none` has n_cells = 0 and carries no
`cell_types`, `cell_offsets` or `cell_connectivity` dataset and no
`cell` dimension (E38). A support of kind `none` has n_nodes = 0 and
no coordinates; it exists so that a slot may declare that it lives on
no support, and scalars do not reference it. It carries no `node`
dimension scale either: a zero-length extent is legal only for `row`
and for a zero-length axis of a dictionary dataset (section 19), so a
`node` dimension of length 0 would not be a legal netCDF-4
dimension.

An `axis` support has exactly one coordinates array and that array
must have `varies = none` (E35). The axis coordinate is part of the
support's identity (section 24), so it cannot differ between rows; a
one-dimensional quantity that does differ between rows is a field on
the axis, not the axis.


21. Named dimensions on disk
----------------------------

Every axis of every dataset in the file must have exactly one HDF5
dimension scale attached to it (E25). This is what makes the file a
netCDF-4 file and what lets a reader in a column-major language
permute by name.

A dimension scale dataset is written exactly as netCDF-C writes a
dimension that has no coordinate variable:

  - one-dimensional, dtype H5T_IEEE_F32BE, length equal to the
    dimension's length;
  - attribute CLASS = "DIMENSION_SCALE";
  - attribute NAME = the 53-character string

      This is a netCDF dimension but not a netCDF variable.

    followed by the dimension's length as a decimal integer right
    justified in ten columns, which is the C format "%s%10d"; the
    value is 63 characters for any length of up to ten digits, and
    longer for a longer one;
  - no values are written into it and a reader must not read any.

Setting CLASS and NAME is what H5DSset_scale does, with NAME given as
the string above; attaching is H5DSattach_scale. A writer that calls
those two functions produces the layout netCDF-C expects.

A dimension scale is created with attribute creation order tracked
and indexed, and with object time tracking off, on its dataset
creation property list. The first of the two is E42; the second is a
writer requirement that nothing validates, for the reason given
below:

    H5Pset_attr_creation_order(dcpl,
        H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
    H5Pset_obj_track_times(dcpl, 0);

which is what netCDF-C sets on every object it creates. The first
call gives the scale a version 2 object header, so that its
REFERENCE_LIST is kept in the file's heap rather than as an object
header message, where an attribute may not exceed 64 KiB. Without it
no scale carries more than 4085 attachments: REFERENCE_LIST grows by
sixteen bytes each time, and the 4086th H5DSattach_scale fails with
"object header message is too large" after it has already deleted the
attribute it was extending, which leaves a file that every reader and
every validator still accepts. The second call keeps the file byte
reproducible, because a version 2 object header records four
timestamps unless it is told not to.

Object time tracking is a writer requirement and not a validated
one. HDF5 stores the flag only in a version 2 object header; a
version 1 header keeps its four timestamps whatever the writer
asked for, and the property read back from one reports tracking on
in every case. So the only object the flag can be read from is a
scale that already obeys E42, where the answer is never the
interesting one, and a check of it would report a second fault on
exactly the files E42 rejects and nothing on any other. E42 is
therefore about attribute creation order alone, and no validator may
reject a file over object times. What the second call is for is byte
reproducibility, and that is checked by writing the file twice and
comparing, not by reading a property list.

Three things about that rule. H5Pset_attr_phase_change looks like the
way to ask for the same thing and is not: under the default library
version bounds it is silently ignored, because dense attribute
storage needs a version 2 object header that those bounds will not
write, and a writer must not rely on it. The library version bounds
themselves stay at the default, for the scale and for every other
object: this rule changes the object header of the scales and of
nothing else, so the superblock and every non-scale object are as
they were. And REFERENCE_LIST is informational. A reader never reads
it; it resolves an axis through the dataset's own DIMENSION_LIST and
the map of this section's last paragraph. A scale whose
REFERENCE_LIST is missing, short or stale is not an error, and no
rule in section 14 is about it.

The dimension's name is the scale dataset's HDF5 link name and not its
NAME attribute, which holds the sentence above in every file. A reader
must take the name from the link, because a reader that took it from
NAME would find every dimension in the file called the same thing.
H5DS API calls and library wrappers that report a "dimension label"
return NAME, so this is worth checking early in each language.

There is a second trap in the same place, and it is worse, because it
does not return a wrong answer. Asking the library for the path of
the scale object attached to an axis, which is the obvious way to
recover the link name, makes HDF5 search the group hierarchy for a
name that leads to it. On a file with a deep chain of groups that
search runs off the stack and takes the process with it, and a reader
that does this passes every well-formed file and dies on a malformed
one. A reader must build a map from each scale's object address or
token to its link name during its own bounded walk of the file
(section 29), and resolve attached scales through that map.
Dereferencing the scale and reading its address is safe; asking for
its path is not.

A dimension scale that is unlimited is chunked with chunk length 1,
which is what netCDF-C writes; a scale that is not unlimited is
chunked with a chunk equal to its length, which is what H5DSset_scale
leaves behind. No value is ever stored in either, so the choice is
visible only in the file's bytes. Only the `row` scale, file-level or
support-local, is unlimited at all (E43).

Where the scales live and what they are called. All file-level scales
are at the root group, because netCDF-4 resolves a dimension in the
group that holds the variable or in any ancestor group, and because
user-chosen names live in subgroups and therefore cannot collide with
them:

  row            unlimited; length equal to the number of rows
  component_<n>  one for each distinct component count n in the file
  draw_<n>       one for each distinct draw count n in the file
  group_<k>      one for each group key k; length equal to the number
                 of categories of k
  category_<t>   one for each category table t under /categories;
                 length equal to the number of entries

Support-local scales are in the support's own group, because their
lengths differ between supports:

  node           length n_nodes; absent when the kind is `none`
  cell           length n_cells; absent when n_cells is 0
  cell_plus_one  length n_cells + 1; absent when n_cells is 0
  index          length of cell_connectivity; absent when n_cells is 0
  row            only in an unaligned file, and only in a support
                 that carries an array with `varies = row`: length
                 equal to the number of rows referencing that
                 support, and unlimited as the file-level `row` is

The support-local `row` shadows the file-level one, which is legal
and intended: netCDF-4 resolves a dimension in the group holding the
variable before any ancestor, so the logical dimension name of such
an array is still `row` and a reader permutes by that name. Section
22 says what an index into it means. An aligned file never has one,
because there every row is on the one support and the two lengths
agree.

Scales for the datasets inside a callable's dictionary are in the
group that holds the dataset and are named `mestra_<dataset>_d<i>`,
where <dataset> is the dataset's link name and i is the axis number
from 0 (section 25). Per-dataset names are needed because two
dictionary datasets in one group may have different lengths on the
same axis number, and the reserved prefix keeps them apart from
dictionary keys.

Logical name to name on disk:

  row           row
  group:<k>     group_<k>
  draw          draw_<n>, n the draw count
  node          node
  cell          cell
  component     component_<n>, n the component count
  cell_plus_one cell_plus_one
  index         index

A reader recovers the logical name from the name on disk by this
table, and must do so by name and never by position.

The instance index of `varies = group:<k>`. Category ids are the
zero-based positions of the entries in the category table: the first
entry is id 0. A key with role categorical, group, split or status
stores category ids, so its values are in [0, number of entries)
(E10). Instance i of an array with `varies = group:<k>` is the
instance for category id i, in category-table order, and the length of
its leading dimension equals the number of categories of k (E34). A
row's instance is therefore the value of key k in that row, used
directly as an index, with no lookup.

Labels without a category table are the exception: their values are
their own categories (CAD face ids, for example) and may be any
integers.

`row` in a zero-row file. `row` is an unlimited dimension in every
file, so the scale dataset has shape (0,) with an unlimited maximum
when the file has no rows, and shape (n,) with an unlimited maximum
otherwise. Making it unlimited in every file has three effects: a
zero-row file is a netCDF-4 file with an empty record dimension rather
than an illegal zero-length fixed dimension; rows can be appended
without rewriting; and the layout is the one netCDF-C reads as an
unlimited dimension.

That last is a claim about reading and not about round trips. A file
that netCDF-C has written or copied is not a conforming file. It
writes every text attribute as ASCII with NUL termination where
section 18 requires fixed-length UTF-8 with NUL padding; it writes
every scalar attribute as a one-element array, so `aligned`, `lower`,
`upper`, `n_nodes`, `n_cells` and `components` all change shape; it
leaves the scale of an unlimited dimension at length 0 where this
section requires the row count; it writes dimension scales contiguous
where this section requires them chunked; and it adds
`_NCProperties` to the root, `_Netcdf4Dimid` to every scale and
`_Netcdf4Coordinates` to every variable. A file that has been through
netCDF-C, including through `nccopy`, must be written again by a
conforming writer before it is a conforming file. This paragraph is
informational: the rules it points at are the normative ones.

The scale dataset's own length must equal the row count, so that a
reader can learn the row count from a file that has no
row-dimensioned datasets at all.
netCDF-C infers the length of an unlimited dimension from the
variables that use it and ignores the scale dataset's length; the two
must agree.


22. Alignment, row_support, and rows with callable slots
--------------------------------------------------------

`aligned` is a boolean root attribute and it is decidable from the
file:

  - a file that declares at most one support must set aligned = true
    and must not have a /row_support dataset;
  - a file that declares more than one support must set
    aligned = false and must have a /row_support dataset (E28, E37).

A file with no rows and one support is aligned; so is a file with no
supports at all.

/row_support is int32 with one dimension, `row`. Its value for a row
is the zero-based position of that row's support in the file's support
order, and the support order is the order of the support group names
under /supports sorted by their UTF-8 bytes. A value outside
[0, number of supports) is an error (E06). Sorting by bytes is used
because it is the one ordering every language produces identically;
HDF5 link order is not.

Row-varying arrays in an unaligned file. An array with
`varies = row` on support s has a leading dimension whose length is
the number of rows that reference s, and it holds those rows in the
file's row order: its element i is the i-th row, counting from zero,
whose `/row_support` value is s. It is not indexed by the file's row
number, because the rows on one support are not contiguous in the
file. That leading dimension is the support-local `row` of section
21, and E16 checks the count per support. In an aligned file the two
counts are equal and nothing about this changes.

Rows and callable slots together. A file may have rows and callable
slots at the same time. When it does, the key columns hold rows, for
example the design the callable was fitted on, and the callable slots
simply hold no data. What decides whether a slot is stored data is its
`source` and nothing else; the row count decides nothing. A file with
rows in which every slot is a callable is valid, and so is a file with
rows in which some slots are data and some are callables. Whether to
publish the training design this way is the producer's choice; it
makes extrapolation warnings and dataset cards possible and it
discloses the design.


23. Chunking, compression and fill
----------------------------------

Every dataset that has a `row` dimension must be chunked, because
`row` is unlimited. Its chunk shape must be (c, ...) where every
non-row extent in the chunk is that dimension's full length and c is
one or more, so that a row range is one contiguous run of chunks and
no other slot is touched. When a single row of a dataset is larger
than 8 MiB, a writer may also chunk the node or cell dimension, and
must then set c to 1.

The default for c: let b be the size in bytes of one row of the
dataset, that is the dtype size times the product of the non-row
extents, with a zero-length extent counted as 1. Then

  c = 1 when the row count is 0;
  otherwise c = floor(1048576 / b), or 1 when that is 0, and then
  c = the row count when c is larger than the row count.

1048576 is 1 MiB. A file with no rows takes c = 1 rather than the
hundred thousand rows a megabyte would hold, because nothing is going
to be written into it in this file and a chunk that large costs a
reader a megabyte of cache for an empty dataset.

The row count in that rule is the length of the row dimension the
dataset's leading axis is attached to, and not the dataset's own
leading extent. In an unaligned file a row-varying array on a support
is attached to that support's own `row` (section 21), so its default
follows the number of rows on that support. A dataset whose leading
extent disagrees with the dimension it is attached to is E16, and its
chunk is still judged against the dimension.

This is a default, not a requirement: a writer may choose another c
and a reader must accept it. A dataset whose chunk shape is not the
default draws a warning (W12), because a corpus file is expected to
use it. W12 applies only to a chunked dataset that carries a row
dimension: a key column, a scalar, an array slot, or /row_support. A
contiguous dataset is E27 or nothing at all, never W12; a dimension
scale and a dataset inside a callable's dictionary never draw it.

A dataset with a zero-length axis is chunked with 1 on every axis,
which is the only chunk HDF5 allows for it.

A dataset with no `row` dimension may be contiguous. When it is
chunked or compressed, the default chunk is the whole dataset if that
is 1 MiB or less, and otherwise the same rule applied to its leading
dimension. A dimension scale dataset is never compressed and is
chunked as section 21 says: chunk length 1 when it is unlimited and a
chunk equal to its length when it is not. Section 21 also fixes its
creation property list, which is the only place in this format where
a property list other than the defaults is required.

Compression. The only filters allowed are gzip at levels 1 to 9 and
shuffle; shuffle may be used with or without gzip. A writer must use
no other filter, must not use fletcher32, and a reader must refuse a
dataset with any other filter (E29). The restriction is portability:
gzip and shuffle are the two filters every HDF5 and netCDF-4 build
has.

Fill. A writer must not set an HDF5 fill value on any dataset, so
every dataset keeps the HDF5 default. A reader must not treat any
value as missing on the strength of a fill value.


24. support_id
--------------

`support_id` is the SHA-256 digest of a byte string built from the
support's arrays, in this order, with nothing between them:

  1. n_nodes as one int64, little-endian;
  2. `cell_types`, as uint8 in storage order;
  3. `cell_offsets`, as int64 little-endian in storage order;
  4. `cell_connectivity`, as int64 little-endian in storage order;
  5. for a support of kind `axis` only, the coordinates array as
     float64 little-endian in storage order, which is (node,
     component) with component of length 1.

A support of kind `axis` or `none` has no cell arrays, so steps 2 to 4
contribute no bytes at all for it; they are not replaced by anything.
Coordinates of a mesh support are not hashed, because they may vary
between rows while the support does not.

When the coordinates of an `axis` support do not have `varies = none`
the file is already invalid (E35); the digest is then computed over
the stored bytes as they are, so that E35 is the only rule such a
file breaks.

The declared kind decides which steps contribute bytes, and not what
the support happens to carry. An `axis` or `none` support that
wrongly carries cell datasets still contributes nothing from steps 2
to 4, so it breaks E38 and its digest still matches. This is the same
choice as the paragraph above, for the same reason: one broken rule
should not make a second rule fire as well.

A support that declares no kind (E39), or a kind section 6 does not
have (E02), has no digest. Nothing says which steps it takes, and an
implementation must not compute one by taking it for a mesh or for
anything else: it refuses, and E08 is not decided for it.

The attribute is the digest in lower-case hexadecimal, 64 characters.
A digest that does not match the stored arrays is an error (E08).

Worked example. A mesh support of six nodes and two quadrilaterals:

  n_nodes            6
  cell_types         [9, 9]
  cell_offsets       [0, 4, 8]
  cell_connectivity  [0, 1, 4, 3, 1, 2, 5, 4]

The byte string is 8 + 2 + 24 + 64 = 98 bytes long and begins
06 00 00 00 00 00 00 00 09 09 00 00 .... Its digest is

  96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c943936c7

An implementation that reproduces that string has the rule right.
Two more, for the cases with no cells:

  an axis support of four nodes at 0.0, 0.5, 1.0, 1.5
  57467fe7370808bdb0ad01b95d963f59e8bc6762f90f96049453ae33bb05a54c

  a support of kind none, n_nodes 0
  af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc


25. The dictionary codec on disk
--------------------------------

Section 17 gives the mapping. This section fixes the cases an
implementer would otherwise decide alone. Throughout, a "number" is a
Python int or float, a MATLAB scalar double or int64, or a C++
arithmetic scalar; an "array" has one dimension or more.

Scalars and their types. A number, a boolean or a string is an
attribute on the enclosing group, encoded as section 18 requires. The
dtype carries the type, and the three cases are distinct:

  int8      a boolean; the value is 0 or 1 and a reader returns a
            boolean, never an integer
  int64     an integer
  float64   a float
  a string  a string

An integer and a float that happen to be equal are therefore different
values in the file and stay different across a round trip. A boolean
array is an int8 dataset, and int8 means boolean there too.

Zero-dimensional arrays. A zero-dimensional array must be written as
an attribute, that is as the number or boolean it holds, and a reader
returns a number. It must not be written as a zero-dimensional
dataset. The reason is that MATLAB and most C++ containers cannot
represent the difference between a scalar and a zero-dimensional
array, so a rule that preserved the difference would round-trip in
Python and not elsewhere, and the corpus could not state one expected
result. Making the choice at write time costs the writer one line and
makes the round trip exact in all three languages.

Empty arrays. An array with a zero in its shape is a dataset with that
shape, and every zero-length axis must be created with an unlimited
maximum so that the dimension is legal in netCDF-4 (section 21). Its
dtype is kept, so an empty float64 array and an empty int64 array are
different values. An empty list with no element type known must be
written as an empty float64 dataset with shape (0,); a producer that
needs an empty list of strings writes an empty fixed-length string
dataset with shape (0,) and size 1.

Lists. A list of numbers of one type is a one-dimensional dataset; a
list of strings is a one-dimensional fixed-length string dataset. A
nested list is allowed only when it is rectangular and numeric, and it
is then the array it describes. A ragged nested list, a list mixing
numbers and strings, a list of dictionaries and a list of nulls are
not representable and the writer must refuse them (E32).

Allowed dtypes inside a dictionary: int8 (boolean), int32, int64,
float64 and fixed-length UTF-8 strings. Any other dtype, including
float32 and unsigned integers, is not representable and the writer
must refuse it (E32).

Unicode. Strings are UTF-8; the declared size of a fixed-length string
is a count of bytes, not of characters, and a writer must never split
a character across the boundary. A string that contains a NUL byte is
not representable (E32); the null sentinel of section 18 is not a
string value.

Portability of non-ASCII strings. One language's binding in common
use, MATLAB's, cannot carry a fixed-length string that is not ASCII:
it refuses a byte above 127 on write and decodes to text on read, so
the bytes do not survive. The format keeps UTF-8 because the format
outlives any one binding, but a producer that wants every language to
read its file keeps the names this format defines, every category
table entry and every string id to ASCII, and puts anything else in a
callable's dictionary or in `/notes`, where a reader that cannot
carry it can copy it unchanged. A reader that cannot recover the
bytes of a string must say so rather than return something else.

Arrays. A numeric array is a dataset in C order, little-endian, with a
dimension scale on each axis named `mestra_<dataset>_d<i>` as section
21 requires.

Key order. Dictionary key order is not significant and a reader must
not depend on it, on HDF5 link order or on attribute order. A writer
visits a dictionary's keys in ascending order of their UTF-8 bytes,
rather than in whatever order the language hands them over, so that
two writers given the same dictionary produce the same file. The
dimension scales the container adds are created alongside the dataset
they belong to and are not dictionary keys.

The reserved prefix. A dictionary key must not begin with `mestra_`.
A reader reconstructing a dictionary skips every member and every
attribute whose name begins with `mestra_`, and also the HDF5 and
netCDF machinery attributes listed in section 18.

`type` and `repr`. They are string attributes on the callable's own
group /callables/<id>, not entries of the dictionary: `type` is
required and `repr` is optional and is one line. The dictionary is
stored on that same group, so at its top level the two names are taken
and a writer must refuse a dictionary whose top-level keys include
`type` or `repr` (E32). Nested dictionaries may use both names
freely.

A reader reconstructs the dictionary, reads `type`, and hands the pair
to the `from_dict` registered for that type. A reader that does not
know the type may still copy the group unchanged, and must not
interpret it.


26. The keys table in each language
-----------------------------------

`call` takes a keys table: one column per key the file declares, in
the file's key order, and any number of rows. The file's key order is
the order of the names under /keys sorted by their UTF-8 bytes. The
row order of the table is the evaluation order, and the outputs follow
it: output row i is the result for table row i.

  Python   a mapping from key name to a one-dimensional numpy array,
           with every array the same length. A caller may instead
           pass a two-dimensional array of shape (rows, keys) together
           with a list of key names in the file's key order; an
           implementation must accept both.
  MATLAB   a table whose variable names are the key names.
  C++      a struct holding the key names beside one vector per key,
           looked up by name, because C++ has no run-time member
           names: for example a vector of names and a vector of
           columns of the same length, or a map from name to column.
           Every column has the same length.
  Julia    a Dict{String,Vector{Float64}} from key name to column. An
           implementation must also accept a NamedTuple of columns
           and a (matrix, names) pair whose names are in the file's
           key order, as Python accepts its second form.

In every language a column of a key with role id may be strings, and
every other column is numeric. A callable reads the columns it
declares and ignores the rest. A missing column that the callable
declares is an error at call time.


27. The affine reference callable
---------------------------------

`affine` is the one callable type this package defines. It exists so
that the protocol, the codec and evaluation can be conformance-tested
in every language with no proprietary model. It is deterministic and
produces no draws.

For each output slot it serves, with x the vector of key values in the
declared key order,

    y = A x + b

and y is reshaped to the slot's shape in C order.

Its `type` is the string `affine`. Its dictionary is exactly:

  keys      a list of key names, as a one-dimensional fixed-length
            UTF-8 string dataset. This is the declared key order, and
            x is built by taking those keys from the keys table in
            that order.
  outputs   a dictionary with one entry per output the callable
            serves. The entry's name is the value of the slots'
            `output` attribute. Each entry is a dictionary with
            exactly:
      A       float64, shape (n_out_flat, n_keys)
      b       float64, shape (n_out_flat,)
      shape   int64, one-dimensional: the slot's dimensions after the
              row dimension and in order. For an array slot on a mesh
              or axis support that is [node count, component count].
              For a scalar slot it is the empty int64 array, shape
              (0,).
            and, when the output carries a band, all three of:
      uncertainty  float64, shape (n_out_flat,): the band, the same
                   for every row
      level        a float in (0, 1)
      method       a string

n_out_flat is the product of `shape`, and the empty product is 1, so a
scalar slot has n_out_flat = 1. n_keys is the length of `keys`. There
is nothing else in the dictionary; a writer must not add to it and a
reader must refuse an `affine` dictionary with any other key, or with
one or two of the three band keys.

Evaluation. For a keys table with R rows, build X of shape
(R, n_keys) from the declared keys as float64, compute
Y = X A' + b broadcast over rows, giving (R, n_out_flat), and reshape
each row of Y to `shape` in C order. That is the `mean` of the
output's prediction (section 10), in the stored form (row, node |
cell, component) for an array and (row) for a scalar. When the entry
carries a band, the prediction's `uncertainty` is `uncertainty`
reshaped to `shape` the same way and repeated for each of the R rows,
with `level` and `method` as stored; otherwise the prediction has no
uncertainty. The band is a constant so that evaluation stays exact and
the implementations can be compared on it bit for bit; it stands in
for whatever a real model knows about its own error.

The dot product is accumulated over the keys in the declared key
order and b is added last, and no fused multiply-add is used. The
order matters because the corpus compares float64 results bit for
bit, and the other orders differ in the last place: the worked cl
below is 1.45 when b is added last and one unit in the last place
above it when b is accumulated first.

Worked example. Two keys in the order mach, alpha. One scalar slot cl
and one node-array slot pressure on a support of six nodes with one
component:

  cl        A = [[2.0, 0.1]]        b = [0.05]     shape = []
  pressure  A = [[1.0, 0.0],        b = [0.0,      shape = [6, 1]
                 [2.0, 0.0],             0.1,
                 [3.0, 0.5],             0.2,
                 [4.0, 0.5],             0.3,
                 [5.0, 1.0],             0.4,
                 [6.0, 1.0]]             0.5]

On the one-row keys table mach = 0.5, alpha = 4.0:

  cl        = 2.0*0.5 + 0.1*4.0 + 0.05 = 1.45
  pressure  = [0.5, 1.1, 3.7, 4.3, 6.9, 7.5], as shape (1, 6, 1)

docs/examples/affine_zero_rows.mes is this callable, written out.

With a band. vectors/cases/affine_band is the same callable with a
band on both outputs: cl carries uncertainty [0.02] at level 0.95 and
pressure carries [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] at level 0.68,
each with the method "constant band". The file adds two slots, cl_band
and pressure_band, with `statistic = band`, `of` naming the base slot
and the same `output` as it. On the one-row table above cl and
pressure are unchanged, cl_band is 0.02 and pressure_band is the
vector just given, and the evaluated file carries `level` and
`method` on both band slots.


28. Version handling
--------------------

The root attribute `format` is "mestra/<major>". A reader of
"mestra/0" must refuse any other major version and say so; it must not
try to read it partially (E01). The major version changes only when a
file that a version-0 reader would read would mean something else.

Within a major version:

  - new optional attributes may be added, and a reader that does not
    know an attribute must ignore it and report it (W11);
  - new optional groups may be added at the root and inside a support,
    and a reader that does not know a group must ignore it and report
    it (W11);
  - new roles, new statistics, new cell type codes, new key words for
    `source` and new required attributes are not additive and must not
    be added within a version;
  - nothing this document requires is ever removed, renamed or given a
    different meaning within a version.

A reader must therefore be written so that an unknown name is a report
and not a failure, and a writer must not rely on an older reader
noticing anything it adds.


29. What a reader must offer
----------------------------

These are requirements on readers. They constrain no byte in the file;
the layout of sections 19 to 23 is what makes them possible.

  - Lazy access. A reader must be able to read one slot, for a range
    of rows, without reading any other slot and without reading the
    rows outside the range. Chunking along `row` with the non-row
    extents full (section 23) is what makes that one contiguous run
    of chunks.
  - Opening a file must not read any array. A reader must be able to
    report the row count, the keys with their roles and bounds, the
    supports with their ids, and every slot with its attributes,
    having read attributes and dataspaces only.
  - Dimensions by name. A reader must expose the logical dimension
    name of every axis it returns (section 21), and must permute by
    name. A column-major reader may return whatever order is natural
    for it, provided the names come with it.
  - Unknown things. A reader must ignore what section 28 says to
    ignore, must report it, and must not fail on it.
  - A reader must not require /notes or /private, and must not
    interpret /private.
  - Untrusted input. A reader treats a file as untrusted input. Its
    own code must not crash, hang, or allocate without bound on a
    malformed one, whatever the file does; the HDF5 library it reads
    through can still fault on a crafted byte before that code runs,
    and an implementation says so in its compatibility notes. Four
    rules make the reader's part reachable.
    It never follows a link that is not a hard link: a soft link,
    resolving or not, and an external link are E40 and are not
    opened. An object it cannot read is reported as E41 with its
    path, and the pass continues rather than stopping at the first
    one, so that one broken object does not hide the rest of the
    file. Recursion over groups and over a callable's dictionary is
    capped at a depth the reader states, and a file deeper than the
    cap is E41 rather than a crash. An eager read refuses a dataset
    whose declared element count is above a maximum the reader
    states, with E41; 2^31 elements is the default to state. A lazy
    read and a row-range read are not subject to that last limit,
    because they never materialise the whole dataset, so the same
    file may be readable one way and E41 the other.


30. Conformance corpus conventions
----------------------------------

The corpus lives under vectors/. One directory per case, named in
lower case with underscores, holding exactly two files:

  vectors/<case>/case.mes        the golden file
  vectors/<case>/expected.json   what every implementation must find

and one vectors/manifest.json listing the cases:

  {"corpus": 0,
   "cases": [{"name": "<case>", "description": "..."}, ...]}

with `cases` sorted by name.

expected.json is canonical JSON (below) with exactly these fields:

  description   a string, one or two sentences
  validator     {"errors": [ids], "warnings": [ids]}, each a sorted
                list of the rule ids of section 14 that the file must
                produce, with no duplicates. A file that must validate
                cleanly has two empty lists.
  support_ids   an object from support group name to the 64-character
                lower-case hexadecimal digest. A support that has no
                digest (section 24) is not in it, and an implementation
                must refuse to compute one for it
  probes        a list of objects, each naming one stored value:
                  slot       the HDF5 path of the dataset, for example
                             "/supports/s0/node_arrays/pressure"
                  row        the row index; omitted when the slot has
                             no row dimension. For a row-varying array
                             in an unaligned file it is the index
                             within that support's rows (section 22)
                  instance   the index along the leading dimension of
                             an array whose `varies` is `group:<k>`;
                             present in place of `row` for such a slot
                  node       the node index, or the cell index for a
                             cell array; omitted when the slot has no
                             node or cell dimension
                  component  the component index; omitted when the
                             slot has no component dimension
                  draw       the draw index; present only when the
                             slot has a draw dimension
                  cell       the cell index, where a probe names a
                             support's own `cell` axis
                  cell_plus_one  the index along the cell offsets
                  index      the index along the flat connectivity
                             axis
                  value      the value as a decimal string: the
                             "%.17e" form below for a float64 slot and
                             the plain decimal form for an integer
                             slot. The rule below that integers are
                             JSON numbers applies to indices and
                             counts, not to a probe's value
                A probe on one of a support's own cell datasets names
                its axis `cell` for `cell_types`, `cell_plus_one` for
                `cell_offsets` and `index` for `cell_connectivity`.
                A dataset inside a callable's dictionary has no
                logical dimension names at all, because the codec's
                scales are the container's and not the dictionary's.
                A probe on one applies whichever index fields it
                carries, in the order row, instance, draw, node,
                component, index, to the dataset's axes in file
                order; with two axes and the fields node and
                component, node is axis 0 and component is axis 1.
  codec         an object from callable id to the round trip of that
                callable's dictionary, in the tagged form below
  evaluation    a list of objects, each a worked evaluation of one
                callable: `callable` the id, `keys` the keys table it
                is evaluated on as an object from key name to a list
                of "%.17e" strings one per table row, and `probes`
                the expected outputs in the probe form above, whose
                `row` is the index into that table and not into any
                row the file stores. The list is empty when there is
                nothing to evaluate

Numbers as text. Every float in expected.json is a string in the C
format "%.17e", which every language produces identically and which
round-trips a float64 exactly: "1.45000000000000000e+00". The three
non-finite values are the strings "nan", "inf" and "-inf". Integers
are JSON numbers. A comparison parses the string to float64 and
requires bit equality, so that no implementation has to agree about
how a float is printed.

Canonical JSON. UTF-8 with no byte order mark; object keys sorted by
their Unicode code points; no whitespace except a single newline at
the end of the file; the separators are "," and ":"; non-ASCII
characters are written literally and not escaped.

The tagged form for a dictionary round trip. Every value is an object
with a tag `t`, so that a dictionary is never confused with a leaf:

  {"t":"dict","v":{ key: value, ... }}
  {"t":"i64","v":42}
  {"t":"f64","v":"2.50000000000000000e+00"}
  {"t":"bool","v":true}
  {"t":"str","v":"affine"}
  {"t":"null"}
  {"t":"array","dtype":"float64","shape":[6,2],"data":[...]}
  {"t":"strings","shape":[2],"data":["mach","alpha"]}

`dtype` is one of "bool", "int32", "int64", "float64". `data` is the
elements flattened in C order; float64 elements are the decimal
strings above, int32 and int64 elements are JSON numbers, bool
elements are true and false.

The hostile subset
------------------

The corpus above is a set of well-formed files, each breaking at most
the rules it means to break. It says nothing about what a reader does
with a file that is not well formed at all, and section 29 requires an
answer: a reader treats a file as untrusted input. The hostile subset
is that answer, written down.

    vectors/hostile/<case>/case.mes        the file
    vectors/hostile/<case>/expected.json   what must happen

with vectors/manifest.json listing them under `hostile` beside
`cases`, sorted by name. expected.json is canonical JSON with exactly
these fields:

  description       a string, one or two sentences
  required_errors   the rule ids of section 14 that must appear,
                    sorted and without duplicates
  allow_extra       true throughout this subset
  timeout_seconds   10

The contract is deliberately looser than the corpus's. A validator
must report at least the required ids; it may report more, because two
readers will reasonably disagree about how much of a broken file is
worth describing. It must exit cleanly within the timeout: not crash,
not hang, not exhaust memory, and not stop at the first bad object.
Opening the file for its metadata alone, and any operation that reads
a slot, must refuse with the same ids rather than return something. A
reader that cannot open the file at all reports the ids and says so;
that is a clean exit.

A file in this subset is not a specimen of the format, and nothing may
be inferred from one about what a valid file looks like.


Golden files must be byte reproducible: two runs of the generator on
one machine must produce identical bytes. Three things are needed.
HDF5 object time tracking must be off, which is `track_times=False` on
every h5py dataset. `created` and `writer` must be fixed values rather
than the time and the version of the run. And the order in which the
generator creates links, attributes and scale attachments must be
fixed, because HDF5 stores the links of a small group in creation
order; an order that comes out of a hash table or a dictionary
iteration is not fixed. The generator is committed, so its order is
the corpus's order and no one has to reconstruct it.

Byte identity across HDF5 versions is not required and must not be
tested, because the HDF5 library decides the superblock and the object
header layout. Implementations link whatever libhdf5 their language
binding brings: 1.12, 1.14 and 2.2 have all been used against this
corpus, and they do not agree byte for byte. The normative comparison
between two files that should be the same is structural equality:

  - the same set of object paths;
  - at each path, the same kind (group or dataset);
  - for each dataset, the same dtype including byte order, character
    set and padding, the same shape, the same maximum shape, the same
    chunk shape, the same filters with the same parameters, and
    element-by-element equal contents, with floats compared as bits so
    that NaN equals NaN;
  - at each path, the same set of attribute names excluding the
    machinery names of section 18, and for each, the same dtype and
    the same value;
  - the same dimension scale attached to each axis of each dataset,
    compared by the dimension's name.


31. Grouped split
-----------------

A grouped split divides the rows of a file into named parts by whole
units of generalisation, so that no unit has rows in two parts. The
helper is `grouped_split(dataset, fractions, seed)` (section 4 of the
API conventions). This section is the algorithm it follows, written
down so that one seed names one split in every language. An
implementation that shuffles with its own language's random generator
is not conforming, however good the generator is: the four in this
repository each chose a different one and no two agreed on a split.

The units. The file names a unit of generalisation, which is a key of
role group (section 7). A key of that role stores category ids
(section 21), so the units are the distinct ids the key's column
holds, and they are put in order by the entries those ids name in the
key's category table, compared as UTF-8 byte strings. Table order is
not the order: two files that hold the same units in tables written
in two orders must split the same way. A file that names no unit of
generalisation cannot be split this way and the call is refused.

The generator. One 64-bit value is drawn per unit from splitmix64,
seeded with the seed, in the unit order above. The generator is
sixty-four bits of state and this update, with every operation on
unsigned 64-bit integers and every shift logical:

    state = state + 0x9E3779B97F4A7C15
    z     = state
    z     = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
    z     = (z XOR (z >> 27)) * 0x94D049BB133111EB
    draw  = z XOR (z >> 31)

The state starts at the seed, and the first draw is the one the first
update returns; the seed itself is never a draw. splitmix64 is used
because it is five lines, has no state beyond the seed, and gives the
same stream in every language that has 64-bit wrapping arithmetic.
The units are then sorted by their draw, as unsigned integers and
ascending, and two units with the same draw are ordered by their
category names as bytes.

The parts. `fractions` names each part and gives it a fraction; the
parts are taken in order of their names as UTF-8 byte strings, which
is the order they are filled in and the order of the result. The
fractions are normalised by their sum, so that 8 and 2 mean what 0.8
and 0.2 mean. A fraction that is negative, or a set of fractions that
sums to zero or less, is refused.

The sizes. With u units and a part whose normalised fraction is f,
the part's exact share is f * u in float64, its base size is the
floor of that, and the remainder is what is left. The base sizes are
handed out first, and the u - sum(base) units still unassigned go one
each to the parts with the largest remainders, ties broken by part
name. Then the floor: while some part has no unit at all, the part
with the fewest -- in name order among equals -- takes one unit from
the part with the most, ties again broken by name. A part with no
unit in it is not a test set, and with at least as many units as
parts some part always has one to give. Fewer units than parts is
refused rather than split: the call cannot do what it was asked.

The assignment. The sorted units are dealt out as consecutive blocks
in part-name order: the first part takes the first block of its size,
the second the next, and so on. A part's rows are then every row
whose unit is in that part, in ascending row order.

A worked example. A file whose unit of generalisation is the key
`rotor`, whose category table holds five entries in the table order
`rotor_b`, `rotor_a`, `rotor_d`, `rotor_c`, `rotor_e`, so that the
ids are 0 to 4 in that order, split with `seed = 0` and the fractions
`{"train": 0.8, "test": 0.2}`.

In unit order, which is by name and not by id, the draws are:

    unit      id   draw
    rotor_a    1   0xe220a8397b1dcdaf
    rotor_b    0   0x6e789e6aa1b965f4
    rotor_c    3   0x06c45d188009454f
    rotor_d    2   0xf88bb8a8724c81ec
    rotor_e    4   0x1b39896a51a8749b

Sorted by the draw, the units are `rotor_c`, `rotor_e`, `rotor_b`,
`rotor_a`, `rotor_d`. The parts in name order are `test` then
`train`; the exact shares are 1.0 and 4.0, so the base sizes are 1
and 4, nothing is left over and no part is empty. Dealing the sorted
units in part-name order gives:

    test    rotor_c
    train   rotor_e, rotor_b, rotor_a, rotor_d

and each part's rows are the rows whose `rotor` value is one of its
units. An implementation that reproduces this table reproduces every
split, because nothing else in the algorithm depends on the file.


32. Units grammar
-----------------

A units string parses when it matches this grammar, whatever names it
uses (W10). It is the UDUNITS-2 grammar in the forms CF files use, with
one widening: the power after "^" or "**" may be any number, so that
"s^0.5" parses where UDUNITS-2 takes only an integer. Deciding the
verdict needs no unit database; converting between units does, and is
a tool's business.

    units      = { sp } product [ shift ] { sp }
    product    = power { [ operator ] power }
    operator   = { sp } ( "*" | "." | "-" | "/" | "per" ) { sp }
               | sp { sp }
    power      = basic [ integer | ( "^" | "**" ) number ]
    basic      = name | number | "(" units ")"
               | ( "log" | "lg" | "ln" | "lb" ) logref
    logref     = "(" { sp } "re" ( ":" { sp } | sp { sp } ) product
                 { sp } ")"
    shift      = { sp } ( "@" | "since" | "from" | "after" | "ref" )
                 { sp } ( timestamp | number )
    timestamp  = date [ ( "T" | sp { sp } ) clock [ { sp } zone ] ]
    date       = [ sign ] digit { digit } "-" digit [ digit ]
                 [ "-" digit [ digit ] ]
    clock      = digit [ digit ] ":" digit digit
                 [ ":" digit digit [ "." { digit } ] ]
    zone       = name | sign digit [ digit ] [ [ ":" ] digit digit ]
    number     = [ sign ] ( digit { digit } [ "." { digit } ]
                          | "." digit { digit } )
                 [ ( "e" | "E" ) [ sign ] digit { digit } ]
    integer    = [ sign ] digit { digit }
    name       = letter { letter | digit }
    letter     = "A" to "Z" | "a" to "z" | "_" | "%" | "'" | '"'
               | any character outside ASCII
    sign       = "+" | "-"
    digit      = "0" to "9"
    sp         = " "

A string is matched from the left, each rule taking as much as it can,
and five readings are fixed so that one string has one verdict:

  - The trailing digits of a name are its power and not part of it:
    "m2" is m squared and "m2s" is one name.
  - An `integer` power follows its basic with nothing between: "m-1" is
    m to the power -1, "m -1" is m times -1, "m-s" is m times s, and
    "m+s" does not parse.
  - "per", "since", "from", "after" and "ref" are operators only where
    an operator can stand, after a power, and only as whole words, not
    followed by a letter or a digit; anywhere else they are names. So
    "per s" is a product of two names, and "m per" lacks the power its
    "per" needs.
  - After a name "log", "lg", "ln" or "lb", a "(" followed by "re" and
    then ":" or a space opens a `logref`; any other "(" opens a group.
  - A shift's operand is a timestamp when one matches and a number
    otherwise, and a clock or a zone that does not match is not part of
    the timestamp.

The empty string and a string of spaces do not parse, and the only
space is U+0020, so a string holding a tab does not parse. Nor does a
string longer than 4096 bytes in UTF-8, or one whose parentheses, of
groups and of log references together, nest more than 32 deep; the
string comes out of a file, and the two bounds keep a recursive parser
safe on it.
