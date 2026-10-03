The conformance corpus
======================

This directory is the conformance corpus of SPEC.md section 15: a set
of small golden files and, for each one, the outcome every
implementation must find. Every reader and every writer, in every
language, runs it in its own test suite.

No implementation is the reference. The specification and this corpus
define conformance between them. When a file here and an
implementation disagree, the specification decides; when the
specification is silent, that is a defect in the specification and it
is fixed there, in the same change that regenerates the corpus.

Nothing here depends on any tool beyond Python, h5py and numpy, and
nothing here is a mestra package. `generate.py` is sections 13 to 30
read literally and written out with h5py, so that an implementer can
compare a file byte for byte against the rules rather than against
somebody's code.


What is in it
-------------

130 cases, one directory each, holding exactly the two files section
30 requires:

    cases/<case>/case.mes        the golden file
    cases/<case>/expected.json   what every implementation must find

and `manifest.json` listing every case with a one-line description,
sorted by name. The cases fall into five groups:

  - the two files listed object by object in docs/example.md,
    `mesh_two_rows` and `affine_zero_rows`, regenerated here. They are
    byte identical to the committed copies under docs/examples, which
    is what makes this generator checkable against a document written
    before it;
  - the five datasets mapped onto the model in docs/mappings.md, at
    toy size: `family_static`, `cascade_varying_geometry`,
    `scalars_only`, `transient_fixed_mesh`, `axis_signature`;
  - the rest of the model: rows together with callable slots, one
    callable filling two slots, a callable serving a support's
    coordinates, two supports, a row-varying field in
    an unaligned file, draws with their summaries, labels with and
    without a category table, a family with time, a derived array,
    a support of kind none, a compressed field, a stored band beside
    its field, the affine callable with a band, a file carrying
    /notes and /private, and `wide_keys`, whose 4200 row-dimensioned
    datasets are past the ceiling that one dimension scale had
    before section 21 fixed how a scale is created;
  - one file per error identifier of section 14, named `err_<id>`,
    each violating that rule and, where the rule cannot be reached
    alone, saying so in its description. A rule that can be broken in
    more than one way has a case for each, named `err_<id>_<how>`:
    E37 has two, `err_e37` and `err_e37_false`, one for each direction
    of the rule, E12 three (a statistic with no `of`, a stored band
    with no level, and a callable slot that declares a draw), and E25
    seven. 81 cases in all;
  - one file per warning identifier, named `warn_<id>`, the same way,
    18 in all.

E07 and W09 were retired on 2026-09-20 and have no case. Their
identifiers are not reused, per section 14: within a major version a
retired rule's identifier is never given to anything else.

Beside the cases there is a second subset, `vectors/hostile`, of
fifteen files that are malformed on purpose. It is described at the
end of this file.

Every file is a few tens of kilobytes. That is almost all HDF5 object
headers and dimension scales; the data in each file is a few hundred
bytes.


Regenerating and checking
-------------------------

    python generate.py                   # rewrite every case
    python generate.py --on-demand       # write only the three
                                         # large files, which are
                                         # not committed
    python check.py                      # compare the committed
                                         # corpus with a fresh run,
                                         # one line per case

Three golden files are generated on demand rather than committed,
because of their size: `cases/wide_keys` at 16 MB and the two deep
hostile files at 31 MB each. Their expected.json is committed like
every other one, so an implementation knows they exist and knows to
generate them first, and `check.py` writes them into place itself if
they are missing. Because they come from whatever libhdf5 the person
running has, they are compared structurally rather than by bytes.

`check.py` regenerates everything into a temporary directory and
compares. It compares bytes first. When the bytes differ it falls back
to the structural equality rule of section 30, which is the normative
comparison, because the HDF5 library decides the superblock and the
object header layout and byte identity across HDF5 versions must not
be tested. It then opens every case that must validate cleanly with
two netCDF-4 readers that share no code and checks that each
variable's dimension names are the link names of the scales attached
to it. It exits non-zero on any difference or failure.

The golden files are byte reproducible on one machine: object time
tracking is off everywhere, `created` and `writer` are fixed strings,
and the order in which the generator creates links, attributes and
scale attachments is fixed in its source. That order is part of the
golden bytes; reordering the statements in a builder changes them.


The schema of expected.json
---------------------------

Canonical JSON as section 30 defines it: UTF-8 with no byte order
mark, object keys sorted by their code points, no whitespace except
one closing newline, the separators "," and ":", non-ASCII written
literally. Every float is a string in the C format "%.17e", and the
three non-finite values are "nan", "inf" and "-inf"; a comparison
parses the string to float64 and requires bit equality.

  description   one or two sentences; the same string appears in
                manifest.json
  validator     {"errors": [ids], "warnings": [ids]}, each sorted and
                without duplicates. Two empty lists means the file
                must validate cleanly
  support_ids   support group name to the 64-character lower-case
                digest of section 24. It is the digest an
                implementation must compute from the stored arrays,
                which for `err_e08` is not the digest the file stores
  probes        a list of stored values, each naming its axes
  codec         callable id to the round trip of that callable's
                dictionary, in the tagged form of section 30
  evaluation    a list of worked evaluations, one per callable

A probe names one value by axis and never by axis position, so that it
means the same thing to a row-major and a column-major reader:

  slot        the HDF5 path of the dataset
  row         the row index, when the slot has a row dimension. For a
              row-varying array in an unaligned file it is the index
              within that support's own rows, which section 22 says
              is not the file's row number
  instance    the index along a `group:<k>` leading dimension
  draw        the draw index, when the slot has a draw dimension
  node        the node index, or the cell index for a cell array
  component   the component index
  index       the index along the flat connectivity axis
  value       the value as a decimal string: the "%.17e" form for a
              float64 slot and the plain decimal form for an integer
              slot

Several probes are chosen so that a transposed read is caught: the
row, the node and the component are all different and the values
differ along every axis, so a reader that took the axes by position
returns a number that is in the file but in the wrong place. The
probes of `two_supports_row_varying` do the same for the row mapping:
its two supports hold file rows 0 and 2, and row 1, and the values
are keyed to the file row, so a reader that took the leading index
for a file row number returns the wrong one.

An evaluation entry states what a callable must produce:

  callable    the callable id
  keys        the keys table it is evaluated on, key name to a list of
              "%.17e" strings, one entry per table row
  probes      the expected outputs, in the probe form above, against
              the slot each output fills. The row index is the index
              into the keys table here and not into any row the file
              stores, which matters for `affine_with_rows`, where the
              table happens to be the design the file also carries


Where the corpus drove a change to the specification
----------------------------------------------------

Building one file per rule found fourteen places where the text was
silent, self-contradictory, or describing something no file could do.
All fourteen are settled in the specification and listed in its
section 16; the corpus follows the settled text and carries no
private convention of its own. The three that shaped this
directory most:

  - expected.json has a sixth field, `evaluation`, without which the
    affine callable of section 27 could not be conformance tested;
  - the probe schema gained `instance` for the leading dimension of a
    group-varying array and `index` for the flat connectivity axis,
    neither of which had a name;
  - W08 gained a number, four times the observed width, in place of a
    tolerance that was stated nowhere. `warn_w08` declares a range
    five thousand times the observed one so that it fires under any
    reading, and no other case comes near the threshold.


Rules a file cannot break on its own
------------------------------------

Four identifiers cannot be reached by a file that breaks nothing
else. Their cases say so in their own description, and the extra
identifier is in the expected list because an implementation will and
should report it:

  E06, E37, W15   need more than one support, so W05 comes with them.
                  `err_e37_false` is the exception: one support with
                  `aligned = false` breaks E37 on its own
  E18             is a rule for a writer. A validator sees only the
                  missing public attribute, which is E39, so
                  `err_e18` expects both

`err_e18` is the clearest instance that could be built: a file that
declares a group key and names its unit of generalisation only under
`/private`, where section 29 forbids a reader to look.


The hostile subset
------------------

`vectors/hostile` holds fifteen files that are not specimens of the
format. Each is malformed in a way a reader has to survive rather than
describe, and nothing may be inferred from one about what a valid file
looks like. They exist because section 29 requires a reader to treat a
file as untrusted input, and a requirement no file tests is a
requirement nobody meets.

    hostile/<case>/case.mes        the file
    hostile/<case>/expected.json   description, required_errors,
                                   allow_extra, timeout_seconds

The contract is looser than the corpus's: a validator must report at
least the required ids, may report more, and must finish cleanly
inside ten seconds without crashing, hanging or exhausting memory.
Opening the file for its metadata alone, and any read of a slot, must
refuse with the same ids.

What they cover: attributes with an array dataspace where section 18
requires a scalar, on the root, on a key and on a slot; an unknown
filter id, and one carrying twelve client data values where some
filter interfaces have room for eight; thirty thousand nested groups
under `/keys` and under `/callables/c0`; a dangling soft link, a
cyclic soft link and an external link, each under `/keys`,
`/scalars`, `/supports` and `/callables`; a `/keys` member that is a
group and a `/supports` member that is a dataset; a slot declaring
10^12 rows, chunked and never written; a category table with a
non-UTF-8 entry and an empty one; a scale attached twice to one axis;
and a scale with CLASS but no NAME.

Two things worth knowing before you run them.

`deep_groups_keys` and `deep_groups_callables` are not committed.
They are 31 MB each, which is thirty thousand HDF5 groups in the
default layout at about a kilobyte apiece. The newer group layout
costs a seventh of that, but it writes four timestamps into the root
object header, and a file that records when it was written is not
byte reproducible. So they are generated on demand instead:

    python generate.py --on-demand

Their expected.json is committed like every other one, so an
implementation knows they exist and knows to generate them first.
Because two people's copies come from two libhdf5 versions, byte
identity is not the comparison for them; `check.py` compares them
structurally, and separately compares the length of the group chain,
which the depth cap would otherwise hide.

The second thing cost a segmentation fault to find, and section 21
now names it. Asking HDF5 for the path of a dimension scale attached
to a dataset makes it search the group hierarchy, and on a file with
thirty thousand nested groups that search runs off the stack and
takes the process with it. Dereferencing the scale and reading its
object address is safe; the link name has to come from a map built
while walking the tree, within the depth cap. A reader that resolves
scale names the obvious way passes all seventy cases and dies on
`deep_groups_keys`. `check.py` does it the safe way, and its walk is
depth capped and follows hard links only, which is what section 29
requires of anything reading a file it did not write.


What a check verifies beyond the bytes
--------------------------------------

`check.py` also reads back, for every case that is not `err_e42`, the
dataset creation property list of every dimension scale, and refuses
a file whose scales are not created with attribute creation order
tracked and indexed. That rule is section 21's E42, and it is the one
thing in this format that a property list rather than a byte position
decides. A writer that forgets it produces files that are correct
until the 4086th dataset attaches to one scale, at which point HDF5
fails the attachment after deleting the REFERENCE_LIST it was
extending, and leaves a file that every reader and every validator
still accepts. `cases/wide_keys` is the case that would not exist
without the rule, and `cases/err_e42` is the case that breaks it on
purpose.

Creation order is all it refuses on. Section 21 asks a writer for
object time tracking off as well, and the check reads that property
too, but a scale it reads as tracking prints a note beside the case
and fails nothing, which is what section 16 settles. HDF5 stores the
flag only in a version 2 object header, and it is tracking the
attribute creation
order that gives a scale one: on a version 1 header the four
timestamps are kept whatever the writer asked for, and the property
reads back as tracking on. So the only scale the flag can be read
from is one that already obeys E42, and a check of it could say
nothing that the E42 line has not said. What the second call is for
is byte reproducibility, and that is decided here by regenerating
each case and comparing it with the committed copy, which is the
first thing this file does.
