# What is portable

Mestra's public schema, readers and conformance corpus are open. The
repository's code is Apache-2.0; the specification text is CC-BY-4.0.
A dataset of stored values can be read without TopoLink or romlib. A
model file uses the same public supports, keys and output slots, with
a named callable supplying the values.

Reading a callable's interface and copying its opaque dictionary do
not require its implementation. Evaluating it does. The software that
owns a callable type defines its execution and licensing requirements.
Mestra does not supply an interpreter for arbitrary model payloads.

| Capability shipped here | Python | C++ | MATLAB | Julia |
|---|---|---|---|---|
| Read/write/validate public stored data | Yes | Yes | Yes, ASCII strings | Yes |
| Preserve supported opaque/private content on read-write | Yes | Yes | Yes, ASCII strings | Yes |
| Fixed-length UTF-8 strings beyond ASCII | Yes | Yes | Refused | Yes |
| Execute the built-in `affine` callable | Yes | Yes | Yes | Yes |
| Execute romlib `composed-surrogate` | With the separate romlib runtime | No shipped runtime | No shipped runtime | No shipped runtime |
| Read evaluated romlib output as stored data | Yes | Yes | Yes, ASCII strings | Yes |
| Incremental `append_rows` API | No | Yes | No | No |

Unsupported or lossy opaque content is refused on rewrite; preservation
does not mean every HDF5 construct is supported. MATLAB's low-level HDF5
interface cannot faithfully transfer non-ASCII fixed strings, so its
reader and writer explicitly refuse them. Passing the shared corpus
(whose strings are ASCII) does not establish full Unicode support.
Convenience post-processing APIs also differ; consult each language's
README. These implementation limits do not change a file's meaning.

## Damaged files

Every reader here treats a file as untrusted input, and the hostile
subset of the corpus holds it to that: a finding or a refusal that
names the rule, never a crash or a hang. That holds for every
construction anybody has thought to build. It does not hold for
every byte: the HDF5 library parses a file's metadata before any code
of this repository runs, and on a crafted change to a single byte of
an otherwise valid file it can fault or spin inside its own parser
instead of returning an error. Measured on 2026-09-27 with
`python/tests/test_mutation.py`: of 424 damaged copies of two corpus
cases (truncations, single-bit flips, zeroed runs), 420 were answered
by rule and 4 were lost in libhdf5 -- three faults and one hang, in
attribute and dimension-scale reads and once at file close -- and the
C++ reader, on a different libhdf5, lost 2 of the same set. Reading
the damaged attribute raw, without the dimension-scale API, faults
the same way, so there is no code path here that avoids it.

What this means for a caller: the process boundary is the defence. The
command line (`mestra validate`, `mestra info`, `mestra read`) is one,
and a program that opens files nobody vouches for should do the same
-- run the read in a child it can kill, as the hostile tests do. The
truncation sweep is not affected: a truncated file is refused by rule
(E01) at every length tried. Every language here links its own
libhdf5, so which bytes are fatal differs between them, and a file
that one reader answers may lose another.

## Replacing files

Checked writers build beside the destination and publish only after
successful construction and validation. Failure before publication
leaves an existing destination intact. A destination that cannot take a
file at all (an empty path, a directory, a missing or unwritable parent)
is refused by name before anything is staged. Publication uses a same-filesystem
rename/atomic move; an unsupported replacement is an error, without a
delete-and-copy fallback. This is not a guarantee of durability after
power loss, nor does it coordinate concurrent writers. Producers must
serialize updates to one destination. C++ reserves its sibling suffixes
`.mestra-writing` and `.mestra-appending` for staging; do not use those
as separate data files. Python and Julia allocate distinct temporary
names, and MATLAB uses a temporary sibling name.

MATLAB checked writes require the JVM for `java.nio.file.Files.move`
with `ATOMIC_MOVE`; the filesystem must support it. Java's contract is
documented in [Files.move](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/nio/file/Files.html#move(java.nio.file.Path,java.nio.file.Path,java.nio.file.CopyOption...)).
Reading and validation do not require this publication step. Python
uses `os.replace`; Julia calls its runtime's native rename primitive
without `mv`'s copy/delete fallback. On platforms that prevent replacing
an open file, close source readers before publishing to the same path.
`check=false` remains the escape hatch for deliberately invalid fixtures;
portable failure-preservation guarantees apply to checked writes.

## Who establishes correspondence

For a mesh, `support_id` hashes node count and connectivity, deliberately
excluding varying coordinates. It identifies the structure under the
specification's definition of a support. It cannot establish that two
independent producers gave node k the same physical meaning. Producers
must establish that correspondence before declaring aligned rows;
consumers compare support IDs when composing known corresponding data.
It is not a geometry fingerprint for caches of geometry-dependent work,
such as Laplace–Beltrami eigenpairs.

## Ownership and modelling policy

A file is a data contract, not a shared mutable workspace. A producer
may maintain its own result file; downstream enrichment should normally
write a separate derived artifact. The tool that owns a file defines
its updates and keeps its lineage in private data or its own records.

Status is per observation. A modelling workflow must deliberately ask
to include a status other than `converged` (SPEC section 3). A mesh
smoother's `iteration_limit` describes that stage, not necessarily the
quality of a subsequently computed physical field. Producers and model
adapters must state their eligibility policy while preserving the
upstream outcome. Reading a file or passing structural validation does
not select its training rows.
