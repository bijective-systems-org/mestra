// Reading and writing `.mes` files, and the lazy access SPEC.md
// section 29 requires of a reader.
//
// A file is untrusted input.  A reader of an open format opens files
// it did not write, and a file is a program's input and not its
// instructions, so this one is written to answer a crafted file rather
// than crash on it, and every limit is a stated number and not
// whatever the stack happened to allow.  What it refuses, and says so:
//
//   - an attribute whose dataspace declares more elements than the
//     encoding of section 18 implies.  Every buffer is sized from the
//     count the file declares, never from the encoding, and a count
//     past a stated maximum is left unread and reported;
//   - a dictionary, a group tree or a dump nested deeper than
//     `kMaxDictDepth` (64).  A stack overflow cannot be caught, so the
//     limit is enforced before the descent, and a dictionary accounts
//     its own nesting as values go in (value.hpp), which makes a
//     deeper one impossible to hold rather than merely unsafe to walk;
//   - an element count or a byte length whose product overflows, or
//     passes the stated maximum of 2^31 elements for an eager read.  A
//     dataset that declares a trillion elements and stores none is
//     refused before anything is allocated.  `read_slot_rows` and the
//     other lazy paths are not subject to that maximum, because they
//     never materialise the whole dataset, so the same file can be
//     readable one way and E41 the other (section 29);
//   - a link in the public tree that is not a hard link: a soft link,
//     whether it resolves, dangles or loops, and an external link,
//     which is never opened, because following one would let a file
//     name another file on the machine and have this reader open it.
//     The link's own type is read before anything is opened, so
//     nothing under a link of either kind is ever asked about.  That
//     is E40;
//   - a member of a container group that is neither the kind that
//     belongs there, a nesting past the cap, a malformed object, or an
//     eager read above the maximum element count.  Each is E41,
//     reported with its path while the pass goes on, so that one
//     broken object does not hide the rest of the file.
//
// `/private` is copied whole and so is held whole: one nested deeper
// than 64, holding more than 65,536 objects or more than one gibibyte
// is E41 rather than a silent truncation or an allocation without
// bound.
#ifndef MESTRA_IO_HPP
#define MESTRA_IO_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mestra/dataset.hpp"

namespace mestra {

// Every failure this library raises.  `rule` carries the identifier of
// section 14 when one applies, so that a caller can say which rule was
// broken and not only that something went wrong; it is empty when no
// rule covers the failure (a missing file, for example).
class Error : public std::runtime_error {
 public:
  Error(std::string rule, const std::string& message);
  const std::string& rule() const { return rule_; }

 private:
  std::string rule_;
};

// What `read` is allowed to accept.
struct ReadOptions {
  // Strict, which is the default, refuses a file that breaks a
  // structural rule -- E01, E16, E19, E25, E26, E29, E30, E40, E41 --
  // and a fault no rule of section 14 covers.  A semantic fault (a
  // missing unit, a split that straddles a generalisation unit) never
  // stops a read, so that `info` and a reader still work on the files
  // a user most needs to look at.
  //
  // `strict = false` reads what it can and lists what it refused in
  // `Dataset::not_read`.
  bool strict = true;
};

// Whether a rule identifier is one of the nine structural rules a
// strict read refuses on, or is empty, which is a fault no rule of
// section 14 covers and is refused the same way.  A tool that opens
// a file for its metadata gates on this and on nothing else, so that
// it works on the files a user most needs to look at.
bool structural_rule(const std::string& id);

// Reads a whole file.  Throws Error carrying the first rule
// identifier, and every structural finding in its message, when a
// strict read meets one; and Error with the rule identifier for the
// layout faults it cannot read past whatever the options say.
//
// A read costs one validation pass over the file.  `read_header`,
// `read_slot` and `read_slot_rows` are the cheap paths and do not
// validate.
Dataset read(const std::string& path,
             const ReadOptions& options = ReadOptions());

// Opens a file and reports what section 29 asks for without reading an
// array: the row count, the keys with their roles and bounds, the
// supports with their ids, and every slot with its attributes and its
// shape, having read attributes and dataspaces only.  Nothing that is
// stored in a dataset comes back: array values, category table
// entries, cell arrays, /row_support and a callable's dictionary are
// all empty, and `Support::computed_support_id` is meaningless on the
// result.  Use `read` for a whole file.
Dataset read_header(const std::string& path);

// Lazy access.  Reads one slot for the half-open row range
// [row_begin, row_end) without reading any other slot and without
// reading the rows outside the range.  `slot` is the HDF5 path, for
// example "/supports/s0/node_arrays/pressure".  The array comes back
// with its dimension names, its leading axis cut to the range.
Array read_slot_rows(const std::string& path, const std::string& slot,
                     std::size_t row_begin, std::size_t row_end);

// Reads one whole dataset by its HDF5 path, with its dimension names.
// Works on a file that does not validate, which is what the
// conformance corpus needs.
Array read_slot(const std::string& path, const std::string& slot);

// What `write` is allowed to do.
struct WriteOptions {
  // Validate before leaving a file behind.  A dataset that breaks a
  // rule of section 14 is refused, with the findings, and `path` is
  // not touched: the file is built beside it and only moved into
  // place once it validates.  `check = false` writes it anyway, for
  // the one caller who wants a file the validator rejects -- a test
  // of a validator, mostly.  The shape checks a builder makes are not
  // part of this and are not skipped: `varies` disagreeing with the
  // shape a slot was built with is refused either way, because such a
  // file is not something a caller can have meant.
  bool check = true;
};

// Writes a conforming file.  Existing content at `path` is replaced.
// Throws Error carrying the first rule identifier, and every finding
// in its message, when the dataset does not validate.
void write(const Dataset& d, const std::string& path,
           const WriteOptions& options = WriteOptions());

// What `append_rows` is allowed to do.
struct AppendOptions {
  // A key of role id whose values name the rows.  A row of the
  // appended dataset whose value equals an existing row's is written
  // over that row instead of after the last one, so a producer that
  // computes a row again keeps one row per id.  Empty appends every
  // row.
  std::string replace_by;
  // Validate the grown file before it replaces the original, as
  // `write` does, and refuse with the findings when it does not
  // validate.  `check = false` moves it into place anyway.
  bool check = true;
};

// Grows the file at `path` by the rows of `rows` without rewriting the
// rows it already holds: the way a producer that computes one row at a
// time keeps one file, and the way a file too large to hold in memory
// is still written.  Returns the file's row count afterwards.
//
// `rows` is built with the same builders as a whole file and must have
// the file's structure: the same keys with the same roles, units,
// category tables and dtypes; the same scalars; the same supports with
// the same connectivity and the same arrays with the same attributes;
// and the same values in every array that does not vary by row.  A
// category id is mapped by its entry's name, so a table built from the
// appended rows alone maps onto the file's table; an entry the file's
// table does not hold is refused.  The file's key bounds widen to cover
// the new values, and a string id column grows to hold a longer id.
// When `rows` carries notes they replace the file's;
// the file's `format`, `writer`, `created` and its `/private` group
// stay as they are.  The result is the file one write of every row
// would have produced, in content though not in bytes: the chunk
// shapes are the ones the file was created with, and the validator
// says so (W12) when they are no longer the default for the row count.
//
// The file is grown beside itself and moved into place once it is
// written and validated, so a failure of any kind, a structural
// difference included, leaves the original as it was.  Throws Error
// with the first rule identifier and every finding when the grown file
// does not validate, and Error with an empty rule and the difference
// when `rows` does not have the file's structure.  An unaligned file
// (several supports, section 22) is not grown by this version and is
// refused the same way.
std::int64_t append_rows(const Dataset& rows, const std::string& path,
                         const AppendOptions& options = AppendOptions());

// The digest of section 24 for one support of a file, computed from
// the stored arrays rather than read from the `support_id` attribute.
std::string support_id_of(const std::string& path,
                          const std::string& support);

// Reads one callable's dictionary.  Works without a registered type.
Dict read_dict(const std::string& path, const std::string& callable_id);

// Writes a dictionary into a file as the only callable, for a codec
// round trip.  The file is a minimal valid container.
void write_dict(const Dict& d, const std::string& type,
                const std::string& callable_id, const std::string& path);

}  // namespace mestra

#endif  // MESTRA_IO_HPP
