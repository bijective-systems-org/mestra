// The validator of SPEC.md section 14.  Every finding carries the rule
// identifier and nothing else identifies it: the identifiers are
// stable within a major version, a rule is never renumbered, and a
// retired identifier (E07, W09) is never emitted.
//
// There is one finding per rule per object, and a rule that could fire
// once per row -- W02, W03, W04 -- reports once with the count and the
// first three row indices in its message (conventions section 5).
//
// The pass reports rather than fails.  One object it cannot read is
// recorded against its path and the rest of the file is still checked,
// because a validator that stops at the first fault tells a caller
// almost nothing; and a fault that no rule of section 14 covers is
// still reported, so that a file is never called clean because the
// thing wrong with it has no identifier.  `mestra-cli validate` prints
// such a fault as `! <path>: <why>`.
#ifndef MESTRA_VALIDATE_HPP
#define MESTRA_VALIDATE_HPP

#include <string>
#include <vector>

namespace mestra {

// One rule broken, at one place.
struct Finding {
  std::string id;       // "E16", "W08", ...
  std::string where;    // the HDF5 path the finding is about
  std::string message;  // one line, plain language
};

struct Report {
  std::vector<Finding> errors;
  std::vector<Finding> warnings;

  bool ok() const { return errors.empty(); }
  // The identifiers, sorted and without duplicates, which is the form
  // the conformance corpus compares (section 30).
  std::vector<std::string> error_ids() const;
  std::vector<std::string> warning_ids() const;
};

// Validates a file.  Never throws for a fault in the file: a file this
// reader cannot open at all comes back as E01 or E17 with a message.
Report validate(const std::string& path);

// The same pass, restricted to what a metadata open sees
// (docs/api-conventions.md section 7): attributes, dataspaces, link
// types and dimension-scale structure, plus a category table in full
// and `/row_support`, which are the two datasets that are not slots
// and that a rule is decided from.  No slot and no dataset inside a
// callable's dictionary is read.  This is what `read_header` refuses
// on, so the metadata open, the whole read and `info` name the same
// rule for the same file.  An object whose declared element count is
// past the maximum an eager read takes is E41 here too, because that
// is a fact of its dataspace (section 29).
Report validate_metadata(const std::string& path);

// The verdict W10 is decided by: true when the string matches the
// units grammar of section 32, whatever names it uses.  It is a
// grammar and deliberately not a units database.
bool units_parse(const std::string& units);

}  // namespace mestra

#endif  // MESTRA_VALIDATE_HPP
