// The validator of SPEC.md section 14.  It reads the file through the
// low-level layer rather than through mestra::read, because most of
// what it has to catch is a file a reader cannot parse.
#include "mestra/validate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "codec.hpp"
#include "h5.hpp"
#include "layout.hpp"
#include "mestra/dataset.hpp"
#include "mestra/io.hpp"
#include "names.hpp"

namespace mestra {
namespace {

using internal::AttrType;
using internal::DsetInfo;
using internal::File;
using internal::Member;
using internal::RawAttr;

// Passed as the chunk row count when W12 does not apply.
const std::size_t kNoChunkCheck = std::numeric_limits<std::size_t>::max();

const RawAttr* find(const std::vector<RawAttr>& attrs,
                    const std::string& name) {
  for (const RawAttr& a : attrs) {
    if (a.name == name) return &a;
  }
  return nullptr;
}

std::string text_of(const std::vector<RawAttr>& attrs,
                    const std::string& name, bool* present = nullptr) {
  const RawAttr* a = find(attrs, name);
  if (present != nullptr) *present = a != nullptr;
  if (a == nullptr || a->value.kind() != AttrValue::Kind::Str) return "";
  return a->value.as_text();
}

// The expected encoding of every attribute this specification names
// (section 18).
enum class Enc { None, Text, Integer, Real, Boolean };

Enc expected_encoding(const std::string& name) {
  if (name == "aligned" || name == "recomputed") return Enc::Boolean;
  if (name == "n_nodes" || name == "n_cells" || name == "components") {
    return Enc::Integer;
  }
  if (name == "lower" || name == "upper" || name == "quantile" ||
      name == "level") {
    return Enc::Real;
  }
  if (name == "format" || name == "writer" || name == "created" ||
      name == "generalisation_group" || name == "role" || name == "units" ||
      name == "category" || name == "trajectory_group" || name == "parent" ||
      name == "source" || name == "output" || name == "statistic" ||
      name == "of" || name == "kind" || name == "support_id" ||
      name == "varies" || name == "derived_from" || name == "recipe" ||
      name == "reference" || name == "type" || name == "repr" ||
      name == "method") {
    return Enc::Text;
  }
  return Enc::None;
}

bool is_null_sentinel_bytes(const std::string& raw) {
  return raw.size() == 5 && raw[0] == '\0' && raw.compare(1, 4, "null") == 0;
}

// A rule that could fire once per row reports once, with the count
// and the first three rows (conventions section 5).  A validator that
// prints 1,800 copies of one sentence is not telling a reader
// anything the count would not, and a reader still needs somewhere to
// start looking, which is what the three indices are for.
class PerRow {
 public:
  explicit PerRow(const char* noun = "row") : noun_(noun) {}

  // The count is of values that broke the rule; the three indices are
  // distinct, because three copies of the same row number tell a
  // reader nothing about where to look next.
  void hit(std::size_t at) {
    ++count_;
    if (first_.size() < 3 && (first_.empty() || first_.back() != at)) {
      first_.push_back(at);
    }
  }
  bool any() const { return count_ != 0; }
  std::size_t count() const { return count_; }

  // "<n> <what>; rows 3, 7, 9"
  std::string message(const std::string& what) const {
    std::string out = internal::format_i64(
                          static_cast<std::int64_t>(count_)) + " " + what;
    if (first_.empty()) return out;
    out += "; ";
    out += noun_;
    if (first_.size() > 1) out += "s";
    out += " ";
    for (std::size_t i = 0; i < first_.size(); ++i) {
      if (i != 0) out += ", ";
      out += internal::format_i64(static_cast<std::int64_t>(first_[i]));
    }
    return out;
  }

 private:
  const char* noun_;
  std::size_t count_ = 0;
  std::vector<std::size_t> first_;
};

// An Error's message without the identifier it carries.  A finding is
// printed as "<id> <path>: <message>" (conventions section 5), so the
// identifier belongs in the first field and not twice.
std::string without_id(const Error& e) {
  const std::string text = e.what();
  const std::string prefix = e.rule() + ": ";
  if (!e.rule().empty() && text.size() >= prefix.size() &&
      text.compare(0, prefix.size(), prefix) == 0) {
    return text.substr(prefix.size());
  }
  return text;
}

// What a slot's axes look like, taken from the dimension scales.
struct Axes {
  std::vector<std::string> logical;   // one per axis, "" when unknown
  std::vector<std::size_t> extent;
  int index_of(const std::string& name) const {
    for (std::size_t i = 0; i < logical.size(); ++i) {
      if (logical[i] == name) return static_cast<int>(i);
    }
    return -1;
  }
};

class Validator {
 public:
  Validator(File& f, Report* r, bool metadata_only = false)
      : f_(f), r_(r), metadata_only_(metadata_only) {}

  void run();

 private:
  void error(const std::string& id, const std::string& where,
             const std::string& message) {
    r_->errors.push_back({id, where, message});
  }

  // One object at a time.  A file is untrusted input and a validator
  // that stops at the first object it cannot read tells a caller
  // almost nothing about the rest of the file, so a failure is
  // recorded against the path it happened at and the pass goes on.
  template <typename Body>
  void guarded(const std::string& path, Body body) {
    try {
      body();
    } catch (const Error& e) {
      error(e.rule().empty() ? std::string("E41") : e.rule(), path,
            without_id(e));
    } catch (const std::exception& e) {
      error("E41", path, e.what());
    }
  }

  // True when a member is an object this reader will look at: a hard
  // link to a group or a dataset.  Anything else is said out loud and
  // left alone -- a soft link is not resolved and an external link is
  // never opened, because opening one would make a crafted file read
  // another file on this machine.
  bool usable(const std::string& parent, const Member& m) {
    const std::string path =
        (parent == "/" ? std::string("/") : parent + "/") + m.name;
    if (m.kind != internal::LinkKind::Hard) {
      error("E40", path,
            std::string("this name is ") + internal::link_kind_name(m.kind) +
                "; a reader never follows one (section 29)");
      return false;
    }
    if (!m.is_group && !m.is_dataset) {
      error("E41", path,
            "this name is neither a group nor a dataset, so there is "
            "nothing here this reader can read");
      return false;
    }
    return true;
  }
  void warn(const std::string& id, const std::string& where,
            const std::string& message) {
    r_->warnings.push_back({id, where, message});
  }

  // E40 for every member of a public group whose members this pass
  // does not otherwise visit: /notes, which holds attributes, and a
  // slot served by a callable, which is a group holding none.  A link
  // there is in the public tree as much as one under /keys is.
  void links_in(const std::string& group) {
    for (const Member& m : f_.members(group)) {
      if (m.kind != internal::LinkKind::Hard) usable(group, m);
    }
  }

  // W10: a units string the parser cannot read.  A `units` that is not
  // a string at all is E19, drawn by check_attribute_encodings, and is
  // not also a string that does not parse.
  void units_warning(const std::string& path,
                     const std::vector<RawAttr>& attrs) {
    const RawAttr* a = find(attrs, "units");
    if (a == nullptr || a->value.kind() != AttrValue::Kind::Str) return;
    const std::string units = a->value.as_text();
    if (!units_parse(units)) {
      warn("W10", path, "the units string \"" + units + "\" does not parse");
    }
  }

  // E12: a statistic with what it needs (section 9), and on a slot a
  // callable serves, one a callable produces (section 10).
  void check_statistic(const std::string& path,
                       const std::vector<RawAttr>& attrs) {
    // A statistic that is not text is E19 and names nothing to check.
    const RawAttr* given = find(attrs, "statistic");
    if (given == nullptr || given->value.kind() != AttrValue::Kind::Str) {
      return;
    }
    const std::string statistic = given->value.as_text();
    // A word section 9 does not name is E02. What a statistic needs
    // follows from its word, so E12 has nothing to decide for it.
    if (statistic != "value" && statistic != "mean" && statistic != "band" &&
        statistic != "std" && statistic != "quantile" &&
        statistic != "draw") {
      error("E02", path,
            "\"" + statistic + "\" is not a statistic of section 9");
      return;
    }
    const std::string source = text_of(attrs, "source");
    const bool served = !internal::callable_of_source(source).empty();
    if (statistic == "quantile" && find(attrs, "quantile") == nullptr) {
      error("E12", path, "a quantile statistic with no `quantile`");
    }
    if (statistic != "value" && statistic != "draw" &&
        find(attrs, "of") == nullptr) {
      error("E12", path, "a statistic other than value or draw with no `of`");
    }
    if (served && (statistic == "std" || statistic == "quantile" ||
                   statistic == "draw")) {
      error("E12", path,
            "a callable returns a mean and at most a band (section 10), so "
            "a slot it serves is value, mean or band, not " + statistic);
    }
    if (statistic == "band" && !served) {
      const RawAttr* level = find(attrs, "level");
      if (level == nullptr) {
        error("E12", path, "a band states the coverage it claims with `level`");
      } else if (level->value.kind() == AttrValue::Kind::Float) {
        const double v = level->value.as_float();
        if (!(v > 0.0 && v < 1.0)) {
          error("E12", path,
                "`level` is a coverage fraction in (0, 1), and this is " +
                    internal::format_f64(v));
        }
      }
      if (find(attrs, "method") == nullptr) {
        error("E12", path, "a band says how it was made with `method`");
      }
    }
  }

  File& f_;
  Report* r_;
  // True for the pass a metadata open makes, which reads no slot and
  // no dataset inside a callable's dictionary (conventions section 7).
  bool metadata_only_ = false;
  std::int64_t n_rows_ = 0;
  bool aligned_ = true;
  std::vector<std::string> support_names_;
  std::map<std::string, std::vector<std::string>> category_tables_;
  std::set<std::string> callable_ids_;
  std::vector<std::int64_t> row_support_;
  bool has_row_support_ = false;
  bool has_group_key_ = false;
  // Objects whose eager read this reader refused because the file
  // declares more of them than the stated maximum (section 29).  What
  // such an object holds was never seen, so no later rule is decided
  // from it.
  std::set<std::string> unread_;

  // The validator reports a fault rather than failing on one, so a
  // read that the library refuses -- a dtype the rule above has
  // already reported, a dataset HDF5 will not convert -- comes back
  // empty and the walk goes on.
  //
  // One refusal is not like those.  Section 29: "An eager read refuses
  // a dataset whose declared element count is above a maximum the
  // reader states, with E41."  That is a rule about this object and
  // not about what it holds, so it is reported here against the path
  // it happened at, and the object is remembered, because everything
  // downstream of it would otherwise be decided from contents this
  // reader never saw: a category table above the maximum read as an
  // empty table puts every category id outside it, which is E10 said
  // of a file whose fault is E41.
  // What a metadata open reads, beside attributes, dataspaces, link
  // types and dimension-scale structure: a category table in full and
  // /row_support.  Those two are the datasets that are not slots --
  // E16 names "a slot ... , a key column, or /row_support" as three
  // different things -- and a rule is decided from each of them: E26
  // and the entry count from the table, and the per-support row count
  // of section 22 from /row_support, which E16 is decided from on an
  // unaligned file.  A slot and a dataset inside a callable's
  // dictionary wait for the read (conventions section 7).
  bool may_read(const std::string& path) const {
    if (!metadata_only_) return true;
    return path == "/row_support" ||
           path.compare(0, 12, "/categories/") == 0;
  }

  template <typename Read>
  auto refusable(const std::string& path, Read read) -> decltype(read()) {
    try {
      if (!may_read(path)) {
        // Whether an eager read of this object would be refused is a
        // fact of its dataspace and not of its contents, so a
        // metadata open decides it without reading anything, and the
        // two passes name E41 on the same files (section 29).
        f_.eager_element_count(path);
        return {};
      }
      return read();
    } catch (const Error& e) {
      // One finding per rule per object, however often the pass asks
      // this object for its contents.
      if (e.rule() == "E41" && unread_.insert(path).second) {
        error("E41", path,
              "this object declares more than an eager read of this reader "
              "takes, which is " +
                  internal::format_i64(static_cast<std::int64_t>(
                      internal::kMaxDatasetElements)) +
                  " elements (section 29); nothing of it was read, so no "
                  "rule below is decided from what it holds. A row-range "
                  "read of it is not subject to that maximum");
      }
      return {};
    } catch (const std::exception&) {
      return {};
    }
  }
  // True when the object at `path` is one of those: no rule may be
  // decided from what it holds.
  bool unread(const std::string& path) const {
    return unread_.count(path) != 0;
  }

  // True when this pass has a dataset's contents in hand.  A rule
  // decided from contents this pass does not have is not decided at
  // all, rather than decided from an empty array: a metadata open
  // has a category table and /row_support and nothing else, and
  // either pass may have been refused an object above the stated
  // maximum element count.  Without this the open would say E23 of a
  // mesh whose cell offsets it never read and E08 of every support in
  // a file that is not wrong about anything.
  bool have_contents(const std::string& path) const {
    return may_read(path) && !unread(path);
  }

  std::vector<double> reals(const std::string& path) {
    return refusable(path, [&] { return f_.read_f64(path); });
  }
  std::vector<std::int64_t> integers(const std::string& path) {
    return refusable(path, [&] { return f_.read_i64(path); });
  }
  std::vector<std::string> texts(const std::string& path) {
    return refusable(path, [&] { return f_.read_strings(path); });
  }
  std::vector<std::string> raw_texts(const std::string& path) {
    return refusable(path, [&] { return f_.read_strings_raw(path); });
  }

  Axes axes_of(const std::string& path, const DsetInfo& info,
               bool report_e25);
  void check_attribute_encodings(const std::string& path,
                                 const std::vector<RawAttr>& attrs);
  void check_string_dataset(const std::string& path, const DsetInfo& info,
                            bool warn_oversize);
  void check_dataset_storage(const std::string& path, const DsetInfo& info,
                             bool is_row_dataset, std::size_t chunk_rows);
  void check_unlimited(const std::string& path, const DsetInfo& info);
  void unknown_dataset(const std::string& path, const DsetInfo& info);

  void root();
  void categories();
  void keys();
  void scalars();
  void row_support_dataset();
  void supports();
  void callables();
  void scales();
  void private_group();
  void check_dict(const std::string& path, bool top_level, int depth);
  void slot(const std::string& path, const std::string& support_kind,
            std::int64_t n_nodes, std::int64_t n_cells,
            std::size_t support_index, bool is_coordinates);
};

// --- helpers ----------------------------------------------------------

Axes Validator::axes_of(const std::string& path, const DsetInfo& info,
                        bool report_e25) {
  Axes a;
  for (std::size_t i = 0; i < info.shape.size(); ++i) {
    a.extent.push_back(static_cast<std::size_t>(info.shape[i]));
    const std::vector<std::string>& attached = info.scales[i];
    if (attached.size() != 1) {
      if (report_e25) {
        const std::string axis =
            internal::format_i64(static_cast<std::int64_t>(i));
        error("E25", path,
              attached.empty()
                  ? "axis " + axis + " has no dimension scale attached"
                  : "axis " + axis +
                        " has more than one dimension scale attached");
      }
      a.logical.push_back(std::string());
    } else if (attached.front().empty()) {
      // One reference, to an object this file gives no name: a scale
      // whose link was deleted after it was attached.  The axis has
      // no dimension a reader can name.
      if (report_e25) {
        error("E25", path,
              "axis " + internal::format_i64(static_cast<std::int64_t>(i)) +
                  " is attached to an object that is not a dimension "
                  "scale of this file");
      }
      a.logical.push_back(std::string());
    } else {
      a.logical.push_back(internal::logical_dim(attached.front()));
    }
  }
  return a;
}

void Validator::check_attribute_encodings(
    const std::string& path, const std::vector<RawAttr>& attrs) {
  for (const RawAttr& a : attrs) {
    if (internal::machinery_attribute(a.name)) continue;
    if (!internal::legal_netcdf_name(a.name)) {
      error("E33", path,
            "the attribute name \"" + a.name +
                "\" is not a legal netCDF-4 name");
    }
    if (a.type.klass == H5T_STRING) {
      if (a.type.variable_length) {
        error("E19", path,
              "the attribute \"" + a.name +
                  "\" is a variable-length string, which section 18 forbids "
                  "anywhere in the file");
        continue;
      }
      if (!is_null_sentinel_bytes(a.raw_bytes)) {
        if (internal::embedded_nul(a.raw_bytes)) {
          error("E26", path,
                "the attribute \"" + a.name +
                    "\" holds a NUL byte outside its trailing padding");
        } else if (!internal::valid_utf8(internal::strip_nul(a.raw_bytes))) {
          error("E26", path,
                "the attribute \"" + a.name + "\" is not valid UTF-8");
        }
      }
    }
    const Enc want = expected_encoding(a.name);
    if (want == Enc::None) continue;
    bool ok = true;
    switch (want) {
      case Enc::Text: ok = internal::is_spec_string(a.type); break;
      case Enc::Integer: ok = internal::is_spec_int64(a.type); break;
      case Enc::Real:
        ok = internal::is_spec_float64(a.type);
        // Section 18: an attribute that declares a bound, a level or
        // a quantile must be finite.  The four names of Enc::Real are
        // exactly those.
        if (ok && a.value.kind() == AttrValue::Kind::Float &&
            !std::isfinite(a.value.as_float())) {
          error("E19", path,
                "the attribute \"" + a.name +
                    "\" declares a bound, a level or a quantile and must "
                    "be finite");
        }
        break;
      case Enc::Boolean:
        ok = internal::is_spec_bool(a.type);
        if (ok && !a.raw_bytes.empty()) {
          const unsigned char v =
              static_cast<unsigned char>(a.raw_bytes[0]);
          ok = v == 0 || v == 1;
        }
        break;
      case Enc::None: break;
    }
    if (!ok) {
      error("E19", path,
            "the attribute \"" + a.name +
                "\" is not stored in the encoding section 18 requires");
    }
  }
}

void Validator::check_string_dataset(const std::string& path,
                                     const DsetInfo& info,
                                     bool warn_oversize) {
  if (info.type.klass != H5T_STRING) return;
  if (info.type.variable_length) {
    error("E19", path,
          "a variable-length string dataset, which section 18 forbids");
    return;
  }
  const std::vector<std::string> raw = raw_texts(path);
  std::size_t longest = 0;
  for (const std::string& s : raw) {
    if (internal::embedded_nul(s)) {
      error("E26", path,
            "an element holds a NUL byte outside its trailing padding");
    } else if (!internal::valid_utf8(internal::strip_nul(s))) {
      error("E26", path, "an element is not valid UTF-8");
    }
    longest = std::max(longest, internal::strip_nul(s).size());
  }
  if (longest == 0) longest = 1;
  if (warn_oversize && !raw.empty() && info.type.size > longest) {
    warn("W13", path,
         "the fixed-length string size is larger than the longest element "
         "needs");
  }
}

// E43.  `row` is the only dimension that may be
// unlimited, file-level or support-local, because every other
// dimension carries its length in its name and one that grows makes
// its own name false.  The check is on the dataspace: an axis whose
// maximum extent is H5S_UNLIMITED is legal only where the scale
// attached to it is `row`, or where the object is the `row` scale
// itself.  Section 25's zero-length dictionary axis is the one
// exception, because HDF5 has no other legal way to write it.
void Validator::check_unlimited(const std::string& path,
                                const DsetInfo& info) {
  const bool in_dictionary = path.compare(0, 11, "/callables/") == 0;
  for (std::size_t axis = 0; axis < info.maxshape.size(); ++axis) {
    if (info.maxshape[axis] != H5S_UNLIMITED) continue;
    if (in_dictionary && axis < info.shape.size() &&
        info.shape[axis] == 0) {
      continue;
    }
    bool is_row = false;
    if (info.is_scale) {
      is_row = internal::basename(path) == "row";
    } else if (axis < info.scales.size()) {
      for (const std::string& name : info.scales[axis]) {
        if (name == "row") is_row = true;
      }
    }
    if (is_row) continue;
    error("E43", path,
          "axis " + internal::format_i64(static_cast<std::int64_t>(axis)) +
              " is unlimited and is not `row`");
  }
}

void Validator::check_dataset_storage(const std::string& path,
                                      const DsetInfo& info,
                                      bool is_row_dataset,
                                      std::size_t chunk_rows) {
  check_unlimited(path, info);
  for (const auto& filter : info.filters) {
    const int id = filter.first;
    if (id == H5Z_FILTER_SHUFFLE) continue;
    if (id == H5Z_FILTER_DEFLATE) {
      const unsigned level = filter.second.empty() ? 0 : filter.second[0];
      if (level >= 1 && level <= 9) continue;
    }
    error("E29", path,
          "a filter other than gzip at level 1 to 9 and shuffle");
  }
  if (is_row_dataset && !info.chunked) {
    error("E27", path, "a row-dimensioned dataset that is not chunked");
  }
  if (chunk_rows == kNoChunkCheck || !is_row_dataset || !info.chunked) {
    return;
  }
  if (info.shape.empty() || info.chunk.empty()) return;
  std::size_t item = info.type.size;
  if (item == 0) item = 1;
  std::vector<std::size_t> rest(info.shape.begin() + 1, info.shape.end());
  std::vector<std::size_t> want;
  want.push_back(default_chunk_rows(item, rest, chunk_rows));
  for (const std::size_t e : rest) want.push_back(e);
  const std::vector<std::size_t> have(info.chunk.begin(), info.chunk.end());
  if (have != want) {
    warn("W12", path, "a chunk shape that is not the default of section 23");
  }
}

// --- passes -----------------------------------------------------------

void Validator::run() {
  for (const Member& m : f_.members("/supports")) {
    if (m.kind == internal::LinkKind::Hard && m.is_group) {
      support_names_.push_back(m.name);
    }
  }
  // Section 22: the support order is the group names sorted by their
  // UTF-8 bytes, which is the one ordering every language produces
  // identically.
  std::sort(support_names_.begin(), support_names_.end(), bytes_less);
  for (const Member& m : f_.members("/callables")) {
    if (m.kind == internal::LinkKind::Hard && m.is_group) {
      callable_ids_.insert(m.name);
    }
  }
  root();
  categories();
  keys();
  row_support_dataset();
  scalars();
  supports();
  callables();
  scales();
  private_group();
}

void Validator::scales() {
  // Section 21 gives a dimension scale a NAME as well as a CLASS.  A
  // scale without one is not the layout netCDF-C writes, and nothing
  // else looks at a scale dataset, so the check lives here.  The walk
  // is bounded and follows hard links only, like every other walk
  // over a file this reader did not write.
  std::vector<std::pair<std::string, int>> todo;
  todo.emplace_back("/", 0);
  while (!todo.empty()) {
    const std::pair<std::string, int> here = todo.back();
    todo.pop_back();
    if (here.second > internal::kMaxGroupDepth) continue;
    // /private is not checked (section 14) and neither is a group
    // this version does not know.
    for (const Member& m : f_.members(here.first)) {
      if (m.kind != internal::LinkKind::Hard) continue;
      const std::string child =
          (here.first == "/" ? std::string("/") : here.first + "/") + m.name;
      if (here.first == "/" && m.is_group &&
          !internal::known_root_group(m.name)) {
        continue;
      }
      if (child == "/private") continue;
      if (m.is_group) {
        todo.emplace_back(child, here.second + 1);
        continue;
      }
      if (!m.is_dataset) continue;
      guarded(child, [&] {
        const DsetInfo info = f_.dataset_info(child);
        if (info.is_scale) {
          bool has_name = false;
          for (const RawAttr& a : f_.attributes(child)) {
            if (a.name == "NAME") has_name = true;
          }
          if (!has_name) {
            error("E25", child,
                  "a dimension scale with no NAME attribute, which section "
                  "21 requires of one");
          }
          // E42.  Nothing about the creation properties
          // is visible in a byte position, so the rule is checked by
          // asking the property list back.  A scale created without
          // them keeps its REFERENCE_LIST in an object header message,
          // where an attribute may not exceed 64 KiB, so it takes at
          // most 4085 attachments and the 4086th destroys the list on
          // its way to failing.
          if (!info.attr_order_tracked || !info.attr_order_indexed) {
            error("E42", child,
                  "a dimension scale created without attribute creation "
                  "order tracked and indexed, which section 21 requires: "
                  "such a scale takes at most 4085 attachments");
          }
          check_unlimited(child, info);
        }
        if (!internal::known_dataset_path(child, info.is_scale)) {
          unknown_dataset(child, info);
        }
      });
    }
  }
}

void Validator::unknown_dataset(const std::string& path,
                                const DsetInfo& info) {
  // A dataset this version does not know, inside a group it does.
  // Section 14 checks the
  // byte-level rules of sections 18 to 25 "on the public objects
  // only", and exempts /private and a *group* this version does not
  // know; section 28 provides for a new attribute and a new group
  // within a version and not for a new dataset.  So this is a public
  // object and it is checked -- not for the rules that follow from a
  // role, because it has none this version can read, but for the ones
  // that follow from being a dataset in this file at all: its name,
  // its attribute encodings, a dimension scale on every axis, valid
  // UTF-8 if it holds fixed-length strings, its filters, and chunking
  // if it carries a row dimension.  A dimension scale in a place
  // section 21 does not name is one of these too, except that a scale
  // carries no scale on its own axis and is not subject to E25
  // (section 14); the dataset attached to it draws E25 instead.
  const std::string name = internal::basename(path);
  if (!internal::legal_netcdf_name(name)) {
    error("E33", path, "\"" + name + "\" is not a legal netCDF-4 name");
  }
  if (internal::reserved_name(name)) {
    error("E33", path,
          "\"" + name + "\" begins with the reserved prefix mestra_");
  }
  check_attribute_encodings(path, f_.attributes(path));
  const Axes axes = axes_of(path, info, !info.is_scale);
  DType dtype = DType::Float64;
  if (internal::dtype_of(info.type, &dtype) && dtype == DType::String) {
    check_string_dataset(path, info, true);
  }
  const bool row_leading = !axes.logical.empty() && axes.logical[0] == "row";
  check_dataset_storage(path, info, row_leading,
                        row_leading ? static_cast<std::size_t>(n_rows_)
                                    : kNoChunkCheck);
}

void Validator::private_group() {
  // E18 is reported beside the rule that found the missing public
  // thing, at each object it is missing from, in a file that also
  // carries /private.  A writer that moved the public thing into the
  // private part is what the rule is about; these are the two facts a
  // reader can see, and it never interprets /private to see them.
  if (!f_.is_group("/private")) return;
  static const char* kCovered[] = {"E02", "E11", "E13", "E15",
                                   "E17", "E31", "E39"};
  // One E18 per object, naming every rule that fired there.
  std::vector<std::pair<std::string, std::string>> missing;
  for (const Finding& f : r_->errors) {
    for (const char* id : kCovered) {
      if (f.id != id) continue;
      auto at = std::find_if(missing.begin(), missing.end(),
                             [&](const auto& m) { return m.first == f.where; });
      if (at == missing.end()) {
        missing.emplace_back(f.where, id);
      } else {
        at->second += std::string(", ") + id;
      }
    }
  }
  for (const auto& [where, ids] : missing) {
    error("E18", where,
          "a required public thing is absent here (" + ids +
              ") in a file that also carries a /private group");
  }
}

void Validator::root() {
  const std::vector<RawAttr> attrs = f_.attributes("/");
  bool has_format = false;
  const std::string format = text_of(attrs, "format", &has_format);
  if (!has_format) {
    error("E01", "/", "the root `format` attribute is missing");
    error("E17", "/", "`format` is missing");
  } else {
    bool ok = format.compare(0, 7, "mestra/") == 0 && format.size() > 7;
    if (ok) {
      for (std::size_t i = 7; i < format.size(); ++i) {
        if (format[i] < '0' || format[i] > '9') ok = false;
      }
    }
    if (!ok) {
      error("E01", "/",
            "`format` is \"" + format + "\" and not \"mestra/<n>\"");
    } else if (format != "mestra/0") {
      error("E01", "/",
            "`format` is \"" + format +
                "\", a major version this reader must refuse");
    }
  }
  bool has_writer = false;
  text_of(attrs, "writer", &has_writer);
  if (!has_writer) error("E17", "/", "`writer` is missing");
  bool has_created = false;
  const std::string created = text_of(attrs, "created", &has_created);
  if (!has_created) {
    error("E17", "/", "`created` is missing");
  } else if (!internal::iso8601_utc(created)) {
    warn("W14", "/", "`created` is not an ISO 8601 UTC timestamp");
  }

  const RawAttr* aligned = find(attrs, "aligned");
  if (aligned == nullptr) {
    error("E39", "/", "`aligned` is missing");
  } else {
    aligned_ = aligned->value.kind() == AttrValue::Kind::Bool
                   ? aligned->value.as_bool()
                   : aligned->value.as_int() != 0;
  }
  check_attribute_encodings("/", attrs);

  for (const RawAttr& a : attrs) {
    if (internal::machinery_attribute(a.name)) continue;
    if (!internal::known_root_attribute(a.name)) {
      warn("W11", "/",
           "the root attribute \"" + a.name +
               "\" is one this version does not know; it is ignored");
    }
  }
  for (const Member& m : f_.members("/")) {
    if (m.kind != internal::LinkKind::Hard) {
      usable("/", m);
      continue;
    }
    if (m.is_group && !internal::known_root_group(m.name)) {
      warn("W11", "/" + m.name,
           "a root group this version does not know; it is ignored");
    }
  }
  // /notes is free-form, so nothing there is unknown (no W11), but
  // section 18 still holds: legal names, no variable-length string
  // anywhere in the file, valid UTF-8, and a string stored as section
  // 18 stores every string (section 11).  A number there is the
  // producer's choice.  /private is not looked at at all, which
  // section 29 requires.
  if (f_.is_group("/notes")) {
    const std::vector<RawAttr> notes = f_.attributes("/notes");
    check_attribute_encodings("/notes", notes);
    for (const RawAttr& a : notes) {
      if (a.type.klass == H5T_STRING && !a.type.variable_length &&
          (a.type.cset != H5T_CSET_UTF8 || a.type.strpad != H5T_STR_NULLPAD)) {
        error("E19", "/notes",
              "the attribute \"" + a.name +
                  "\" is a string that is not UTF-8 padded with NUL");
      }
    }
    links_in("/notes");
  }

  if (f_.is_dataset("/row")) {
    const DsetInfo info = f_.dataset_info("/row");
    n_rows_ = info.shape.empty()
                  ? 0
                  : static_cast<std::int64_t>(info.shape[0]);
    if (info.maxshape.empty() || info.maxshape[0] != H5S_UNLIMITED) {
      error("E27", "/row", "`row` is not an unlimited dimension");
    }
  } else {
    error("E39", "/", "the `row` dimension scale is missing");
  }
}

void Validator::categories() {
  for (const Member& m : f_.members("/categories")) {
    if (!usable("/categories", m)) continue;
    const std::string p = "/categories/" + m.name;
    if (!m.is_dataset) {
      error("E41", p, "a category table must be a dataset, and this is a "
                      "group, so there is nothing here to read");
      continue;
    }
    guarded(p, [&] {
    const DsetInfo info = f_.dataset_info(p);
    if (!internal::legal_netcdf_name(m.name)) {
      error("E33", p, "\"" + m.name + "\" is not a legal netCDF-4 name");
    }
    if (internal::reserved_name(m.name)) {
      error("E33", p,
            "\"" + m.name + "\" begins with the reserved prefix mestra_");
    }
    // A table of the wrong dtype is E20, and is still a public dataset
    // the byte-level rules hold for: its scale and its storage are
    // checked below whatever it holds.
    const bool is_text = info.type.klass == H5T_STRING;
    if (!is_text) {
      error("E20", p, "a category table that is not a string dataset");
    } else {
      check_string_dataset(p, info, true);
      category_tables_[m.name] = texts(p);
    }
    axes_of(p, info, true);
    if (!info.scales.empty() && info.scales[0].size() == 1 &&
        info.scales[0].front() != "category_" + m.name) {
      error("E25", p,
            "the dimension scale on this table is not `category_" + m.name +
                "`");
    }
    check_dataset_storage(p, info, false, kNoChunkCheck);
    });
  }
}

namespace {

struct KeyInfo {
  std::string name;
  std::string role;
  std::string category;
  std::string trajectory_group;
  DType dtype = DType::Float64;
  // False when the stored dtype is not one section 19 allows for the
  // role (E20): the values are then not read as ids, and E10 and W07,
  // which are about ids, have nothing to say.
  bool dtype_allowed = true;
  std::vector<double> f64;
  std::vector<std::int64_t> i64;
  bool has_lower = false;
  bool has_upper = false;
  double lower = 0.0;
  double upper = 0.0;
};

}  // namespace

void Validator::keys() {
  static const char* kRoles[] = {"design",      "condition", "time",
                                 "categorical", "group",     "split",
                                 "id",          "status"};
  std::map<std::string, int> role_count;
  std::vector<KeyInfo> infos;

  for (const Member& m : f_.members("/keys")) {
    if (!usable("/keys", m)) continue;
    const std::string p = "/keys/" + m.name;
    if (!m.is_dataset) {
      error("E41", p, "a key must be a dataset, and this is a group, so "
                      "there is nothing here to read");
      continue;
    }
    bool read_failed = false;
    KeyInfo scratch;
    guarded(p, [&] {
    const DsetInfo info = f_.dataset_info(p);
    const std::vector<RawAttr> attrs = f_.attributes(p);
    check_attribute_encodings(p, attrs);
    if (!internal::legal_netcdf_name(m.name)) {
      error("E33", p, "\"" + m.name + "\" is not a legal netCDF-4 name");
    }
    if (internal::reserved_name(m.name)) {
      error("E33", p,
            "\"" + m.name + "\" begins with the reserved prefix mestra_");
    }
    for (const RawAttr& a : attrs) {
      if (internal::machinery_attribute(a.name)) continue;
      if (!internal::known_key_attribute(a.name)) {
        warn("W11", p,
             "the attribute \"" + a.name +
                 "\" is one this version does not know; it is ignored");
      }
    }

    KeyInfo k;
    k.name = m.name;
    bool has_role = false;
    k.role = text_of(attrs, "role", &has_role);
    bool known_role = false;
    for (const char* r : kRoles) {
      if (k.role == r) known_role = true;
    }
    if (!has_role || !known_role) {
      error("E02", p,
            has_role ? "the role \"" + k.role + "\" is not in section 3"
                     : "no role attribute");
      k.role.clear();
    } else {
      role_count[k.role] += 1;
    }

    const bool has_units = find(attrs, "units") != nullptr;
    if (k.role == "design" || k.role == "condition" || k.role == "time") {
      if (!has_units) {
        error("E39", p, "a " + k.role + " key with no `units`");
      }
    }
    units_warning(p, attrs);

    bool has_category = false;
    k.category = text_of(attrs, "category", &has_category);
    if ((k.role == "categorical" || k.role == "group" ||
         k.role == "split" || k.role == "status") &&
        !has_category) {
      error("E39", p, "a " + k.role + " key with no `category`");
    }
    k.trajectory_group = text_of(attrs, "trajectory_group");

    DType dtype = DType::Float64;
    const bool dtype_ok = internal::dtype_of(info.type, &dtype);
    k.dtype = dtype;
    if (!k.role.empty()) {
      bool allowed = true;
      if (k.role == "design" || k.role == "condition" || k.role == "time") {
        allowed = dtype_ok && dtype == DType::Float64;
      } else if (k.role == "categorical" || k.role == "group" ||
                 k.role == "split" || k.role == "status") {
        allowed = dtype_ok &&
                  (dtype == DType::Int32 || dtype == DType::Int64);
      } else if (k.role == "id") {
        allowed = dtype_ok &&
                  (dtype == DType::Int64 || dtype == DType::String);
      }
      if (!allowed) {
        error("E20", p,
              "the dtype is not one section 19 allows for role " + k.role);
      }
      k.dtype_allowed = allowed;
    }

    const RawAttr* lower = find(attrs, "lower");
    const RawAttr* upper = find(attrs, "upper");
    if (lower != nullptr) {
      k.has_lower = true;
      k.lower = lower->value.as_float();
    }
    if (upper != nullptr) {
      k.has_upper = true;
      k.upper = upper->value.as_float();
    }

    axes_of(p, info, true);
    if (!info.scales.empty() && info.scales[0].size() == 1 &&
        info.scales[0].front() != "row") {
      error("E25", p, "the dimension of a key is not `row`");
    }
    if (info.shape.size() != 1) {
      error("E16", p, "a key must have exactly one dimension, `row`");
    } else if (static_cast<std::int64_t>(info.shape[0]) != n_rows_) {
      error("E16", p,
            "a key of " +
                internal::format_i64(
                    static_cast<std::int64_t>(info.shape[0])) +
                " elements in a file of " + internal::format_i64(n_rows_) +
                " rows");
    }
    check_dataset_storage(p, info, true,
                          static_cast<std::size_t>(n_rows_));
    if (dtype == DType::String) check_string_dataset(p, info, true);

    if (dtype == DType::Float64) {
      k.f64 = reals(p);
    } else if (dtype != DType::String) {
      k.i64 = integers(p);
    }
    scratch = std::move(k);
    });
    if (!read_failed) infos.push_back(std::move(scratch));
  }

  for (const char* role : {"time", "split", "id", "status"}) {
    if (role_count[role] > 1) {
      error("E03", "/keys",
            "more than one key with the role " + std::string(role));
    }
  }
  has_group_key_ = role_count["group"] > 0;

  const std::vector<RawAttr> root_attrs = f_.attributes("/");
  const bool has_gen = find(root_attrs, "generalisation_group") != nullptr;
  if (has_group_key_ && !has_gen) {
    error("E39", "/",
          "the file declares a group key and no `generalisation_group`");
  }

  const KeyInfo* time_key = nullptr;
  const KeyInfo* split_key = nullptr;
  const KeyInfo* status_key = nullptr;
  for (const KeyInfo& k : infos) {
    if (k.role == "time") time_key = &k;
    if (k.role == "split") split_key = &k;
    if (k.role == "status") status_key = &k;
  }
  if (time_key != nullptr && has_group_key_ &&
      time_key->trajectory_group.empty()) {
    error("E39", "/keys/" + time_key->name,
          "the time key names no `trajectory_group` in a file that declares "
          "a group key");
  }

  for (const KeyInfo& k : infos) {
    const std::string p = "/keys/" + k.name;
    if ((k.role == "categorical" || k.role == "group" ||
         k.role == "split" || k.role == "status") &&
        !k.category.empty()) {
      const auto it = category_tables_.find(k.category);
      if (it == category_tables_.end()) {
        error("E39", p,
              "the category table \"" + k.category +
                  "\" is not in the file");
      } else if (unread("/categories/" + k.category)) {
        // The table is E41 and its entries were never read, so
        // nothing here can be said about the ids in this column.
      } else if (!k.dtype_allowed) {
        // E20 already: the column is not category ids.
      } else {
        const std::int64_t n = static_cast<std::int64_t>(it->second.size());
        bool outside = false;
        for (const std::int64_t v : k.i64) {
          if (v < 0 || v >= n) outside = true;
        }
        if (outside) error("E10", p, "a value outside its category table");
        if (k.role == "group" && have_contents(p)) {
          const std::set<std::int64_t> used(k.i64.begin(), k.i64.end());
          for (std::int64_t i = 0; i < n; ++i) {
            if (used.count(i) == 0) {
              warn("W07", p,
                   "the category table has an entry no row uses: \"" +
                       it->second[static_cast<std::size_t>(i)] + "\"");
              break;
            }
          }
        }
      }
    }

    if (k.dtype == DType::Float64 && (k.has_lower || k.has_upper)) {
      PerRow outside;
      double lo = 0.0;
      double hi = 0.0;
      bool any = false;
      for (std::size_t i = 0; i < k.f64.size(); ++i) {
        const double v = k.f64[i];
        if (!std::isfinite(v)) continue;
        if ((k.has_lower && v < k.lower) || (k.has_upper && v > k.upper)) {
          outside.hit(i);
        }
        if (!any) {
          lo = v;
          hi = v;
          any = true;
        } else {
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
      if (outside.any()) {
        warn("W04", p,
             outside.message("key value(s) outside the declared bounds"));
      } else if (k.has_lower && k.has_upper && any && hi > lo) {
        // More than a factor of four in width.  A value outside the
        // bounds is W04 and not W08, and a zero observed width
        // takes the rule out altogether,
        // which covers a file with no rows, a key with one distinct
        // value and a key with no finite value.
        if ((k.upper - k.lower) > 4.0 * (hi - lo)) {
          warn("W08", p,
               "declared bounds more than four times wider than the "
               "observed range");
        }
      }
    }
  }

  if (time_key != nullptr) {
    const KeyInfo* group = nullptr;
    for (const KeyInfo& k : infos) {
      if (k.name == time_key->trajectory_group) group = &k;
    }
    std::map<std::int64_t, double> last;
    bool bad = false;
    for (std::size_t i = 0; i < time_key->f64.size(); ++i) {
      const std::int64_t g =
          (group != nullptr && i < group->i64.size()) ? group->i64[i] : 0;
      const auto it = last.find(g);
      if (it != last.end() && !(time_key->f64[i] > it->second)) bad = true;
      last[g] = time_key->f64[i];
    }
    if (bad) {
      error("E09", "/keys/" + time_key->name,
            "time is not strictly increasing within a trajectory");
    }
  }

  if (status_key != nullptr && !status_key->category.empty() &&
      !unread("/categories/" + status_key->category)) {
    const auto it = category_tables_.find(status_key->category);
    if (it != category_tables_.end()) {
      std::int64_t converged = -1;
      for (std::size_t i = 0; i < it->second.size(); ++i) {
        if (it->second[i] == "converged") {
          converged = static_cast<std::int64_t>(i);
        }
      }
      PerRow other;
      for (std::size_t i = 0; i < status_key->i64.size(); ++i) {
        if (status_key->i64[i] != converged) other.hit(i);
      }
      if (other.any()) {
        warn("W02", "/keys/" + status_key->name,
             other.message("row(s) whose status is not converged"));
      }
    }
  }

  if (split_key != nullptr) {
    const std::string gen = text_of(root_attrs, "generalisation_group");
    const KeyInfo* unit = nullptr;
    for (const KeyInfo& k : infos) {
      if (k.name == gen && k.role == "group") unit = &k;
    }
    if (unit != nullptr) {
      std::map<std::int64_t, std::set<std::int64_t>> sides;
      const std::size_t n =
          std::min(unit->i64.size(), split_key->i64.size());
      for (std::size_t i = 0; i < n; ++i) {
        sides[unit->i64[i]].insert(split_key->i64[i]);
      }
      // Name the leaked unit and then say why it matters, which is
      // what turns a warning into something a user acts on.
      // Conventions section 6 makes Python's W01 the model message for
      // every language, so the entry is named bare and not quoted; a
      // unit with no name in the table falls back to its id.
      const auto table = category_tables_.find(unit->category);
      auto name_of = [&table, this](std::int64_t id) {
        if (table != category_tables_.end() && id >= 0 &&
            static_cast<std::size_t>(id) < table->second.size()) {
          return table->second[static_cast<std::size_t>(id)];
        }
        return internal::format_i64(id);
      };
      std::vector<std::int64_t> leaked;
      for (const auto& entry : sides) {
        if (entry.second.size() > 1) leaked.push_back(entry.first);
      }
      if (!leaked.empty()) {
        std::string message = "the rows of " + unit->name + " ";
        for (std::size_t i = 0; i < leaked.size() && i < 3; ++i) {
          if (i != 0) message += ", ";
          message += name_of(leaked[i]);
        }
        if (leaked.size() > 3) {
          message += " and " +
                     internal::format_i64(
                         static_cast<std::int64_t>(leaked.size() - 3)) +
                     " more";
        }
        message += leaked.size() == 1 ? " are" : " are each";
        message +=
            " on both sides of the split, so this is not a "
            "generalisation test";
        warn("W01", "/keys/" + split_key->name, message);
      }
    }
  }
}

void Validator::scalars() {
  for (const Member& m : f_.members("/scalars")) {
    if (!usable("/scalars", m)) continue;
    const std::string p = "/scalars/" + m.name;
    guarded(p, [&] {
    if (m.is_group) links_in(p);
    const std::vector<RawAttr> attrs = f_.attributes(p);
    check_attribute_encodings(p, attrs);
    if (!internal::legal_netcdf_name(m.name)) {
      error("E33", p, "\"" + m.name + "\" is not a legal netCDF-4 name");
    }
    if (internal::reserved_name(m.name)) {
      error("E33", p,
            "\"" + m.name + "\" begins with the reserved prefix mestra_");
    }
    for (const RawAttr& a : attrs) {
      if (internal::machinery_attribute(a.name)) continue;
      if (!internal::known_scalar_attribute(a.name)) {
        warn("W11", p,
             "the attribute \"" + a.name +
                 "\" is one this version does not know; it is ignored");
      }
    }

    if (find(attrs, "units") == nullptr) {
      error("E11", p, "a scalar with no `units`");
    }
    units_warning(p, attrs);

    bool has_source = false;
    const std::string source = text_of(attrs, "source", &has_source);
    if (!has_source) {
      error("E39", p, "a scalar with no `source`");
    } else {
      const std::string id = internal::callable_of_source(source);
      if (source != "data" && id.empty()) {
        error("E36", p,
              "`source` is \"" + source +
                  "\", which is neither `data` nor `callable:<id>`");
      } else if (!id.empty()) {
        if (callable_ids_.count(id) == 0) {
          error("E14", p,
                "`source` names the callable \"" + id +
                    "\", which the file does not hold");
        }
        if (find(attrs, "output") == nullptr) {
          error("E39", p, "a callable-served slot with no `output`");
        }
        if (m.is_dataset) {
          error("E30", p, "a callable-served slot stored as a dataset");
        }
      } else if (source == "data" && m.is_group) {
        error("E30", p, "a slot whose source is data stored as a group");
      }
    }

    check_statistic(p, attrs);

    if (!m.is_dataset) return;
    const DsetInfo info = f_.dataset_info(p);
    DType dtype = DType::Float64;
    if (!internal::dtype_of(info.type, &dtype) || dtype != DType::Float64) {
      error("E20", p, "a scalar that is not float64");
    }
    axes_of(p, info, true);
    if (!info.scales.empty() && info.scales[0].size() == 1 &&
        info.scales[0].front() != "row") {
      error("E25", p, "the dimension of a scalar is not `row`");
    }
    if (info.shape.size() != 1) {
      error("E16", p, "a scalar must have exactly one dimension, `row`");
    } else if (static_cast<std::int64_t>(info.shape[0]) != n_rows_) {
      error("E16", p,
            "a scalar of " +
                internal::format_i64(
                    static_cast<std::int64_t>(info.shape[0])) +
                " elements in a file of " + internal::format_i64(n_rows_) +
                " rows");
    }
    check_dataset_storage(p, info, true,
                          static_cast<std::size_t>(n_rows_));
    if (dtype == DType::Float64) {
      PerRow missing;
      const std::vector<double> values = reals(p);
      for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) missing.hit(i);
      }
      if (missing.any()) {
        warn("W03", p,
             missing.message("non-finite value(s) in a scalar, which is "
                             "how this format spells missing "
                             "floating-point data"));
      }
      if (text_of(attrs, "statistic") == "band") {
        PerRow negative;
        for (std::size_t i = 0; i < values.size(); ++i) {
          if (values[i] < 0.0) negative.hit(i);
        }
        if (negative.any()) {
          warn("W16", p,
               negative.message("value(s) below zero in a band, which is a "
                                "half-width and is never negative"));
        }
      }
    }
    });
  }
}

void Validator::row_support_dataset() {
  has_row_support_ = f_.is_dataset("/row_support");
  const std::size_t n_supports = support_names_.size();
  if (has_row_support_ && aligned_) {
    error("E28", "/row_support",
          "/row_support is present in a file that sets aligned = true");
  }
  if (!has_row_support_ && !aligned_) {
    error("E28", "/",
          "/row_support is absent in a file that sets aligned = false");
  }
  if (aligned_ != (n_supports <= 1)) {
    error("E37", "/",
          "`aligned` disagrees with the " +
              internal::format_i64(static_cast<std::int64_t>(n_supports)) +
              " supports the file declares");
  }
  if (n_supports > 1) {
    warn("W05", "/supports",
         "the file declares more than one support, so index-aligned "
         "operations are not available");
  }
  if (!has_row_support_) return;

  const std::string p = "/row_support";
  const DsetInfo info = f_.dataset_info(p);
  DType dtype = DType::Float64;
  if (!internal::dtype_of(info.type, &dtype) || dtype != DType::Int32) {
    error("E20", p, "/row_support must be int32");
  }
  axes_of(p, info, true);
  if (!info.scales.empty() && info.scales[0].size() == 1 &&
      info.scales[0].front() != "row") {
    error("E25", p, "the dimension of /row_support is not `row`");
  }
  if (!info.shape.empty() &&
      static_cast<std::int64_t>(info.shape[0]) != n_rows_) {
    error("E16", p, "/row_support does not have one entry per row");
  }
  check_dataset_storage(p, info, true, static_cast<std::size_t>(n_rows_));
  row_support_ = integers(p);
  std::set<std::int64_t> used;
  for (const std::int64_t v : row_support_) {
    if (v < 0 || v >= static_cast<std::int64_t>(n_supports)) {
      error("E06", p,
            "a row references support " + internal::format_i64(v) +
                ", which the file does not declare");
    } else {
      used.insert(v);
    }
  }
  for (std::size_t i = 0; i < n_supports; ++i) {
    if (used.count(static_cast<std::int64_t>(i)) == 0) {
      warn("W15", "/supports/" + support_names_[i],
           "a declared support that no row references");
    }
  }
}

void Validator::supports() {
  for (const Member& m : f_.members("/supports")) {
    if (!usable("/supports", m)) continue;
    if (!m.is_group) {
      error("E41", "/supports/" + m.name,
            "a support must be a group, and this is a dataset, so there "
            "is nothing here to read");
    }
  }
  for (std::size_t index = 0; index < support_names_.size(); ++index) {
    const std::string name = support_names_[index];
    const std::string sp = "/supports/" + name;
    guarded(sp, [&] {
    const std::vector<RawAttr> attrs = f_.attributes(sp);
    check_attribute_encodings(sp, attrs);
    if (!internal::legal_netcdf_name(name)) {
      error("E33", sp, "\"" + name + "\" is not a legal netCDF-4 name");
    }
    if (internal::reserved_name(name)) {
      error("E33", sp,
            "\"" + name + "\" begins with the reserved prefix mestra_");
    }
    for (const RawAttr& a : attrs) {
      if (internal::machinery_attribute(a.name)) continue;
      if (!internal::known_support_attribute(a.name)) {
        warn("W11", sp,
             "the attribute \"" + a.name +
                 "\" is one this version does not know; it is ignored");
      }
    }
    for (const Member& g : f_.members(sp)) {
      if (g.kind != internal::LinkKind::Hard) {
        usable(sp, g);
        continue;
      }
      if (g.is_group && !internal::known_support_group(g.name)) {
        warn("W11", sp + "/" + g.name,
             "a group inside a support this version does not know");
      }
    }

    bool has_kind = false;
    const std::string kind = text_of(attrs, "kind", &has_kind);
    if (!has_kind) error("E39", sp, "a support with no `kind`");
    const RawAttr* kind_attr = find(attrs, "kind");
    if (kind_attr != nullptr &&
        kind_attr->value.kind() == AttrValue::Kind::Str && kind != "mesh" &&
        kind != "axis" && kind != "none") {
      error("E02", sp, "\"" + kind + "\" is not a kind of support of section 6");
    }
    const RawAttr* n_nodes_attr = find(attrs, "n_nodes");
    const RawAttr* n_cells_attr = find(attrs, "n_cells");
    if (n_nodes_attr == nullptr) error("E39", sp, "no `n_nodes`");
    if (n_cells_attr == nullptr) error("E39", sp, "no `n_cells`");
    bool has_sid = false;
    const std::string stored_sid = text_of(attrs, "support_id", &has_sid);
    if (!has_sid) error("E39", sp, "a support with no `support_id`");
    const std::int64_t n_nodes =
        n_nodes_attr != nullptr ? n_nodes_attr->value.as_int() : 0;
    const std::int64_t n_cells =
        n_cells_attr != nullptr ? n_cells_attr->value.as_int() : 0;

    const bool has_types = f_.is_dataset(sp + "/cell_types");
    const bool has_offsets = f_.is_dataset(sp + "/cell_offsets");
    const bool has_conn = f_.is_dataset(sp + "/cell_connectivity");
    if (kind == "mesh") {
      if (!has_types || !has_offsets || !has_conn) {
        error("E38", sp,
              "a mesh support missing cell_types, cell_offsets or "
              "cell_connectivity");
      }
    } else if (kind == "axis" || kind == "none") {
      if (has_types || has_offsets || has_conn ||
          f_.is_dataset(sp + "/cell")) {
        error("E38", sp,
              "a support of kind " + kind + " carrying cell arrays");
      }
    }

    std::vector<std::uint8_t> types;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> conn;
    if (has_types) {
      const std::string p = sp + "/cell_types";
      const DsetInfo info = f_.dataset_info(p);
      DType dtype = DType::Float64;
      if (!internal::dtype_of(info.type, &dtype) || dtype != DType::UInt8) {
        error("E20", p, "cell_types must be uint8");
      }
      axes_of(p, info, true);
      if (!info.scales.empty() && info.scales[0].size() == 1 &&
          info.scales[0].front() != "cell") {
        error("E25", p, "the dimension of cell_types is not `cell`");
      }
      check_dataset_storage(p, info, false, kNoChunkCheck);
      for (const std::int64_t v : integers(p)) {
        types.push_back(static_cast<std::uint8_t>(v));
      }
    }
    if (has_offsets) {
      const std::string p = sp + "/cell_offsets";
      const DsetInfo info = f_.dataset_info(p);
      DType dtype = DType::Float64;
      if (!internal::dtype_of(info.type, &dtype) || dtype != DType::Int64) {
        error("E20", p, "cell_offsets must be int64");
      }
      axes_of(p, info, true);
      if (!info.scales.empty() && info.scales[0].size() == 1 &&
          info.scales[0].front() != "cell_plus_one") {
        error("E25", p,
              "the dimension of cell_offsets is not `cell_plus_one`");
      }
      check_dataset_storage(p, info, false, kNoChunkCheck);
      offsets = integers(p);
    }
    if (has_conn) {
      const std::string p = sp + "/cell_connectivity";
      const DsetInfo info = f_.dataset_info(p);
      DType dtype = DType::Float64;
      if (!internal::dtype_of(info.type, &dtype) || dtype != DType::Int64) {
        error("E20", p, "cell_connectivity must be int64");
      }
      axes_of(p, info, true);
      if (!info.scales.empty() && info.scales[0].size() == 1 &&
          info.scales[0].front() != "index") {
        error("E25", p,
              "the dimension of cell_connectivity is not `index`");
      }
      check_dataset_storage(p, info, false, kNoChunkCheck);
      conn = integers(p);
    }

    bool bad_code = false;
    for (const std::uint8_t code : types) {
      if (cell_type_nodes(code) < 0) bad_code = true;
    }
    if (bad_code) {
      error("E21", sp + "/cell_types",
            "a cell type code that is not in the table of section 20");
    }
    const bool have_types =
        has_types && have_contents(sp + "/cell_types");
    if (kind == "mesh" && have_types &&
        static_cast<std::int64_t>(types.size()) != n_cells) {
      // The cell count a mesh declares is the length of its
      // cell_types.  A cell array is checked against the declaration;
      // the declaration is checked here against the one dataset that
      // defines it, or a mesh with no cell arrays could declare any
      // count at all.  A cell dataset on another kind is E38 alone.
      error("E05", sp + "/cell_types",
            "the support declares " + internal::format_i64(n_cells) +
                " cells and cell_types holds " +
                internal::format_i64(static_cast<std::int64_t>(types.size())));
    }
    const bool have_offsets =
        has_offsets && have_contents(sp + "/cell_offsets");
    const bool have_conn =
        has_conn && have_contents(sp + "/cell_connectivity");
    if (have_offsets) {
      bool bad = offsets.empty() || offsets.front() != 0;
      for (std::size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) bad = true;
      }
      if (!offsets.empty() && have_conn &&
          offsets.back() != static_cast<std::int64_t>(conn.size())) {
        bad = true;
      }
      if (bad) {
        error("E23", sp + "/cell_offsets",
              "cell_offsets does not start at 0, is not non-decreasing, or "
              "does not end at the length of cell_connectivity");
      } else if (have_types && offsets.size() == types.size() + 1) {
        bool bad_count = false;
        for (std::size_t j = 0; j < types.size(); ++j) {
          const int want = cell_type_nodes(types[j]);
          if (want < 0) continue;
          const std::int64_t got = offsets[j + 1] - offsets[j];
          if (types[j] == 7) {
            if (got < 3) bad_count = true;
          } else if (got != want) {
            bad_count = true;
          }
        }
        if (bad_count) {
          error("E22", sp + "/cell_offsets",
                "a cell whose node count disagrees with its cell type");
        }
      }
    }
    for (const std::int64_t v : conn) {
      if (v < 0 || v >= n_nodes) {
        error("E24", sp + "/cell_connectivity",
              "a connectivity value outside [0, n_nodes)");
        break;
      }
    }

    const bool has_coordinates = f_.is_dataset(sp + "/coordinates") ||
                                 f_.is_group(sp + "/coordinates");
    if ((kind == "mesh" || kind == "axis") && !has_coordinates) {
      error("E03", sp, "a " + kind + " support with no coordinates array");
    }
    if (has_coordinates) {
      const std::string cp = sp + "/coordinates";
      if (f_.is_group(cp)) guarded(cp, [&] { links_in(cp); });
      guarded(cp, [&] { slot(cp, kind, n_nodes, n_cells, index, true); });
    }
    for (int which = 0; which < 2; ++which) {
      const std::string gp =
          sp + (which == 0 ? "/node_arrays" : "/cell_arrays");
      for (const Member& a : f_.members(gp)) {
        if (!usable(gp, a)) continue;
        const std::string ap = gp + "/" + a.name;
        if (a.is_group) guarded(ap, [&] { links_in(ap); });
        guarded(ap, [&] { slot(ap, kind, n_nodes, n_cells, index, false); });
      }
    }

    // The digest of section 24 is the stored arrays, so a pass that
    // does not have them does not compute one.
    const bool digest_decided =
        (!has_types || have_types) && (!has_offsets || have_offsets) &&
        (!has_conn || have_conn) &&
        (!f_.is_dataset(sp + "/coordinates") ||
         have_contents(sp + "/coordinates"));
    if (has_sid && digest_decided) {
      // Section 24: the kind decides which arrays are hashed, so a
      // file that wrongly puts cell arrays on an axis support breaks
      // E38 and not E08 as well.
      std::string computed;
      const std::vector<std::uint8_t> no_types;
      const std::vector<std::int64_t> no_ints;
      if (kind == "axis" && f_.is_dataset(sp + "/coordinates")) {
        const std::vector<double> coords = reals(sp + "/coordinates");
        computed =
            support_id_digest(n_nodes, no_types, no_ints, no_ints, &coords);
      } else if (kind == "axis" || kind == "none") {
        computed =
            support_id_digest(n_nodes, no_types, no_ints, no_ints, nullptr);
      } else {
        computed = support_id_digest(n_nodes, types, offsets, conn, nullptr);
      }
      if (computed != stored_sid) {
        error("E08", sp,
              "the stored support_id does not match the stored arrays");
      }
    }
    });
  }
}

void Validator::slot(const std::string& path,
                     const std::string& support_kind, std::int64_t n_nodes,
                     std::int64_t n_cells, std::size_t support_index,
                     bool is_coordinates) {
  const bool is_dataset = f_.is_dataset(path);
  const std::vector<RawAttr> attrs = f_.attributes(path);
  check_attribute_encodings(path, attrs);
  const std::string name = internal::basename(path);
  if (!internal::legal_netcdf_name(name)) {
    error("E33", path, "\"" + name + "\" is not a legal netCDF-4 name");
  }
  if (internal::reserved_name(name)) {
    error("E33", path,
          "\"" + name + "\" begins with the reserved prefix mestra_");
  }
  for (const RawAttr& a : attrs) {
    if (internal::machinery_attribute(a.name)) continue;
    if (!internal::known_array_attribute(a.name)) {
      warn("W11", path,
           "the attribute \"" + a.name +
               "\" is one this version does not know; it is ignored");
    }
  }

  static const char* kRoles[] = {"coordinates", "field",  "label",
                                 "weight",      "normal", "derived"};
  bool has_role = false;
  const std::string role = text_of(attrs, "role", &has_role);
  bool known_role = false;
  for (const char* r : kRoles) {
    if (role == r) known_role = true;
  }
  if (!has_role || !known_role) {
    error("E02", path,
          has_role ? "the role \"" + role + "\" is not in section 3"
                   : "no role attribute");
  }

  bool has_varies = false;
  const std::string varies = text_of(attrs, "varies", &has_varies);
  if (!has_varies) error("E39", path, "no `varies`");

  const bool has_units = find(attrs, "units") != nullptr;
  if (role == "field") {
    if (!has_units) error("E11", path, "a field with no `units`");
  } else if (role == "derived" || role == "coordinates") {
    if (!has_units) {
      error("E39", path, "a " + role + " array with no `units`");
    }
  }
  units_warning(path, attrs);

  if (role == "derived" && (find(attrs, "derived_from") == nullptr ||
                            find(attrs, "recipe") == nullptr)) {
    error("E13", path, "a derived array without `derived_from` and `recipe`");
  }
  if (role == "weight" || role == "normal") {
    const RawAttr* r = find(attrs, "recomputed");
    const bool marked =
        r != nullptr && (r->value.kind() == AttrValue::Kind::Bool
                             ? r->value.as_bool()
                             : r->value.as_int() != 0);
    if (!marked) {
      warn("W06", path,
           "a " + role + " array that does not say it was recomputed");
    }
  }

  check_statistic(path, attrs);

  bool has_source = false;
  const std::string source = text_of(attrs, "source", &has_source);
  if (!has_source) {
    error("E39", path, "no `source`");
  } else {
    const std::string id = internal::callable_of_source(source);
    if (source != "data" && id.empty()) {
      error("E36", path,
            "`source` is \"" + source +
                "\", which is neither `data` nor `callable:<id>`");
    } else if (!id.empty()) {
      if (callable_ids_.count(id) == 0) {
        error("E14", path,
              "`source` names the callable \"" + id +
                  "\", which the file does not hold");
      }
      if (find(attrs, "output") == nullptr) {
        error("E39", path, "a callable-served slot with no `output`");
      }
      if (is_dataset) {
        error("E30", path, "a callable-served slot stored as a dataset");
      }
    } else if (source == "data" && !is_dataset) {
      error("E30", path, "a slot whose source is data stored as a group");
    }
  }

  const RawAttr* components_attr = find(attrs, "components");
  if (components_attr == nullptr) {
    error("E31", path, "an array slot with no `components`");
  }
  // Whether there is a number to compare is its own fact: a stored -1
  // is a count no axis has, not the absence of one.
  const bool has_components =
      components_attr != nullptr &&
      components_attr->value.kind() == AttrValue::Kind::Int;
  const std::int64_t components =
      has_components ? components_attr->value.as_int() : 0;

  // Section 5 names three values of `varies`, and no leading dimension
  // agrees with any other word.
  if (has_varies && varies != "none" && varies != "row" &&
      varies.compare(0, 6, "group:") != 0) {
    error("E04", path,
          "`varies` is none, row or group:<k>, and this is \"" + varies +
              "\"");
  }

  if (is_coordinates && support_kind == "axis" && has_varies &&
      varies != "none") {
    error("E35", path,
          "an axis support's coordinates must have varies = none");
  }

  if (!is_dataset) return;
  const DsetInfo info = f_.dataset_info(path);
  const Axes axes = axes_of(path, info, true);

  DType dtype = DType::Float64;
  const bool dtype_ok = internal::dtype_of(info.type, &dtype);
  if (role == "coordinates" || role == "field" || role == "derived" ||
      role == "weight" || role == "normal") {
    if (!dtype_ok || dtype != DType::Float64) {
      error("E20", path, "an array with role " + role + " must be float64");
    }
  } else if (role == "label") {
    if (!dtype_ok || (dtype != DType::Int32 && dtype != DType::Int64)) {
      error("E20", path, "a label must be int32 or int64");
    }
  }

  const std::string leading = axes.logical.empty() ? "" : axes.logical[0];
  if (has_varies && !axes.logical.empty()) {
    if (varies == "none") {
      if (leading == "row" || leading.compare(0, 6, "group:") == 0) {
        error("E04", path,
              "`varies` is none and the leading dimension is \"" + leading +
                  "\"");
      }
    } else if (varies == "row") {
      if (leading != "row") {
        error("E04", path,
              "`varies` is row and the leading dimension is \"" + leading +
                  "\"");
      }
    } else if (varies.compare(0, 6, "group:") == 0 && leading != varies) {
      error("E04", path,
            "`varies` is \"" + varies + "\" and the leading dimension is \"" +
                leading + "\"");
    }
  }

  // E25: the name section 21 allows at each axis position of a slot,
  // which is (row | group:<k>, [draw], node | cell, component) with the
  // last two always present.  The leading axis is E04's against
  // `varies` and is not repeated here; the count on the node, cell and
  // component axes is E05's and E31's, but the name is this rule's,
  // or an array whose node axis carried a stranger's scale would pass
  // every count check by never being counted.
  if (axes.logical.size() >= 2) {
    const std::string location =
        path.find("/cell_arrays/") != std::string::npos ? "cell" : "node";
    for (std::size_t i = 0; i < axes.logical.size(); ++i) {
      const std::string& have = axes.logical[i];
      if (have.empty()) continue;   // no scale, or two: reported above
      std::string want;
      if (i + 1 == axes.logical.size()) {
        want = "component";
      } else if (i + 2 == axes.logical.size()) {
        want = location;
      } else if (i == 0) {
        continue;
      } else {
        want = "draw";
      }
      if (have != want) {
        error("E25", path,
              "axis " + internal::format_i64(static_cast<std::int64_t>(i)) +
                  " carries the dimension \"" + have +
                  "\", where section 21 puts `" + want + "`");
      }
    }
  }

  const int node_axis = axes.index_of("node");
  const int cell_axis = axes.index_of("cell");
  if (node_axis >= 0 &&
      static_cast<std::int64_t>(
          axes.extent[static_cast<std::size_t>(node_axis)]) != n_nodes) {
    error("E05", path, "the node count disagrees with the support");
  }
  if (cell_axis >= 0 &&
      static_cast<std::int64_t>(
          axes.extent[static_cast<std::size_t>(cell_axis)]) != n_cells) {
    error("E05", path, "the cell count disagrees with the support");
  }

  const int comp_axis = axes.index_of("component");
  if (comp_axis >= 0 && has_components &&
      static_cast<std::int64_t>(
          axes.extent[static_cast<std::size_t>(comp_axis)]) != components) {
    error("E31", path,
          "`components` disagrees with the length of the component "
          "dimension");
  }

  if (varies.compare(0, 6, "group:") == 0 && !axes.extent.empty()) {
    const std::string key = varies.substr(6);
    const std::vector<RawAttr> key_attrs =
        f_.is_dataset("/keys/" + key) ? f_.attributes("/keys/" + key)
                                      : std::vector<RawAttr>();
    if (text_of(key_attrs, "role") != "group") {
      error("E04", path,
            "`varies` names \"" + key +
                "\", which the file does not declare as a group key");
    }
    const std::string table = text_of(key_attrs, "category");
    const auto it = category_tables_.find(table);
    if (it != category_tables_.end() && !unread("/categories/" + table) &&
        axes.extent[0] != it->second.size()) {
      error("E34", path,
            "the leading dimension differs from the number of categories of "
            "the group key");
    }
  }

  const bool row_leading = leading == "row";
  std::size_t want_rows = static_cast<std::size_t>(n_rows_);
  if (row_leading && !aligned_ && has_row_support_) {
    want_rows = 0;
    for (const std::int64_t v : row_support_) {
      if (v == static_cast<std::int64_t>(support_index)) ++want_rows;
    }
  }
  if (row_leading && !axes.extent.empty() && axes.extent[0] != want_rows) {
    error("E16", path,
          "the leading dimension does not match the number of rows this "
          "slot must have");
  }

  // Section 23: the row count in the chunk default is the length of
  // the row dimension the leading axis is attached to, not the number
  // of rows /row_support puts on the support -- a dataset that
  // disagrees with that is E16, and its chunk is still judged against
  // the dimension.
  std::size_t chunk_rows = want_rows;
  if (row_leading && !info.scale_paths.empty() &&
      info.scale_paths[0].size() == 1 &&
      f_.is_dataset(info.scale_paths[0].front())) {
    const DsetInfo scale = f_.dataset_info(info.scale_paths[0].front());
    if (scale.shape.size() == 1) {
      chunk_rows = static_cast<std::size_t>(scale.shape[0]);
    }
  }
  check_dataset_storage(path, info, row_leading,
                        row_leading ? chunk_rows : kNoChunkCheck);

  if ((role == "field" || role == "derived") && dtype_ok &&
      dtype == DType::Float64) {
    // Per row where the array has rows, and per stored element where
    // it has none, so that the three indices a reader is given always
    // point at something they can look up.
    std::size_t per_row = 1;
    for (std::size_t i = 1; i < axes.extent.size(); ++i) {
      per_row *= axes.extent[i] == 0 ? 1 : axes.extent[i];
    }
    PerRow missing(row_leading ? "row" : "index");
    const std::vector<double> values = reals(path);
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (!std::isfinite(values[i])) {
        missing.hit(row_leading ? i / per_row : i);
      }
    }
    if (missing.any()) {
      warn("W03", path,
           missing.message("non-finite value(s) in a " + role +
                           ", which is how this format spells missing "
                           "floating-point data"));
    }
    if (text_of(attrs, "statistic") == "band") {
      PerRow negative(row_leading ? "row" : "index");
      for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] < 0.0) negative.hit(row_leading ? i / per_row : i);
      }
      if (negative.any()) {
        warn("W16", path,
             negative.message("value(s) below zero in a band, which is a "
                              "half-width and is never negative"));
      }
    }
  }

  if (role == "label") {
    const std::string table = text_of(attrs, "category");
    if (!table.empty()) {
      const auto it = category_tables_.find(table);
      if (it == category_tables_.end()) {
        error("E39", path,
              "the category table \"" + table + "\" is not in the file");
      } else if (unread("/categories/" + table)) {
        // E41 on the table; nothing here is decidable from it.
      } else {
        const std::int64_t n = static_cast<std::int64_t>(it->second.size());
        for (const std::int64_t v : integers(path)) {
          if (v < 0 || v >= n) {
            error("E10", path, "a label value outside its category table");
            break;
          }
        }
      }
    }
  }
}

void Validator::callables() {
  for (const Member& m : f_.members("/callables")) {
    if (!usable("/callables", m)) continue;
    const std::string p = "/callables/" + m.name;
    if (!m.is_group) {
      error("E41", p, "a callable must be a group, and this is a dataset, "
                      "so there is nothing here to read");
      continue;
    }
    guarded(p, [&] {
    const std::vector<RawAttr> attrs = f_.attributes(p);
    if (!internal::legal_netcdf_name(m.name)) {
      error("E33", p, "\"" + m.name + "\" is not a legal netCDF-4 name");
    }
    if (internal::reserved_name(m.name)) {
      error("E33", p,
            "\"" + m.name + "\" begins with the reserved prefix mestra_");
    }
    if (find(attrs, "type") == nullptr) {
      error("E15", p, "a callable group with no `type`");
    }
    check_dict(p, true, 0);
    });
  }
}

void Validator::check_dict(const std::string& path, bool top_level,
                           int depth) {
  if (depth >= kMaxDictDepth) {
    error("E41", path,
          "nested deeper than this reader walks, which is " +
              internal::format_i64(kMaxDictDepth) + " groups");
    return;
  }
  // A callable's dictionary is opaque to a reader that does not own its
  // type, so nothing here interprets it: these are the rules of
  // section 25 about what is representable at all.
  for (const RawAttr& a : f_.attributes(path)) {
    if (internal::machinery_attribute(a.name)) continue;
    if (internal::reserved_name(a.name)) continue;
    if (top_level && (a.name == "type" || a.name == "repr")) {
      if (!internal::is_spec_string(a.type)) {
        error("E19", path,
              "the attribute \"" + a.name +
                  "\" is not stored in the encoding section 18 requires");
      }
      continue;
    }
    if (!internal::legal_netcdf_name(a.name)) {
      error("E33", path,
            "the dictionary key \"" + a.name +
                "\" is not a legal netCDF-4 name");
    }
    if (a.type.klass == H5T_STRING) {
      if (a.type.variable_length) {
        error("E19", path,
              "the dictionary entry \"" + a.name +
                  "\" is a variable-length string");
      } else if (!is_null_sentinel_bytes(a.raw_bytes)) {
        if (internal::embedded_nul(a.raw_bytes)) {
          error("E32", path,
                "the dictionary string \"" + a.name +
                    "\" holds an embedded NUL");
        } else if (!internal::valid_utf8(internal::strip_nul(a.raw_bytes))) {
          error("E26", path,
                "the dictionary string \"" + a.name + "\" is not valid "
                                                      "UTF-8");
        }
      }
      continue;
    }
    // Section 25: int8 is a boolean, int64 an integer, float64 a
    // float.  Anything else is not representable.
    const bool ok = internal::is_spec_bool(a.type) ||
                    internal::is_spec_int64(a.type) ||
                    internal::is_spec_float64(a.type);
    if (!ok) {
      error("E32", path,
            "the dictionary entry \"" + a.name +
                "\" is stored in a type section 25 does not allow");
    }
  }

  for (const Member& m : f_.members(path)) {
    if (internal::reserved_name(m.name)) continue;
    const std::string child = path + "/" + m.name;
    if (!internal::legal_netcdf_name(m.name)) {
      error("E33", child,
            "the dictionary key \"" + m.name +
                "\" is not a legal netCDF-4 name");
    }
    if (!usable(path, m)) continue;
    if (m.is_group) {
      check_dict(child, false, depth + 1);
      continue;
    }
    if (!m.is_dataset) continue;
    const DsetInfo info = f_.dataset_info(child);
    if (info.shape.empty()) {
      error("E32", child,
            "a zero-dimensional dataset, which section 25 requires to be "
            "written as an attribute");
      continue;
    }
    DType dtype = DType::Float64;
    const bool ok = internal::dtype_of(info.type, &dtype) &&
                    (dtype == DType::Bool || dtype == DType::Int32 ||
                     dtype == DType::Int64 || dtype == DType::Float64 ||
                     dtype == DType::String);
    if (!ok) {
      error("E32", child,
            "a dtype section 25 does not allow inside a dictionary");
      continue;
    }
    if (dtype == DType::String) check_string_dataset(child, info, true);
    const Axes axes = axes_of(child, info, true);
    for (std::size_t axis = 0; axis < info.shape.size(); ++axis) {
      if (info.scales[axis].size() != 1) continue;
      const std::string want =
          "mestra_" + m.name + "_d" +
          internal::format_i64(static_cast<std::int64_t>(axis));
      if (info.scales[axis].front() != want) {
        error("E25", child,
              "axis " + internal::format_i64(
                            static_cast<std::int64_t>(axis)) +
                  " does not carry the scale `" + want + "`");
      }
    }
    (void)axes;
    check_dataset_storage(child, info, false, kNoChunkCheck);
  }
}

}  // namespace

namespace {

Report validate_pass(const std::string& path, bool metadata_only) {
  Report r;
  try {
    File f;
    try {
      f = File::open_read(path);
    } catch (const Error& e) {
      // E01, as the Python and Julia readers say: a file that will not
      // open as HDF5 has no `format`, so it is not a mestra file. (This
      // used to say E41, which is for an object inside a file.)
      r.errors.push_back({"E01", path, without_id(e)});
      return r;
    }
    Validator v(f, &r, metadata_only);
    v.run();
  } catch (const Error& e) {
    r.errors.push_back({e.rule(), path, e.what()});
  } catch (const std::exception& e) {
    r.errors.push_back({std::string(), path, e.what()});
  }
  return r;
}

}  // namespace

Report validate(const std::string& path) {
  return validate_pass(path, false);
}

Report validate_metadata(const std::string& path) {
  return validate_pass(path, true);
}

std::vector<std::string> Report::error_ids() const {
  std::vector<std::string> ids;
  for (const Finding& f : errors) {
    if (!f.id.empty()) ids.push_back(f.id);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

std::vector<std::string> Report::warning_ids() const {
  std::vector<std::string> ids;
  for (const Finding& f : warnings) {
    if (!f.id.empty()) ids.push_back(f.id);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

}  // namespace mestra
