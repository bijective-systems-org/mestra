#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>

#include "codec.hpp"
#include "h5.hpp"
#include "layout.hpp"
#include "mestra/io.hpp"
#include "mestra/mestra.hpp"
#include "mestra/validate.hpp"
#include "names.hpp"

namespace mestra {
namespace {

using internal::File;

// The order in which links, attributes and scale attachments are
// created is fixed here and nowhere else, because HDF5 stores the
// links of a small group in creation order and section 30 asks a
// writer to be byte reproducible.  Reordering the statements below
// changes the bytes of every file this writer produces.
class Writer {
 public:
  Writer(const Dataset& d, File& f) : d_(d), f_(f) {}

  void run() {
    root_attributes();
    root_scales();
    categories();
    keys();
    scalars();
    row_support();
    supports();
    callables();
    trailing_groups();
  }

 private:
  // --- helpers ------------------------------------------------------

  std::size_t rows() const { return static_cast<std::size_t>(d_.n_rows); }

  void put(const std::string& path, const std::string& name,
           const std::string& value) {
    f_.write_attr(path, name, AttrValue::text(value));
  }
  void put_int(const std::string& path, const std::string& name,
               std::int64_t value) {
    f_.write_attr(path, name, AttrValue::integer(value));
  }
  void put_real(const std::string& path, const std::string& name,
                double value) {
    f_.write_attr(path, name, AttrValue::real(value));
  }
  void put_bool(const std::string& path, const std::string& name,
                bool value) {
    f_.write_attr(path, name, AttrValue::boolean(value));
  }
  void put_extra(const std::string& path, const AttrMap& extra) {
    for (const auto& entry : extra) {
      f_.write_attr(path, entry.first, entry.second);
    }
  }

  // The scale a logical dimension name refers to, for a dataset that
  // sits in `support` (empty for a root-level dataset).
  std::string scale_path(const std::string& logical, std::size_t length,
                         const Support* support, bool local_row) const {
    if (logical == "row") {
      if (local_row && support != nullptr) {
        return "/supports/" + support->name + "/row";
      }
      return "/row";
    }
    if (logical == "component") {
      return "/component_" + internal::format_i64(
                                 static_cast<std::int64_t>(length));
    }
    if (logical == "draw") {
      return "/draw_" +
             internal::format_i64(static_cast<std::int64_t>(length));
    }
    if (logical.compare(0, 6, "group:") == 0) {
      return "/group_" + logical.substr(6);
    }
    if (support != nullptr) return "/supports/" + support->name + "/" +
                                   logical;
    return "/" + logical;
  }

  std::vector<std::size_t> chunk_for(const std::string& path,
                                     std::size_t item,
                                     const std::vector<std::size_t>& shape,
                                     std::size_t row_count) const {
    const auto it = d_.chunk_overrides.find(path);
    if (it != d_.chunk_overrides.end()) return it->second;
    std::vector<std::size_t> rest(shape.begin() + 1, shape.end());
    std::vector<std::size_t> chunk;
    chunk.push_back(default_chunk_rows(item, rest, row_count));
    for (const std::size_t e : rest) chunk.push_back(e);
    return chunk;
  }

  // The filters the dataset at `path` came with, or none.  Section 23
  // makes compression optional, so a dataset built from vectors gets
  // none; what a round trip must not do is drop one the file had.
  FilterPipeline filters_for(const std::string& path) const {
    const auto it = d_.filters.find(path);
    return it == d_.filters.end() ? FilterPipeline() : it->second;
  }

  // Section 23 for a dataset with no `row` dimension: contiguous,
  // unless the file chunked it or it carries a filter, which HDF5
  // allows on a chunked dataset only.  The default then is the whole
  // dataset when that is 1 MiB or less, and otherwise the row rule
  // applied to its leading dimension.
  std::vector<std::size_t> fixed_chunk_for(
      const std::string& path, std::size_t item,
      const std::vector<std::size_t>& shape) const {
    const auto it = d_.chunk_overrides.find(path);
    if (it != d_.chunk_overrides.end()) return it->second;
    if (shape.empty() || filters_for(path).empty()) return {};
    std::vector<std::size_t> chunk(shape);
    std::size_t bytes = item;
    for (std::size_t& e : chunk) {
      if (e == 0) e = 1;
      bytes *= e;
    }
    if (bytes <= 1048576) return chunk;
    const std::vector<std::size_t> rest(chunk.begin() + 1, chunk.end());
    chunk[0] = default_chunk_rows(item, rest, chunk[0]);
    return chunk;
  }

  static std::vector<hsize_t> to_h(const std::vector<std::size_t>& v) {
    return std::vector<hsize_t>(v.begin(), v.end());
  }

  // A dimension scale, with whatever storage the dataset it came from
  // had when a reader recorded one (sections 21 and 23 otherwise).
  void scale(const std::string& path, std::size_t length, bool unlimited) {
    const auto it = d_.chunk_overrides.find(path);
    f_.make_scale(path, static_cast<hsize_t>(length), unlimited,
                  it == d_.chunk_overrides.end()
                      ? std::vector<hsize_t>()
                      : to_h(it->second));
  }

  static std::size_t bytes_of(DType t, std::size_t string_size) {
    switch (t) {
      case DType::Bool:
      case DType::UInt8: return 1;
      case DType::Int32: return 4;
      case DType::String: return string_size == 0 ? 1 : string_size;
      default: return 8;
    }
  }

  // Writes one dataset over the row dimension and attaches its scales.
  void write_row_dataset(const std::string& path, DType dtype,
                         const std::vector<std::size_t>& shape,
                         const std::vector<std::string>& dims,
                         const std::vector<double>& f64,
                         const std::vector<std::int64_t>& i64,
                         const std::vector<std::string>& str,
                         std::size_t string_size, std::size_t row_count,
                         const Support* support, bool local_row) {
    const std::size_t item = bytes_of(dtype, string_size);
    const std::vector<std::size_t> chunk =
        chunk_for(path, item, shape, row_count);
    std::vector<hsize_t> maxshape = to_h(shape);
    maxshape[0] = H5S_UNLIMITED;
    write_any(path, dtype, shape, maxshape, chunk, f64, i64, str,
              string_size);
    attach(path, shape, dims, support, local_row);
  }

  void write_fixed_dataset(const std::string& path, DType dtype,
                           const std::vector<std::size_t>& shape,
                           const std::vector<std::string>& dims,
                           const std::vector<double>& f64,
                           const std::vector<std::int64_t>& i64,
                           const std::vector<std::string>& str,
                           std::size_t string_size,
                           const Support* support) {
    const std::vector<std::size_t> chunk =
        fixed_chunk_for(path, bytes_of(dtype, string_size), shape);
    write_any(path, dtype, shape, {}, chunk, f64, i64, str, string_size);
    attach(path, shape, dims, support, false);
  }

  void write_any(const std::string& path, DType dtype,
                 const std::vector<std::size_t>& shape,
                 const std::vector<hsize_t>& maxshape,
                 const std::vector<std::size_t>& chunk,
                 const std::vector<double>& f64,
                 const std::vector<std::int64_t>& i64,
                 const std::vector<std::string>& str,
                 std::size_t string_size) {
    const std::vector<hsize_t> h_shape = to_h(shape);
    const std::vector<hsize_t> h_chunk = to_h(chunk);
    const FilterPipeline pipeline = filters_for(path);
    switch (dtype) {
      case DType::Float64:
        f_.write_f64(path, h_shape, maxshape, h_chunk, f64, pipeline);
        return;
      case DType::String: {
        std::size_t item = string_size;
        if (item == 0) {
          for (const std::string& s : str) item = std::max(item, s.size());
        }
        f_.write_strings(path, item == 0 ? 1 : item, h_shape, maxshape,
                         h_chunk, str, pipeline);
        return;
      }
      default:
        f_.write_ints(path, dtype, h_shape, maxshape, h_chunk, i64,
                      pipeline);
        return;
    }
  }

  void attach(const std::string& path,
              const std::vector<std::size_t>& shape,
              const std::vector<std::string>& dims, const Support* support,
              bool local_row) {
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      const std::string logical =
          axis < dims.size() ? dims[axis] : std::string();
      if (logical.empty()) continue;
      const std::string s =
          scale_path(logical, shape[axis], support, local_row);
      f_.attach_scale(path, s, static_cast<unsigned>(axis));
    }
  }

  // --- the file, in order -------------------------------------------

  void root_attributes() {
    put("/", "created", d_.created);
    put("/", "format", d_.format);
    put("/", "writer", d_.writer);
    put_bool("/", "aligned", d_.aligned);
    if (d_.generalisation_group.has_value()) {
      put("/", "generalisation_group", *d_.generalisation_group);
    }
    put_extra("/", d_.root_extra);
  }

  void root_scales() {
    scale("/row", static_cast<std::size_t>(d_.n_rows), true);
    for (const std::size_t n : internal::component_lengths(d_)) {
      scale("/component_" +
                internal::format_i64(static_cast<std::int64_t>(n)),
            n, false);
    }
    for (const std::size_t n : internal::draw_lengths(d_)) {
      scale("/draw_" + internal::format_i64(static_cast<std::int64_t>(n)),
            n, false);
    }
    for (const Key& k : d_.keys) {
      if (k.role != "group") continue;
      std::size_t length = 0;
      if (k.category.has_value()) {
        const CategoryTable* t = d_.category(*k.category);
        if (t != nullptr) length = t->entries.size();
      }
      scale("/group_" + k.name, length, false);
    }
    for (const CategoryTable& t : d_.categories) {
      scale("/category_" + t.name, t.entries.size(), false);
    }
  }

  void categories() {
    if (d_.categories.empty() &&
        d_.container_groups.count("/categories") == 0) {
      return;
    }
    f_.make_group("/categories");
    for (const CategoryTable& t : d_.categories) {
      const std::string p = "/categories/" + t.name;
      std::size_t item = t.string_size.value_or(0);
      if (item == 0) {
        for (const std::string& s : t.entries) item = std::max(item, s.size());
      }
      const std::vector<std::size_t> shape{t.entries.size()};
      const std::vector<std::size_t> chunk =
          fixed_chunk_for(p, item == 0 ? 1 : item, shape);
      f_.write_strings(p, item == 0 ? 1 : item, to_h(shape), {},
                       to_h(chunk), t.entries, filters_for(p));
      f_.attach_scale(p, "/category_" + t.name, 0);
    }
  }

  void keys() {
    if (d_.keys.empty() && d_.container_groups.count("/keys") == 0) return;
    f_.make_group("/keys");
    for (const Key& k : d_.keys) {
      const std::string p = "/keys/" + k.name;
      const std::vector<std::size_t> shape{rows()};
      write_row_dataset(p, k.dtype, shape, {"row"}, k.f64, k.i64, k.str,
                        k.string_size.value_or(0), rows(), nullptr, false);
      put(p, "role", k.role);
      if (k.units.has_value()) put(p, "units", *k.units);
      if (k.lower.has_value()) put_real(p, "lower", *k.lower);
      if (k.upper.has_value()) put_real(p, "upper", *k.upper);
      if (k.category.has_value()) put(p, "category", *k.category);
      if (k.trajectory_group.has_value()) {
        put(p, "trajectory_group", *k.trajectory_group);
      }
      if (k.parent.has_value()) put(p, "parent", *k.parent);
      put_extra(p, k.extra);
    }
  }

  void scalars() {
    if (d_.scalars.empty() && d_.container_groups.count("/scalars") == 0) {
      return;
    }
    f_.make_group("/scalars");
    for (const Scalar& s : d_.scalars) {
      const std::string p = "/scalars/" + s.name;
      if (s.is_callable()) {
        f_.make_group(p);
      } else {
        const std::vector<std::size_t> shape{rows()};
        write_row_dataset(p, DType::Float64, shape, {"row"}, s.values, {},
                          {}, 0, rows(), nullptr, false);
      }
      put(p, "units", s.units);
      put(p, "source", s.source);
      if (s.output.has_value()) put(p, "output", *s.output);
      if (s.statistic.has_value()) put(p, "statistic", *s.statistic);
      if (s.of.has_value()) put(p, "of", *s.of);
      if (s.quantile.has_value()) put_real(p, "quantile", *s.quantile);
      if (s.level.has_value()) put_real(p, "level", *s.level);
      if (s.method.has_value()) put(p, "method", *s.method);
      put_extra(p, s.extra);
    }
  }

  void row_support() {
    if (!d_.row_support.has_value()) return;
    std::vector<std::int64_t> values(d_.row_support->begin(),
                                     d_.row_support->end());
    const std::vector<std::size_t> shape{values.size()};
    write_row_dataset("/row_support", DType::Int32, shape, {"row"}, {},
                      values, {}, 0, rows(), nullptr, false);
  }

  void supports() {
    if (d_.supports.empty() &&
        d_.container_groups.count("/supports") == 0) {
      return;
    }
    f_.make_group("/supports");
    for (std::size_t i = 0; i < d_.supports.size(); ++i) {
      write_support(d_.supports[i], i);
    }
  }

  void write_support(const Support& s, std::size_t index) {
    const std::string sp = "/supports/" + s.name;
    f_.make_group(sp);
    // Section 20: a support of kind `none` carries no `node` scale,
    // because a zero-length fixed dimension is not legal.
    if (s.kind != "none") {
      scale(sp + "/node", static_cast<std::size_t>(s.n_nodes), false);
    }
    const bool has_cells = !s.cell_types.empty() || s.n_cells > 0;
    if (has_cells) {
      scale(sp + "/cell", s.cell_types.size(), false);
      scale(sp + "/cell_plus_one", s.cell_offsets.size(), false);
      scale(sp + "/index", s.cell_connectivity.size(), false);
      std::vector<std::int64_t> types(s.cell_types.begin(),
                                      s.cell_types.end());
      write_fixed_dataset(sp + "/cell_types", DType::UInt8,
                          {s.cell_types.size()}, {"cell"}, {}, types, {}, 0,
                          &s);
      write_fixed_dataset(sp + "/cell_offsets", DType::Int64,
                          {s.cell_offsets.size()}, {"cell_plus_one"}, {},
                          s.cell_offsets, {}, 0, &s);
      write_fixed_dataset(sp + "/cell_connectivity", DType::Int64,
                          {s.cell_connectivity.size()}, {"index"}, {},
                          s.cell_connectivity, {}, 0, &s);
    }
    put(sp, "kind", s.kind);
    put_int(sp, "n_nodes", s.n_nodes);
    put_int(sp, "n_cells", s.n_cells);
    put(sp, "support_id", s.support_id);
    put_extra(sp, s.extra);

    const bool local_row = internal::needs_local_row(d_, s);
    const std::size_t support_rows = internal::rows_on_support(d_, index);
    if (local_row) {
      scale(sp + "/row", support_rows, true);
    }

    if (s.coordinates.has_value()) {
      write_slot(sp + "/coordinates", *s.coordinates, &s, local_row,
                 support_rows);
    }
    if (!s.node_arrays.empty()) {
      f_.make_group(sp + "/node_arrays");
      for (const ArraySlot& a : s.node_arrays) {
        write_slot(sp + "/node_arrays/" + a.name, a, &s, local_row,
                   support_rows);
      }
    }
    if (!s.cell_arrays.empty()) {
      f_.make_group(sp + "/cell_arrays");
      for (const ArraySlot& a : s.cell_arrays) {
        write_slot(sp + "/cell_arrays/" + a.name, a, &s, local_row,
                   support_rows);
      }
    }
    // Groups this version does not know, put back as they came
    // (section 28), as at the root.
    for (const std::string& name : s.unknown_groups) {
      const auto copy = s.unknown_group_copies.find(name);
      if (copy != s.unknown_group_copies.end() && !copy->second.empty()) {
        internal::restore_group(f_, sp + "/" + name, copy->second);
      } else {
        f_.make_group(sp + "/" + name);
      }
    }
  }

  void write_slot(const std::string& p, const ArraySlot& a,
                  const Support* s, bool local_row,
                  std::size_t support_rows) {
    if (a.is_callable()) {
      // Section 19: a slot served by a callable is an empty group
      // carrying the slot's attributes and no data.
      f_.make_group(p);
    } else if (a.varies == "row") {
      write_row_dataset(p, a.data.dtype, a.data.shape, a.data.dims,
                        a.data.f64, a.data.i64, a.data.str, 0,
                        local_row ? support_rows : rows(), s, local_row);
    } else {
      write_fixed_dataset(p, a.data.dtype, a.data.shape, a.data.dims,
                          a.data.f64, a.data.i64, a.data.str, 0, s);
    }
    put(p, "role", a.role);
    put(p, "varies", a.varies);
    if (a.units.has_value()) put(p, "units", *a.units);
    put_int(p, "components", a.components);
    put(p, "source", a.source);
    if (a.output.has_value()) put(p, "output", *a.output);
    if (a.statistic.has_value()) put(p, "statistic", *a.statistic);
    if (a.of.has_value()) put(p, "of", *a.of);
    if (a.quantile.has_value()) put_real(p, "quantile", *a.quantile);
    if (a.level.has_value()) put_real(p, "level", *a.level);
    if (a.method.has_value()) put(p, "method", *a.method);
    if (a.category.has_value()) put(p, "category", *a.category);
    if (a.recomputed.has_value()) put_bool(p, "recomputed", *a.recomputed);
    if (a.derived_from.has_value()) put(p, "derived_from", *a.derived_from);
    if (a.recipe.has_value()) put(p, "recipe", *a.recipe);
    if (a.reference.has_value()) put(p, "reference", *a.reference);
    put_extra(p, a.extra);
  }

  void callables() {
    if (d_.callables.empty() &&
        d_.container_groups.count("/callables") == 0) {
      return;
    }
    f_.make_group("/callables");
    for (const StoredCallable& c : d_.callables) {
      const std::string p = "/callables/" + c.id;
      f_.make_group(p);
      put(p, "type", c.type);
      if (c.repr.has_value()) put(p, "repr", *c.repr);
      const internal::StoredStorage stored{&d_.chunk_overrides,
                                           &d_.filters};
      internal::write_dict_group(f_, p, c.dict, &stored);
    }
  }

  void trailing_groups() {
    if (d_.has_notes) {
      f_.make_group("/notes");
      put_extra("/notes", d_.notes);
    }
    // The private group goes back exactly as it came, object by
    // object, without this writer having decided what any of it means
    // (sections 12 and 29).  A dataset built from vectors carries none
    // and nothing is written.
    if (!d_.private_group.empty()) {
      internal::restore_group(f_, "/private", d_.private_group);
    } else if (d_.container_groups.count("/private") != 0) {
      f_.make_group("/private");
    }
    // A group this version does not know goes back as it came: its
    // name is reported (W11) and nothing in it is interpreted, and a
    // rewrite that kept the name and dropped the contents would lose
    // them in silence.
    for (const std::string& name : d_.unknown_root_groups) {
      const auto copy = d_.unknown_root_copies.find(name);
      if (copy != d_.unknown_root_copies.end() && !copy->second.empty()) {
        internal::restore_group(f_, "/" + name, copy->second);
      } else {
        f_.make_group("/" + name);
      }
    }
  }

  const Dataset& d_;
  File& f_;
};

}  // namespace

namespace internal {

void check_dataset_names(const Dataset& d) {
  auto check = [](const std::string& name) {
    if (!internal::legal_netcdf_name(name)) {
      throw Error("E33", "\"" + name + "\" is not a legal netCDF-4 name");
    }
    if (internal::reserved_name(name)) {
      throw Error("E33", "\"" + name +
                             "\" begins with the reserved prefix mestra_");
    }
  };
  for (const Key& k : d.keys) check(k.name);
  for (const Scalar& s : d.scalars) check(s.name);
  for (const CategoryTable& t : d.categories) check(t.name);
  for (const StoredCallable& c : d.callables) check(c.id);
  for (const Support& s : d.supports) {
    check(s.name);
    for (const ArraySlot& a : s.node_arrays) check(a.name);
    for (const ArraySlot& a : s.cell_arrays) check(a.name);
  }
}

// The one thing a caller cannot say in a Dataset and mean: a slot
// whose `varies` or `components` disagrees with the shape it was
// built with.  The builders derive both from `dims`, so the only way
// to get here is to have assigned to the member afterwards and
// expected the array to be reshaped.  It is refused before anything is
// opened, with the rule the validator would give the file, because a
// writable field that is read only at construction time is a trap
// (conventions section 1).
void check_dataset_shapes(const Dataset& d) {
  for (const Support& s : d.supports) {
    std::vector<const ArraySlot*> slots;
    if (s.coordinates.has_value()) slots.push_back(&*s.coordinates);
    for (const ArraySlot& a : s.node_arrays) slots.push_back(&a);
    for (const ArraySlot& a : s.cell_arrays) slots.push_back(&a);
    for (const ArraySlot* a : slots) {
      if (a->is_callable() || a->data.dims.empty()) continue;
      const std::string axis =
          a->location == Location::Node ? "node" : "cell";
      const std::string path = "/supports/" + s.name +
                               (a->role == "coordinates"
                                    ? "/"
                                    : (a->location == Location::Node
                                           ? "/node_arrays/"
                                           : "/cell_arrays/")) +
                               a->name;
      // A slot holding draws carries the draw axis first when nothing
      // leads it (section 19: (row | group | -, [draw], node | cell,
      // component)), so a shared set of draws starts with `draw`.
      const bool draws = a->statistic.has_value() && *a->statistic == "draw";
      const std::string want =
          a->varies == "none" ? (draws ? std::string("draw") : axis)
                              : a->varies;
      if (a->data.dims.front() != want) {
        throw Error(
            "E04",
            path + ": `varies` is \"" + a->varies +
                "\" and the array was built with a leading \"" +
                a->data.dims.front() +
                "\" axis; a slot's shape is fixed when it is built, so "
                "build it again with the `dims` you mean rather than "
                "assigning to `varies`");
      }
      for (std::size_t i = 0; i < a->data.dims.size(); ++i) {
        if (a->data.dims[i] != "component") continue;
        if (static_cast<std::int64_t>(a->data.shape[i]) != a->components) {
          throw Error(
              "E31",
              path + ": `components` is " +
                  internal::format_i64(a->components) +
                  " and the array was built with a component axis of " +
                  internal::format_i64(
                      static_cast<std::int64_t>(a->data.shape[i])) +
                  "; build it again with the `dims` you mean rather than "
                  "assigning to `components`");
        }
      }
    }
  }
}

}  // namespace internal

namespace {

std::string findings_text(const std::string& path, const Report& r) {
  std::string out = "mestra::write refused \"" + path + "\": " +
                    internal::format_i64(
                        static_cast<std::int64_t>(r.errors.size())) +
                    " error(s)";
  for (const Finding& f : r.errors) {
    out += "\n  " + (f.id.empty() ? std::string("!") : f.id) + " " +
           f.where + ": " + f.message;
  }
  return out;
}

void write_file(const Dataset& d, const std::string& path) {
  File f = File::create(path);
  Writer w(d, f);
  w.run();
}

// The staging and the publish both happen in the destination's
// directory (docs/compatibility.md, "Replacing files"), so a
// destination that cannot take a file is refused here, naming `path`,
// before anything is validated, staged or created.  An I/O failure
// after this point is the operating system's error, and leaves an
// existing destination as it was.  The Python and Julia writers refuse
// the same four cases in the same words.
void check_destination(const std::string& path) {
  namespace fs = std::filesystem;
  if (path.empty()) {
    throw Error("", "path is empty; pass the name of the file to write");
  }
  std::error_code ec;
  if (fs::is_directory(path, ec)) {
    throw Error("", "path names a directory, which a file cannot replace; "
                    "pass a file name: \"" + path + "\"");
  }
  fs::path parent = fs::absolute(path, ec).parent_path();
  if (!fs::is_directory(parent, ec)) {
    throw Error("", "the directory " + parent.string() + " does not exist; "
                    "the file is staged and published there, so create it "
                    "first: \"" + path + "\"");
  }
  // A directory whose entries cannot be added to.  Permissions on
  // Windows are not expressed this way, so the check is a POSIX one and
  // a refusal there comes from the operating system at staging time.
  const fs::perms mode = fs::status(parent, ec).permissions();
  const bool writable = (mode & (fs::perms::owner_write | fs::perms::group_write |
                                 fs::perms::others_write)) != fs::perms::none;
  if (!ec && !writable) {
    throw Error("", "the directory " + parent.string() + " is not writable; "
                    "the file is staged and published there: \"" + path +
                    "\"");
  }
}

}  // namespace

void write(const Dataset& d, const std::string& path,
           const WriteOptions& options) {
  internal::check_dataset_names(d);
  internal::check_dataset_shapes(d);
  check_destination(path);
  // What a read had to leave out of a callable's dictionary would be
  // dropped by a rewrite, so a checked write refuses rather than lose
  // it in silence (conventions section 7).
  if (options.check) {
    for (const Finding& f : d.not_read) {
      if (f.id == "E32") {
        throw Error("E32", "\"" + path + "\" was not written: " + f.where +
                               " " + f.message +
                               ", so this dataset holds the dictionary "
                               "without it; pass check = false to write "
                               "it so");
      }
    }
  }
  if (!options.check) {
    write_file(d, path);
    return;
  }
  // Conventions section 2: validate first and refuse on any error.
  // The file is built beside the name the caller gave and moved into
  // place only once it validates, so a refusal never leaves a file
  // the validator would reject where the caller asked for one.
  const std::string beside = path + ".mestra-writing";
  std::remove(beside.c_str());
  try {
    write_file(d, beside);
  } catch (...) {
    std::remove(beside.c_str());
    throw;
  }
  const Report r = validate(beside);
  if (!r.ok()) {
    std::remove(beside.c_str());
    throw Error(r.errors.front().id, findings_text(path, r));
  }
  if (std::rename(beside.c_str(), path.c_str()) != 0) {
    std::remove(beside.c_str());
    throw Error("", "cannot move the written file into place at \"" + path +
                        "\"");
  }
}

void write_dict(const Dict& dict, const std::string& type,
                const std::string& callable_id, const std::string& path) {
  Dataset d;
  d.writer = library_version();
  d.created = "1970-01-01T00:00:00Z";
  d.aligned = true;
  d.n_rows = 0;
  d.container_groups.insert("/keys");
  d.add_callable(callable_id, type, dict);
  write(d, path);
}

}  // namespace mestra
