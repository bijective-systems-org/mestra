#include "mestra/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "mestra/callable.hpp"
#include "mestra/io.hpp"
#include "mestra/sha256.hpp"
#include "names.hpp"

namespace mestra {
namespace {

// A builder refuses at build time, with the rule identifier, whatever
// the validator would refuse at read time (conventions section 1), and
// says which argument to change (section 6).  Error puts the rule
// identifier in front, so what a caller reads is the identifier, then
// the object path, then what to do about it.
[[noreturn]] void refuse(const std::string& rule, const std::string& path,
                         const std::string& message) {
  throw Error(rule, path + ": " + message);
}

void check_builder_name(const std::string& path, const std::string& name) {
  if (!internal::legal_netcdf_name(name)) {
    refuse("E33", path,
           "\"" + name +
               "\" is not a legal netCDF-4 name; change the `name` "
               "argument");
  }
  if (internal::reserved_name(name)) {
    refuse("E33", path,
           "\"" + name +
               "\" begins with the reserved prefix mestra_; change the "
               "`name` argument");
  }
}

// E11: a scalar carries units, and "1" is how this format spells a
// dimensionless quantity, so an empty string is a caller who meant to
// say something and did not.  Refused here rather than at write time,
// because conventions section 1 asks a builder to refuse at build
// time whatever the validator would refuse at read time, and section 6
// asks it to say which argument to change.
void check_builder_units(const std::string& path, const std::string& units) {
  if (units.empty()) {
    refuse("E11", path,
           "a scalar carries units; pass units (\"1\" for a "
           "dimensionless one)");
  }
}

const ArraySlot* slot_named(const std::vector<ArraySlot>& v,
                            const std::string& wanted) {
  for (const ArraySlot& a : v) {
    if (a.name == wanted) return &a;
  }
  return nullptr;
}

void put_i64_le(Sha256& h, std::int64_t v) {
  unsigned char bytes[8];
  std::uint64_t u = static_cast<std::uint64_t>(v);
  for (int i = 0; i < 8; ++i) {
    bytes[i] = static_cast<unsigned char>((u >> (8 * i)) & 0xffu);
  }
  h.update(bytes, 8);
}

void put_f64_le(Sha256& h, double v) {
  // IEEE 754 float64, little-endian, which is what every platform this
  // builds on stores natively; the bytes are taken through memcpy so
  // that no strict-aliasing rule is bent.
  unsigned char bytes[8];
  std::memcpy(bytes, &v, 8);
  h.update(bytes, 8);
}

}  // namespace

// --- small value types ----------------------------------------------

std::size_t Key::rows() const {
  switch (dtype) {
    case DType::Float64: return f64.size();
    case DType::String: return str.size();
    default: return i64.size();
  }
}

bool Scalar::is_callable() const {
  return !internal::callable_of_source(source).empty();
}

std::string Scalar::callable_id() const {
  return internal::callable_of_source(source);
}

bool ArraySlot::is_callable() const {
  return !internal::callable_of_source(source).empty();
}

std::string ArraySlot::callable_id() const {
  return internal::callable_of_source(source);
}

int CategoryTable::index_of(const std::string& entry) const {
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (entries[i] == entry) return static_cast<int>(i);
  }
  return -1;
}

// --- support_id ------------------------------------------------------

std::string support_id_digest(
    std::int64_t n_nodes, const std::vector<std::uint8_t>& cell_types,
    const std::vector<std::int64_t>& cell_offsets,
    const std::vector<std::int64_t>& connectivity,
    const std::vector<double>* axis_coordinates) {
  Sha256 h;
  put_i64_le(h, n_nodes);
  if (!cell_types.empty()) h.update(cell_types.data(), cell_types.size());
  for (const std::int64_t v : cell_offsets) put_i64_le(h, v);
  for (const std::int64_t v : connectivity) put_i64_le(h, v);
  if (axis_coordinates != nullptr) {
    for (const double v : *axis_coordinates) put_f64_le(h, v);
  }
  return h.hex();
}

const ArraySlot* Support::node_array(const std::string& wanted) const {
  return slot_named(node_arrays, wanted);
}

const ArraySlot* Support::cell_array(const std::string& wanted) const {
  return slot_named(cell_arrays, wanted);
}

ArraySlot* Support::node_array(const std::string& wanted) {
  return const_cast<ArraySlot*>(
      static_cast<const Support*>(this)->node_array(wanted));
}

ArraySlot* Support::cell_array(const std::string& wanted) {
  return const_cast<ArraySlot*>(
      static_cast<const Support*>(this)->cell_array(wanted));
}

std::string Support::computed_support_id() const {
  // Section 24: "a support of kind `axis` or `none` has no cell
  // arrays, so steps 2 to 4 contribute no bytes at all for it".  The
  // kind decides, not what the file happens to carry, so a file that
  // wrongly puts cell arrays on an axis support breaks E38 and not
  // E08 as well.  A support that declares no kind, or one section 6
  // does not have, has no digest at all.
  if (kind.empty()) {
    throw Error("E39", "support \"" + name +
                           "\" does not say its kind, and the kind decides "
                           "which of its arrays section 24 hashes, so it "
                           "has no support_id");
  }
  if (kind != "mesh" && kind != "axis" && kind != "none") {
    throw Error("E02", "support \"" + name + "\" is of kind \"" + kind +
                           "\", which section 6 does not have, so it has "
                           "no support_id");
  }
  static const std::vector<std::uint8_t> kNoTypes;
  static const std::vector<std::int64_t> kNoInts;
  const bool has_cells = kind == "mesh";
  if (kind == "axis" && coordinates.has_value()) {
    return support_id_digest(n_nodes, kNoTypes, kNoInts, kNoInts,
                             &coordinates->data.f64);
  }
  if (!has_cells) {
    return support_id_digest(n_nodes, kNoTypes, kNoInts, kNoInts, nullptr);
  }
  return support_id_digest(n_nodes, cell_types, cell_offsets,
                           cell_connectivity, nullptr);
}

int cell_type_nodes(std::uint8_t code) {
  switch (code) {
    case 1: return 1;    // vertex
    case 3: return 2;    // line
    case 5: return 3;    // triangle
    case 7: return 0;    // polygon: three or more
    case 9: return 4;    // quadrilateral
    case 10: return 4;   // tetrahedron
    case 12: return 8;   // hexahedron
    case 13: return 6;   // wedge
    case 14: return 5;   // pyramid
    case 21: return 3;   // quadratic line
    case 22: return 6;   // quadratic triangle
    case 23: return 8;   // quadratic quadrilateral
    case 24: return 10;  // quadratic tetrahedron
    case 25: return 20;  // quadratic hexahedron
    case 26: return 15;  // quadratic wedge
    case 27: return 13;  // quadratic pyramid
    default: return -1;
  }
}

std::size_t default_chunk_rows(std::size_t item_bytes,
                               const std::vector<std::size_t>& other_extents,
                               std::size_t row_count) {
  if (row_count == 0) return 1;
  std::size_t b = item_bytes;
  for (const std::size_t e : other_extents) b *= (e == 0 ? 1 : e);
  if (b == 0) b = 1;
  std::size_t c = 1048576 / b;
  if (c < 1) c = 1;
  if (c > row_count) c = row_count;
  return c;
}

// --- Dataset lookup --------------------------------------------------

const Key* Dataset::key(const std::string& name) const {
  for (const Key& k : keys) {
    if (k.name == name) return &k;
  }
  return nullptr;
}

const Scalar* Dataset::scalar(const std::string& name) const {
  for (const Scalar& s : scalars) {
    if (s.name == name) return &s;
  }
  return nullptr;
}

const Support* Dataset::support(const std::string& name) const {
  for (const Support& s : supports) {
    if (s.name == name) return &s;
  }
  return nullptr;
}

Key* Dataset::key(const std::string& name) {
  return const_cast<Key*>(static_cast<const Dataset*>(this)->key(name));
}

Scalar* Dataset::scalar(const std::string& name) {
  return const_cast<Scalar*>(static_cast<const Dataset*>(this)->scalar(name));
}

Support* Dataset::support(const std::string& name) {
  return const_cast<Support*>(
      static_cast<const Dataset*>(this)->support(name));
}

const CategoryTable* Dataset::category(const std::string& name) const {
  for (const CategoryTable& c : categories) {
    if (c.name == name) return &c;
  }
  return nullptr;
}

const StoredCallable* Dataset::callable(const std::string& id) const {
  for (const StoredCallable& c : callables) {
    if (c.id == id) return &c;
  }
  return nullptr;
}

const Support* Dataset::support_of_row(std::size_t row) const {
  if (supports.empty()) return nullptr;
  if (!row_support.has_value()) return &supports.front();
  if (row >= row_support->size()) return nullptr;
  const std::int32_t at = (*row_support)[row];
  if (at < 0 || static_cast<std::size_t>(at) >= supports.size()) {
    return nullptr;
  }
  return &supports[static_cast<std::size_t>(at)];
}

std::vector<std::string> Dataset::key_order() const {
  std::vector<std::string> names;
  names.reserve(keys.size());
  for (const Key& k : keys) names.push_back(k.name);
  std::sort(names.begin(), names.end(), bytes_less);
  return names;
}

// --- Dataset building ------------------------------------------------

namespace {

template <typename T>
T& insert_sorted(std::vector<T>& into, T value) {
  auto at = std::lower_bound(
      into.begin(), into.end(), value,
      [](const T& a, const T& b) { return bytes_less(a.name, b.name); });
  return *into.insert(at, std::move(value));
}

}  // namespace

Key& Dataset::add_key(const std::string& name, std::vector<double> values,
                      const std::string& role,
                      const std::string& units) {
  check_builder_name("/keys/" + name, name);
  Key k;
  k.name = name;
  k.role = role;
  k.units = units;
  k.dtype = DType::Float64;
  k.f64 = std::move(values);
  // Conventions section 1: with no bounds given, the observed finite
  // range is what the file records, so that the same arrays give the
  // same file in every language and W04 and W08 are decidable.
  bool any = false;
  double lo = 0.0;
  double hi = 0.0;
  for (const double v : k.f64) {
    if (!std::isfinite(v)) continue;
    if (!any) {
      lo = v;
      hi = v;
      any = true;
    } else {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  if (any) {
    k.lower = lo;
    k.upper = hi;
  }
  if (n_rows == 0) n_rows = static_cast<std::int64_t>(k.f64.size());
  container_groups.insert("/keys");
  return insert_sorted(keys, std::move(k));
}

Key& Dataset::add_category_key(const std::string& name,
                               std::vector<std::int64_t> ids,
                               const std::string& role,
                               const std::string& category_table,
                               DType dtype) {
  check_builder_name("/keys/" + name, name);
  Key k;
  k.name = name;
  k.role = role;
  k.category = category_table;
  k.dtype = dtype;
  k.i64 = std::move(ids);
  if (n_rows == 0) n_rows = static_cast<std::int64_t>(k.i64.size());
  container_groups.insert("/keys");
  return insert_sorted(keys, std::move(k));
}

Scalar& Dataset::add_scalar(const std::string& name,
                            std::vector<double> values,
                            const std::string& units) {
  check_builder_name("/scalars/" + name, name);
  check_builder_units("/scalars/" + name, units);
  Scalar s;
  s.name = name;
  s.units = units;
  s.source = "data";
  s.values = std::move(values);
  if (n_rows == 0) n_rows = static_cast<std::int64_t>(s.values.size());
  container_groups.insert("/scalars");
  return insert_sorted(scalars, std::move(s));
}

CategoryTable& Dataset::add_category_table(
    const std::string& name, std::vector<std::string> entries) {
  check_builder_name("/categories/" + name, name);
  CategoryTable c;
  c.name = name;
  c.entries = std::move(entries);
  container_groups.insert("/categories");
  return insert_sorted(categories, std::move(c));
}

void Dataset::set_generalisation_group(const std::string& name) {
  generalisation_group = name;
}

Support& Dataset::add_mesh_support(
    const std::string& name, std::int64_t n_nodes,
    std::vector<std::uint8_t> cell_types,
    std::vector<std::int64_t> cell_offsets,
    std::vector<std::int64_t> connectivity) {
  Support s;
  s.name = name;
  s.kind = "mesh";
  s.n_nodes = n_nodes;
  s.n_cells = static_cast<std::int64_t>(cell_types.size());
  s.cell_types = std::move(cell_types);
  s.cell_offsets = std::move(cell_offsets);
  s.cell_connectivity = std::move(connectivity);
  s.support_id = s.computed_support_id();
  container_groups.insert("/supports");
  Support& ref = insert_sorted(supports, std::move(s));
  aligned = supports.size() <= 1;
  return ref;
}

Support& Dataset::add_axis_support(const std::string& name,
                                   const std::vector<double>& coordinates,
                                   const std::string& units) {
  Support s;
  s.name = name;
  s.kind = "axis";
  s.n_nodes = static_cast<std::int64_t>(coordinates.size());
  s.n_cells = 0;
  container_groups.insert("/supports");
  Support& ref = insert_sorted(supports, std::move(s));
  // Section 5: the coordinates of an axis support vary along nothing,
  // because they are part of that support's identity.
  set_coordinates(ref, coordinates, units, {"node"});
  ref.support_id = ref.computed_support_id();
  aligned = supports.size() <= 1;
  return ref;
}

Support& Dataset::add_none_support(const std::string& name) {
  Support s;
  s.name = name;
  s.kind = "none";
  s.n_nodes = 0;
  s.n_cells = 0;
  s.support_id = s.computed_support_id();
  container_groups.insert("/supports");
  Support& ref = insert_sorted(supports, std::move(s));
  aligned = supports.size() <= 1;
  return ref;
}

StoredCallable& Dataset::add_callable(const std::string& id,
                                      const std::string& type, Dict dict) {
  StoredCallable c;
  c.id = id;
  c.type = type;
  c.dict = std::move(dict);
  container_groups.insert("/callables");
  auto at = std::lower_bound(callables.begin(), callables.end(), c,
                             [](const StoredCallable& a,
                                const StoredCallable& b) {
                               return bytes_less(a.id, b.id);
                             });
  return *callables.insert(at, std::move(c));
}

StoredCallable& Dataset::add_callable(const std::string& id,
                                      const Callable& c) {
  StoredCallable& stored = add_callable(id, c.type(), c.to_dict());
  const std::string line = c.repr();
  if (!line.empty()) stored.repr = line;
  return stored;
}

// --- support helpers -------------------------------------------------

namespace {

// The logical name of the axis a support contributes at `where`.
const char* support_axis(Location where) {
  return where == Location::Node ? "node" : "cell";
}

std::string count_text(std::size_t n) {
  return internal::format_i64(static_cast<std::int64_t>(n));
}

// What a builder works out from the `dims` it was given: the stored
// shape of section 4, the `varies` and the `components` that follow
// from it, and the permutation from the caller's axis order into the
// stored one.  Nothing here is guessed: an axis whose length neither
// the support nor the values settle is refused by name.
struct Resolved {
  std::string varies = "none";
  std::int64_t components = 1;
  std::vector<std::string> dims;        // stored order
  std::vector<std::size_t> shape;       // stored order
  std::vector<std::size_t> source_shape; // the caller's order
  std::vector<int> source_axis;         // stored axis -> caller axis, -1
};

Resolved resolve(const std::string& path, const Dims& dims, Location where,
                 std::int64_t support_extent, std::size_t values,
                 bool have_values) {
  const std::string here = support_axis(where);
  const std::string there = where == Location::Node ? "cell" : "node";

  std::vector<std::string> names;
  std::vector<std::int64_t> extent;
  int leading = -1;
  int draw = -1;
  int support_at = -1;
  int component = -1;
  for (std::size_t i = 0; i < dims.size(); ++i) {
    const std::string& n = dims[i].name;
    const int at = static_cast<int>(i);
    for (const std::string& seen : names) {
      if (seen == n) {
        refuse("E25", path,
               "`dims` names the axis \"" + n + "\" twice");
      }
    }
    if (n == "row" || n.compare(0, 6, "group:") == 0) {
      if (n.size() == 6) {
        refuse("E04", path,
               "\"group:\" in `dims` names no group key; write "
               "\"group:<key>\" with the name of a key of role group");
      }
      if (leading >= 0) {
        refuse("E04", path,
               "`dims` names both \"" + names[static_cast<std::size_t>(
                   leading)] + "\" and \"" + n +
                   "\"; an array varies along row, along one group, or "
                   "along neither");
      }
      leading = at;
    } else if (n == "draw") {
      draw = at;
    } else if (n == here) {
      support_at = at;
    } else if (n == there) {
      refuse("E25", path,
             "`dims` names the \"" + there + "\" axis on a " + here +
                 " array; it carries the \"" + here + "\" axis");
    } else if (n == "component") {
      component = at;
    } else {
      refuse("E25", path,
             "\"" + n +
                 "\" is not a dimension name of section 4; `dims` takes "
                 "row, group:<key>, draw, " + here + " and component");
    }
    names.push_back(n);
    extent.push_back(dims[i].extent);
  }
  if (support_at < 0) {
    refuse("E25", path,
           "`dims` does not name the \"" + here +
               "\" axis, which every array on a support carries");
  }

  // The support settles its own axis; a caller who states it as well
  // must state it right.
  const std::size_t sa = static_cast<std::size_t>(support_at);
  if (extent[sa] >= 0 && extent[sa] != support_extent) {
    refuse("E05", path,
           "`dims` gives the \"" + here + "\" axis " +
               internal::format_i64(extent[sa]) +
               " and the support has " +
               internal::format_i64(support_extent) +
               "; leave the length off and the support decides it");
  }
  extent[sa] = support_extent;

  // Whatever is left unknown has to follow from the number of values,
  // and only one thing can.
  std::size_t known = 1;
  std::vector<std::size_t> unknown;
  for (std::size_t i = 0; i < extent.size(); ++i) {
    if (extent[i] < 0) {
      unknown.push_back(i);
    } else {
      known *= static_cast<std::size_t>(extent[i]);
    }
  }
  const bool component_unknown =
      component >= 0 && extent[static_cast<std::size_t>(component)] < 0;
  const std::string count_rule = leading >= 0 ? "E04" : "E05";

  if (!have_values) {
    // A slot served by a callable stores nothing, so only `varies` and
    // `components` have to come out; but `components` is an attribute
    // of the slot and there are no values to work it out from.
    if (component_unknown) {
      refuse("E31", path,
             "the \"component\" axis has no length and a callable slot "
             "carries no values to work one out from; write it as "
             "{\"component\", <n>}");
    }
    for (const std::size_t i : unknown) extent[i] = 1;
  } else if (unknown.size() > 1) {
    std::string listed;
    for (std::size_t at = 0; at < unknown.size(); ++at) {
      if (at != 0) listed += at + 1 == unknown.size() ? "\" and \"" : "\", \"";
      listed += names[unknown[at]];
    }
    // The axis to suggest is the component one when it is among them,
    // because a caller knows how many components they have and not
    // always how many rows.
    const std::string suggest =
        component_unknown ? std::string("component")
                          : names[unknown.front()];
    refuse(component_unknown ? "E31" : count_rule, path,
           "the length of \"" + listed + "\" cannot " +
               (unknown.size() == 2 ? "both" : "all") +
               " be worked out from " + count_text(values) +
               " values; give one of them a length in `dims`, as "
               "{\"" + suggest + "\", <n>}");
  } else if (unknown.size() == 1) {
    const std::size_t i = unknown.front();
    if (known == 0 || values % known != 0) {
      refuse(count_rule, path,
             count_text(values) + " values do not divide into blocks of " +
                 count_text(known) + ", which is what one \"" + names[i] +
                 "\" holds; change `values` or the lengths in `dims`");
    }
    extent[i] = static_cast<std::int64_t>(values / known);
  } else if (known != values) {
    refuse(count_rule, path,
           count_text(values) + " values do not fill a shape of " +
               count_text(known) +
               "; change `values` or the lengths in `dims`");
  }

  Resolved r;
  r.varies = leading >= 0
                 ? names[static_cast<std::size_t>(leading)]
                 : std::string("none");
  r.components =
      component >= 0 ? extent[static_cast<std::size_t>(component)] : 1;
  if (r.components <= 0) {
    refuse("E31", path,
           "the \"component\" axis has length " +
               internal::format_i64(r.components) +
               "; an array has one component or more");
  }
  auto push = [&r, &extent](int caller_axis, const std::string& name) {
    r.dims.push_back(name);
    r.shape.push_back(
        caller_axis < 0
            ? std::size_t(1)
            : static_cast<std::size_t>(
                  extent[static_cast<std::size_t>(caller_axis)]));
    r.source_axis.push_back(caller_axis);
  };
  if (leading >= 0) push(leading, r.varies);
  if (draw >= 0) push(draw, "draw");
  push(support_at, here);
  push(component, "component");
  for (const std::int64_t e : extent) {
    r.source_shape.push_back(static_cast<std::size_t>(e));
  }
  return r;
}

// The caller's flattening put the axes in the caller's order; the file
// wants the order of section 4.  One pass, and the identity case costs
// the copy it would have cost anyway.
template <typename T>
std::vector<T> in_stored_order(const std::vector<T>& in, const Resolved& r) {
  std::size_t total = 1;
  for (const std::size_t e : r.shape) total *= e;
  if (total == 0 || in.empty()) return std::vector<T>();
  std::vector<std::size_t> stride(r.source_shape.size(), 1);
  for (std::size_t i = r.source_shape.size(); i-- > 1;) {
    stride[i - 1] = stride[i] * r.source_shape[i];
  }
  std::vector<T> out(total);
  std::vector<std::size_t> at(r.shape.size(), 0);
  for (std::size_t f = 0; f < total; ++f) {
    std::size_t source = 0;
    for (std::size_t i = 0; i < r.shape.size(); ++i) {
      const int axis = r.source_axis[i];
      if (axis >= 0) {
        source += at[i] * stride[static_cast<std::size_t>(axis)];
      }
    }
    out[f] = source < in.size() ? in[source] : T();
    for (std::size_t i = r.shape.size(); i-- > 0;) {
      if (++at[i] < r.shape[i]) break;
      at[i] = 0;
    }
  }
  return out;
}

ArraySlot& add_slot(Support& s, Location where, ArraySlot slot) {
  std::vector<ArraySlot>& into =
      where == Location::Node ? s.node_arrays : s.cell_arrays;
  auto at = std::lower_bound(
      into.begin(), into.end(), slot,
      [](const ArraySlot& a, const ArraySlot& b) {
        return bytes_less(a.name, b.name);
      });
  return *into.insert(at, std::move(slot));
}

std::int64_t extent_of(const Support& s, Location where) {
  return where == Location::Node ? s.n_nodes : s.n_cells;
}

ArraySlot built_slot(const std::string& name, Location where,
                     const Resolved& r) {
  ArraySlot a;
  a.name = name;
  a.varies = r.varies;
  a.components = r.components;
  a.source = "data";
  a.location = where;
  a.data.dims = r.dims;
  a.data.shape = r.shape;
  return a;
}

}  // namespace

void set_coordinates(Support& s, const std::vector<double>& values,
                     const std::string& units, const Dims& dims) {
  const std::string path = "/supports/" + s.name + "/coordinates";
  const Resolved r =
      resolve(path, dims, Location::Node, s.n_nodes, values.size(), true);
  ArraySlot c = built_slot("coordinates", Location::Node, r);
  c.role = "coordinates";
  c.units = units;
  c.data.dtype = DType::Float64;
  c.data.f64 = in_stored_order(values, r);
  s.coordinates = std::move(c);
}

void set_callable_coordinates(Support& s, const std::string& units,
                              const Dims& dims,
                              const std::string& callable_id,
                              const std::string& output) {
  const std::string path = "/supports/" + s.name + "/coordinates";
  if (s.kind != "mesh") {
    throw Error("E03", path + ": a callable may serve the coordinates of "
                       "a mesh support; this support is of kind " +
                       s.kind + ", whose coordinates are stored");
  }
  const Resolved r =
      resolve(path, dims, Location::Node, s.n_nodes, 0, false);
  ArraySlot c = built_slot("coordinates", Location::Node, r);
  c.role = "coordinates";
  c.units = units;
  c.source = "callable:" + callable_id;
  c.output = output;
  c.data = Array();
  s.coordinates = std::move(c);
}

namespace {

ArraySlot& add_array(Support& s, Location where, const std::string& name,
                     const std::vector<double>& values,
                     const std::string& units, const Dims& dims) {
  const std::string path = "/supports/" + s.name +
                           (where == Location::Node ? "/node_arrays/"
                                                    : "/cell_arrays/") +
                           name;
  check_builder_name(path, name);
  const Resolved r = resolve(path, dims, where, extent_of(s, where),
                             values.size(), true);
  ArraySlot a = built_slot(name, where, r);
  a.role = "field";
  a.units = units;
  a.data.dtype = DType::Float64;
  a.data.f64 = in_stored_order(values, r);
  return add_slot(s, where, std::move(a));
}

ArraySlot& add_label(Support& s, Location where, const std::string& name,
                     const std::vector<std::int64_t>& values,
                     std::optional<std::string> category, const Dims& dims,
                     DType dtype) {
  const std::string path = "/supports/" + s.name +
                           (where == Location::Node ? "/node_arrays/"
                                                    : "/cell_arrays/") +
                           name;
  check_builder_name(path, name);
  const Resolved r = resolve(path, dims, where, extent_of(s, where),
                             values.size(), true);
  ArraySlot a = built_slot(name, where, r);
  a.role = "label";
  a.category = std::move(category);
  a.data.dtype = dtype;
  a.data.i64 = in_stored_order(values, r);
  return add_slot(s, where, std::move(a));
}

ArraySlot& add_callable_array(Support& s, Location where,
                              const std::string& name,
                              const std::string& units, const Dims& dims,
                              const std::string& callable_id,
                              const std::string& output) {
  const std::string path = "/supports/" + s.name +
                           (where == Location::Node ? "/node_arrays/"
                                                    : "/cell_arrays/") +
                           name;
  check_builder_name(path, name);
  const Resolved r =
      resolve(path, dims, where, extent_of(s, where), 0, false);
  ArraySlot a = built_slot(name, where, r);
  a.role = "field";
  a.units = units;
  a.source = "callable:" + callable_id;
  a.output = output;
  a.data = Array();
  return add_slot(s, where, std::move(a));
}

}  // namespace

ArraySlot& add_node_array(Support& s, const std::string& name,
                          const std::vector<double>& values,
                          const std::string& units, const Dims& dims) {
  return add_array(s, Location::Node, name, values, units, dims);
}

ArraySlot& add_cell_array(Support& s, const std::string& name,
                          const std::vector<double>& values,
                          const std::string& units, const Dims& dims) {
  return add_array(s, Location::Cell, name, values, units, dims);
}

ArraySlot& add_node_label(Support& s, const std::string& name,
                          const std::vector<std::int64_t>& values,
                          std::optional<std::string> category,
                          const Dims& dims, DType dtype) {
  return add_label(s, Location::Node, name, values, std::move(category),
                   dims, dtype);
}

ArraySlot& add_cell_label(Support& s, const std::string& name,
                          const std::vector<std::int64_t>& values,
                          std::optional<std::string> category,
                          const Dims& dims, DType dtype) {
  return add_label(s, Location::Cell, name, values, std::move(category),
                   dims, dtype);
}

ArraySlot& add_callable_node_array(Support& s, const std::string& name,
                                   const std::string& units,
                                   const Dims& dims,
                                   const std::string& callable_id,
                                   const std::string& output) {
  return add_callable_array(s, Location::Node, name, units, dims,
                            callable_id, output);
}

ArraySlot& add_callable_cell_array(Support& s, const std::string& name,
                                   const std::string& units,
                                   const Dims& dims,
                                   const std::string& callable_id,
                                   const std::string& output) {
  return add_callable_array(s, Location::Cell, name, units, dims,
                            callable_id, output);
}

Scalar& add_callable_scalar(Dataset& d, const std::string& name,
                            const std::string& units,
                            const std::string& callable_id,
                            const std::string& output) {
  check_builder_name("/scalars/" + name, name);
  check_builder_units("/scalars/" + name, units);
  Scalar s;
  s.name = name;
  s.units = units;
  s.source = "callable:" + callable_id;
  s.output = output;
  d.container_groups.insert("/scalars");
  auto at = std::lower_bound(
      d.scalars.begin(), d.scalars.end(), s,
      [](const Scalar& a, const Scalar& b) {
        return bytes_less(a.name, b.name);
      });
  return *d.scalars.insert(at, std::move(s));
}

}  // namespace mestra
