// The dataset value type: what `mestra::read` returns and what
// `mestra::write` puts back on disk.  It is the model of SPEC.md
// sections 2 to 12 with the byte-level facts of sections 18 to 25 kept
// where a writer needs them (the stored integer width of a key, the
// declared byte size of a category table, a chunk shape that is not
// the default).  Nothing here reads or writes a file.
#ifndef MESTRA_DATASET_HPP
#define MESTRA_DATASET_HPP

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "mestra/validate.hpp"
#include "mestra/value.hpp"

namespace mestra {

class Callable;

// One step of a dataset's compression pipeline.  Section 23 allows
// two filters and no others: shuffle, and gzip at levels 1 to 9,
// either with or without the other.  The order is the order HDF5
// applies them in and is part of what the file says about its own
// layout, so a reader keeps it and a writer puts it back.
struct FilterStep {
  enum Kind { Shuffle, Gzip };
  Kind kind = Shuffle;
  int level = 0;    // the gzip level, 1 to 9; unused for shuffle

  static FilterStep shuffle() { return FilterStep{Shuffle, 0}; }
  static FilterStep gzip(int level) { return FilterStep{Gzip, level}; }
  bool operator==(const FilterStep& o) const {
    return kind == o.kind && (kind != Gzip || level == o.level);
  }
  bool operator!=(const FilterStep& o) const { return !(*this == o); }
};

// A dataset's whole pipeline, in order.  Empty means no filter, which
// is what a dataset built from vectors gets.
using FilterPipeline = std::vector<FilterStep>;

// Where an array sits: under `node_arrays` or under `cell_arrays`.
// It is not an attribute (section 19); the group the slot sits in says
// it.
enum class Location { Node, Cell };

// A per-row column that locates the row in the sampled space.
// `role` is one of design, condition, time, categorical, group, split,
// id, status (section 3).  A key stores category ids when its role is
// categorical, group, split or status: the zero-based position of the
// entry in its category table (section 21).
struct Key {
  std::string name;
  std::string role;
  std::optional<std::string> units;
  std::optional<double> lower;
  std::optional<double> upper;
  std::optional<std::string> category;          // category table name
  std::optional<std::string> trajectory_group;  // on the time key
  std::optional<std::string> parent;            // on a group key
  DType dtype = DType::Float64;
  std::vector<double> f64;
  std::vector<std::int64_t> i64;
  std::vector<std::string> str;                 // a string id column
  // The declared byte size of a string id column.  Left unset the
  // writer uses the longest element, which is what section 19 asks
  // for; a larger size is legal and draws W13.
  std::optional<std::size_t> string_size;
  AttrMap extra;   // attributes this version does not know (W11)

  std::size_t rows() const;
};

// A per-row quantity of interest.  `source` is "data" or
// "callable:<id>"; a callable-served scalar carries no values.
struct Scalar {
  std::string name;
  std::string units;
  std::string source = "data";
  std::optional<std::string> output;     // when served by a callable
  std::optional<std::string> statistic;
  std::optional<std::string> of;
  std::optional<double> quantile;
  std::optional<double> level;           // on a band: its coverage
  std::optional<std::string> method;     // on a band: how it was made
  std::vector<double> values;
  AttrMap extra;

  bool is_callable() const;
  std::string callable_id() const;       // "" when source is data
};

// A field-like quantity on a support, or a support's coordinates.
// `varies` is "none", "row" or "group:<k>" (section 5).  `data`
// carries the stored values with their dimension names; it is empty
// when the slot is served by a callable.
//
// `varies` and `components` describe the shape `data` was built with:
// the builders derive both from the `dims` they are given, and
// assigning to either on a slot that already holds data does not
// reshape it.  Such a slot is refused by `write` with E04 or E31
// before any file is opened, rather than written out for the
// validator to reject afterwards.  To change the shape, build the
// slot again with the `dims` you meant.
struct ArraySlot {
  std::string name;
  std::string role;                      // coordinates|field|label|...
  std::string varies = "none";
  std::optional<std::string> units;
  std::int64_t components = 1;
  std::string source = "data";
  std::optional<std::string> output;
  std::optional<std::string> statistic;
  std::optional<std::string> of;
  std::optional<double> quantile;
  std::optional<double> level;           // on a band: its coverage
  std::optional<std::string> method;     // on a band: how it was made
  std::optional<std::string> category;   // a label's table
  std::optional<bool> recomputed;        // weight and normal
  std::optional<std::string> derived_from;
  std::optional<std::string> recipe;
  std::optional<std::string> reference;  // "row=<i>" or "group:<k>=<c>"
  Location location = Location::Node;
  Array data;
  AttrMap extra;

  bool is_callable() const;
  std::string callable_id() const;
};

// A group carried from one file to another without being looked
// into.  Sections 12 and 29 forbid a reader to interpret `/private`;
// they do not forbid it to copy it, and a producer that round-trips a
// file through this library keeps its own records.
//
// `image` is the bytes of an HDF5 file holding a copy of the group and
// nothing else, made by the library's own object copy, so every dtype,
// shape, chunk, filter, attribute and subgroup comes back as it was
// without this code having decided what any of it means.  The one
// thing that copy does not carry is a dimension scale attachment,
// which is a pair of attributes holding object references: those are
// recorded here by path and remade on the way out.
struct OpaqueGroup {
  struct Attachment {
    std::string dataset;     // absolute path in the source file
    std::size_t axis = 0;
    std::string scale;       // absolute path; may be outside the group
  };

  std::vector<char> image;
  std::vector<Attachment> attachments;

  bool empty() const { return image.empty(); }
};

// The structure a field lives on.  `kind` is mesh, axis or none.
// The cell arrays are present only for a mesh (section 20).
struct Support {
  std::string name;                      // the group name, e.g. "s0"
  std::string kind = "mesh";
  std::int64_t n_nodes = 0;
  std::int64_t n_cells = 0;
  std::string support_id;                // as stored, lower-case hex
  std::vector<std::uint8_t> cell_types;
  std::vector<std::int64_t> cell_offsets;
  std::vector<std::int64_t> cell_connectivity;
  std::optional<ArraySlot> coordinates;
  std::vector<ArraySlot> node_arrays;
  std::vector<ArraySlot> cell_arrays;
  AttrMap extra;
  std::vector<std::string> unknown_groups;   // ignored and reported
  // An opaque copy of each of those groups, by name, so that a rewrite
  // puts back what it holds as well as its name (section 28: ignored
  // and reported, not dropped).  Filled by a whole read.
  std::map<std::string, OpaqueGroup> unknown_group_copies;

  // The digest of section 24, computed from the arrays above rather
  // than read from `support_id`.
  std::string computed_support_id() const;

  // One array by name, or nullptr when the support carries none of
  // that name at that location; the same lookup `Dataset::key`,
  // `scalar` and `support` give, because a slot is met by name
  // everywhere else in this library and a caller should not have to
  // search a vector for it.  Adding an array keeps the vector sorted
  // and so invalidates the pointer: fetch again by name rather than
  // hold one across an `add_`.
  const ArraySlot* node_array(const std::string& wanted) const;
  const ArraySlot* cell_array(const std::string& wanted) const;
  ArraySlot* node_array(const std::string& wanted);
  ArraySlot* cell_array(const std::string& wanted);
};

// A category table under /categories.  `string_size` is the declared
// byte size of the fixed-length strings; left unset the writer uses
// the longest element, which is what section 19 asks for.
struct CategoryTable {
  std::string name;
  std::vector<std::string> entries;
  std::optional<std::size_t> string_size;

  int index_of(const std::string& entry) const;   // -1 when absent
};

// A callable as the file stores it: a public `type`, an optional
// one-line `repr`, and the dictionary the codec round-trips.
struct StoredCallable {
  std::string id;
  std::string type;
  std::optional<std::string> repr;
  Dict dict;
};

// One file.
struct Dataset {
  std::string format = "mestra/0";
  std::string writer;
  std::string created;
  bool aligned = true;
  std::optional<std::string> generalisation_group;

  // The length of the `row` dimension.  Every row-dimensioned dataset
  // in an aligned file has this many entries (section 21).
  std::int64_t n_rows = 0;

  std::vector<Key> keys;
  std::vector<Scalar> scalars;
  std::vector<CategoryTable> categories;
  std::vector<Support> supports;
  std::vector<StoredCallable> callables;

  // Present exactly when the file declares more than one support
  // (section 22).  Values are positions in the support order, which
  // is the support group names sorted by their UTF-8 bytes.
  std::optional<std::vector<std::int32_t>> row_support;

  AttrMap notes;                 // /notes, free-form
  bool has_notes = false;
  // /private exists in the file this came from.  Section 29 forbids a
  // reader to interpret it and this library does not: `private_group`
  // holds an opaque copy of it, which `write` puts back, so that a
  // round trip keeps a producer's own records without this code ever
  // deciding what any of them means.  A whole read fills both; the
  // header read of section 29 reads no array and fills neither.
  bool has_private = false;
  OpaqueGroup private_group;
  AttrMap root_extra;            // root attributes this version does
                                 // not know (W11)
  std::vector<std::string> unknown_root_groups;
  // An opaque copy of each of those groups, by name, as for a
  // support's unknown groups.  Filled by a whole read.
  std::map<std::string, OpaqueGroup> unknown_root_copies;
  // What a read left out.  After a non-strict read, the structural
  // findings a strict read would have thrown for.  After any read, an
  // entry of a callable's dictionary section 25 cannot represent
  // (E32), which a semantic rule does not stop a read for; a checked
  // write refuses while one is listed, since it would drop it.  Empty
  // after a build from vectors.
  std::vector<Finding> not_read;
  // Container groups the file carries even when they hold nothing, so
  // that a round trip does not add or drop an object path.
  std::set<std::string> container_groups;
  // A chunk shape that is not the default of section 23, by the HDF5
  // path of the dataset it belongs to.  A reader fills this in for
  // every dataset whose chunk it finds is not the default, so that a
  // round trip reproduces the file; a caller building a dataset from
  // vectors leaves it alone and gets the default everywhere.
  std::map<std::string, std::vector<std::size_t>> chunk_overrides;
  // The compression filters a dataset carries, by its HDF5 path, the
  // same way and for the same reason as the chunk shape above: a
  // reader records what it found so that a writer puts the same file
  // back.  Section 23 makes compression optional, so a dataset built
  // from vectors carries none and is written uncompressed; what a
  // round trip must not do is silently drop a filter the file had,
  // which on a real dataset grows it by a sixth.
  std::map<std::string, FilterPipeline> filters;

  // --- lookup ------------------------------------------------------
  const Key* key(const std::string& name) const;
  const Scalar* scalar(const std::string& name) const;
  const Support* support(const std::string& name) const;
  // The same three by name, for a caller that wants to change one.
  // Adding a key, a scalar or a support keeps the vectors sorted and
  // therefore invalidates references into them, so fetch again by
  // name rather than holding a reference across an `add_`.
  Key* key(const std::string& name);
  Scalar* scalar(const std::string& name);
  Support* support(const std::string& name);
  const CategoryTable* category(const std::string& name) const;
  const StoredCallable* callable(const std::string& id) const;
  // The support of a row, by /row_support or by the single support.
  const Support* support_of_row(std::size_t row) const;
  // Key names sorted by their UTF-8 bytes: the file's key order
  // (section 26).
  std::vector<std::string> key_order() const;

  // --- building ----------------------------------------------------
  // Each of these fills in the dimension names, the shape and, where
  // the format decides it, the support id, so that a caller builds a
  // conforming file from plain vectors.  The argument order is the
  // one docs/api-conventions.md section 1 fixes for every language:
  // the name, then the values, then what they mean.
  //
  // Bounds: when the caller gives none, `add_key` records the observed
  // finite minimum and maximum as `lower` and `upper`, so that every
  // writer produces the same file from the same arrays and W04 and W08
  // are decidable on the result.  A caller who wants a wider domain of
  // validity assigns `lower` and `upper` on the key it gets back;
  // those are plain attributes and assigning them takes effect.
  // `units` has no default: section 3 requires it on a key of role
  // design, condition or time, and those are the roles this builder
  // is for.  "1" is the dimensionless unit and is said out loud.
  Key& add_key(const std::string& name, std::vector<double> values,
               const std::string& role, const std::string& units);
  // The convenience of section 1 for a key that names a category table
  // instead of units: add the table first with `add_category_table`
  // and name it here.  It is `add_key` with an integer column and
  // `category` in the place of `units`.
  Key& add_category_key(const std::string& name,
                        std::vector<std::int64_t> ids,
                        const std::string& role,
                        const std::string& category_table,
                        DType dtype = DType::Int32);
  Scalar& add_scalar(const std::string& name, std::vector<double> values,
                     const std::string& units);
  CategoryTable& add_category_table(const std::string& name,
                                    std::vector<std::string> entries);
  // The unit of generalisation is a property of the dataset and this
  // is how it is set (section 7, conventions section 1).  The name is
  // a key of role `group`; that it is one is checked when the file is
  // written, because the key may be added after this call.
  void set_generalisation_group(const std::string& name);
  // A mesh support; `support_id` is computed from the arrays.
  Support& add_mesh_support(const std::string& name,
                            std::int64_t n_nodes,
                            std::vector<std::uint8_t> cell_types,
                            std::vector<std::int64_t> cell_offsets,
                            std::vector<std::int64_t> connectivity);
  Support& add_axis_support(const std::string& name,
                            const std::vector<double>& coordinates,
                            const std::string& units);
  Support& add_none_support(const std::string& name);
  StoredCallable& add_callable(const std::string& id,
                               const std::string& type, Dict dict);
  // The same from a live callable, which is the form the conventions
  // show: `add_callable(id, callable)`.  The type, the dictionary and
  // the optional one-line `repr` all come from the object.
  StoredCallable& add_callable(const std::string& id, const Callable& c);
};

// --- naming the axes of an array a builder is given -----------------

// One axis of the array a caller hands an array builder: its logical
// dimension name (section 4) and, where the builder cannot work it out
// from the length of the values, its length.
//
//     {"row", "node", {"component", 3}}
//
// A bare name is an axis whose length the builder derives: `node` and
// `cell` from the support, and the one remaining unknown from the
// number of values.  Two unknown lengths are refused at build time,
// naming the axis to give a length to, rather than guessed at.
//
// The names may be in any order.  They name the axes of the array the
// caller flattened, so a caller holding (node, row) data says so and
// the builder permutes into the stored order of section 4 rather than
// asking the caller to.
struct Dim {
  std::string name;
  std::int64_t extent = -1;             // -1: the builder derives it

  Dim(const char* n) : name(n) {}                        // NOLINT
  Dim(std::string n) : name(std::move(n)) {}             // NOLINT
  Dim(std::string n, std::int64_t e) : name(std::move(n)), extent(e) {}
};

using Dims = std::vector<Dim>;

// --- support helpers ------------------------------------------------

// Set a support's coordinates.  `dims` names the axes of `values`;
// `varies` and `components` are derived from it and are not separately
// settable, because a shape that has been built cannot be changed by
// assigning to the slot afterwards.
void set_coordinates(Support& s, const std::vector<double>& values,
                     const std::string& units, const Dims& dims);

// Coordinates a callable serves rather than data: `set_coordinates`
// with the values dropped and the callable and its output added, the
// way `add_callable_node_array` is `add_node_array` -- a model of the
// geometry itself.  The support was added with its node count, and
// `dims` still says what the slot's shape will be, so its component
// axis needs a length.  An axis support's coordinates are part of its
// identity (section 24) and are always stored.
void set_callable_coordinates(Support& s, const std::string& units,
                              const Dims& dims,
                              const std::string& callable_id,
                              const std::string& output = "coordinates");

// Add a float64 array: role `field` unless the caller changes it on
// the slot that comes back.  `values` is the array flattened in the
// caller's own axis order, which `dims` names.
ArraySlot& add_node_array(Support& s, const std::string& name,
                          const std::vector<double>& values,
                          const std::string& units, const Dims& dims);
ArraySlot& add_cell_array(Support& s, const std::string& name,
                          const std::vector<double>& values,
                          const std::string& units, const Dims& dims);

// Add an integer label.  A label names a category table instead of
// carrying units, and when it names none its values are their own
// categories (section 3).
ArraySlot& add_node_label(Support& s, const std::string& name,
                          const std::vector<std::int64_t>& values,
                          std::optional<std::string> category,
                          const Dims& dims, DType dtype = DType::Int32);
ArraySlot& add_cell_label(Support& s, const std::string& name,
                          const std::vector<std::int64_t>& values,
                          std::optional<std::string> category,
                          const Dims& dims, DType dtype = DType::Int32);

// Add a slot served by a callable: it carries the attributes and no
// data (section 19).  The argument order is the array builders' with
// the values dropped and the callable and its output added.  `dims`
// still says what the slot's shape will be, so a component axis needs
// its length: there are no values to derive it from.
ArraySlot& add_callable_node_array(Support& s, const std::string& name,
                                   const std::string& units,
                                   const Dims& dims,
                                   const std::string& callable_id,
                                   const std::string& output);
ArraySlot& add_callable_cell_array(Support& s, const std::string& name,
                                   const std::string& units,
                                   const Dims& dims,
                                   const std::string& callable_id,
                                   const std::string& output);
Scalar& add_callable_scalar(Dataset& d, const std::string& name,
                            const std::string& units,
                            const std::string& callable_id,
                            const std::string& output);

// The digest of section 24 over the arrays given, without a file.
std::string support_id_digest(std::int64_t n_nodes,
                              const std::vector<std::uint8_t>& cell_types,
                              const std::vector<std::int64_t>& cell_offsets,
                              const std::vector<std::int64_t>& connectivity,
                              const std::vector<double>* axis_coordinates);

// The number of nodes a VTK cell type code takes, or 0 for a polygon
// (code 7, three or more) and -1 for a code section 20 does not allow.
int cell_type_nodes(std::uint8_t code);

// The default chunk length along `row` of section 23.
std::size_t default_chunk_rows(std::size_t item_bytes,
                               const std::vector<std::size_t>& other_extents,
                               std::size_t row_count);

}  // namespace mestra

#endif  // MESTRA_DATASET_HPP
