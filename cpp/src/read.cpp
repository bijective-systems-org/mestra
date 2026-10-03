#include <algorithm>
#include <functional>

#include "codec.hpp"
#include "h5.hpp"
#include "layout.hpp"
#include "mestra/io.hpp"
#include "mestra/validate.hpp"
#include "names.hpp"

namespace mestra {

Error::Error(std::string rule, const std::string& message)
    : std::runtime_error(rule.empty() ? message : rule + ": " + message),
      rule_(std::move(rule)) {}

const char* library_version() { return "mestra c++ 0.1"; }

namespace {

using internal::AttrType;
using internal::DsetInfo;
using internal::File;
using internal::Member;
using internal::RawAttr;

// A small view over one object's attributes.
class Attrs {
 public:
  Attrs(const File& f, const std::string& path) : raw_(f.attributes(path)) {}

  const RawAttr* find(const std::string& name) const {
    for (const RawAttr& a : raw_) {
      if (a.name == name) return &a;
    }
    return nullptr;
  }
  bool has(const std::string& name) const { return find(name) != nullptr; }

  std::optional<std::string> text(const std::string& name) const {
    const RawAttr* a = find(name);
    if (a == nullptr || a->value.kind() != AttrValue::Kind::Str) {
      return std::nullopt;
    }
    return a->value.as_text();
  }
  std::optional<std::int64_t> integer(const std::string& name) const {
    const RawAttr* a = find(name);
    if (a == nullptr) return std::nullopt;
    if (a->value.kind() == AttrValue::Kind::Int) return a->value.as_int();
    if (a->value.kind() == AttrValue::Kind::Bool) {
      return a->value.as_bool() ? 1 : 0;
    }
    return std::nullopt;
  }
  std::optional<double> real(const std::string& name) const {
    const RawAttr* a = find(name);
    if (a == nullptr) return std::nullopt;
    if (a->value.kind() == AttrValue::Kind::Float) return a->value.as_float();
    if (a->value.kind() == AttrValue::Kind::Int) {
      return static_cast<double>(a->value.as_int());
    }
    return std::nullopt;
  }
  std::optional<bool> boolean(const std::string& name) const {
    const RawAttr* a = find(name);
    if (a == nullptr) return std::nullopt;
    if (a->value.kind() == AttrValue::Kind::Bool) return a->value.as_bool();
    if (a->value.kind() == AttrValue::Kind::Int) {
      return a->value.as_int() != 0;
    }
    return std::nullopt;
  }

  // Everything the caller did not ask about, in file order, so that a
  // writer can put it back (section 28).
  template <typename Known>
  AttrMap unknown(Known known) const {
    AttrMap out;
    for (const RawAttr& a : raw_) {
      if (internal::machinery_attribute(a.name)) continue;
      if (known(a.name)) continue;
      out.emplace_back(a.name, a.value);
    }
    return out;
  }


 private:
  std::vector<RawAttr> raw_;
};

// The dimension names of a dataset, as logical names.
std::vector<std::string> dims_of(const DsetInfo& info) {
  std::vector<std::string> dims;
  for (const std::vector<std::string>& attached : info.scales) {
    if (attached.size() == 1) {
      dims.push_back(internal::logical_dim(attached.front()));
    } else {
      dims.push_back(std::string());   // E25 territory; read it anyway
    }
  }
  return dims;
}

Array read_array_at(const File& f, const std::string& path,
                    const DsetInfo& info) {
  Array a;
  a.shape.assign(info.shape.begin(), info.shape.end());
  a.dims = dims_of(info);
  DType dtype = DType::Float64;
  if (!internal::dtype_of(info.type, &dtype)) {
    throw Error("E20", "\"" + path + "\" has a dtype this format does not "
                                     "allow");
  }
  a.dtype = dtype;
  switch (dtype) {
    case DType::Float64: a.f64 = f.read_f64(path); break;
    case DType::String: a.str = f.read_strings(path); break;
    default: a.i64 = f.read_i64(path); break;
  }
  return a;
}

std::size_t item_bytes(DType t, const DsetInfo& info) {
  switch (t) {
    case DType::Bool:
    case DType::UInt8: return 1;
    case DType::Int32: return 4;
    case DType::String: return info.type.size;
    default: return 8;
  }
}

// Records the chunk of a row-dimensioned dataset when it is not the
// default of section 23, so that a round trip reproduces the file.
void note_chunk(Dataset* d, const std::string& path, const DsetInfo& info,
                DType dtype, std::size_t row_count) {
  if (!info.chunked || info.chunk.empty() || info.shape.empty()) return;
  std::vector<std::size_t> chunk(info.chunk.begin(), info.chunk.end());
  std::vector<std::size_t> rest(info.shape.begin() + 1, info.shape.end());
  std::vector<std::size_t> want;
  want.push_back(
      default_chunk_rows(item_bytes(dtype, info), rest, row_count));
  for (const std::size_t e : rest) want.push_back(e);
  if (chunk != want) d->chunk_overrides[path] = chunk;
}

void fill_slot_attributes(const Attrs& at, ArraySlot* slot) {
  slot->role = at.text("role").value_or(std::string());
  slot->varies = at.text("varies").value_or(std::string("none"));
  slot->units = at.text("units");
  slot->components = at.integer("components").value_or(1);
  slot->source = at.text("source").value_or(std::string("data"));
  slot->output = at.text("output");
  slot->statistic = at.text("statistic");
  slot->of = at.text("of");
  slot->quantile = at.real("quantile");
  slot->level = at.real("level");
  slot->method = at.text("method");
  slot->category = at.text("category");
  slot->recomputed = at.boolean("recomputed");
  slot->derived_from = at.text("derived_from");
  slot->recipe = at.text("recipe");
  slot->reference = at.text("reference");
  slot->extra = at.unknown(internal::known_array_attribute);
}

Dataset read_impl(const std::string& path, bool with_data) {
  File f = File::open_read(path);
  Dataset d;

  const Attrs root(f, "/");
  d.format = root.text("format").value_or(std::string());
  if (d.format.compare(0, 7, "mestra/") != 0) {
    throw Error("E01", "the root `format` attribute is \"" + d.format +
                           "\" and not \"mestra/<n>\"");
  }
  if (d.format != "mestra/0") {
    throw Error("E01", "this reader accepts mestra/0 and the file is \"" +
                           d.format + "\"; it must not be read partially");
  }
  d.writer = root.text("writer").value_or(std::string());
  d.created = root.text("created").value_or(std::string());
  d.aligned = root.boolean("aligned").value_or(true);
  d.generalisation_group = root.text("generalisation_group");
  d.root_extra = root.unknown(internal::known_root_attribute);

  // Section 21: the row count is the length of the /row scale, so that
  // a file with no row-dimensioned dataset still states it.
  if (f.is_dataset("/row")) {
    const DsetInfo info = f.dataset_info("/row");
    d.n_rows = info.shape.empty()
                   ? 0
                   : static_cast<std::int64_t>(info.shape[0]);
  }
  const std::size_t n_rows = static_cast<std::size_t>(d.n_rows);

  for (const Member& m : f.members("/")) {
    if (m.is_group) {
      if (internal::known_root_group(m.name)) {
        d.container_groups.insert("/" + m.name);
        if (m.name == "notes") d.has_notes = true;
        if (m.name == "private") d.has_private = true;
      } else {
        d.unknown_root_groups.push_back(m.name);
        if (with_data) {
          d.unknown_root_copies[m.name] =
              internal::capture_group(f, "/" + m.name);
        }
      }
    }
  }
  if (d.has_notes) {
    const Attrs notes(f, "/notes");
    d.notes = notes.unknown([](const std::string&) { return false; });
  }
  // Sections 12 and 29 forbid a reader to interpret /private and not to
  // copy it, so a whole read takes an opaque copy and `write` puts it
  // back.  A header read reads no array (section 29) and takes none.
  if (d.has_private && with_data) {
    d.private_group = internal::capture_group(f, "/private");
  }

  // --- category tables ---------------------------------------------
  for (const Member& m : f.members("/categories")) {
    if (!m.is_dataset) continue;
    const std::string p = "/categories/" + m.name;
    const DsetInfo info = f.dataset_info(p);
    CategoryTable t;
    t.name = m.name;
    if (with_data) t.entries = f.read_strings(p);
    t.string_size = info.type.size;
    d.categories.push_back(std::move(t));
  }

  // --- keys ---------------------------------------------------------
  for (const Member& m : f.members("/keys")) {
    if (!m.is_dataset) continue;
    const std::string p = "/keys/" + m.name;
    const DsetInfo info = f.dataset_info(p);
    const Attrs at(f, p);
    Key k;
    k.name = m.name;
    k.role = at.text("role").value_or(std::string());
    k.units = at.text("units");
    k.lower = at.real("lower");
    k.upper = at.real("upper");
    k.category = at.text("category");
    k.trajectory_group = at.text("trajectory_group");
    k.parent = at.text("parent");
    k.extra = at.unknown(internal::known_key_attribute);
    DType dtype = DType::Float64;
    if (!internal::dtype_of(info.type, &dtype)) dtype = DType::Float64;
    k.dtype = dtype;
    if (dtype == DType::String) k.string_size = info.type.size;
    if (with_data) {
      switch (dtype) {
        case DType::Float64: k.f64 = f.read_f64(p); break;
        case DType::String: k.str = f.read_strings(p); break;
        default: k.i64 = f.read_i64(p); break;
      }
    }
    note_chunk(&d, p, info, dtype, n_rows);
    d.keys.push_back(std::move(k));
  }

  // --- scalars ------------------------------------------------------
  for (const Member& m : f.members("/scalars")) {
    const std::string p = "/scalars/" + m.name;
    const Attrs at(f, p);
    Scalar s;
    s.name = m.name;
    s.units = at.text("units").value_or(std::string());
    s.source = at.text("source").value_or(std::string("data"));
    s.output = at.text("output");
    s.statistic = at.text("statistic");
    s.of = at.text("of");
    s.quantile = at.real("quantile");
    s.level = at.real("level");
    s.method = at.text("method");
    s.extra = at.unknown(internal::known_scalar_attribute);
    if (m.is_dataset) {
      const DsetInfo info = f.dataset_info(p);
      if (with_data) s.values = f.read_f64(p);
      note_chunk(&d, p, info, DType::Float64, n_rows);
    }
    d.scalars.push_back(std::move(s));
  }

  // --- row_support --------------------------------------------------
  if (with_data && f.is_dataset("/row_support")) {
    const DsetInfo info = f.dataset_info("/row_support");
    std::vector<std::int64_t> values = f.read_i64("/row_support");
    std::vector<std::int32_t> narrow;
    narrow.reserve(values.size());
    for (const std::int64_t v : values) {
      narrow.push_back(static_cast<std::int32_t>(v));
    }
    d.row_support = std::move(narrow);
    note_chunk(&d, "/row_support", info, DType::Int32, n_rows);
  }

  // --- supports -----------------------------------------------------
  for (const Member& m : f.members("/supports")) {
    if (!m.is_group) continue;
    const std::string sp = "/supports/" + m.name;
    const Attrs at(f, sp);
    Support s;
    s.name = m.name;
    s.kind = at.text("kind").value_or(std::string());
    s.n_nodes = at.integer("n_nodes").value_or(0);
    s.n_cells = at.integer("n_cells").value_or(0);
    s.support_id = at.text("support_id").value_or(std::string());
    s.extra = at.unknown(internal::known_support_attribute);

    if (with_data && f.is_dataset(sp + "/cell_types")) {
      for (const std::int64_t v : f.read_i64(sp + "/cell_types")) {
        s.cell_types.push_back(static_cast<std::uint8_t>(v));
      }
    }
    if (with_data && f.is_dataset(sp + "/cell_offsets")) {
      s.cell_offsets = f.read_i64(sp + "/cell_offsets");
    }
    if (with_data && f.is_dataset(sp + "/cell_connectivity")) {
      s.cell_connectivity = f.read_i64(sp + "/cell_connectivity");
    }
    if (f.is_dataset(sp + "/coordinates")) {
      const std::string p = sp + "/coordinates";
      const DsetInfo info = f.dataset_info(p);
      const Attrs cat(f, p);
      ArraySlot c;
      c.name = "coordinates";
      c.location = Location::Node;
      fill_slot_attributes(cat, &c);
      if (with_data) {
        c.data = read_array_at(f, p, info);
      } else {
        c.data.shape.assign(info.shape.begin(), info.shape.end());
        c.data.dims = dims_of(info);
        DType dtype = DType::Float64;
        internal::dtype_of(info.type, &dtype);
        c.data.dtype = dtype;
      }
      note_chunk(&d, p, info, c.data.dtype, n_rows);
      s.coordinates = std::move(c);
    } else if (f.is_group(sp + "/coordinates")) {
      // Served by a callable: the attributes and no data, like a
      // served node array (section 19).
      const std::string p = sp + "/coordinates";
      const Attrs cat(f, p);
      ArraySlot c;
      c.name = "coordinates";
      c.location = Location::Node;
      fill_slot_attributes(cat, &c);
      s.coordinates = std::move(c);
    }

    for (const Member& g : f.members(sp)) {
      if (!g.is_group) continue;
      if (!internal::known_support_group(g.name)) {
        s.unknown_groups.push_back(g.name);
        if (with_data) {
          s.unknown_group_copies[g.name] =
              internal::capture_group(f, sp + "/" + g.name);
        }
      }
    }

    for (int which = 0; which < 2; ++which) {
      const Location where = which == 0 ? Location::Node : Location::Cell;
      const std::string gp =
          sp + (which == 0 ? "/node_arrays" : "/cell_arrays");
      for (const Member& a : f.members(gp)) {
        const std::string p = gp + "/" + a.name;
        const Attrs aat(f, p);
        ArraySlot slot;
        slot.name = a.name;
        slot.location = where;
        fill_slot_attributes(aat, &slot);
        if (a.is_dataset) {
          const DsetInfo info = f.dataset_info(p);
          if (with_data) {
            slot.data = read_array_at(f, p, info);
          } else {
            slot.data.shape.assign(info.shape.begin(), info.shape.end());
            slot.data.dims = dims_of(info);
            DType dtype = DType::Float64;
            internal::dtype_of(info.type, &dtype);
            slot.data.dtype = dtype;
          }
          std::size_t count = n_rows;
          if (!d.aligned && slot.varies == "row" && !info.shape.empty()) {
            count = static_cast<std::size_t>(info.shape[0]);
          }
          note_chunk(&d, p, info, slot.data.dtype, count);
        }
        if (where == Location::Node) {
          s.node_arrays.push_back(std::move(slot));
        } else {
          s.cell_arrays.push_back(std::move(slot));
        }
      }
    }
    d.supports.push_back(std::move(s));
  }

  // --- callables ----------------------------------------------------
  for (const Member& m : f.members("/callables")) {
    if (!m.is_group) continue;
    const std::string p = "/callables/" + m.name;
    const Attrs at(f, p);
    StoredCallable c;
    c.id = m.name;
    c.type = at.text("type").value_or(std::string());
    c.repr = at.text("repr");
    // A callable's dictionary holds no field data; it is what a
    // reader needs to hand to `from_dict`, so it is read either way.
    if (with_data) {
      c.dict = internal::read_dict_group(f, p, true, 0, &d.not_read);
    }
    d.callables.push_back(std::move(c));
  }

  // What the file chose about its own storage, for the objects whose
  // choice is not decided elsewhere: a dimension scale's chunk, and
  // every dataset's filters.
  //
  // Sections 21 and 23 make a scale contiguous unless it is
  // unlimited, in which case its chunk length is 1.  A file that
  // stores one another way is recorded so that a round trip
  // reproduces it rather than silently restoring the default.
  //
  // Filters are recorded for every dataset, because section 23 leaves
  // compression to the writer and a round trip that dropped gzip
  // would grow a real dataset by a sixth.  The chunk of a data
  // dataset is recorded by note_chunk beside the slot it belongs to;
  // one walk decides the two things that walk can see.  /private is
  // copied whole and is never rewritten object by object, so it is
  // not walked here.
  std::function<void(const std::string&, int)> note_storage =
      [&](const std::string& group, int depth) {
        if (depth > internal::kMaxGroupDepth) return;
        for (const Member& m : f.members(group)) {
          if (m.kind != internal::LinkKind::Hard) continue;
          const std::string p =
              (group == "/" ? std::string("/") : group + "/") + m.name;
          if (p == "/private") continue;
          if (m.is_group) {
            note_storage(p, depth + 1);
            continue;
          }
          if (!m.is_dataset) continue;
          const DsetInfo info = f.dataset_info(p);
          const FilterPipeline pipeline = internal::pipeline_of(info);
          if (!pipeline.empty()) d.filters[p] = pipeline;
          if (!info.is_scale || info.shape.empty()) continue;
          const bool unlimited =
              !info.maxshape.empty() && info.maxshape[0] == H5S_UNLIMITED;
          const hsize_t want = unlimited
                                   ? 1
                                   : (info.shape[0] == 0 ? 1 : info.shape[0]);
          const bool default_layout = info.chunked &&
                                      info.chunk.size() == 1 &&
                                      info.chunk[0] == want;
          if (!default_layout && info.chunked) {
            d.chunk_overrides[p] =
                std::vector<std::size_t>(info.chunk.begin(),
                                         info.chunk.end());
          }
        }
      };
  note_storage("/", 0);

  // The support order is the group names sorted by their UTF-8 bytes
  // (section 22); HDF5 link order is not it.
  std::sort(d.supports.begin(), d.supports.end(),
            [](const Support& a, const Support& b) {
              return bytes_less(a.name, b.name);
            });
  std::sort(d.keys.begin(), d.keys.end(),
            [](const Key& a, const Key& b) {
              return bytes_less(a.name, b.name);
            });
  std::sort(d.scalars.begin(), d.scalars.end(),
            [](const Scalar& a, const Scalar& b) {
              return bytes_less(a.name, b.name);
            });
  std::sort(d.categories.begin(), d.categories.end(),
            [](const CategoryTable& a, const CategoryTable& b) {
              return bytes_less(a.name, b.name);
            });
  std::sort(d.callables.begin(), d.callables.end(),
            [](const StoredCallable& a, const StoredCallable& b) {
              return bytes_less(a.id, b.id);
            });
  return d;
}

}  // namespace

// The structural rules of the reading convention: the ones that say a
// file is not this format, rather than that what it says is wrong.
bool structural_rule(const std::string& id) {
  if (id.empty()) return true;    // a fault no rule of section 14 covers
  for (const char* rule : {"E01", "E16", "E19", "E25", "E26", "E29",
                           "E30", "E40", "E41"}) {
    if (id == rule) return true;
  }
  return false;
}

namespace {

// The structural findings of a pass, and the refusal they make.
std::vector<Finding> structural_findings(const Report& r) {
  std::vector<Finding> out;
  for (const Finding& f : r.errors) {
    if (structural_rule(f.id)) out.push_back(f);
  }
  return out;
}

void refuse(const std::string& who, const std::string& path,
            const std::vector<Finding>& refused) {
  std::string message = who + " refused \"" + path + "\": " +
                        internal::format_i64(static_cast<std::int64_t>(
                            refused.size())) +
                        " structural error(s)";
  for (const Finding& f : refused) {
    message += "\n  " + (f.id.empty() ? std::string("!") : f.id) + " " +
               f.where + ": " + f.message;
  }
  throw Error(refused.front().id, message);
}

}  // namespace

Dataset read(const std::string& path, const ReadOptions& options) {
  std::vector<Finding> refused = structural_findings(validate(path));
  if (options.strict && !refused.empty()) {
    refuse("mestra::read", path, refused);
  }
  Dataset d = read_impl(path, true);
  d.not_read.insert(d.not_read.begin(), refused.begin(), refused.end());
  return d;
}

Dataset read_header(const std::string& path) {
  // Section 30's hostile contract: opening a file for its metadata
  // alone must refuse what a read refuses, with the same identifiers,
  // rather than return something.  Conventions section 7 says what an
  // open may read to decide that, and `validate_metadata` is that
  // much and no more, so this costs no slot and no dictionary.
  const std::vector<Finding> refused =
      structural_findings(validate_metadata(path));
  if (!refused.empty()) refuse("mestra::read_header", path, refused);
  return read_impl(path, false);
}

Array read_slot(const std::string& path, const std::string& slot) {
  File f = File::open_read(path);
  if (!f.is_dataset(slot)) {
    throw Error("", "\"" + slot + "\" is not a dataset in this file");
  }
  const DsetInfo info = f.dataset_info(slot);
  return read_array_at(f, slot, info);
}

Array read_slot_rows(const std::string& path, const std::string& slot,
                     std::size_t row_begin, std::size_t row_end) {
  File f = File::open_read(path);
  if (!f.is_dataset(slot)) {
    throw Error("", "\"" + slot + "\" is not a dataset in this file");
  }
  const DsetInfo info = f.dataset_info(slot);
  if (info.shape.empty()) {
    throw Error("", "\"" + slot + "\" has no leading dimension to slice");
  }
  Array a;
  a.dims = dims_of(info);
  if (a.dims.empty() || a.dims.front() != "row") {
    throw Error("", "\"" + slot + "\" has no `row` dimension");
  }
  DType dtype = DType::Float64;
  if (!internal::dtype_of(info.type, &dtype)) {
    throw Error("E20", "\"" + slot + "\" has a dtype this format does not "
                                     "allow");
  }
  a.dtype = dtype;
  a.shape.assign(info.shape.begin(), info.shape.end());
  a.shape[0] = row_end > row_begin ? row_end - row_begin : 0;
  if (dtype == DType::Float64) {
    a.f64 = f.read_f64_rows(slot, row_begin, row_end);
  } else if (dtype == DType::String) {
    throw Error("", "a lazy read of a string slot is not supported");
  } else {
    a.i64 = f.read_i64_rows(slot, row_begin, row_end);
  }
  return a;
}

std::string support_id_of(const std::string& path,
                          const std::string& support) {
  File f = File::open_read(path);
  const std::string sp = "/supports/" + support;
  if (!f.is_group(sp)) {
    throw Error("", "the file has no support \"" + support + "\"");
  }
  const Attrs at(f, sp);
  const std::int64_t n_nodes = at.integer("n_nodes").value_or(0);
  // The declared kind decides which arrays are hashed, so a support
  // that declares none, or one section 6 does not have, has no digest
  // (section 24) and is not taken for a mesh.
  const std::optional<std::string> declared = at.text("kind");
  if (!declared.has_value()) {
    throw Error("E39", "support \"" + support +
                           "\" does not say its kind, so it has no "
                           "support_id");
  }
  const std::string kind = *declared;
  if (kind != "mesh" && kind != "axis" && kind != "none") {
    throw Error("E02", "support \"" + support + "\" is of kind \"" + kind +
                           "\", which section 6 does not have, so it has "
                           "no support_id");
  }
  std::vector<std::uint8_t> types;
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> conn;
  // Section 24: for an axis support the digest is over
  // the stored coordinate bytes as they are, even when `varies` is
  // wrong, so that such a file breaks E35 and nothing else.
  if (kind == "axis" && f.is_dataset(sp + "/coordinates")) {
    const std::vector<double> coords = f.read_f64(sp + "/coordinates");
    return support_id_digest(n_nodes, types, offsets, conn, &coords);
  }
  if (kind == "axis" || kind == "none") {
    return support_id_digest(n_nodes, types, offsets, conn, nullptr);
  }
  if (f.is_dataset(sp + "/cell_types")) {
    for (const std::int64_t v : f.read_i64(sp + "/cell_types")) {
      types.push_back(static_cast<std::uint8_t>(v));
    }
  }
  if (f.is_dataset(sp + "/cell_offsets")) {
    offsets = f.read_i64(sp + "/cell_offsets");
  }
  if (f.is_dataset(sp + "/cell_connectivity")) {
    conn = f.read_i64(sp + "/cell_connectivity");
  }
  return support_id_digest(n_nodes, types, offsets, conn, nullptr);
}

Dict read_dict(const std::string& path, const std::string& callable_id) {
  File f = File::open_read(path);
  const std::string p = "/callables/" + callable_id;
  if (!f.is_group(p)) {
    throw Error("E14", "the file has no callable \"" + callable_id + "\"");
  }
  return internal::read_dict_group(f, p, true);
}

}  // namespace mestra
