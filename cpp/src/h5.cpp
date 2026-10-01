#include "h5.hpp"

#include <hdf5_hl.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "mestra/io.hpp"
#include "names.hpp"

namespace mestra {
namespace internal {
namespace {

// HDF5 prints its own error stack on any failure; this library
// reports failures through mestra::Error instead.
struct SilenceHdf5 {
  SilenceHdf5() { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
};
const SilenceHdf5 kSilence;

void need(bool ok, const std::string& message) {
  if (!ok) throw Error("", message);
}

std::string parent_of(const std::string& path) {
  const std::size_t at = path.find_last_of('/');
  if (at == std::string::npos || at == 0) return "/";
  return path.substr(0, at);
}

Id open_object(hid_t file, const std::string& path) {
  Id id(H5Oopen(file, path.c_str(), H5P_DEFAULT));
  need(id.valid(), "cannot open \"" + path + "\"");
  return id;
}

struct ScaleVisit {
  std::vector<std::string>* names;
  std::vector<std::string>* paths;
  const File* file;
};

// An object's token as a printable key, which is how two identifiers
// are told to be the same object without asking the library for a
// path it would have to search for.
std::string token_key(hid_t where, const H5O_token_t& token) {
  char* text = nullptr;
  if (H5Otoken_to_str(where, &token, &text) < 0 || text == nullptr) {
    return std::string();
  }
  std::string out(text);
  H5free_memory(text);
  return out;
}

herr_t collect_scale(hid_t /*did*/, unsigned /*dim*/, hid_t dsid,
                     void* data) {
  ScaleVisit* visit = static_cast<ScaleVisit*>(data);
  // The dimension's name is the scale dataset's HDF5 link name and
  // never its NAME attribute (section 21).  It is looked up by object
  // token; see File::dataset_link_name for why not by H5Iget_name.
  visit->names->push_back(visit->file->dataset_link_name(dsid));
  visit->paths->push_back(visit->file->dataset_path(dsid));
  return 0;
}

// One member of a group, as H5Literate2 hands it over.  The link's own
// type is read from the link and nothing is resolved: a soft link is
// not followed and an external link is never opened, so neither can
// make this reader touch anything the caller did not name.
herr_t collect_member(hid_t group, const char* name,
                      const H5L_info2_t* linfo, void* data) {
  std::vector<Member>* out = static_cast<std::vector<Member>*>(data);
  Member m;
  m.name = name == nullptr ? std::string() : std::string(name);
  if (linfo == nullptr) {
    m.kind = LinkKind::Missing;
    out->push_back(m);
    return 0;
  }
  switch (linfo->type) {
    case H5L_TYPE_HARD: m.kind = LinkKind::Hard; break;
    case H5L_TYPE_SOFT: m.kind = LinkKind::Soft; break;
    case H5L_TYPE_EXTERNAL: m.kind = LinkKind::External; break;
    default: m.kind = LinkKind::Other; break;
  }
  if (m.kind == LinkKind::Hard) {
    H5O_info2_t oinfo;
    if (H5Oget_info_by_name3(group, m.name.c_str(), &oinfo, H5O_INFO_BASIC,
                             H5P_DEFAULT) >= 0) {
      m.is_group = oinfo.type == H5O_TYPE_GROUP;
      m.is_dataset = oinfo.type == H5O_TYPE_DATASET;
    }
  }
  out->push_back(m);
  return 0;
}

herr_t collect_attr_name(hid_t /*object*/, const char* name,
                         const H5A_info_t* /*info*/, void* data) {
  std::vector<std::string>* out = static_cast<std::vector<std::string>*>(data);
  if (name != nullptr) out->push_back(name);
  return 0;
}

AttrType type_of(hid_t type_id, hid_t space_id) {
  AttrType t;
  t.klass = H5Tget_class(type_id);
  t.size = H5Tget_size(type_id);
  if (t.klass == H5T_STRING) {
    t.variable_length = H5Tis_variable_str(type_id) > 0;
    t.cset = H5Tget_cset(type_id);
    t.strpad = H5Tget_strpad(type_id);
  } else if (t.klass == H5T_INTEGER) {
    t.is_signed = H5Tget_sign(type_id) == H5T_SGN_2;
    t.order = H5Tget_order(type_id);
  } else if (t.klass == H5T_FLOAT) {
    t.order = H5Tget_order(type_id);
  }
  if (space_id >= 0) {
    t.scalar_dataspace = H5Sget_simple_extent_type(space_id) == H5S_SCALAR;
    const hssize_t points = H5Sget_simple_extent_npoints(space_id);
    t.points = points > 0 ? static_cast<std::size_t>(points) : 0;
  }
  return t;
}

}  // namespace

void Id::close() {
  if (id_ >= 0) {
    H5Idec_ref(id_);
    id_ = H5I_INVALID_HID;
  }
}

std::string scale_name_attribute(hsize_t length) {
  // Section 21 gives the C format "%s%10d".  The conversion is done as
  // a long long so that a dimension longer than INT_MAX is printed
  // rather than converted out of range; for every length "%10d" can
  // represent the two produce the same 63 characters.
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "%s%10lld",
                "This is a netCDF dimension but not a netCDF variable.",
                static_cast<long long>(length));
  return std::string(buffer);
}

FilterPipeline pipeline_of(const DsetInfo& info) {
  FilterPipeline out;
  for (const auto& filter : info.filters) {
    if (filter.first == H5Z_FILTER_SHUFFLE) {
      out.push_back(FilterStep::shuffle());
    } else if (filter.first == H5Z_FILTER_DEFLATE) {
      const unsigned level = filter.second.empty() ? 0 : filter.second[0];
      if (level >= 1 && level <= 9) {
        out.push_back(FilterStep::gzip(static_cast<int>(level)));
      }
    }
  }
  return out;
}

Id string_type(std::size_t size) {
  Id t(H5Tcopy(H5T_C_S1));
  need(t.valid(), "cannot make a string type");
  H5Tset_size(t.get(), size == 0 ? 1 : size);
  H5Tset_strpad(t.get(), H5T_STR_NULLPAD);
  H5Tset_cset(t.get(), H5T_CSET_UTF8);
  return t;
}

bool is_spec_string(const AttrType& t) {
  return t.klass == H5T_STRING && !t.variable_length &&
         t.cset == H5T_CSET_UTF8 && t.strpad == H5T_STR_NULLPAD &&
         t.scalar_dataspace;
}

bool is_spec_int64(const AttrType& t) {
  return t.klass == H5T_INTEGER && t.size == 8 && t.is_signed &&
         t.order == H5T_ORDER_LE && t.scalar_dataspace;
}

bool is_spec_float64(const AttrType& t) {
  return t.klass == H5T_FLOAT && t.size == 8 && t.order == H5T_ORDER_LE &&
         t.scalar_dataspace;
}

bool is_spec_bool(const AttrType& t) {
  return t.klass == H5T_INTEGER && t.size == 1 && t.is_signed &&
         t.scalar_dataspace;
}

bool dtype_of(const AttrType& t, DType* out) {
  if (t.klass == H5T_STRING) {
    if (t.variable_length) return false;
    *out = DType::String;
    return true;
  }
  if (t.klass == H5T_FLOAT) {
    if (t.size != 8 || t.order != H5T_ORDER_LE) return false;
    *out = DType::Float64;
    return true;
  }
  if (t.klass == H5T_INTEGER) {
    if (t.size == 1) {
      *out = t.is_signed ? DType::Bool : DType::UInt8;
      return true;
    }
    if (t.order != H5T_ORDER_LE || !t.is_signed) return false;
    if (t.size == 4) {
      *out = DType::Int32;
      return true;
    }
    if (t.size == 8) {
      *out = DType::Int64;
      return true;
    }
  }
  return false;
}

// --- File ------------------------------------------------------------

File File::open_read(const std::string& path) {
  File f;
  f.id_ = Id(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  if (!f.id_.valid()) {
    // E01, as the Python and Julia readers say: a file that is not HDF5
    // has no `format`, so it is not a mestra file. (It used to carry no
    // rule and print as E41, which is for an object inside a file.)
    throw Error("E01", "cannot open \"" + path + "\" as an HDF5 file");
  }
  return f;
}

File File::open_write(const std::string& path) {
  File f;
  f.id_ = Id(H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
  if (!f.id_.valid()) {
    throw Error("", "cannot open \"" + path + "\" for writing");
  }
  return f;
}

File File::create(const std::string& path) {
  File f;
  // The root group is created with the file, so the file creation
  // property list is where its object header is told not to record
  // the clock: the same dataset written twice must give the same
  // bytes, which every other object below asks for on its own list.
  Id fcpl(H5Pcreate(H5P_FILE_CREATE));
  H5Pset_obj_track_times(fcpl.get(), 0);
  f.id_ = Id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, fcpl.get(),
                       H5P_DEFAULT));
  if (!f.id_.valid()) {
    throw Error("", "cannot create \"" + path + "\"");
  }
  return f;
}

const char* link_kind_name(LinkKind kind) {
  switch (kind) {
    case LinkKind::Missing: return "missing";
    case LinkKind::Hard: return "a hard link";
    case LinkKind::Soft: return "a soft link";
    case LinkKind::External: return "an external link";
    case LinkKind::Other: return "a link of a kind this reader does not "
                                 "know";
  }
  return "a link";
}

LinkKind File::link_kind(const std::string& path) const {
  if (path == "/") return LinkKind::Hard;
  // Component by component, so that the walk stops at the first link
  // that is not hard and never asks the library about anything below
  // it.  Asking about a path under an external link is what would
  // open another file on the machine.
  std::size_t at = 1;
  LinkKind last = LinkKind::Hard;
  while (at <= path.size()) {
    const std::size_t next = path.find('/', at);
    const std::string prefix =
        next == std::string::npos ? path : path.substr(0, next);
    H5L_info2_t info;
    if (H5Lget_info2(id_.get(), prefix.c_str(), &info, H5P_DEFAULT) < 0) {
      return LinkKind::Missing;
    }
    switch (info.type) {
      case H5L_TYPE_HARD: last = LinkKind::Hard; break;
      case H5L_TYPE_SOFT: last = LinkKind::Soft; break;
      case H5L_TYPE_EXTERNAL: last = LinkKind::External; break;
      default: last = LinkKind::Other; break;
    }
    if (next == std::string::npos) break;
    if (last != LinkKind::Hard) return last;
    at = next + 1;
  }
  return last;
}

bool File::exists(const std::string& path) const {
  return link_kind(path) != LinkKind::Missing;
}

bool File::is_group(const std::string& path) const {
  if (link_kind(path) != LinkKind::Hard) return false;
  H5O_info2_t info;
  if (H5Oget_info_by_name3(id_.get(), path.c_str(), &info, H5O_INFO_BASIC,
                           H5P_DEFAULT) < 0) {
    return false;
  }
  return info.type == H5O_TYPE_GROUP;
}

bool File::is_dataset(const std::string& path) const {
  if (link_kind(path) != LinkKind::Hard) return false;
  H5O_info2_t info;
  if (H5Oget_info_by_name3(id_.get(), path.c_str(), &info, H5O_INFO_BASIC,
                           H5P_DEFAULT) < 0) {
    return false;
  }
  return info.type == H5O_TYPE_DATASET;
}

std::vector<Member> File::members(const std::string& path) const {
  std::vector<Member> out;
  if (!is_group(path)) return out;
  Id group(H5Gopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(group.valid(), "cannot open the group \"" + path + "\"");
  // One iteration over the links rather than one lookup per index.
  // Asking for the i-th link by index makes HDF5 order the group's
  // links, and a group whose links are in the newer dense storage --
  // which is what a writer using the newer object header layout
  // leaves behind -- is ordered by building the whole table again for
  // every index.  That is linear work per member and quadratic per
  // group, and it is what made a file written that way cost the
  // square of its column count to read.  H5Literate2 orders the group
  // once and hands back every link, which is linear in both layouts.
  hsize_t at = 0;
  H5Literate2(group.get(), H5_INDEX_NAME, H5_ITER_INC, &at, collect_member,
              &out);
  return out;
}

std::vector<RawAttr> File::attributes(const std::string& path) const {
  std::vector<RawAttr> out;
  if (link_kind(path) != LinkKind::Hard) return out;
  Id object = open_object(id_.get(), path);
  // The names in one ordered pass, and then each attribute by name.
  // Opening the i-th attribute makes HDF5 order the object's
  // attributes again for every index, which is quadratic on an object
  // carrying many of them in the newer layout's dense storage; this is
  // the same trap as File::members and the same way out of it.
  std::vector<std::string> names;
  hsize_t at = 0;
  H5Aiterate2(object.get(), H5_INDEX_NAME, H5_ITER_INC, &at, collect_attr_name,
              &names);
  for (const std::string& attr_name : names) {
    Id attr(H5Aopen(object.get(), attr_name.c_str(), H5P_DEFAULT));
    if (!attr.valid()) continue;
    RawAttr a;
    a.name = attr_name;
    Id type(H5Aget_type(attr.get()));
    Id space(H5Aget_space(attr.get()));
    a.type = type_of(type.get(), space.get());

    // H5Aread fills as many elements as the attribute's dataspace
    // declares, and that count is in the file.  Every buffer below is
    // sized from it; nothing here reads into a single scalar slot on
    // the strength of the encoding section 18 asks for, because a
    // crafted file names an encoding and stores four thousand of them.
    const std::size_t points = a.type.points;
    const bool readable = points >= 1 && points <= kMaxAttributeElements;
    if (points > kMaxAttributeElements) {
      // Left unread and unvalued.  The validator reports the
      // dataspace (E19 for an attribute section 18 names) and a
      // reader carries nothing it could not read.
      a.too_large = true;
    }

    if (a.type.klass == H5T_STRING && readable) {
      if (a.type.variable_length) {
        std::vector<char*> values(points, nullptr);
        Id mem(H5Tcopy(H5T_C_S1));
        H5Tset_size(mem.get(), H5T_VARIABLE);
        H5Tset_cset(mem.get(), a.type.cset);
        if (H5Aread(attr.get(), mem.get(), values.data()) >= 0) {
          if (values[0] != nullptr) a.raw_bytes = values[0];
        }
        // Every element the library allocated is freed, not just the
        // one whose value is kept.
        for (char* p : values) {
          if (p != nullptr) H5free_memory(p);
        }
      } else {
        if (a.type.size > 0 && points > kMaxAttributeBytes / a.type.size) {
          a.too_large = true;
        } else {
          std::string bytes(a.type.size * points, '\0');
          if (!bytes.empty() &&
              H5Aread(attr.get(), type.get(), &bytes[0]) >= 0) {
            // Only the first element is a value this format has a
            // place for; the rest are read so that the library writes
            // inside the buffer and are then dropped.
            a.raw_bytes = bytes.substr(0, a.type.size);
          }
        }
      }
      a.value = AttrValue::raw_text(strip_nul(a.raw_bytes));
    } else if (a.type.klass == H5T_INTEGER && readable) {
      std::vector<std::int64_t> values(points, 0);
      if (H5Aread(attr.get(), H5T_NATIVE_INT64, values.data()) >= 0) {
        // Section 25: int8 means a boolean everywhere it appears, and
        // int64 means an integer.
        if (a.type.size == 1 && a.type.is_signed) {
          a.value = AttrValue::boolean(values[0] != 0);
          a.raw_bytes = std::string(1, static_cast<char>(values[0] & 0xff));
        } else {
          a.value = AttrValue::integer(values[0]);
        }
      }
    } else if (a.type.klass == H5T_FLOAT && readable) {
      std::vector<double> values(points, 0.0);
      if (H5Aread(attr.get(), H5T_NATIVE_DOUBLE, values.data()) >= 0) {
        a.value = AttrValue::real(values[0]);
      }
    }
    out.push_back(a);
  }
  return out;
}

DsetInfo File::dataset_info(const std::string& path) const {
  DsetInfo info;
  if (!is_dataset(path)) return info;
  info.is_dataset = true;
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open the dataset \"" + path + "\"");
  Id type(H5Dget_type(dset.get()));
  Id space(H5Dget_space(dset.get()));
  info.type = type_of(type.get(), space.get());
  const int rank = H5Sget_simple_extent_ndims(space.get());
  if (rank > 0) {
    info.shape.resize(static_cast<std::size_t>(rank));
    info.maxshape.resize(static_cast<std::size_t>(rank));
    H5Sget_simple_extent_dims(space.get(), info.shape.data(),
                              info.maxshape.data());
  }
  Id dcpl(H5Dget_create_plist(dset.get()));
  const H5D_layout_t layout = H5Pget_layout(dcpl.get());
  info.chunked = layout == H5D_CHUNKED;
  info.contiguous = layout == H5D_CONTIGUOUS;
  if (info.chunked && rank > 0) {
    info.chunk.resize(static_cast<std::size_t>(rank));
    H5Pget_chunk(dcpl.get(), rank, info.chunk.data());
  }
  const int filters = H5Pget_nfilters(dcpl.get());
  for (int i = 0; i < filters; ++i) {
    // cd_nelmts is in and out: on return it is the number of values
    // the filter defines, which may be more than the buffer held.
    // Asking with no buffer first is what keeps a file-chosen count
    // from sizing a read out of a fixed array.
    unsigned flags = 0;
    unsigned config = 0;
    std::size_t count = 0;
    if (H5Pget_filter2(dcpl.get(), static_cast<unsigned>(i), &flags, &count,
                       nullptr, 0, nullptr, &config) < 0) {
      continue;
    }
    if (count > kMaxFilterParameters) count = kMaxFilterParameters;
    std::vector<unsigned> params(count, 0u);
    std::size_t given = count;
    const H5Z_filter_t id = H5Pget_filter2(
        dcpl.get(), static_cast<unsigned>(i), &flags, &given,
        params.empty() ? nullptr : params.data(), 0, nullptr, &config);
    if (id < 0) continue;
    if (given < params.size()) params.resize(given);
    info.filters.emplace_back(static_cast<int>(id), params);
  }
  H5D_fill_value_t fill = H5D_FILL_VALUE_DEFAULT;
  if (H5Pfill_value_defined(dcpl.get(), &fill) >= 0) {
    info.has_fill_value_set = fill == H5D_FILL_VALUE_USER_DEFINED;
  }

  // Section 21 (E42): the creation properties of a dimension scale.
  // Nothing about them is visible in a byte position, so the only way
  // to check the rule is to ask the property list back.
  unsigned crt_order = 0;
  if (H5Pget_attr_creation_order(dcpl.get(), &crt_order) >= 0) {
    info.attr_order_tracked = (crt_order & H5P_CRT_ORDER_TRACKED) != 0;
    info.attr_order_indexed = (crt_order & H5P_CRT_ORDER_INDEXED) != 0;
  }

  info.is_scale = H5DSis_scale(dset.get()) > 0;
  info.scales.resize(static_cast<std::size_t>(rank > 0 ? rank : 0));
  info.scale_paths.resize(static_cast<std::size_t>(rank > 0 ? rank : 0));
  if (!info.is_scale) {
    for (int axis = 0; axis < rank; ++axis) {
      const std::size_t a = static_cast<std::size_t>(axis);
      ScaleVisit visit{&info.scales[a], &info.scale_paths[a], this};
      int index = 0;
      H5DSiterate_scales(dset.get(), static_cast<unsigned>(axis), &index,
                         collect_scale, &visit);
    }
  }
  return info;
}

namespace {

// The extents come out of the file, so the product is built with an
// overflow check at every step and refused past a stated maximum.
// Without it a crafted shape wraps to a small number and the buffer
// that follows is far too small for the read.
std::size_t checked_product(const std::string& path,
                            const std::vector<hsize_t>& shape,
                            std::size_t limit) {
  if (shape.empty()) return 1;
  std::size_t n = 1;
  for (const hsize_t raw : shape) {
    const std::size_t e = static_cast<std::size_t>(raw);
    if (e != 0 && n > limit / e) {
      throw Error("E41", "\"" + path + "\" declares more elements than this "
                                        "reader will read");
    }
    n *= e;
  }
  if (n > limit) {
    throw Error("E41", "\"" + path + "\" declares more elements than this "
                                      "reader will read");
  }
  return n;
}

}  // namespace

std::size_t File::eager_element_count(const std::string& path) const {
  const DsetInfo info = dataset_info(path);
  return checked_product(path, info.shape, kMaxDatasetElements);
}

std::vector<double> File::read_f64(const std::string& path) const {
  const DsetInfo info = dataset_info(path);
  std::vector<double> out(
      checked_product(path, info.shape, kMaxDatasetElements));
  if (out.empty()) return out;
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open \"" + path + "\"");
  need(H5Dread(dset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               out.data()) >= 0,
       "cannot read \"" + path + "\"");
  return out;
}

std::vector<std::int64_t> File::read_i64(const std::string& path) const {
  const DsetInfo info = dataset_info(path);
  std::vector<std::int64_t> out(
      checked_product(path, info.shape, kMaxDatasetElements));
  if (out.empty()) return out;
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open \"" + path + "\"");
  need(H5Dread(dset.get(), H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               out.data()) >= 0,
       "cannot read \"" + path + "\"");
  return out;
}

std::vector<std::string> File::read_strings_raw(
    const std::string& path) const {
  const DsetInfo info = dataset_info(path);
  const std::size_t count =
      checked_product(path, info.shape, kMaxDatasetElements);
  std::vector<std::string> out;
  if (count == 0) return out;
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open \"" + path + "\"");
  Id type(H5Dget_type(dset.get()));
  if (info.type.variable_length) {
    std::vector<char*> raw(count, nullptr);
    Id mem(H5Tcopy(H5T_C_S1));
    H5Tset_size(mem.get(), H5T_VARIABLE);
    H5Tset_cset(mem.get(), info.type.cset);
    need(H5Dread(dset.get(), mem.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 raw.data()) >= 0,
         "cannot read \"" + path + "\"");
    for (char* p : raw) {
      out.push_back(p ? std::string(p) : std::string());
      if (p) H5free_memory(p);
    }
    return out;
  }
  const std::size_t item = info.type.size;
  // Both factors come out of the file, so the product is checked
  // before it sizes the buffer the library then writes into.
  if (item != 0 && count > kMaxDatasetBytes / item) {
    throw Error("E41", "\"" + path + "\" declares more bytes than this "
                                     "reader will read");
  }
  std::string buffer(item * count, '\0');
  need(H5Dread(dset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
               &buffer[0]) >= 0,
       "cannot read \"" + path + "\"");
  for (std::size_t i = 0; i < count; ++i) {
    out.push_back(buffer.substr(i * item, item));
  }
  return out;
}

std::vector<std::string> File::read_strings(const std::string& path) const {
  std::vector<std::string> out = read_strings_raw(path);
  for (std::string& s : out) s = strip_nul(s);
  return out;
}

namespace {

void select_rows(hid_t space, const std::vector<hsize_t>& shape,
                 std::size_t begin, std::size_t end,
                 std::vector<hsize_t>* count) {
  std::vector<hsize_t> start(shape.size(), 0);
  *count = shape;
  start[0] = begin;
  (*count)[0] = static_cast<hsize_t>(end - begin);
  H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), nullptr,
                      count->data(), nullptr);
}

}  // namespace

std::vector<double> File::read_f64_rows(const std::string& path,
                                        std::size_t begin,
                                        std::size_t end) const {
  const DsetInfo info = dataset_info(path);
  need(info.is_dataset && !info.shape.empty(),
       "\"" + path + "\" has no rows to read");
  if (end > static_cast<std::size_t>(info.shape[0])) {
    throw Error("", "row range outside \"" + path + "\"");
  }
  if (end <= begin) return {};
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  Id space(H5Dget_space(dset.get()));
  std::vector<hsize_t> count;
  select_rows(space.get(), info.shape, begin, end, &count);
  Id mem(H5Screate_simple(static_cast<int>(count.size()), count.data(),
                          nullptr));
  std::vector<double> out(
      checked_product(path, count, kMaxDatasetElements));
  need(H5Dread(dset.get(), H5T_NATIVE_DOUBLE, mem.get(), space.get(),
               H5P_DEFAULT, out.data()) >= 0,
       "cannot read rows of \"" + path + "\"");
  return out;
}

std::vector<std::int64_t> File::read_i64_rows(const std::string& path,
                                              std::size_t begin,
                                              std::size_t end) const {
  const DsetInfo info = dataset_info(path);
  need(info.is_dataset && !info.shape.empty(),
       "\"" + path + "\" has no rows to read");
  if (end > static_cast<std::size_t>(info.shape[0])) {
    throw Error("", "row range outside \"" + path + "\"");
  }
  if (end <= begin) return {};
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  Id space(H5Dget_space(dset.get()));
  std::vector<hsize_t> count;
  select_rows(space.get(), info.shape, begin, end, &count);
  Id mem(H5Screate_simple(static_cast<int>(count.size()), count.data(),
                          nullptr));
  std::vector<std::int64_t> out(
      checked_product(path, count, kMaxDatasetElements));
  need(H5Dread(dset.get(), H5T_NATIVE_INT64, mem.get(), space.get(),
               H5P_DEFAULT, out.data()) >= 0,
       "cannot read rows of \"" + path + "\"");
  return out;
}

// --- writing ---------------------------------------------------------

void File::make_group(const std::string& path) {
  if (path == "/" || exists(path)) return;
  make_group(parent_of(path));
  // No modification times in the group's object header, as on every
  // dataset: a file's bytes depend on what it holds and not on when.
  Id gcpl(H5Pcreate(H5P_GROUP_CREATE));
  H5Pset_obj_track_times(gcpl.get(), 0);
  Id group(H5Gcreate2(id_.get(), path.c_str(), H5P_DEFAULT, gcpl.get(),
                      H5P_DEFAULT));
  need(group.valid(), "cannot create the group \"" + path + "\"");
}

void File::write_raw_string_attr(const std::string& path,
                                 const std::string& name,
                                 const std::string& bytes) {
  Id object = open_object(id_.get(), path);
  const std::size_t size = bytes.empty() ? 1 : bytes.size();
  Id type = string_type(size);
  Id space(H5Screate(H5S_SCALAR));
  Id attr(H5Acreate2(object.get(), name.c_str(), type.get(), space.get(),
                     H5P_DEFAULT, H5P_DEFAULT));
  need(attr.valid(), "cannot create the attribute \"" + name + "\"");
  std::string padded = bytes;
  padded.resize(size, '\0');
  need(H5Awrite(attr.get(), type.get(), padded.data()) >= 0,
       "cannot write the attribute \"" + name + "\"");
}

void File::write_attr(const std::string& path, const std::string& name,
                      const AttrValue& value) {
  switch (value.kind()) {
    case AttrValue::Kind::Str:
      write_raw_string_attr(path, name, value.as_text());
      return;
    case AttrValue::Kind::Bool: {
      Id object = open_object(id_.get(), path);
      Id space(H5Screate(H5S_SCALAR));
      Id attr(H5Acreate2(object.get(), name.c_str(), H5T_STD_I8LE,
                         space.get(), H5P_DEFAULT, H5P_DEFAULT));
      need(attr.valid(), "cannot create the attribute \"" + name + "\"");
      const signed char v = value.as_bool() ? 1 : 0;
      H5Awrite(attr.get(), H5T_NATIVE_SCHAR, &v);
      return;
    }
    case AttrValue::Kind::Int: {
      Id object = open_object(id_.get(), path);
      Id space(H5Screate(H5S_SCALAR));
      Id attr(H5Acreate2(object.get(), name.c_str(), H5T_STD_I64LE,
                         space.get(), H5P_DEFAULT, H5P_DEFAULT));
      need(attr.valid(), "cannot create the attribute \"" + name + "\"");
      const std::int64_t v = value.as_int();
      H5Awrite(attr.get(), H5T_NATIVE_INT64, &v);
      return;
    }
    case AttrValue::Kind::Float: {
      Id object = open_object(id_.get(), path);
      Id space(H5Screate(H5S_SCALAR));
      Id attr(H5Acreate2(object.get(), name.c_str(), H5T_IEEE_F64LE,
                         space.get(), H5P_DEFAULT, H5P_DEFAULT));
      need(attr.valid(), "cannot create the attribute \"" + name + "\"");
      const double v = value.as_float();
      H5Awrite(attr.get(), H5T_NATIVE_DOUBLE, &v);
      return;
    }
  }
}

namespace {

Id make_dcpl(const std::vector<hsize_t>& chunk,
             const FilterPipeline& filters) {
  Id dcpl(H5Pcreate(H5P_DATASET_CREATE));
  // Section 30: object time tracking off, so that two runs of a writer
  // produce identical bytes.
  H5Pset_obj_track_times(dcpl.get(), 0);
  if (!chunk.empty()) {
    H5Pset_chunk(dcpl.get(), static_cast<int>(chunk.size()), chunk.data());
    // Section 23: gzip at levels 1 to 9 and shuffle, and no other
    // filter.  They go on in the pipeline order the file they came
    // from had, because HDF5 applies them in the order they are set
    // and a round trip must put the same file back.  The library's own
    // calls are used rather than H5Pset_filter, so that the flags and
    // the client data are the ones every other writer of this format
    // produces.
    for (const FilterStep& step : filters) {
      if (step.kind == FilterStep::Shuffle) {
        need(H5Pset_shuffle(dcpl.get()) >= 0,
             "cannot set the shuffle filter");
      } else {
        need(step.level >= 1 && step.level <= 9,
             "gzip level " + std::to_string(step.level) +
                 " is outside the 1 to 9 section 23 allows");
        need(H5Pset_deflate(dcpl.get(),
                            static_cast<unsigned>(step.level)) >= 0,
             "cannot set the gzip filter");
      }
    }
    // Section 23 forbids setting a fill value and this sets none.  The
    // fill *time* is a different property, which the format does not
    // mention; ALLOC on a chunked dataset and the library default on a
    // contiguous one is what the HDF5 wrappers this format
    // interoperates with write, so two writers agree byte for byte and
    // not only structurally.
    H5Pset_fill_time(dcpl.get(), H5D_FILL_TIME_ALLOC);
  }
  return dcpl;
}

// H5Dwrite reads as many elements as the dataspace declares, so a
// caller that hands over fewer than the shape says would have it read
// past the end of the vector.  Every write goes through here first.
void need_elements(const std::string& path,
                   const std::vector<hsize_t>& shape, std::size_t given) {
  const std::size_t want =
      checked_product(path, shape, kMaxDatasetElements);
  if (given != want) {
    throw Error("", "\"" + path + "\" declares " + std::to_string(want) +
                        " elements and was given " + std::to_string(given));
  }
}

Id make_space(const std::vector<hsize_t>& shape,
              const std::vector<hsize_t>& maxshape) {
  if (shape.empty()) return Id(H5Screate(H5S_SCALAR));
  const hsize_t* max = maxshape.empty() ? nullptr : maxshape.data();
  return Id(H5Screate_simple(static_cast<int>(shape.size()), shape.data(),
                             max));
}

}  // namespace

void File::write_f64(const std::string& path,
                     const std::vector<hsize_t>& shape,
                     const std::vector<hsize_t>& maxshape,
                     const std::vector<hsize_t>& chunk,
                     const std::vector<double>& data,
                     const FilterPipeline& filters) {
  need_elements(path, shape, data.size());
  make_group(parent_of(path));
  Id space = make_space(shape, maxshape);
  Id dcpl = make_dcpl(chunk, filters);
  Id dset(H5Dcreate2(id_.get(), path.c_str(), H5T_IEEE_F64LE, space.get(),
                     H5P_DEFAULT, dcpl.get(), H5P_DEFAULT));
  need(dset.valid(), "cannot create the dataset \"" + path + "\"");
  if (!data.empty()) {
    need(H5Dwrite(dset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                  H5P_DEFAULT, data.data()) >= 0,
         "cannot write \"" + path + "\"");
  }
}

void File::write_ints(const std::string& path, DType dtype,
                      const std::vector<hsize_t>& shape,
                      const std::vector<hsize_t>& maxshape,
                      const std::vector<hsize_t>& chunk,
                      const std::vector<std::int64_t>& data,
                      const FilterPipeline& filters) {
  need_elements(path, shape, data.size());
  make_group(parent_of(path));
  hid_t file_type = H5T_STD_I64LE;
  switch (dtype) {
    case DType::Bool: file_type = H5T_STD_I8LE; break;
    case DType::UInt8: file_type = H5T_STD_U8LE; break;
    case DType::Int32: file_type = H5T_STD_I32LE; break;
    default: file_type = H5T_STD_I64LE; break;
  }
  Id space = make_space(shape, maxshape);
  Id dcpl = make_dcpl(chunk, filters);
  Id dset(H5Dcreate2(id_.get(), path.c_str(), file_type, space.get(),
                     H5P_DEFAULT, dcpl.get(), H5P_DEFAULT));
  need(dset.valid(), "cannot create the dataset \"" + path + "\"");
  if (!data.empty()) {
    need(H5Dwrite(dset.get(), H5T_NATIVE_INT64, H5S_ALL, H5S_ALL,
                  H5P_DEFAULT, data.data()) >= 0,
         "cannot write \"" + path + "\"");
  }
}

void File::write_strings(const std::string& path, std::size_t item_size,
                         const std::vector<hsize_t>& shape,
                         const std::vector<hsize_t>& maxshape,
                         const std::vector<hsize_t>& chunk,
                         const std::vector<std::string>& data,
                         const FilterPipeline& filters) {
  need_elements(path, shape, data.size());
  make_group(parent_of(path));
  const std::size_t item = item_size == 0 ? 1 : item_size;
  Id type = string_type(item);
  Id space = make_space(shape, maxshape);
  Id dcpl = make_dcpl(chunk, filters);
  Id dset(H5Dcreate2(id_.get(), path.c_str(), type.get(), space.get(),
                     H5P_DEFAULT, dcpl.get(), H5P_DEFAULT));
  need(dset.valid(), "cannot create the dataset \"" + path + "\"");
  if (!data.empty()) {
    std::string buffer(item * data.size(), '\0');
    for (std::size_t i = 0; i < data.size(); ++i) {
      const std::size_t n = data[i].size() < item ? data[i].size() : item;
      std::memcpy(&buffer[i * item], data[i].data(), n);
    }
    need(H5Dwrite(dset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  buffer.data()) >= 0,
         "cannot write \"" + path + "\"");
  }
}

// --- growing a file in place ------------------------------------------

namespace {

// The rows `data` holds against a dataset's non-leading extents, and the
// check that it holds whole rows and nothing else.
std::size_t rows_held(const std::string& path, const DsetInfo& info,
                      std::size_t elements) {
  need(info.is_dataset && !info.shape.empty(),
       "\"" + path + "\" has no rows to write");
  std::size_t stride = 1;
  for (std::size_t i = 1; i < info.shape.size(); ++i) {
    const std::size_t e = static_cast<std::size_t>(info.shape[i]);
    if (e != 0 && stride > kMaxDatasetElements / e) {
      throw Error("E41", "\"" + path + "\" declares more elements than this "
                                        "writer will write");
    }
    stride *= e;
  }
  if (stride == 0) return 0;
  if (elements % stride != 0) {
    throw Error("", "\"" + path + "\" takes " + std::to_string(stride) +
                        " elements per row and was given " +
                        std::to_string(elements));
  }
  return elements / stride;
}

// The hyperslab of rows [row, row + count), with a memory space to match.
void select_row_block(const std::string& path, const DsetInfo& info,
                      std::size_t row, std::size_t count, Id* space,
                      Id* mem) {
  const std::size_t have = static_cast<std::size_t>(info.shape[0]);
  if (row > have || count > have - row) {
    throw Error("", "rows " + std::to_string(row) + " to " +
                        std::to_string(row + count) + " lie outside \"" +
                        path + "\", which has " + std::to_string(have));
  }
  std::vector<hsize_t> start(info.shape.size(), 0);
  std::vector<hsize_t> block = info.shape;
  start[0] = static_cast<hsize_t>(row);
  block[0] = static_cast<hsize_t>(count);
  H5Sselect_hyperslab(space->get(), H5S_SELECT_SET, start.data(), nullptr,
                      block.data(), nullptr);
  *mem = Id(H5Screate_simple(static_cast<int>(block.size()), block.data(),
                             nullptr));
}

}  // namespace

void File::extend_rows(const std::string& path, hsize_t rows) {
  const DsetInfo info = dataset_info(path);
  need(info.is_dataset && !info.shape.empty(),
       "\"" + path + "\" has no rows to extend");
  need(info.chunked && !info.maxshape.empty() &&
           info.maxshape[0] == H5S_UNLIMITED,
       "\"" + path + "\" is not unlimited along its leading dimension");
  need(rows >= info.shape[0], "\"" + path + "\" can only grow");
  if (rows == info.shape[0]) return;
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open \"" + path + "\"");
  std::vector<hsize_t> shape = info.shape;
  shape[0] = rows;
  need(H5Dset_extent(dset.get(), shape.data()) >= 0,
       "cannot extend \"" + path + "\"");
}

void File::write_f64_rows(const std::string& path, std::size_t row,
                          const std::vector<double>& data) {
  const DsetInfo info = dataset_info(path);
  const std::size_t count = rows_held(path, info, data.size());
  if (count == 0) return;
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open \"" + path + "\"");
  Id space(H5Dget_space(dset.get()));
  Id mem;
  select_row_block(path, info, row, count, &space, &mem);
  need(H5Dwrite(dset.get(), H5T_NATIVE_DOUBLE, mem.get(), space.get(),
                H5P_DEFAULT, data.data()) >= 0,
       "cannot write rows of \"" + path + "\"");
}

void File::write_i64_rows(const std::string& path, std::size_t row,
                          const std::vector<std::int64_t>& data) {
  const DsetInfo info = dataset_info(path);
  const std::size_t count = rows_held(path, info, data.size());
  if (count == 0) return;
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open \"" + path + "\"");
  Id space(H5Dget_space(dset.get()));
  Id mem;
  select_row_block(path, info, row, count, &space, &mem);
  need(H5Dwrite(dset.get(), H5T_NATIVE_INT64, mem.get(), space.get(),
                H5P_DEFAULT, data.data()) >= 0,
       "cannot write rows of \"" + path + "\"");
}

void File::write_string_rows(const std::string& path, std::size_t row,
                             const std::vector<std::string>& data) {
  const DsetInfo info = dataset_info(path);
  const std::size_t count = rows_held(path, info, data.size());
  if (count == 0) return;
  need(!info.type.variable_length && info.type.size > 0,
       "\"" + path + "\" is not a fixed-length string dataset");
  const std::size_t item = info.type.size;
  for (const std::string& s : data) {
    if (s.size() > item) {
      throw Error("", "\"" + path + "\" holds strings of " +
                          std::to_string(item) + " bytes and \"" + s +
                          "\" is longer");
    }
  }
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open \"" + path + "\"");
  Id type(H5Dget_type(dset.get()));
  Id space(H5Dget_space(dset.get()));
  Id mem;
  select_row_block(path, info, row, count, &space, &mem);
  std::string buffer(item * data.size(), '\0');
  for (std::size_t i = 0; i < data.size(); ++i) {
    std::memcpy(&buffer[i * item], data[i].data(), data[i].size());
  }
  need(H5Dwrite(dset.get(), type.get(), mem.get(), space.get(), H5P_DEFAULT,
                buffer.data()) >= 0,
       "cannot write rows of \"" + path + "\"");
}

void File::replace_attr(const std::string& path, const std::string& name,
                        const AttrValue& value) {
  {
    Id object = open_object(id_.get(), path);
    if (H5Aexists(object.get(), name.c_str()) > 0) {
      need(H5Adelete(object.get(), name.c_str()) >= 0,
           "cannot replace the attribute \"" + name + "\"");
    }
  }
  write_attr(path, name, value);
}

void File::remove_attributes(const std::string& path) {
  Id object = open_object(id_.get(), path);
  H5O_info2_t info;
  need(H5Oget_info3(object.get(), &info, H5O_INFO_NUM_ATTRS) >= 0,
       "cannot count the attributes of \"" + path + "\"");
  std::vector<std::string> names;
  for (hsize_t i = 0; i < info.num_attrs; ++i) {
    // Names are collected first and deleted after, since deleting
    // while iterating by index would skip every other one.
    const ssize_t n = H5Aget_name_by_idx(object.get(), ".", H5_INDEX_NAME,
                                         H5_ITER_INC, i, nullptr, 0,
                                         H5P_DEFAULT);
    if (n < 0) continue;
    std::string name(static_cast<std::size_t>(n) + 1, '\0');
    H5Aget_name_by_idx(object.get(), ".", H5_INDEX_NAME, H5_ITER_INC, i,
                       &name[0], name.size(), H5P_DEFAULT);
    name.resize(static_cast<std::size_t>(n));
    names.push_back(name);
  }
  for (const std::string& name : names) {
    need(H5Adelete(object.get(), name.c_str()) >= 0,
         "cannot delete the attribute \"" + name + "\" of \"" + path +
             "\"");
  }
}

void File::set_scale_length(const std::string& path, hsize_t length) {
  extend_rows(path, length);
  Id dset(H5Dopen2(id_.get(), path.c_str(), H5P_DEFAULT));
  need(dset.valid(), "cannot open the dimension scale \"" + path + "\"");
  // NAME as H5DSset_scale writes it: a NUL-terminated string one byte
  // longer than the sentence, so that the grown scale carries the same
  // attribute the writer gave it, with the new length spelled out.
  const std::string name = scale_name_attribute(length);
  if (H5Aexists(dset.get(), "NAME") > 0) {
    need(H5Adelete(dset.get(), "NAME") >= 0,
         "cannot replace the NAME of \"" + path + "\"");
  }
  Id type(H5Tcopy(H5T_C_S1));
  H5Tset_size(type.get(), name.size() + 1);
  Id space(H5Screate(H5S_SCALAR));
  Id attr(H5Acreate2(dset.get(), "NAME", type.get(), space.get(),
                     H5P_DEFAULT, H5P_DEFAULT));
  need(attr.valid(), "cannot write the NAME of \"" + path + "\"");
  need(H5Awrite(attr.get(), type.get(), name.c_str()) >= 0,
       "cannot write the NAME of \"" + path + "\"");
}

void File::remove_link(const std::string& path) {
  need(H5Ldelete(id_.get(), path.c_str(), H5P_DEFAULT) >= 0,
       "cannot remove \"" + path + "\"");
}

void File::build_object_index() const {
  object_index_built_ = true;
  // One walk, bounded in depth and in count, recording every dataset's
  // token against its link name.  Iterative, so a file that nests
  // groups deeply costs stack here as well as inside the library.
  std::vector<std::pair<std::string, int>> todo;
  todo.emplace_back("/", 0);
  while (!todo.empty()) {
    const std::pair<std::string, int> here = todo.back();
    todo.pop_back();
    if (here.second > kMaxGroupDepth) continue;
    if (object_paths_.size() >= kMaxIndexedObjects) return;
    for (const Member& m : members(here.first)) {
      if (m.kind != LinkKind::Hard) continue;
      const std::string child =
          (here.first == "/" ? std::string("/") : here.first + "/") + m.name;
      if (m.is_group) {
        todo.emplace_back(child, here.second + 1);
        continue;
      }
      if (!m.is_dataset) continue;
      H5O_info2_t info;
      if (H5Oget_info_by_name3(id_.get(), child.c_str(), &info,
                               H5O_INFO_BASIC, H5P_DEFAULT) < 0) {
        continue;
      }
      const std::string key = token_key(id_.get(), info.token);
      if (!key.empty()) object_paths_[key] = child;
      if (object_paths_.size() >= kMaxIndexedObjects) return;
    }
  }
}

std::string File::dataset_path(hid_t object) const {
  if (!object_index_built_) build_object_index();
  H5O_info2_t info;
  if (H5Oget_info3(object, &info, H5O_INFO_BASIC) < 0) return std::string();
  const std::string key = token_key(id_.get(), info.token);
  if (key.empty()) return std::string();
  const auto it = object_paths_.find(key);
  return it == object_paths_.end() ? std::string() : it->second;
}

std::string File::dataset_link_name(hid_t object) const {
  const std::string path = dataset_path(object);
  const std::size_t at = path.find_last_of('/');
  return at == std::string::npos ? path : path.substr(at + 1);
}

void File::make_scale(const std::string& path, hsize_t length,
                      bool unlimited, const std::vector<hsize_t>& chunk) {
  make_group(parent_of(path));
  // Section 21: one-dimensional H5T_IEEE_F32BE, no value
  // ever written, chunked with chunk length 1 when it is unlimited and
  // with a chunk equal to its length when it is not, which is what
  // H5DSset_scale leaves behind.
  const hsize_t max = unlimited ? H5S_UNLIMITED : length;
  Id space(H5Screate_simple(1, &length, &max));
  Id dcpl(H5Pcreate(H5P_DATASET_CREATE));
  // Section 21.  Tracking attribute creation order gives
  // the scale a version 2 object header, which is what lets its
  // REFERENCE_LIST live in the file's fractal heap instead of in an
  // object header message, where an attribute may not exceed 64 KiB.
  // Without it no scale carries more than 4085 attachments.  The
  // library version bounds stay at the default here and everywhere
  // else, so this changes the object header of the scales and of
  // nothing else.
  //
  // H5Pset_attr_phase_change looks like the way to ask for the same
  // thing and is silently ignored under the default bounds; it must
  // not be relied on.
  H5Pset_attr_creation_order(dcpl.get(),
                             H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
  // Not optional, and not only section 30's byte reproducibility: a
  // version 2 object header records four timestamps unless it is told
  // not to, so the call above would make every file record when it
  // was written.
  H5Pset_obj_track_times(dcpl.get(), 0);
  const hsize_t fallback = unlimited ? 1 : (length == 0 ? 1 : length);
  const hsize_t* use = chunk.empty() ? &fallback : chunk.data();
  H5Pset_chunk(dcpl.get(), 1, use);
  H5Pset_fill_time(dcpl.get(), H5D_FILL_TIME_ALLOC);
  Id dset(H5Dcreate2(id_.get(), path.c_str(), H5T_IEEE_F32BE, space.get(),
                     H5P_DEFAULT, dcpl.get(), H5P_DEFAULT));
  need(dset.valid(), "cannot create the dimension scale \"" + path + "\"");
  const std::string name = scale_name_attribute(length);
  need(H5DSset_scale(dset.get(), name.c_str()) >= 0,
       "cannot mark \"" + path + "\" as a dimension scale");
}

void File::attach_scale(const std::string& dataset, const std::string& scale,
                        unsigned axis) {
  Id d(H5Dopen2(id_.get(), dataset.c_str(), H5P_DEFAULT));
  need(d.valid(), "cannot open \"" + dataset + "\"");
  Id s(H5Dopen2(id_.get(), scale.c_str(), H5P_DEFAULT));
  need(s.valid(), "cannot open the dimension scale \"" + scale + "\"");
  // An attachment that fails has already deleted the REFERENCE_LIST it
  // was extending, so what is on disk is a file every reader and every
  // validator still accepts and netCDF-C can no longer rebuild the
  // dimension from.  The caller has to delete it, so the message says
  // what was lost and where the rule is.
  need(H5DSattach_scale(d.get(), s.get(), axis) >= 0,
       "cannot attach \"" + scale + "\" to \"" + dataset +
           "\": the scale's REFERENCE_LIST was deleted before the "
           "attachment failed and has not been written back, so this file "
           "is incomplete and must be deleted rather than kept (section "
           "21)");
}

namespace {

// Every object under `root`, by path, from a walk bounded in depth and
// in count the way every other walk over file-controlled structure
// here is.  A link that is not a hard link is recorded and not
// followed, which is also what the library's own copy does with one.
void walk_opaque(const File& f, const std::string& root,
                 std::vector<std::string>* groups,
                 std::vector<std::string>* datasets) {
  std::vector<std::pair<std::string, int>> todo;
  todo.emplace_back(root, 0);
  groups->push_back(root);
  while (!todo.empty()) {
    const std::pair<std::string, int> here = todo.back();
    todo.pop_back();
    if (here.second >= kMaxGroupDepth) {
      throw Error("E41", here.first +
                             ": nested deeper than this reader walks (" +
                             std::to_string(kMaxGroupDepth) + ")");
    }
    for (const Member& m : f.members(here.first)) {
      if (groups->size() + datasets->size() >= kMaxOpaqueObjects) {
        throw Error("E41", root + ": more objects than this reader copies (" +
                               std::to_string(kMaxOpaqueObjects) + ")");
      }
      if (m.kind != LinkKind::Hard) continue;
      const std::string child =
          (here.first == "/" ? std::string("/") : here.first + "/") + m.name;
      if (m.is_group) {
        groups->push_back(child);
        todo.emplace_back(child, here.second + 1);
      } else if (m.is_dataset) {
        datasets->push_back(child);
      }
    }
  }
}

// The machinery attributes that hold object references.  The library's
// object copy carries their bytes and not what they point at, so they
// are dropped from the copy and the attachments they stood for are
// remade from the record this file keeps beside the image.
void drop_reference_attributes(hid_t file, const std::string& path) {
  Id object(H5Oopen(file, path.c_str(), H5P_DEFAULT));
  if (!object.valid()) return;
  for (const char* name : {"DIMENSION_LIST", "REFERENCE_LIST"}) {
    if (H5Aexists(object.get(), name) > 0) H5Adelete(object.get(), name);
  }
}

}  // namespace

OpaqueGroup capture_group(const File& f, const std::string& path) {
  OpaqueGroup out;
  if (!f.is_group(path)) return out;

  // The walk first, so that the depth, the object count and the size
  // are known before the library is asked to copy anything.  The copy
  // itself recurses inside HDF5, where a stack overflow cannot be
  // caught, which is the trap section 21 names in another place.
  std::vector<std::string> groups;
  std::vector<std::string> datasets;
  walk_opaque(f, path, &groups, &datasets);

  std::size_t bytes = 0;
  for (const std::string& p : datasets) {
    const DsetInfo info = f.dataset_info(p);
    std::size_t elements = 1;
    for (const hsize_t extent : info.shape) {
      const std::size_t e = static_cast<std::size_t>(extent);
      if (e != 0 && elements > kMaxOpaqueBytes / e) {
        throw Error("E41", p + ": larger than this reader copies");
      }
      elements *= e;
    }
    const std::size_t size = info.type.size == 0 ? 1 : info.type.size;
    if (elements > (kMaxOpaqueBytes - bytes) / size) {
      throw Error("E41", path + ": more than this reader copies (" +
                             std::to_string(kMaxOpaqueBytes) + " bytes)");
    }
    bytes += elements * size;
    for (std::size_t axis = 0; axis < info.scale_paths.size(); ++axis) {
      for (const std::string& scale : info.scale_paths[axis]) {
        if (scale.empty()) continue;
        OpaqueGroup::Attachment a;
        a.dataset = p;
        a.axis = axis;
        a.scale = scale;
        out.attachments.push_back(a);
      }
    }
  }

  // An HDF5 file that never reaches the disk, holding the copy and
  // nothing else.  What goes into it is the library's own object copy,
  // so no dtype, filter, chunk, attribute or subgroup here is decided
  // by this code, which is what section 12 asks of a reader that
  // carries a private group.
  Id fapl(H5Pcreate(H5P_FILE_ACCESS));
  need(fapl.valid(), "cannot make a file access property list");
  need(H5Pset_fapl_core(fapl.get(), 1u << 16, 0) >= 0,
       "cannot make an in-memory HDF5 file");
  Id mem(H5Fcreate("mestra-opaque.mem", H5F_ACC_TRUNC, H5P_DEFAULT,
                   fapl.get()));
  need(mem.valid(), "cannot make an in-memory HDF5 file");
  // The copy keeps its path, and a group that is not at the root --
  // one a support carries -- needs its parents made in the image.
  Id lcpl(H5Pcreate(H5P_LINK_CREATE));
  need(lcpl.valid() && H5Pset_create_intermediate_group(lcpl.get(), 1) >= 0,
       "cannot make a link creation property list");
  need(H5Ocopy(f.get(), path.c_str(), mem.get(), path.c_str(), H5P_DEFAULT,
               lcpl.get()) >= 0,
       "cannot copy \"" + path + "\"");
  for (const std::string& p : groups) drop_reference_attributes(mem.get(), p);
  for (const std::string& p : datasets) drop_reference_attributes(mem.get(), p);
  need(H5Fflush(mem.get(), H5F_SCOPE_GLOBAL) >= 0,
       "cannot flush the copy of \"" + path + "\"");

  const ssize_t size = H5Fget_file_image(mem.get(), nullptr, 0);
  need(size > 0, "cannot take the copy of \"" + path + "\"");
  if (static_cast<std::size_t>(size) > kMaxOpaqueBytes) {
    throw Error("E41", path + ": more than this reader copies (" +
                           std::to_string(kMaxOpaqueBytes) + " bytes)");
  }
  out.image.resize(static_cast<std::size_t>(size));
  need(H5Fget_file_image(mem.get(), out.image.data(), out.image.size()) > 0,
       "cannot take the copy of \"" + path + "\"");
  return out;
}

void restore_group(File& f, const std::string& path, const OpaqueGroup& g) {
  if (g.empty()) return;
  // The image is this library's own, written by capture_group above.
  // The library takes its own copy of the buffer, so nothing here
  // depends on how long the caller keeps the dataset alive.
  Id mem(H5LTopen_file_image(const_cast<char*>(g.image.data()),
                             g.image.size(), 0));
  need(mem.valid(), "cannot open the copy of \"" + path + "\"");
  need(H5Ocopy(mem.get(), path.c_str(), f.get(), path.c_str(), H5P_DEFAULT,
               H5P_DEFAULT) >= 0,
       "cannot put \"" + path + "\" back");
  // An attachment whose scale did not come with the group -- the
  // file-level `row` scale, say -- is remade against the one this
  // writer wrote, and one whose scale is nowhere in the new file is
  // left undone rather than guessed at.
  for (const OpaqueGroup::Attachment& a : g.attachments) {
    if (!f.is_dataset(a.dataset) || !f.is_dataset(a.scale)) continue;
    f.attach_scale(a.dataset, a.scale, static_cast<unsigned>(a.axis));
  }
}

}  // namespace internal
}  // namespace mestra
