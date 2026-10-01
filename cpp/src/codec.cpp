#include "codec.hpp"

#include <algorithm>
#include <cstdio>

#include "mestra/io.hpp"
#include "names.hpp"

namespace mestra {
namespace internal {

// Section 18: the null sentinel is one NUL byte followed by "null",
// with size 5.  No other string may contain a NUL byte, so it cannot
// be confused with a value.
static const char kNullSentinel[5] = {'\0', 'n', 'u', 'l', 'l'};

bool machinery_attribute(const std::string& name) {
  static const char* kNames[] = {
      "CLASS",           "NAME",
      "DIMENSION_LIST",  "REFERENCE_LIST",
      "DIMENSION_LABELS", "_Netcdf4Dimid",
      "_Netcdf4Coordinates", "_nc3_strict",
      "_NCProperties"};
  for (const char* n : kNames) {
    if (name == n) return true;
  }
  return false;
}

namespace {

bool is_null_sentinel(const RawAttr& a) {
  return a.type.klass == H5T_STRING && !a.type.variable_length &&
         a.raw_bytes.size() == 5 &&
         std::equal(a.raw_bytes.begin(), a.raw_bytes.end(), kNullSentinel);
}

std::string scale_name(const std::string& dataset, std::size_t axis) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "_d%zu", axis);
  return "mestra_" + dataset + std::string(buffer);
}

}  // namespace

Dict read_dict_group(const File& f, const std::string& path, bool top_level,
                     int depth) {
  if (depth >= kMaxDictDepth) {
    throw Error("E41", "\"" + path + "\" is nested deeper than this reader "
                                      "walks");
  }
  Dict d;
  for (const RawAttr& a : f.attributes(path)) {
    if (machinery_attribute(a.name)) continue;
    if (reserved_name(a.name)) continue;
    if (top_level && (a.name == "type" || a.name == "repr")) continue;
    if (is_null_sentinel(a)) {
      d.set(a.name, Value::null());
      continue;
    }
    switch (a.value.kind()) {
      case AttrValue::Kind::Bool:
        d.set(a.name, Value::boolean(a.value.as_bool()));
        break;
      case AttrValue::Kind::Int:
        d.set(a.name, Value::integer(a.value.as_int()));
        break;
      case AttrValue::Kind::Float:
        d.set(a.name, Value::real(a.value.as_float()));
        break;
      case AttrValue::Kind::Str:
        d.set(a.name, Value::text(a.value.as_text()));
        break;
    }
  }
  for (const Member& m : f.members(path)) {
    if (reserved_name(m.name)) continue;
    const std::string child = path + "/" + m.name;
    if (m.kind != LinkKind::Hard) {
      // A dictionary is data, and this reader follows a hard link and
      // nothing else.
      throw Error("E40", "\"" + child + "\" is " +
                             link_kind_name(m.kind) +
                             "; a reader never follows one");
    }
    if (m.is_group) {
      d.set(m.name,
            Value::dict(read_dict_group(f, child, false, depth + 1)));
      continue;
    }
    if (!m.is_dataset) continue;
    const DsetInfo info = f.dataset_info(child);
    if (info.shape.empty()) {
      throw Error("E32",
                  "\"" + child +
                      "\" is a zero-dimensional dataset, which section 25 "
                      "requires to be written as an attribute");
    }
    Array a;
    a.shape.assign(info.shape.begin(), info.shape.end());
    DType dtype = DType::Float64;
    if (!dtype_of(info.type, &dtype)) {
      throw Error("E32", "\"" + child + "\" has a dtype no dictionary may "
                                        "hold");
    }
    a.dtype = dtype;
    if (dtype == DType::String) {
      a.str = f.read_strings(child);
      d.set(m.name, Value::strings(a));
    } else if (dtype == DType::Float64) {
      a.f64 = f.read_f64(child);
      d.set(m.name, Value::numbers(a));
    } else if (dtype == DType::Int32 || dtype == DType::Int64 ||
               dtype == DType::Bool) {
      a.i64 = f.read_i64(child);
      d.set(m.name, Value::numbers(a));
    } else {
      throw Error("E32", "\"" + child + "\" has a dtype no dictionary may "
                                        "hold");
    }
  }
  return d;
}

std::vector<hsize_t> StoredStorage::chunk_of(const std::string& path) const {
  std::vector<hsize_t> out;
  if (chunks == nullptr) return out;
  const auto it = chunks->find(path);
  if (it != chunks->end()) out.assign(it->second.begin(), it->second.end());
  return out;
}

FilterPipeline StoredStorage::filters_of(const std::string& path) const {
  if (filters == nullptr) return FilterPipeline();
  const auto it = filters->find(path);
  return it == filters->end() ? FilterPipeline() : it->second;
}

void write_dict_group(File& f, const std::string& path, const Dict& d,
                      const StoredStorage* stored, int depth) {
  if (depth >= kMaxDictDepth) {
    throw Error("E41", "\"" + path + "\" is nested deeper than this writer "
                                      "walks");
  }
  const StoredStorage none;
  const StoredStorage& found = stored == nullptr ? none : *stored;
  // Section 25: a writer visits a dictionary's keys in ascending order
  // of their UTF-8 bytes, which is the order Dict iterates in.
  for (const auto& entry : d) {
    const std::string& key = entry.first;
    const Value& v = entry.second;
    if (reserved_name(key)) {
      throw Error("E33", "the dictionary key \"" + key +
                             "\" begins with the reserved prefix");
    }
    if (!legal_netcdf_name(key)) {
      throw Error("E33", "the dictionary key \"" + key +
                             "\" is not a legal netCDF-4 name");
    }
    const std::string child = path + "/" + key;
    switch (v.kind()) {
      case Value::Kind::Null:
        f.write_raw_string_attr(path, key,
                                std::string(kNullSentinel, 5));
        break;
      case Value::Kind::Bool:
        f.write_attr(path, key, AttrValue::boolean(v.as_bool()));
        break;
      case Value::Kind::Int:
        f.write_attr(path, key, AttrValue::integer(v.as_int()));
        break;
      case Value::Kind::Float:
        f.write_attr(path, key, AttrValue::real(v.as_float()));
        break;
      case Value::Kind::Str: {
        const std::string& text = v.as_text();
        if (text.find('\0') != std::string::npos) {
          throw Error("E32", "a dictionary string with an embedded NUL is "
                             "not representable");
        }
        f.write_attr(path, key, AttrValue::text(text));
        break;
      }
      case Value::Kind::Numbers:
      case Value::Kind::Strings: {
        const Array& a = v.as_array();
        if (a.shape.empty()) {
          throw Error("E32", "a zero-dimensional array must be written as "
                             "the number it holds");
        }
        std::vector<hsize_t> shape(a.shape.begin(), a.shape.end());
        const bool empty =
            std::find(a.shape.begin(), a.shape.end(), std::size_t(0)) !=
            a.shape.end();
        // Section 25: every zero-length axis is created with an
        // unlimited maximum, so that the dimension is legal in
        // netCDF-4.  Only that axis: E43 allows no other dimension
        // but `row` to be unlimited, so an axis of length 2 beside it
        // keeps its length as its maximum.
        std::vector<hsize_t> maxshape;
        std::vector<hsize_t> chunk;
        if (empty) {
          maxshape = shape;
          for (hsize_t& e : maxshape) {
            if (e == 0) e = H5S_UNLIMITED;
          }
          chunk.assign(shape.size(), 1);
        }
        // A dictionary dataset is not row dimensioned, so section 23
        // leaves it contiguous unless the file it came from chunked
        // it -- which it must have, to carry a filter at all.
        const FilterPipeline pipeline = found.filters_of(child);
        const std::vector<hsize_t> kept = found.chunk_of(child);
        if (!kept.empty()) {
          chunk = kept;
        } else if (!pipeline.empty() && chunk.empty()) {
          chunk = shape;
          for (hsize_t& e : chunk) {
            if (e == 0) e = 1;
          }
        }
        if (v.kind() == Value::Kind::Strings) {
          std::size_t item = 1;
          for (const std::string& s : a.str) {
            if (s.find('\0') != std::string::npos) {
              throw Error("E32", "a dictionary string with an embedded NUL "
                                 "is not representable");
            }
            item = std::max(item, s.size());
          }
          f.write_strings(child, item, shape, maxshape, chunk, a.str,
                          pipeline);
        } else if (a.dtype == DType::Float64) {
          f.write_f64(child, shape, maxshape, chunk, a.f64, pipeline);
        } else if (a.dtype == DType::Int32 || a.dtype == DType::Int64 ||
                   a.dtype == DType::Bool) {
          f.write_ints(child, a.dtype, shape, maxshape, chunk, a.i64,
                       pipeline);
        } else {
          throw Error("E32", "a dictionary array of dtype " +
                                 std::string(dtype_name(a.dtype)) +
                                 " is not representable");
        }
        for (std::size_t axis = 0; axis < a.shape.size(); ++axis) {
          const std::string s = path + "/" + scale_name(key, axis);
          f.make_scale(s, static_cast<hsize_t>(a.shape[axis]),
                       a.shape[axis] == 0, found.chunk_of(s));
          f.attach_scale(child, s, static_cast<unsigned>(axis));
        }
        break;
      }
      case Value::Kind::Dict:
        f.make_group(child);
        write_dict_group(f, child, v.as_dict(), stored, depth + 1);
        break;
    }
  }
}

}  // namespace internal
}  // namespace mestra
