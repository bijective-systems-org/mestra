#include "layout.hpp"

#include <algorithm>
#include <set>

#include "names.hpp"

namespace mestra {
namespace internal {

std::string logical_dim(const std::string& disk) {
  if (disk.compare(0, 10, "component_") == 0) return "component";
  if (disk.compare(0, 5, "draw_") == 0) return "draw";
  if (disk.compare(0, 6, "group_") == 0) return "group:" + disk.substr(6);
  return disk;   // row, node, cell, index, cell_plus_one
}

std::size_t rows_on_support(const Dataset& d, std::size_t index) {
  if (!d.row_support.has_value()) {
    return static_cast<std::size_t>(d.n_rows);
  }
  std::size_t n = 0;
  for (const std::int32_t at : *d.row_support) {
    if (at >= 0 && static_cast<std::size_t>(at) == index) ++n;
  }
  return n;
}

// Section 21: a support-local `row` exists in an unaligned file in a
// support that carries an array with varies = row.  A slot a callable
// serves is such an array even though it holds no data yet, so it
// counts too, as it does in every other implementation.
bool needs_local_row(const Dataset& d, const Support& s) {
  if (d.aligned) return false;
  if (s.coordinates.has_value() && s.coordinates->varies == "row") {
    return true;
  }
  for (const ArraySlot& a : s.node_arrays) {
    if (a.varies == "row") return true;
  }
  for (const ArraySlot& a : s.cell_arrays) {
    if (a.varies == "row") return true;
  }
  return false;
}

namespace {

void collect(const ArraySlot& a, std::set<std::size_t>* components,
             std::set<std::size_t>* draws) {
  if (a.is_callable()) return;          // a slot with no data has no axes
  const int comp = a.data.axis("component");
  if (comp >= 0) {
    components->insert(a.data.shape[static_cast<std::size_t>(comp)]);
  }
  const int draw = a.data.axis("draw");
  if (draw >= 0) {
    draws->insert(a.data.shape[static_cast<std::size_t>(draw)]);
  }
}

}  // namespace

std::vector<std::size_t> component_lengths(const Dataset& d) {
  std::set<std::size_t> components;
  std::set<std::size_t> draws;
  for (const Support& s : d.supports) {
    if (s.coordinates.has_value()) collect(*s.coordinates, &components,
                                           &draws);
    for (const ArraySlot& a : s.node_arrays) collect(a, &components, &draws);
    for (const ArraySlot& a : s.cell_arrays) collect(a, &components, &draws);
  }
  return std::vector<std::size_t>(components.begin(), components.end());
}

std::vector<std::size_t> draw_lengths(const Dataset& d) {
  std::set<std::size_t> components;
  std::set<std::size_t> draws;
  for (const Support& s : d.supports) {
    if (s.coordinates.has_value()) collect(*s.coordinates, &components,
                                           &draws);
    for (const ArraySlot& a : s.node_arrays) collect(a, &components, &draws);
    for (const ArraySlot& a : s.cell_arrays) collect(a, &components, &draws);
  }
  return std::vector<std::size_t>(draws.begin(), draws.end());
}

namespace {

bool one_of(const std::string& name, const std::vector<const char*>& list) {
  for (const char* n : list) {
    if (name == n) return true;
  }
  return false;
}

}  // namespace

bool known_root_attribute(const std::string& name) {
  return one_of(name, {"format", "writer", "created", "aligned",
                       "generalisation_group"});
}

bool known_key_attribute(const std::string& name) {
  return one_of(name, {"role", "units", "lower", "upper", "category",
                       "trajectory_group", "parent"});
}

bool known_scalar_attribute(const std::string& name) {
  return one_of(name, {"units", "source", "output", "statistic", "of",
                       "quantile", "level", "method"});
}

bool known_support_attribute(const std::string& name) {
  return one_of(name, {"kind", "n_nodes", "n_cells", "support_id"});
}

bool known_array_attribute(const std::string& name) {
  return one_of(name, {"role", "varies", "units", "components", "source",
                       "output", "statistic", "of", "quantile", "level",
                       "method", "category", "recomputed", "derived_from",
                       "recipe", "reference"});
}

bool known_callable_attribute(const std::string& name) {
  return one_of(name, {"type", "repr"});
}

bool known_root_group(const std::string& name) {
  return one_of(name, {"keys", "scalars", "categories", "supports",
                       "callables", "notes", "private"});
}

bool known_support_group(const std::string& name) {
  // `coordinates` is a dataset when stored and a group when a callable
  // serves it (section 19), so as a group it is known, not W11.
  return one_of(name, {"node_arrays", "cell_arrays", "coordinates"});
}

bool root_scale_name(const std::string& name) {
  if (name == "row") return true;
  return name.compare(0, 10, "component_") == 0 ||
         name.compare(0, 5, "draw_") == 0 ||
         name.compare(0, 6, "group_") == 0 ||
         name.compare(0, 9, "category_") == 0;
}

bool support_scale_name(const std::string& name) {
  return one_of(name, {"node", "cell", "cell_plus_one", "index", "row"});
}

bool known_dataset_path(const std::string& path, bool is_scale) {
  // The path split into its components, which is all this needs.
  std::vector<std::string> part;
  std::size_t at = 0;
  while (at < path.size()) {
    const std::size_t next = path.find('/', at + 1);
    const std::size_t end = next == std::string::npos ? path.size() : next;
    if (end > at + 1) part.push_back(path.substr(at + 1, end - at - 1));
    at = end;
  }
  if (part.empty()) return false;
  if (part.size() == 1) {
    // At the root: a dimension scale of section 21, or /row_support.
    if (part[0] == "row_support") return true;
    return is_scale && root_scale_name(part[0]);
  }
  if (part.size() == 2) {
    if (part[0] == "keys" || part[0] == "scalars" ||
        part[0] == "categories") {
      return true;
    }
    return false;
  }
  if (part[0] == "callables") return true;   // a dictionary, section 25
  if (part[0] != "supports") return false;
  if (part.size() == 3) {
    if (part[2] == "coordinates" || part[2] == "cell_types" ||
        part[2] == "cell_offsets" || part[2] == "cell_connectivity") {
      return true;
    }
    return is_scale && support_scale_name(part[2]);
  }
  if (part.size() == 4) {
    return part[2] == "node_arrays" || part[2] == "cell_arrays";
  }
  return false;
}

}  // namespace internal
}  // namespace mestra
