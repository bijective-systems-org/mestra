// mestra-cli: the command line over the library, and what the
// conformance driver of cpp/tests/run_corpus.py talks to.  It prints
// plain text so that a test script needs no JSON code on this side.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "mestra/mestra.hpp"

namespace {

int usage() {
  std::cout <<
      "mestra-cli COMMAND ARGUMENTS\n"
      "\n"
      "  read FILE                        check, then read the whole file and say what\n"
      "                                   came back\n"
      "  validate [--ids] [--metadata] FILE\n"
      "                                   one finding per line, as\n"
      "                                   \"<id> <path>: <message>\", then\n"
      "                                   \"<n> error(s), <m> warning(s)\";\n"
      "                                   --ids prints the identifiers alone,\n"
      "                                   --metadata decides from what a\n"
      "                                   metadata open reads and no slot\n"
      "  info FILE                        what the file holds, in the fields\n"
      "                                   docs/api-conventions.md section 5\n"
      "                                   lists; it refuses a file that breaks\n"
      "                                   a structural rule, as `read` does, and\n"
      "                                   shows one with a semantic fault; it\n"
      "                                   reads attributes and dataspaces only,\n"
      "                                   so it opens a large file as fast as a\n"
      "                                   small one\n"
      "  integrate FILE SLOT [WEIGHT]     one slot integrated over its support;\n"
      "                                   it says which weight array it used and\n"
      "                                   whether it had to compute one\n"
      "  stats FILE SLOT [BY]             one slot summarised, grouped by a\n"
      "                                   label; the grouping column is headed\n"
      "                                   with the label's own name\n"
      "  probe FILE SLOT ROW NODE COMPONENT [DRAW]\n"
      "                                   one stored value\n"
      "  support-id FILE SUPPORT          the digest of section 24\n"
      "  roundtrip IN OUT                 read IN and write OUT\n"
      "  evaluate FILE KEYS.csv OUT       evaluate every callable\n"
      "  dict-dump FILE CALLABLE          a callable's dictionary\n"
      "  dict-roundtrip FILE CALLABLE OUT write the dictionary back\n"
      "  rows FILE SLOT BEGIN END         a lazy read of a row range\n"
      "  cost FILE [SLOT]                 seconds for a metadata open, a\n"
      "                                   validation and one lazy row read\n"
      "  callable-types                   the types this build knows\n"
      "\n"
      "A probe index may be `-` when the slot has no such axis, and "
      "any\n"
      "index may also be given as NAME=VALUE, with NAME one of row,\n"
      "instance, draw, node, cell, cell_plus_one, component or index.\n"
      "A float64 slot\n"
      "prints its value in the C format \"%.17e\" and an integer slot\n"
      "in plain decimal, which is what section 30 asks a probe for.\n"
      "\n"
      "A command exits 1 when the file is rejected and 0 otherwise, so a\n"
      "warning alone still exits 0.  A fault that no rule of section 14\n"
      "covers carries no identifier and is printed as \"! <path>: <why>\",\n"
      "so a file is never reported clean because the thing wrong with it\n"
      "has no name.\n";
  return 2;
}

// The axes a probe may name, in stored order.
struct ProbeIndex {
  bool has_row = false;
  bool has_instance = false;
  bool has_draw = false;
  bool has_node = false;
  bool has_component = false;
  bool has_cell_plus_one = false;
  bool has_index = false;
  std::size_t row = 0, instance = 0, draw = 0, node = 0, component = 0,
              cell_plus_one = 0, index = 0;
};

bool parse_index(const std::string& text, std::size_t* out) {
  if (text == "-" || text.empty()) return false;
  *out = static_cast<std::size_t>(std::strtoull(text.c_str(), nullptr, 10));
  return true;
}

// Builds the subscript list in stored order, which is
// (row | instance, [draw], node | cell, component) with `index` the
// flat connectivity axis (section 30).
std::vector<std::size_t> subscripts(const ProbeIndex& p) {
  std::vector<std::size_t> out;
  if (p.has_row) out.push_back(p.row);
  if (p.has_instance) out.push_back(p.instance);
  if (p.has_draw) out.push_back(p.draw);
  if (p.has_node) out.push_back(p.node);
  if (p.has_component) out.push_back(p.component);
  if (p.has_cell_plus_one) out.push_back(p.cell_plus_one);
  if (p.has_index) out.push_back(p.index);
  return out;
}

// One finding per line, as "<id> <path>: <message>", which is the
// form docs/api-conventions.md section 5 fixes for every language's
// tool.  A failure no rule of section 14 covers carries no identifier
// and is printed as "! ", so that a file is never reported clean
// because the thing wrong with it has no name.
void print_finding(const mestra::Finding& f) {
  std::cout << (f.id.empty() ? std::string("!") : f.id) << " " << f.where
            << ": " << f.message << "\n";
}

void print_report(const mestra::Report& r) {
  for (const mestra::Finding& f : r.errors) print_finding(f);
  for (const mestra::Finding& f : r.warnings) print_finding(f);
  std::cout << r.errors.size() << " error(s), " << r.warnings.size()
            << " warning(s)\n";
}

// The identifiers alone, which is what a script wants and what this
// tool printed before there was a human form.
void print_ids(const mestra::Report& r) {
  for (const std::string& id : r.error_ids()) std::cout << "E " << id << "\n";
  for (const std::string& id : r.warning_ids()) {
    std::cout << "W " << id << "\n";
  }
  for (const mestra::Finding& f : r.errors) {
    if (f.id.empty()) {
      std::cout << "! " << f.where << ": " << f.message << "\n";
    }
  }
}

// `metadata_only` is the pass a metadata open makes (conventions
// section 7): it reads attributes, dataspaces, link types and
// dimension-scale structure, and of datasets only a category table
// and /row_support.  It answers the question "would `info` and
// `read_header` refuse this file, and with which identifiers", which
// a caller cannot otherwise ask without reading the whole file.
int cmd_validate(const std::string& path, bool ids_only,
                 bool metadata_only) {
  const mestra::Report r = metadata_only ? mestra::validate_metadata(path)
                                         : mestra::validate(path);
  if (ids_only) {
    print_ids(r);
  } else {
    print_report(r);
  }
  // Exit 1 when the file is rejected, so that a shell can tell.
  return r.ok() ? 0 : 1;
}

// Section 30, the hostile subset: opening a file for its metadata
// alone, and any operation that reads a slot, must refuse a file the
// validator rejects with the same identifiers rather than return
// something.  So both check before they read.  The library's own
// `read_header` still reads no dataset; this is the tool's policy and
// it is what a caller of a command-line tool should want.
bool refused(const std::string& path) {
  const mestra::Report r = mestra::validate(path);
  if (r.ok()) return false;
  print_report(r);
  return true;
}

// Reads everything, which is what `info` deliberately does not do.
int cmd_read(const std::string& path) {
  if (refused(path)) return 1;
  const mestra::Dataset d = mestra::read(path);
  std::size_t arrays = 0;
  std::size_t values = 0;
  for (const mestra::Support& s : d.supports) {
    if (s.coordinates.has_value()) {
      ++arrays;
      values += s.coordinates->data.f64.size();
    }
    for (const mestra::ArraySlot& a : s.node_arrays) {
      ++arrays;
      values += a.data.f64.size() + a.data.i64.size();
    }
    for (const mestra::ArraySlot& a : s.cell_arrays) {
      ++arrays;
      values += a.data.f64.size() + a.data.i64.size();
    }
  }
  std::cout << "rows " << d.n_rows << "\n"
            << "keys " << d.keys.size() << "\n"
            << "scalars " << d.scalars.size() << "\n"
            << "categories " << d.categories.size() << "\n"
            << "supports " << d.supports.size() << "\n"
            << "callables " << d.callables.size() << "\n"
            << "arrays " << arrays << "\n"
            << "values " << values << "\n";
  return 0;
}

// A bound or a quantile as a reader would write it: the shortest
// decimal that reads back as the same float64, so that 0.1 prints as
// 0.1 and a bound a reader is checking W08 against is not quietly
// rounded to six digits.
std::string number_text(double v) {
  char buffer[64];
  for (int precision = 15; precision <= 17; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*g", precision, v);
    if (std::strtod(buffer, nullptr) == v) break;
  }
  return std::string(buffer);
}

// A slot's shape with its axes named: "(row, node, component) 6x8x1".
// A first-time reader of a file they did not write wants to know how
// big it is, and the dimension names are what makes the extents mean
// anything (section 4).  A callable slot stores nothing and has no
// shape to print.
std::string shape_text(const mestra::Array& a) {
  if (a.dims.empty() && a.shape.empty()) return std::string();
  std::string out = "(";
  for (std::size_t i = 0; i < a.dims.size(); ++i) {
    if (i != 0) out += ", ";
    out += a.dims[i].empty() ? "?" : a.dims[i];
  }
  out += ") ";
  for (std::size_t i = 0; i < a.shape.size(); ++i) {
    if (i != 0) out += "x";
    out += std::to_string(a.shape[i]);
  }
  return out;
}

void print_slot(const char* kind, const mestra::ArraySlot& a) {
  std::cout << "  " << kind;
  if (a.name != kind) std::cout << " " << a.name;
  const std::string shape = shape_text(a.data);
  if (!shape.empty()) std::cout << " " << shape;
  std::cout << " role=" << a.role << " varies=" << a.varies
            << " components=" << a.components;
  if (a.units.has_value()) std::cout << " units=" << *a.units;
  if (a.category.has_value()) std::cout << " category=" << *a.category;
  if (a.recomputed.has_value()) {
    std::cout << " recomputed=" << (*a.recomputed ? "true" : "false");
  }
  std::cout << " source=" << a.source;
  if (a.is_callable()) {
    std::cout << " callable=" << a.callable_id();
    if (a.output.has_value()) std::cout << " output=" << *a.output;
  }
  if (a.statistic.has_value()) std::cout << " statistic=" << *a.statistic;
  if (a.of.has_value()) std::cout << " of=" << *a.of;
  if (a.quantile.has_value()) {
    std::cout << " quantile=" << number_text(*a.quantile);
  }
  if (a.level.has_value()) std::cout << " level=" << number_text(*a.level);
  if (a.method.has_value()) std::cout << " method=\"" << *a.method << "\"";
  if (a.derived_from.has_value()) {
    std::cout << " derived_from=" << *a.derived_from;
  }
  if (a.recipe.has_value()) std::cout << " recipe=" << *a.recipe;
  std::cout << "\n";
}

// `info` is a metadata open, and it refuses on the structural rules
// alone, as a read does (conventions section 2): a missing unit or a
// bad split is what a user runs `info` to find, and refusing the file
// for it would hide the survey behind the finding.  What it refuses
// on is decided from what a metadata open reads (section 7), so the
// same file is refused here and by `read_header`.
bool refused_for_info(const std::string& path) {
  const mestra::Report r = mestra::validate_metadata(path);
  std::size_t structural = 0;
  for (const mestra::Finding& f : r.errors) {
    if (!mestra::structural_rule(f.id)) continue;
    print_finding(f);
    ++structural;
  }
  if (structural == 0) return false;
  std::cout << structural << " error(s), 0 warning(s)\n";
  return true;
}

int cmd_info(const std::string& path) {
  if (refused_for_info(path)) return 1;
  const mestra::Dataset d = mestra::read_header(path);
  std::cout << "format " << d.format << "\n";
  std::cout << "writer " << d.writer << "\n";
  std::cout << "created " << d.created << "\n";
  std::cout << "aligned " << (d.aligned ? "true" : "false") << "\n";
  std::cout << "rows " << d.n_rows << "\n";
  if (d.generalisation_group.has_value()) {
    std::cout << "generalisation_group " << *d.generalisation_group << "\n";
  }
  for (const mestra::Key& k : d.keys) {
    std::cout << "key " << k.name << " role=" << k.role;
    if (k.units.has_value()) std::cout << " units=" << *k.units;
    if (k.lower.has_value()) {
      std::cout << " lower=" << number_text(*k.lower);
    }
    if (k.upper.has_value()) {
      std::cout << " upper=" << number_text(*k.upper);
    }
    if (k.category.has_value()) std::cout << " category=" << *k.category;
    // The attribute that makes a file a set of trajectories rather
    // than a pile of rows, and the one that says a group nests.
    if (k.trajectory_group.has_value()) {
      std::cout << " trajectory_group=" << *k.trajectory_group;
    }
    if (k.parent.has_value()) std::cout << " parent=" << *k.parent;
    std::cout << "\n";
  }
  for (const mestra::Scalar& s : d.scalars) {
    std::cout << "scalar " << s.name;
    if (!s.is_callable()) std::cout << " (row) " << d.n_rows;
    std::cout << " units=" << s.units << " source=" << s.source;
    if (s.is_callable()) {
      std::cout << " callable=" << s.callable_id();
      if (s.output.has_value()) std::cout << " output=" << *s.output;
    }
    if (s.statistic.has_value()) std::cout << " statistic=" << *s.statistic;
    if (s.of.has_value()) std::cout << " of=" << *s.of;
    if (s.quantile.has_value()) {
      std::cout << " quantile=" << number_text(*s.quantile);
    }
    if (s.level.has_value()) std::cout << " level=" << number_text(*s.level);
    if (s.method.has_value()) {
      std::cout << " method=\"" << *s.method << "\"";
    }
    std::cout << "\n";
  }
  for (const mestra::CategoryTable& t : d.categories) {
    // `info` opens the file without reading a dataset (section 29), so
    // the table's entries are not among what it can print.
    std::cout << "categories " << t.name << "\n";
  }
  for (const mestra::Support& s : d.supports) {
    std::cout << "support " << s.name << " kind=" << s.kind
              << " n_nodes=" << s.n_nodes << " n_cells=" << s.n_cells
              << " support_id=" << s.support_id << "\n";
    if (s.coordinates.has_value()) print_slot("coordinates", *s.coordinates);
    for (const mestra::ArraySlot& a : s.node_arrays) {
      print_slot("node_array", a);
    }
    for (const mestra::ArraySlot& a : s.cell_arrays) {
      print_slot("cell_array", a);
    }
  }
  for (const mestra::StoredCallable& c : d.callables) {
    std::cout << "callable " << c.id << " type=" << c.type;
    if (c.repr.has_value()) std::cout << " repr=" << *c.repr;
    std::cout << "\n";
  }
  return 0;
}

std::string value_text(const mestra::Array& a,
                       const std::vector<std::size_t>& at) {
  char buffer[64];
  if (a.dtype == mestra::DType::Float64) {
    const double v = a.at_f64(at);
    // Section 30: the three non-finite values are spelled "nan",
    // "inf" and "-inf".
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::snprintf(buffer, sizeof(buffer), "%.17e", v);
    return std::string(buffer);
  }
  std::snprintf(buffer, sizeof(buffer), "%lld",
                static_cast<long long>(a.at_i64(at)));
  return std::string(buffer);
}

int cmd_probe(int argc, char** argv) {
  // probe FILE SLOT ROW NODE COMPONENT [DRAW], or any number of
  // NAME=VALUE arguments after SLOT.
  if (argc < 4) return usage();
  const std::string path = argv[2];
  const std::string slot = argv[3];
  ProbeIndex p;
  bool keyword = false;
  for (int i = 4; i < argc; ++i) {
    if (std::strchr(argv[i], '=') != nullptr) keyword = true;
  }
  if (keyword) {
    for (int i = 4; i < argc; ++i) {
      const std::string arg = argv[i];
      const std::size_t at = arg.find('=');
      const std::string name = arg.substr(0, at);
      const std::string text = arg.substr(at + 1);
      std::size_t value = 0;
      if (!parse_index(text, &value)) continue;
      if (name == "row") {
        p.has_row = true;
        p.row = value;
      } else if (name == "instance") {
        p.has_instance = true;
        p.instance = value;
      } else if (name == "draw") {
        p.has_draw = true;
        p.draw = value;
      } else if (name == "node" || name == "cell") {
        p.has_node = true;
        p.node = value;
      } else if (name == "component") {
        p.has_component = true;
        p.component = value;
      } else if (name == "cell_plus_one") {
        p.has_cell_plus_one = true;
        p.cell_plus_one = value;
      } else if (name == "index") {
        p.has_index = true;
        p.index = value;
      } else {
        std::cerr << "mestra-cli: unknown probe axis \"" << name << "\"\n";
        return 2;
      }
    }
  } else {
    if (argc < 7) return usage();
    p.has_row = parse_index(argv[4], &p.row);
    p.has_node = parse_index(argv[5], &p.node);
    p.has_component = parse_index(argv[6], &p.component);
    if (argc > 7) p.has_draw = parse_index(argv[7], &p.draw);
  }
  const mestra::Array a = mestra::read_slot(path, slot);
  std::cout << value_text(a, subscripts(p)) << "\n";
  return 0;
}

// What this file costs to work with: the metadata open of section 29,
// a validation, and one lazy row read of one slot.  Each is the
// library call alone and in one process, because `info` validates
// before it prints (see `refused`) and a shell loop over three
// commands would time three interpreter starts as well.  The scale
// study needed a program of its own to ask this; it belongs in the
// tool, so that a change which makes an open cost the square of the
// dataset count is caught where the corpus is run.
int cmd_cost(const std::string& path, const std::string& slot) {
  using Clock = std::chrono::steady_clock;
  auto since = [](Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  const Clock::time_point t_open = Clock::now();
  const mestra::Dataset d = mestra::read_header(path);
  const double open = since(t_open);

  const Clock::time_point t_validate = Clock::now();
  const mestra::Report r = mestra::validate(path);
  const double validate = since(t_validate);

  double rows = -1.0;
  if (!slot.empty() && d.n_rows > 0) {
    const Clock::time_point t_rows = Clock::now();
    const mestra::Array a = mestra::read_slot_rows(path, slot, 0, 1);
    rows = since(t_rows);
    if (a.shape.empty()) rows = -1.0;
  }
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "open %.3f\nvalidate %.3f\n", open,
                validate);
  std::cout << buffer;
  if (rows < 0.0) {
    std::cout << "rows -\n";
  } else {
    std::snprintf(buffer, sizeof(buffer), "rows %.3f\n", rows);
    std::cout << buffer;
  }
  std::cout << "findings " << r.errors.size() << " " << r.warnings.size()
            << "\n";
  return 0;
}

int cmd_rows(const std::string& path, const std::string& slot,
             const std::string& begin, const std::string& end) {
  const mestra::Array a = mestra::read_slot_rows(
      path, slot, static_cast<std::size_t>(std::atoll(begin.c_str())),
      static_cast<std::size_t>(std::atoll(end.c_str())));
  for (std::size_t i = 0; i < a.dims.size(); ++i) {
    std::cout << (i == 0 ? "dims " : " ") << a.dims[i];
  }
  std::cout << "\nshape";
  for (const std::size_t e : a.shape) std::cout << " " << e;
  std::cout << "\n";
  char buffer[64];
  if (a.dtype == mestra::DType::Float64) {
    for (const double v : a.f64) {
      std::snprintf(buffer, sizeof(buffer), "%.17e", v);
      std::cout << buffer << "\n";
    }
  } else {
    for (const std::int64_t v : a.i64) std::cout << v << "\n";
  }
  return 0;
}

int cmd_evaluate(const std::string& path, const std::string& csv,
                 const std::string& out) {
  const mestra::Dataset d = mestra::read(path);
  const mestra::KeysTable keys = mestra::read_keys_csv(d, csv);
  const mestra::Dataset evaluated = mestra::evaluate(d, keys);
  mestra::write(evaluated, out);
  // Every other subcommand says something; this one used to succeed
  // in silence.
  std::cout << "wrote " << out << ": " << evaluated.n_rows << " row(s), "
            << evaluated.callables.size() << " callable(s) evaluated\n";
  return 0;
}

int cmd_integrate(const std::string& path, const std::string& slot,
                  const std::string& weight) {
  const mestra::Dataset d = mestra::read(path);
  mestra::IntegrateOptions options;
  options.weight = weight;
  const mestra::Integral r = mestra::integrate(d, slot, options);
  std::cout << "weight " << r.weight
            << (r.weight_recomputed ? " (computed here: the file carries "
                                      "none)"
                                    : " (from the file)")
            << "\n";
  if (!r.units.empty()) std::cout << "units " << r.units << "\n";
  for (std::size_t i = 0; i < r.dims.size(); ++i) {
    std::cout << (i == 0 ? "dims " : " ") << r.dims[i];
  }
  std::cout << "\nshape";
  for (const std::size_t e : r.shape) std::cout << " " << e;
  std::cout << "\n";
  char buffer[64];
  for (const double v : r.values) {
    std::snprintf(buffer, sizeof(buffer), "%.17e", v);
    std::cout << buffer << "\n";
  }
  return 0;
}

int cmd_stats(const std::string& path, const std::string& slot,
              const std::string& by) {
  const mestra::Dataset d = mestra::read(path);
  const mestra::FieldStatistics r = mestra::field_statistics(d, slot, by);
  // The grouping column is named after the label and never by a fixed
  // word (conventions section 4), and there is no such column at all
  // when nothing was grouped by.
  if (!r.group_by.empty()) std::cout << r.group_by << " ";
  std::cout << "count missing minimum mean maximum deviation";
  if (!r.units.empty()) std::cout << "   units " << r.units;
  std::cout << "\n";
  char buffer[64];
  for (std::size_t i = 0; i < r.size(); ++i) {
    if (!r.group_by.empty()) std::cout << r.groups[i] << " ";
    std::cout << r.count[i] << " " << r.missing[i];
    const double values[4] = {r.minimum[i], r.mean[i], r.maximum[i],
                              r.deviation[i]};
    for (const double v : values) {
      std::snprintf(buffer, sizeof(buffer), " %.17e", v);
      std::cout << buffer;
    }
    std::cout << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string command = argv[1];
  try {
    if (command == "validate" && argc >= 3) {
      bool ids_only = false;
      bool metadata_only = false;
      int at = 2;
      for (; at < argc; ++at) {
        const std::string option = argv[at];
        if (option == "--ids") {
          ids_only = true;
        } else if (option == "--metadata") {
          metadata_only = true;
        } else {
          break;
        }
      }
      if (at + 1 == argc) {
        return cmd_validate(argv[at], ids_only, metadata_only);
      }
      return usage();
    }
    if (command == "info" && argc == 3) return cmd_info(argv[2]);
    if (command == "integrate" && (argc == 4 || argc == 5)) {
      return cmd_integrate(argv[2], argv[3], argc == 5 ? argv[4] : "");
    }
    if (command == "stats" && (argc == 4 || argc == 5)) {
      return cmd_stats(argv[2], argv[3], argc == 5 ? argv[4] : "");
    }
    if (command == "read" && argc == 3) return cmd_read(argv[2]);
    if (command == "probe") return cmd_probe(argc, argv);
    if (command == "support-id" && argc == 4) {
      std::cout << mestra::support_id_of(argv[2], argv[3]) << "\n";
      return 0;
    }
    if (command == "roundtrip" && argc == 4) {
      mestra::write(mestra::read(argv[2]), argv[3]);
      return 0;
    }
    if (command == "evaluate" && argc == 5) {
      return cmd_evaluate(argv[2], argv[3], argv[4]);
    }
    if (command == "dict-dump" && argc == 4) {
      std::cout << mestra::dump_dict(mestra::read_dict(argv[2], argv[3]));
      return 0;
    }
    if (command == "dict-roundtrip" && argc == 5) {
      const mestra::Dict d = mestra::read_dict(argv[2], argv[3]);
      const mestra::Dataset source = mestra::read_header(argv[2]);
      const mestra::StoredCallable* stored = source.callable(argv[3]);
      mestra::write_dict(d, stored != nullptr ? stored->type : std::string(),
                         argv[3], argv[4]);
      return 0;
    }
    if (command == "rows" && argc == 6) {
      return cmd_rows(argv[2], argv[3], argv[4], argv[5]);
    }
    if (command == "cost" && (argc == 3 || argc == 4)) {
      return cmd_cost(argv[2], argc == 4 ? argv[3] : "");
    }
    if (command == "callable-types") {
      mestra::Affine::register_type();
      for (const std::string& t : mestra::CallableRegistry::types()) {
        std::cout << t << "\n";
      }
      return 0;
    }
  } catch (const mestra::Error& e) {
    // Section 6: the rule identifier first, then the object path, then
    // what to do.  Error puts the identifier in front of its own
    // message already, so it is not repeated here.
    std::cerr << "mestra-cli: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "mestra-cli: " << e.what() << "\n";
    return 1;
  }
  return usage();
}
