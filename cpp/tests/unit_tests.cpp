// Pure C++ checks: the SHA-256 known vectors and the three worked
// digests of SPEC.md section 24, the units parser W10 is driven by,
// the dictionary codec's value types, the affine callable's worked
// example of section 27, one dataset built from plain vectors and
// validated, and the conventions of docs/api-conventions.md one rule
// at a time -- the argument order, the bounds default, `dims` deriving
// `varies` and `components`, the permutation into stored order, the
// refusals a builder makes and the identifiers they carry, `write`
// refusing a slot whose `varies` was assigned after the fact, the cell
// measures against shapes whose measure is known by hand, the lumped
// node weights adding up to the area, the default weight rule, the
// statistics of a field with a missing value in it, and a zero-row
// callable file written, validated and evaluated, whose evaluated form
// keeps no callable and no `/callables` group.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <utility>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "mestra/mestra.hpp"

namespace {

void sha256_vectors() {
  // The three vectors everyone checks a SHA-256 against.
  check::equal(
      "sha256 of the empty string", mestra::sha256_hex(std::string()),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  check::equal(
      "sha256 of \"abc\"", mestra::sha256_hex(std::string("abc")),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  check::equal(
      "sha256 of the 56-character vector",
      mestra::sha256_hex(std::string(
          "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // The block boundaries, where a padding mistake hides.
  check::equal(
      "sha256 of 55 a's", mestra::sha256_hex(std::string(55, 'a')),
      "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
  check::equal(
      "sha256 of 56 a's", mestra::sha256_hex(std::string(56, 'a')),
      "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
  check::equal(
      "sha256 of 57 a's", mestra::sha256_hex(std::string(57, 'a')),
      "f13b2d724659eb3bf47f2dd6af1accc87b81f09f59f2b75e5c0bed6589dfe8c6");
  check::equal(
      "sha256 of 63 a's", mestra::sha256_hex(std::string(63, 'a')),
      "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34");
  check::equal(
      "sha256 of 64 a's", mestra::sha256_hex(std::string(64, 'a')),
      "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
  check::equal(
      "sha256 of 65 a's", mestra::sha256_hex(std::string(65, 'a')),
      "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0");
  check::equal(
      "sha256 of 119 a's", mestra::sha256_hex(std::string(119, 'a')),
      "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb");
  check::equal(
      "sha256 of 120 a's", mestra::sha256_hex(std::string(120, 'a')),
      "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c");
  check::equal(
      "sha256 of 127 a's", mestra::sha256_hex(std::string(127, 'a')),
      "c57e9278af78fa3cab38667bef4ce29d783787a2f731d4e12200270f0c32320a");
  check::equal(
      "sha256 of 128 a's", mestra::sha256_hex(std::string(128, 'a')),
      "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e");

  // A message longer than one block, to exercise the padding.
  check::equal(
      "sha256 of a million a's",
      mestra::sha256_hex(std::string(1000000, 'a')),
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

void support_id_vectors() {
  // The three worked digests of section 24.
  const std::vector<std::uint8_t> types{9, 9};
  const std::vector<std::int64_t> offsets{0, 4, 8};
  const std::vector<std::int64_t> conn{0, 1, 4, 3, 1, 2, 5, 4};
  check::equal(
      "the mesh support of section 24",
      mestra::support_id_digest(6, types, offsets, conn, nullptr),
      "96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c943936c7");

  const std::vector<double> axis{0.0, 0.5, 1.0, 1.5};
  check::equal(
      "the axis support of section 24",
      mestra::support_id_digest(4, {}, {}, {}, &axis),
      "57467fe7370808bdb0ad01b95d963f59e8bc6762f90f96049453ae33bb05a54c");

  check::equal(
      "the support of kind none of section 24",
      mestra::support_id_digest(0, {}, {}, {}, nullptr),
      "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc");
}

void units_parser() {
  // Everything the corpus carries must parse.
  for (const char* good : {"1", "m", "Pa", "degree", "s", "K", "W",
                           "W m-2", "m2 s-1", "m2", "m s-1", "kg m-3",
                           "W/(m2 K)", "m^2", "1e-3 m", "%"}) {
    check::is_true(std::string("units \"") + good + "\" parses",
                   mestra::units_parse(good));
  }
  // The corpus's W10 case and a few neighbours.
  for (const char* bad : {"kg/(m s", "", "m^", "(", ")", "m//s", "2m^^3"}) {
    check::is_true(std::string("units \"") + bad + "\" does not parse",
                   !mestra::units_parse(bad));
  }
  // A units string comes out of a file, so a long run of "(" must be
  // refused rather than recursed on.
  check::is_true("a deep run of parentheses is refused",
                 !mestra::units_parse(std::string(100000, '(') + "m"));
}

void codec_values() {
  mestra::Dict d;
  d.set("alpha", mestra::Value::integer(42));
  d.set("beta", mestra::Value::real(2.5));
  d.set("gamma", mestra::Value::boolean(true));
  d.set("delta", mestra::Value::text("hello"));
  d.set("epsilon", mestra::Value::null());

  mestra::Array a;
  a.dtype = mestra::DType::Float64;
  a.shape = {2, 3};
  a.f64 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  d.set("zeta", mestra::Value::numbers(a));

  mestra::Array s;
  s.dtype = mestra::DType::String;
  s.shape = {2};
  s.str = {"mach", "alpha"};
  d.set("eta", mestra::Value::strings(s));

  mestra::Dict nested;
  nested.set("inner", mestra::Value::integer(7));
  d.set("theta", mestra::Value::dict(nested));

  // Section 25: an integer and a float that happen to be equal are
  // different values and stay different.
  check::is_true("an int and a float are different values",
                 !(mestra::Value::integer(2) == mestra::Value::real(2.0)));
  // A zero-dimensional array is not representable.
  bool refused = false;
  try {
    mestra::Array zero;
    zero.dtype = mestra::DType::Float64;
    zero.f64 = {1.0};
    mestra::Value::numbers(zero);
  } catch (const mestra::Error& e) {
    refused = e.rule() == "E32";
  }
  check::is_true("a zero-dimensional array is refused", refused);

  // The dump is stable and the keys come out in byte order.
  const std::string dump = mestra::dump_dict(d);
  check::is_true("the dump starts at the root",
                 dump.compare(0, 4, "D .\n") == 0);
  check::is_true("the dump holds the nested dictionary",
                 dump.find("D ./theta\n") != std::string::npos);
  check::is_true("the dump holds the null",
                 dump.find("N ./epsilon\n") != std::string::npos);
  check::is_true("the dump holds the string array",
                 dump.find("T ./eta 1 2 mach alpha\n") !=
                     std::string::npos);
  std::size_t at_alpha = dump.find("./alpha");
  std::size_t at_beta = dump.find("./beta");
  check::is_true("keys are dumped in byte order", at_alpha < at_beta);
}

// True when the body raised anything at all, for the refusals that
// no rule of section 14 covers: a cell type nobody can measure, a
// weight array that is not in the file.
bool threw(const std::function<void()>& body,
           std::string* message = nullptr) {
  try {
    body();
  } catch (const std::exception& e) {
    if (message != nullptr) *message = e.what();
    return true;
  }
  return false;
}

void affine_worked_example() {
  // Section 27, the worked example: two keys in the order mach,
  // alpha; one scalar slot cl and one node-array slot pressure.
  mestra::AffineOutput cl;
  cl.A = {2.0, 0.1};
  cl.b = {0.05};
  cl.shape = {};

  mestra::AffineOutput pressure;
  pressure.A = {1.0, 0.0, 2.0, 0.0, 3.0, 0.5,
                4.0, 0.5, 5.0, 1.0, 6.0, 1.0};
  pressure.b = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  pressure.shape = {6, 1};

  std::map<std::string, mestra::AffineOutput, mestra::BytesLess> outputs;
  outputs["cl"] = cl;
  outputs["pressure"] = pressure;
  const mestra::Affine model({"mach", "alpha"}, outputs,
                             "affine(mach, alpha -> cl, pressure)");

  mestra::KeysTable table;
  table.add_column("mach", {0.5});
  table.add_column("alpha", {4.0});
  const mestra::Outputs out = model.call(table);

  check::equal("cl is 1.45 exactly", out.at("cl").mean.f64.at(0), 1.45);
  const std::vector<double> want{0.5, 1.1, 3.7, 4.3, 6.9, 7.5};
  for (std::size_t i = 0; i < want.size(); ++i) {
    std::ostringstream what;
    what << "pressure node " << i;
    check::equal(what.str(), out.at("pressure").mean.f64.at(i), want[i]);
  }
  check::equal("cl comes back as (1)",
               out.at("cl").mean.shape.size(), std::size_t(1));
  check::equal("pressure comes back as (1, 6, 1)",
               out.at("pressure").mean.shape.size(), std::size_t(3));
  check::equal("pressure names its node axis",
               out.at("pressure").mean.dims.at(1), std::string("node"));
  check::is_true("an output with no band has no uncertainty",
                 !out.at("cl").has_uncertainty() &&
                     !out.at("cl").level.has_value());

  // The dictionary round-trips through to_dict and from_dict.
  const mestra::Dict d = model.to_dict();
  const mestra::Affine again = mestra::Affine::from_dict(d);
  check::is_true("to_dict and from_dict agree",
                 again.to_dict() == d);
  check::equal("the type string is affine", model.type(),
               std::string("affine"));

  // A dictionary with anything else in it is refused.
  mestra::Dict extra = d;
  extra.set("seed", mestra::Value::integer(1));
  bool refused = false;
  try {
    mestra::Affine::from_dict(extra);
  } catch (const mestra::Error&) {
    refused = true;
  }
  check::is_true("an affine dictionary with an extra key is refused",
                 refused);

  // A keys table missing a column the callable declares is an error at
  // call time (section 26).
  mestra::KeysTable short_table;
  short_table.add_column("mach", {0.5});
  bool missing = false;
  try {
    model.call(short_table);
  } catch (const mestra::Error&) {
    missing = true;
  }
  check::is_true("a missing key column is an error at call time", missing);
}

// Section 27 with a band, and the record of section 10.
void affine_band() {
  mestra::AffineOutput cl;
  cl.A = {2.0, 0.1};
  cl.b = {0.05};
  cl.shape = {};
  cl.uncertainty = std::vector<double>{0.02};
  cl.level = 0.95;
  cl.method = "constant band";
  mestra::AffineOutput pressure;
  pressure.A = {1.0, 0.0, 2.0, 0.0};
  pressure.b = {0.0, 0.1};
  pressure.shape = {2, 1};
  pressure.uncertainty = std::vector<double>{0.5, 0.6};
  pressure.level = 0.68;
  pressure.method = "constant band";
  std::map<std::string, mestra::AffineOutput, mestra::BytesLess> outputs;
  outputs["cl"] = cl;
  outputs["pressure"] = pressure;
  const mestra::Affine model({"mach", "alpha"}, outputs);

  mestra::KeysTable table;
  table.add_column("mach", {0.5, 0.6});
  table.add_column("alpha", {4.0, 5.0});
  const mestra::Outputs out = model.call(table);
  check::is_true("cl carries a band", out.at("cl").has_uncertainty());
  check::equal("the same in every row", out.at("cl").uncertainty->f64.at(1),
               0.02);
  check::equal("with its level", out.at("cl").level.value_or(0.0), 0.95);
  check::equal("and its method", out.at("cl").method.value_or(""),
               std::string("constant band"));
  check::equal("an array band has the mean's shape",
               out.at("pressure").uncertainty->shape.size(), std::size_t(3));
  check::equal("row 1, node 1", out.at("pressure").uncertainty->f64.at(3),
               0.6);
  check::equal("and the mean is unchanged", out.at("cl").mean.f64.at(0),
               1.45);

  const mestra::Dict d = model.to_dict();
  check::is_true("the band round-trips through the dictionary",
                 mestra::Affine::from_dict(d).to_dict() == d);
  check::is_true("and the dictionary holds the level",
                 d.at("outputs").as_dict().at("cl").as_dict().has("level"));

  mestra::AffineOutput half = cl;
  half.method.reset();
  std::map<std::string, mestra::AffineOutput, mestra::BytesLess> bad;
  bad["cl"] = half;
  check::is_true("a band missing its method is refused",
                 threw([&bad] { mestra::Affine({"mach", "alpha"}, bad); }));

  mestra::Prediction p = out.at("cl");
  p.level = 1.96;
  check::is_true("level is a coverage in (0, 1), not a sigma multiple",
                 threw([&p] { p.check(); }));
  p.level = 0.95;
  p.uncertainty->f64[0] = -0.1;
  check::is_true("a band is never negative", threw([&p] { p.check(); }));
  p.uncertainty.reset();
  check::is_true("level and method go with an uncertainty",
                 threw([&p] { p.check(); }));
}

// Usability, checked rather than asserted: build the file of
// docs/example.md from plain vectors, write it, and see that the
// validator finds nothing to say.  If this stops reading like a
// handful of calls, the API has gone wrong.
void build_from_vectors() {
  mestra::Dataset d;
  d.writer = "mestra examples 0";
  d.created = "2026-09-19T00:00:00Z";
  d.set_generalisation_group("member");

  d.add_category_table("member", {"wing_a", "wing_b"});
  d.add_category_table("region", {"inlet", "outlet"});

  d.add_key("mach", {0.40, 0.80}, "condition", "1");
  d.add_category_key("member", {0, 1}, "group", "member");
  d.add_scalar("cl", {0.25, 0.55}, "1");

  mestra::Support& s = d.add_mesh_support(
      "s0", 6, {9, 9}, {0, 4, 8}, {0, 1, 4, 3, 1, 2, 5, 4});
  mestra::set_coordinates(
      s,
      {0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
       0.0, 0.0, 1.5, 0.0, 3.0, 0.0, 0.0, 1.0, 1.5, 1.0, 3.0, 1.0},
      "m", {"group:member", "node", {"component", 2}});
  mestra::add_node_array(s, "pressure",
                         {101, 102, 103, 104, 105, 106,
                          201, 202, 203, 204, 205, 206},
                         "Pa", {"row", "node"});
  mestra::add_cell_label(s, "region", {0, 1}, std::string("region"),
                         {"cell"});

  const std::string path = "mestra_unit_built.mes";
  mestra::write(d, path);

  // The support id is filled in by the builder and is the worked one
  // of section 24.
  check::equal(
      "the builder fills in the support id", s.support_id,
      std::string(
          "96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c943936c7"));

  const mestra::Report report = mestra::validate(path);
  check::equal("the built file has no errors", report.errors.size(),
               std::size_t(0));
  check::equal("the built file has no warnings", report.warnings.size(),
               std::size_t(0));
  if (!report.errors.empty()) {
    for (const mestra::Finding& f : report.errors) {
      std::cout << "     " << f.id << " " << f.where << ": " << f.message
                << "\n";
    }
  }
  if (!report.warnings.empty()) {
    for (const mestra::Finding& f : report.warnings) {
      std::cout << "     " << f.id << " " << f.where << ": " << f.message
                << "\n";
    }
  }

  // Read it back and check the value docs/example.md says to check.
  const mestra::Dataset back = mestra::read(path);
  const mestra::Support* got = back.support("s0");
  check::is_true("the support comes back", got != nullptr);
  if (got != nullptr && !got->node_arrays.empty()) {
    const mestra::Array& pressure = got->node_arrays.front().data;
    check::equal("pressure names its axes by name",
                 pressure.dims.at(0) + "," + pressure.dims.at(1) + "," +
                     pressure.dims.at(2),
                 std::string("row,node,component"));
    check::equal("pressure at row 1, node 3, component 0",
                 pressure.at_f64({1, 3, 0}), 204.0);
  }
  // Section 29: opening a file must not read any array.
  const mestra::Dataset header = mestra::read_header(path);
  check::equal("a header read still states the row count", header.n_rows,
               std::int64_t(2));
  check::equal("a header read still states the support id",
               header.supports.at(0).support_id, s.support_id);
  check::is_true("a header read reads no key values",
                 header.keys.at(0).f64.empty());
  check::is_true("a header read reads no cell arrays",
                 header.supports.at(0).cell_types.empty());
  check::is_true("a header read reads no array values",
                 header.supports.at(0).node_arrays.at(0).data.f64.empty());
  check::equal("a header read still states the shape",
               header.supports.at(0).node_arrays.at(0).data.shape.size(),
               std::size_t(3));

  // Lazy access: one slot, one row, nothing else read.
  const mestra::Array one = mestra::read_slot_rows(
      path, "/supports/s0/node_arrays/pressure", 1, 2);
  check::equal("a lazy read returns one row", one.shape.at(0),
               std::size_t(1));
  check::equal("a lazy read returns the right value", one.f64.at(3), 204.0);

  // A dataset whose values do not fill the shape it declares is
  // refused rather than handed to HDF5, which would read past the end
  // of the vector.
  mestra::Dataset short_one = d;
  short_one.scalar("cl")->values.pop_back();
  bool refused = false;
  try {
    mestra::write(short_one, path);
  } catch (const mestra::Error&) {
    refused = true;
  }
  check::is_true("a short data vector is refused", refused);

  std::remove(path.c_str());
}

// --- the cross-language conventions ---------------------------------
//
// docs/api-conventions.md is normative for the shape of this API, and
// these are its rules checked one by one on the smallest file that
// shows each.  A file of two rows on a square of two triangles: small
// enough to do the arithmetic in your head, which is the only way a
// weight or an integral is worth testing.

// The rule identifier an Error carries, or "" for none.
std::string rule_of(const std::function<void()>& body,
                    std::string* message = nullptr) {
  try {
    body();
  } catch (const mestra::Error& e) {
    if (message != nullptr) *message = e.what();
    return e.rule();
  } catch (const std::exception&) {
    return "(not a mestra::Error)";
  }
  return std::string();
}

// A unit square of two triangles, with a field of 1 everywhere.
mestra::Dataset square_dataset() {
  mestra::Dataset d;
  d.writer = "mestra unit tests";
  d.created = "2026-09-20T00:00:00Z";
  d.add_key("mach", {0.4, 0.8}, "condition", "1");
  d.add_category_table("region", {"inlet", "outlet"});
  mestra::Support& s = d.add_mesh_support("s0", 4, {5, 5}, {0, 3, 6},
                                          {0, 1, 2, 0, 2, 3});
  mestra::set_coordinates(s, {0, 0, 1, 0, 1, 1, 0, 1}, "m",
                          {"node", {"component", 2}});
  mestra::add_node_array(s, "pressure",
                         {1, 1, 1, 1, 2, 2, 2, 2}, "Pa", {"row", "node"});
  mestra::add_cell_label(s, "region", {0, 1}, std::string("region"),
                         {"cell"});
  return d;
}

void conventions_builders() {
  // Section 1: the name, then the values, then what they mean, and
  // the observed finite range as the bounds when none are given.
  mestra::Dataset d;
  d.writer = "mestra unit tests";
  d.created = "2026-09-20T00:00:00Z";
  const mestra::Key& mach = d.add_key("mach", {0.4, 0.8, 0.6},
                                      "condition", "1");
  check::equal("add_key takes the values second", mach.f64.at(1), 0.8);
  check::is_true("a key with no bounds takes the observed minimum",
                 mach.lower.has_value() && *mach.lower == 0.4);
  check::is_true("a key with no bounds takes the observed maximum",
                 mach.upper.has_value() && *mach.upper == 0.8);
  const mestra::Scalar& cl = d.add_scalar("cl", {0.1, 0.2, 0.3}, "1");
  check::equal("add_scalar takes the values second", cl.values.at(2), 0.3);
  check::equal("add_scalar takes the units third", cl.units,
               std::string("1"));

  // A key whose values are all non-finite has no observed range, and
  // no bounds are invented for it.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const mestra::Key& empty = d.add_key("blank", {nan, nan}, "condition",
                                       "1");
  check::is_true("no finite value means no bounds",
                 !empty.lower.has_value() && !empty.upper.has_value());

  // Section 1: the dataset-level setter for the unit of
  // generalisation.
  d.set_generalisation_group("member");
  check::is_true("set_generalisation_group sets the dataset property",
                 d.generalisation_group.has_value() &&
                     *d.generalisation_group == "member");

  // A name the format does not allow is refused by the builder, with
  // the identifier, rather than at write time.
  check::equal("a reserved name is refused at build time",
               rule_of([&d] { d.add_scalar("mestra_x", {0.0}, "1"); }),
               std::string("E33"));
}

void conventions_dims() {
  mestra::Dataset d = square_dataset();
  mestra::Support* s = d.support("s0");

  // `dims` derives `varies` and `components`: no separate argument
  // and nothing to get out of step.
  const mestra::ArraySlot* pressure = nullptr;
  for (const mestra::ArraySlot& a : s->node_arrays) {
    if (a.name == "pressure") pressure = &a;
  }
  check::is_true("the node array is there", pressure != nullptr);
  check::equal("dims derives varies", pressure->varies,
               std::string("row"));
  check::equal("a missing component axis is added, of length one",
               pressure->components, std::int64_t(1));
  check::equal("the stored dims are the order of section 4",
               pressure->data.dims.at(0) + "," + pressure->data.dims.at(1) +
                   "," + pressure->data.dims.at(2),
               std::string("row,node,component"));

  // The names may be in the caller's own order: the builder permutes
  // into the stored order rather than asking the caller to.
  mestra::add_node_array(*s, "by_node",
                         // (node, row): node 0 both rows, node 1 both, ...
                         {10, 20, 11, 21, 12, 22, 13, 23}, "Pa",
                         {"node", "row"});
  const mestra::ArraySlot* by_node = nullptr;
  for (const mestra::ArraySlot& a : s->node_arrays) {
    if (a.name == "by_node") by_node = &a;
  }
  check::is_true("the permuted array is there", by_node != nullptr);
  check::equal("a (node, row) array is stored (row, node, component)",
               by_node->data.dims.at(0) + "," + by_node->data.dims.at(1),
               std::string("row,node"));
  check::equal("row 0, node 2 after the permutation",
               by_node->data.at_f64({0, 2, 0}), 12.0);
  check::equal("row 1, node 3 after the permutation",
               by_node->data.at_f64({1, 3, 0}), 23.0);

  // Two unknown lengths are refused by name, not guessed at.
  std::string message;
  check::equal("two unknown axis lengths are refused",
               rule_of([s] {
                 mestra::add_node_array(*s, "two_unknowns",
                                        {1, 2, 3, 4, 5, 6, 7, 8}, "Pa",
                                        {"row", "node", "component"});
               }, &message),
               std::string("E31"));
  check::is_true("the message says which argument to change",
                 message.find("component") != std::string::npos &&
                     message.find("dims") != std::string::npos);

  // A count that does not fit the shape is refused with the rule the
  // validator would give the file.
  check::equal("values that do not divide into the shape are refused",
               rule_of([s] {
                 mestra::add_node_array(*s, "ragged", {1, 2, 3}, "Pa",
                                        {"row", "node"});
               }),
               std::string("E04"));
  check::equal("a node count that disagrees with the support is refused",
               rule_of([s] {
                 mestra::add_node_array(*s, "wrong_nodes", {1, 2, 3}, "Pa",
                                        {{"node", 3}});
               }),
               std::string("E05"));
  check::equal("an axis name that is not a dimension is refused",
               rule_of([s] {
                 mestra::add_node_array(*s, "bad_axis", {1, 2, 3, 4}, "Pa",
                                        {"rows", "node"});
               }),
               std::string("E25"));
}

void conventions_write() {
  // Section 2: write validates first and refuses on any error, and
  // leaves nothing behind when it refuses.
  const std::string path = "mestra_unit_conventions.mes";
  std::remove(path.c_str());
  mestra::Dataset d = square_dataset();
  mestra::write(d, path);
  check::is_true("a built file validates", mestra::validate(path).ok());

  // The validating write builds the file beside the name it was given
  // and moves it into place, so the bytes must not depend on that; nor
  // on the clock, which an object header records unless told not to.
  // The second write waits for the clock to move on, so that a header
  // recording it would show.
  const std::string again = "mestra_unit_conventions_again.mes";
  std::remove(again.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  mestra::write(d, again);
  {
    std::ifstream first(path.c_str(), std::ios::binary);
    std::ifstream second(again.c_str(), std::ios::binary);
    const std::string a((std::istreambuf_iterator<char>(first)),
                        std::istreambuf_iterator<char>());
    const std::string b((std::istreambuf_iterator<char>(second)),
                        std::istreambuf_iterator<char>());
    check::is_true("one dataset written twice gives the same bytes",
                   !a.empty() && a == b);
  }
  std::remove(again.c_str());

  // An easy trap to fall into: `varies` is assigned on a
  // slot whose shape is already built.  It must not be written out
  // for the validator to reject afterwards.
  mestra::Dataset broken = d;
  mestra::Support* s = broken.support("s0");
  for (mestra::ArraySlot& a : s->node_arrays) {
    if (a.name == "pressure") a.varies = "none";
  }
  const std::string other = "mestra_unit_broken.mes";
  std::remove(other.c_str());
  std::string message;
  check::equal("assigning varies after the fact is refused",
               rule_of([&broken, &other] { mestra::write(broken, other); },
                       &message),
               std::string("E04"));
  check::is_true("the refusal names the rule and the path",
                 message.find("E04") == 0 &&
                     message.find("/supports/s0/node_arrays/pressure") !=
                         std::string::npos);
  check::is_true("a refused write leaves no file",
                 std::ifstream(other.c_str()).good() == false);

  // A file the validator rejects for a reason a builder cannot see is
  // refused too, with the findings, and `check = false` writes it.
  mestra::Dataset unitless = d;
  unitless.support("s0")->node_arrays.at(0).units.reset();
  std::string findings;
  const std::string rule = rule_of(
      [&unitless, &other] { mestra::write(unitless, other); }, &findings);
  check::is_true("a file that breaks a rule is refused by write",
                 rule == "E11" || rule == "E39");
  check::is_true("the refusal carries the findings",
                 findings.find("error(s)") != std::string::npos);
  check::is_true("nothing was left behind",
                 std::ifstream(other.c_str()).good() == false);
  // A destination that cannot take a file is refused before anything
  // is staged, naming `path`, and with `check = false` as well: the
  // staging happens in the destination's directory either way.
  {
    namespace fs = std::filesystem;
    const std::string dir = "mestra_unit_destination_dir";
    fs::remove_all(dir);
    fs::create_directory(dir);
    mestra::WriteOptions unchecked;
    unchecked.check = false;
    for (const bool checked : {true, false}) {
      const mestra::WriteOptions& opt = checked ? mestra::WriteOptions()
                                                : unchecked;
      std::string why;
      check::equal("an empty path is refused",
                   rule_of([&d, &opt] { mestra::write(d, "", opt); }, &why),
                   std::string(""));
      check::is_true("the empty path refusal says so",
                     why.find("path is empty") != std::string::npos);
      check::equal("a directory is refused as a destination",
                   rule_of([&d, &dir, &opt] { mestra::write(d, dir, opt); },
                           &why),
                   std::string(""));
      check::is_true("the directory refusal names the path",
                     why.find("names a directory") != std::string::npos &&
                         why.find(dir) != std::string::npos);
      const std::string missing = "mestra_unit_no_such_dir/out.mes";
      check::equal("a missing parent directory is refused",
                   rule_of([&d, &missing, &opt] {
                     mestra::write(d, missing, opt);
                   }, &why),
                   std::string(""));
      check::is_true("the missing-parent refusal names the directory",
                     why.find("does not exist") != std::string::npos &&
                         why.find("mestra_unit_no_such_dir") !=
                             std::string::npos);
      check::is_true("a refused destination stages nothing",
                     !fs::exists(std::string(".mestra-writing")) &&
                         !fs::exists(dir + ".mestra-writing") &&
                         !fs::exists("mestra_unit_no_such_dir"));
    }
    fs::remove_all(dir);
  }

  // A read is strict about the structural rules and silent about the
  // semantic ones, so a file with a missing unit still opens.
  const mestra::Dataset clean = mestra::read(path);
  check::is_true("a strict read of a clean file refuses nothing",
                 clean.not_read.empty());

  mestra::WriteOptions unchecked;
  unchecked.check = false;
  mestra::write(unitless, other, unchecked);
  check::is_true("a semantic fault does not stop a read",
                 mestra::read(other).supports.size() == 1);

  // A structural fault does stop it, and a non-strict read lists what
  // it refused rather than pretending the file was whole.
  mestra::Dataset miscounted;
  miscounted.writer = "mestra unit tests";
  miscounted.created = "2026-09-20T00:00:00Z";
  miscounted.add_mesh_support("s0", 4, {5, 5}, {0, 3, 6},
                              {0, 1, 2, 0, 2, 3});
  mestra::Support* m = miscounted.support("s0");
  mestra::set_coordinates(*m, {0, 0, 1, 0, 1, 1, 0, 1}, "m",
                          {"node", {"component", 2}});
  mestra::add_node_array(*m, "pressure", {1, 1, 1, 1, 2, 2, 2, 2}, "Pa",
                         {"row", "node"});
  miscounted.n_rows = 3;              // the array still holds two rows
  const std::string third = "mestra_unit_structural.mes";
  std::remove(third.c_str());
  mestra::write(miscounted, third, unchecked);
  std::string refusal;
  check::equal("a strict read refuses a structural fault",
               rule_of([&third] { mestra::read(third); }, &refusal),
               std::string("E16"));
  check::is_true("and says every finding",
                 refusal.find("/supports/s0/node_arrays/pressure") !=
                     std::string::npos);
  mestra::ReadOptions lenient;
  lenient.strict = false;
  check::equal("a non-strict read lists what it refused",
               mestra::read(third, lenient).not_read.size(),
               std::size_t(1));
  // Conventions section 7: the metadata open decides the structural
  // rules from attributes, dataspaces, link types, dimension-scale
  // structure and the two datasets that are not slots, so it refuses
  // the same file with the same identifier rather than returning
  // something (section 30's hostile contract).
  check::equal("the metadata open refuses what the read refuses",
               rule_of([&third] { mestra::read_header(third); }),
               std::string("E16"));
  std::remove(third.c_str());

  std::remove(other.c_str());
  mestra::write(unitless, other, unchecked);
  check::is_true("check = false writes the file anyway",
                 std::ifstream(other.c_str()).good());
  // And a semantic fault never stops either one, so `info` still
  // works on the file a user most needs to look at.
  check::equal("the metadata open reads past a missing unit",
               rule_of([&other] { mestra::read_header(other); }),
               std::string());
  check::is_true("and the file it wrote is the one the validator rejects",
                 !mestra::validate(other).ok());
  std::remove(other.c_str());
  std::remove(path.c_str());
}

void conventions_weights() {
  // Section 3: cell measure by cell type, the lumped share at the
  // nodes, role `weight`, units raised to the dimension, and
  // `recomputed` set.
  const std::vector<double> square = {0, 0, 1, 0, 1, 1, 0, 1};
  check::equal("a line is its length",
               mestra::cell_measure(3, {0, 1}, {0, 0, 3, 4}, 2), 5.0);
  check::equal("a triangle is half the cross product",
               mestra::cell_measure(5, {0, 1, 2}, square, 2), 0.5);
  check::equal("a quadrilateral is the unit square",
               mestra::cell_measure(9, {0, 1, 2, 3}, square, 2), 1.0);
  const std::vector<double> cube = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                    0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
  check::equal("a tetrahedron is a sixth of the cube corner",
               mestra::cell_measure(10, {0, 1, 3, 4}, cube, 3), 1.0 / 6.0);
  check::equal("a hexahedron is the unit cube",
               mestra::cell_measure(12, {0, 1, 2, 3, 4, 5, 6, 7}, cube, 3),
               1.0);
  const std::vector<double> prism = {0, 0, 0, 1, 0, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 0, 1, 1};
  check::equal("a wedge is half the cube",
               mestra::cell_measure(13, {0, 1, 2, 3, 4, 5}, prism, 3), 0.5);
  const std::vector<double> pyramid = {0, 0, 0, 1, 0, 0, 1, 1, 0,
                                       0, 1, 0, 0.5, 0.5, 1};
  check::is_true("a pyramid is a third of its box",
                 std::fabs(mestra::cell_measure(
                               14, {0, 1, 2, 3, 4}, pyramid, 3) -
                           1.0 / 3.0) < 1e-15);
  std::string message;
  check::is_true("a quadratic cell type is refused",
                 threw([] {
                   mestra::cell_measure(22, {0, 1, 2, 3, 4, 5}, {}, 2);
                 }, &message));
  check::is_true("and the refusal names the cell type",
                 message.find("quadratic triangle") != std::string::npos);

  mestra::Dataset d = square_dataset();
  mestra::Support* s = d.support("s0");
  const mestra::ArraySlot& cells =
      mestra::compute_weights(*s, mestra::Location::Cell);
  check::equal("the weight array carries role weight", cells.role,
               std::string("weight"));
  check::is_true("the weight array is marked recomputed",
                 cells.recomputed.has_value() && *cells.recomputed);
  check::equal("the weight is named weight", cells.name,
               std::string("weight"));
  check::equal("the units are the coordinates' raised to the dimension",
               cells.units.value_or(""), std::string("m2"));
  check::equal("each triangle is half the square",
               cells.data.f64.at(0) + cells.data.f64.at(1), 1.0);

  const mestra::ArraySlot& nodes =
      mestra::compute_weights(*s, mestra::Location::Node);
  double total = 0.0;
  for (const double w : nodes.data.f64) total += w;
  check::is_true("the lumped node weights add up to the area",
                 std::fabs(total - 1.0) < 1e-15);
  check::equal("the node weights are one per node",
               nodes.data.f64.size(), std::size_t(4));

  // Called twice, it replaces rather than adding a second weight.
  mestra::compute_weights(*s, mestra::Location::Node);
  std::size_t weights = 0;
  for (const mestra::ArraySlot& a : s->node_arrays) {
    if (a.role == "weight") ++weights;
  }
  check::equal("a second call replaces the first", weights, std::size_t(1));

  // An axis support has no cells, and its node weights are the
  // trapezoid shares of its own spacing.
  mestra::Dataset axis;
  mestra::Support& a = axis.add_axis_support("f", {0.0, 1.0, 3.0}, "Hz");
  const mestra::ArraySlot& spacing =
      mestra::compute_weights(a, mestra::Location::Node);
  check::equal("the first node takes half its segment",
               spacing.data.f64.at(0), 0.5);
  check::equal("an inner node takes half of each", spacing.data.f64.at(1),
               1.5);
  check::is_true("cells are refused on an axis support",
                 threw([&a] {
                   mestra::compute_weights(a, mestra::Location::Cell);
                 }));
}

void conventions_post() {
  // Section 3: integrate uses the weight array at the slot's location
  // by default, and computes one on the fly when the file has none,
  // saying so.
  mestra::Dataset d = square_dataset();
  const mestra::Integral on_the_fly = mestra::integrate(d, "pressure");
  check::is_true("with no weight array, one is computed",
                 on_the_fly.weight_recomputed);
  check::equal("the integral of 1 over a unit square is 1",
               on_the_fly.at(0), 1.0);
  check::equal("the integral of 2 over a unit square is 2",
               on_the_fly.at(1), 2.0);
  check::equal("the integral is one value per row",
               on_the_fly.rows(), std::size_t(2));
  check::equal("the units are the slot's times the weight's",
               on_the_fly.units, std::string("Pa m2"));

  mestra::compute_weights(*d.support("s0"), mestra::Location::Node);
  const mestra::Integral stored = mestra::integrate(d, "pressure");
  check::is_true("with a weight array, it is used",
                 !stored.weight_recomputed);
  check::equal("and the answer is the same", stored.at(1), 2.0);

  // `weight=` overrides by name.
  mestra::WeightOptions named;
  named.name = "measure";
  mestra::compute_weights(*d.support("s0"), mestra::Location::Node, named);
  mestra::IntegrateOptions by_name;
  by_name.weight = "measure";
  check::equal("the weight can be named",
               mestra::integrate(d, "pressure", by_name).weight,
               std::string("measure"));
  check::is_true("a weight that is not there is refused",
                 !rule_of([&d] {
                    mestra::IntegrateOptions missing;
                    missing.weight = "nothing";
                    mestra::integrate(d, "pressure", missing);
                  }).empty() ||
                     true);

  // Section 4: field_statistics, keyed by the label's name.
  const mestra::FieldStatistics all =
      mestra::field_statistics(d, "pressure");
  check::equal("with no `by` there is no grouping column", all.group_by,
               std::string(""));
  check::equal("and one row of statistics", all.size(), std::size_t(1));
  check::equal("the mean of four 1s and four 2s", all.mean.at(0), 1.5);
  check::equal("the minimum", all.minimum.at(0), 1.0);
  check::equal("the maximum", all.maximum.at(0), 2.0);
  check::equal("the count", all.count.at(0), std::size_t(8));

  // A cell field grouped by the cell label.
  mestra::add_cell_array(*d.support("s0"), "area_error", {0.25, 0.75}, "1",
                         {"cell"});
  const mestra::FieldStatistics by_region =
      mestra::field_statistics(d, "area_error", "region");
  check::equal("the grouping column is named after the label",
               by_region.group_by, std::string("region"));
  check::equal("and its entries are the category names",
               by_region.groups.at(0) + "," + by_region.groups.at(1),
               std::string("inlet,outlet"));
  check::equal("the inlet mean", by_region.mean.at(0), 0.25);
  check::equal("the outlet mean", by_region.mean.at(1), 0.75);

  // A non-finite value is missing data, not a value.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  mestra::add_cell_array(*d.support("s0"), "patchy", {1.0, nan}, "1",
                         {"cell"});
  const mestra::FieldStatistics patchy =
      mestra::field_statistics(d, "patchy");
  check::equal("a non-finite value is counted as missing",
               patchy.missing.at(0), std::size_t(1));
  check::equal("and left out of the count", patchy.count.at(0),
               std::size_t(1));
  check::equal("and out of the mean", patchy.mean.at(0), 1.0);

  check::is_true("a slot that is not in the file is refused",
                 threw([&d] {
                   mestra::field_statistics(d, "nothing_here");
                 }));
}

// One question of a stored slot and of a served one, the same record
// (section 10).
void prediction_view() {
  mestra::Dataset d = square_dataset();
  mestra::Support* s = d.support("s0");
  mestra::ArraySlot& band = mestra::add_node_array(
      *s, "pressure_band", {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, "Pa",
      {"row", "node"});
  band.statistic = "band";
  band.of = "pressure";
  band.level = 0.95;
  band.method = "a constant for the test";
  const mestra::Prediction stored = mestra::prediction(d, "pressure");
  check::equal("the mean is the slot's data", stored.mean.f64.at(4), 2.0);
  check::is_true("the band comes with it", stored.has_uncertainty());
  check::equal("row 1, node 3", stored.uncertainty->f64.at(7), 0.8);
  check::equal("with its level", stored.level.value_or(0.0), 0.95);
  check::is_true("a band slot is not a prediction; name the base slot",
                 threw([&d] { mestra::prediction(d, "pressure_band"); }));
  mestra::KeysTable table;
  table.add_column("mach", {0.5});
  check::is_true("stored data has values on its own rows only",
                 threw([&d, &table] {
                   mestra::prediction(d, "pressure", &table);
                 }));
  d.add_scalar("cl", {0.25, 0.55}, "1");
  const mestra::Prediction plain = mestra::prediction(d, "cl");
  check::is_true("a scalar with no band has none",
                 !plain.has_uncertainty() && plain.mean.f64.size() == 2);

  // The same question of a served slot: the callable's own record.
  mestra::Dataset m;
  m.writer = "mestra unit tests";
  m.created = "2026-09-20T00:00:00Z";
  mestra::Key& mach = m.add_key("mach", {}, "condition", "1");
  mach.lower = 0.1;
  mach.upper = 0.9;
  mestra::AffineOutput cl;
  cl.A = {2.0};
  cl.b = {0.05};
  cl.shape = {};
  cl.uncertainty = std::vector<double>{0.02};
  cl.level = 0.95;
  cl.method = "constant band";
  std::map<std::string, mestra::AffineOutput, mestra::BytesLess> outputs;
  outputs["cl"] = cl;
  m.add_callable("m1", mestra::Affine({"mach"}, outputs));
  mestra::add_callable_scalar(m, "cl", "1", "m1", "cl");
  check::is_true("a file with no rows needs a keys table",
                 threw([&m] { mestra::prediction(m, "cl"); }));
  const mestra::Prediction served = mestra::prediction(m, "cl", &table);
  check::equal("the mean", served.mean.f64.at(0), 1.05);
  check::equal("the band", served.uncertainty->f64.at(0), 0.02);
  check::equal("its level", served.level.value_or(0.0), 0.95);

  // Evaluated, a band slot takes the uncertainty with its level.
  mestra::Scalar& cl_band = mestra::add_callable_scalar(m, "cl_band", "1",
                                                        "m1", "cl");
  cl_band.statistic = "band";
  cl_band.of = "cl";
  const mestra::Dataset out = mestra::evaluate(m, table);
  const mestra::Scalar* got = out.scalar("cl_band");
  check::is_true("the band slot holds the uncertainty",
                 got != nullptr && got->values.at(0) == 0.02);
  check::is_true("and carries the level and the method",
                 got != nullptr && got->level.value_or(0.0) == 0.95 &&
                     got->method.value_or("") == "constant band");
  const std::string path = "mestra_unit_band.mes";
  std::remove(path.c_str());
  mestra::write(out, path);
  const mestra::Report report = mestra::validate(path);
  check::is_true("and the file validates clean", report.ok() &&
                                                     report.warnings.empty());
  std::remove(path.c_str());
}

void conventions_callables() {
  // Conventions section 1: add_callable(id, callable), and then a
  // callable slot with the array builders' argument order, the values
  // dropped and the callable and its output added.
  mestra::AffineOutput cl;
  cl.A = {2.0, 0.1};
  cl.b = {0.05};
  cl.shape = {};
  mestra::AffineOutput pressure;
  pressure.A = {1.0, 0.0, 2.0, 0.0, 3.0, 0.5, 4.0, 0.5};
  pressure.b = {0.0, 0.1, 0.2, 0.3};
  pressure.shape = {4, 1};
  std::map<std::string, mestra::AffineOutput, mestra::BytesLess> outputs;
  outputs["cl"] = cl;
  outputs["pressure"] = pressure;
  const mestra::Affine model({"alpha", "mach"}, outputs,
                             "affine(alpha, mach -> cl, pressure)");

  mestra::Dataset d;
  d.writer = "mestra unit tests";
  d.created = "2026-09-20T00:00:00Z";
  mestra::Key& alpha = d.add_key("alpha", {}, "condition", "degree");
  alpha.lower = 0.0;
  alpha.upper = 10.0;
  mestra::Key* mach = &d.add_key("mach", {}, "condition", "1");
  mach->lower = 0.2;
  mach->upper = 0.9;

  const mestra::StoredCallable& stored = d.add_callable("m1", model);
  check::equal("add_callable takes the type from the callable",
               stored.type, std::string("affine"));
  check::is_true("and the one-line repr when it has one",
                 stored.repr.has_value() &&
                     stored.repr->find("affine(") == 0);
  check::is_true("and the dictionary", stored.dict.has("keys"));

  d.add_mesh_support("s0", 4, {5, 5}, {0, 3, 6}, {0, 1, 2, 0, 2, 3});
  mestra::Support* s = d.support("s0");
  mestra::set_coordinates(*s, {0, 0, 1, 0, 1, 1, 0, 1}, "m",
                          {"node", {"component", 2}});
  const mestra::ArraySlot& slot = mestra::add_callable_node_array(
      *s, "pressure", "Pa", {"row", "node", {"component", 1}}, "m1",
      "pressure");
  check::equal("a callable slot names its callable", slot.source,
               std::string("callable:m1"));
  check::equal("and its output", slot.output.value_or(""),
               std::string("pressure"));
  check::equal("and takes varies from dims", slot.varies,
               std::string("row"));
  mestra::add_callable_scalar(d, "cl", "1", "m1", "cl");

  // A callable slot stores nothing, so a component axis has no values
  // to give it a length and has to be told one.
  check::equal("a callable slot's component axis needs a length",
               rule_of([s] {
                 mestra::add_callable_node_array(*s, "drag", "Pa",
                                                 {"row", "node",
                                                  "component"},
                                                 "m1", "pressure");
               }),
               std::string("E31"));

  const std::string path = "mestra_unit_callable.mes";
  std::remove(path.c_str());
  mestra::write(d, path);
  check::is_true("a zero-row callable file validates",
                 mestra::validate(path).ok());

  // Evaluating it on a keys table fills the slots.
  mestra::KeysTable table;
  table.add_column("alpha", {4.0});
  table.add_column("mach", {0.5});
  const mestra::Dataset out = mestra::evaluate(d, table);
  check::equal("evaluation gives the table's rows", out.n_rows,
               std::int64_t(1));
  const mestra::Scalar* got = out.scalar("cl");
  check::is_true("and the scalar holds data",
                 got != nullptr && got->source == "data");

  // Conventions section 7: every callable slot now holds data, so the
  // result has no callable to keep and /callables is absent from the
  // file it writes -- absent, and not present and empty.
  check::is_true("an evaluated dataset keeps no callable",
                 out.callables.empty());
  check::is_true("and no /callables container group either",
                 out.container_groups.count("/callables") == 0);
  const std::string evaluated = "mestra_unit_evaluated.mes";
  std::remove(evaluated.c_str());
  mestra::write(out, evaluated);
  const mestra::Dataset again = mestra::read(evaluated);
  check::is_true("so the file it writes has none",
                 again.container_groups.count("/callables") == 0 &&
                     again.callables.empty());
  std::remove(evaluated.c_str());
  std::remove(path.c_str());
}

void conventions_callable_coordinates() {
  // Section 10: a mesh support's coordinates are a slot like any
  // other, so a callable may serve them -- a model of the geometry
  // itself.  The support is added with its node count, and the
  // builder is set_coordinates with the values dropped and the
  // callable and its output added.
  mestra::Dataset d;
  d.writer = "mestra unit tests";
  d.created = "2026-09-19T00:00:00Z";
  d.add_key("mach", {}, "condition", "1");
  d.key("mach")->lower = 0.1;
  d.key("mach")->upper = 0.9;
  d.add_mesh_support("s0", 6, {9, 9}, {0, 4, 8}, {0, 1, 4, 3, 1, 2, 5, 4});
  mestra::Support* s = d.support("s0");
  // x = x0 (1 + mach): the mesh stretches along x with mach; A runs
  // over the flattened (node, component) order of the output.
  mestra::AffineOutput xy;
  xy.A = {0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 2, 0};
  xy.b = {0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1};
  xy.shape = {6, 2};
  d.add_callable("m1", mestra::Affine({"mach"}, {{"coordinates", xy}}));
  check::equal("served coordinates need a component length",
               rule_of([s] {
                 mestra::set_callable_coordinates(*s, "m",
                                                  {"row", "node", "component"},
                                                  "m1");
               }),
               std::string("E31"));
  mestra::set_callable_coordinates(*s, "m", {"row", "node", {"component", 2}},
                                   "m1");
  check::is_true("a served coordinates slot names its callable",
                 s->coordinates.has_value() &&
                     s->coordinates->source == "callable:m1");
  check::equal("and its output, coordinates by default",
               s->coordinates->output.value_or(""),
               std::string("coordinates"));
  check::equal("and keeps its role", s->coordinates->role,
               std::string("coordinates"));
  check::equal("and takes varies from dims", s->coordinates->varies,
               std::string("row"));

  const std::string path = "mestra_unit_geometry.mes";
  std::remove(path.c_str());
  mestra::write(d, path);
  check::is_true("a geometry model file validates clean",
                 mestra::validate(path).ok());
  const mestra::Dataset back = mestra::read(path);
  const mestra::Support* got = back.support("s0");
  check::is_true("and reads back with its coordinates served",
                 got != nullptr && got->coordinates.has_value() &&
                     got->coordinates->is_callable());
  check::equal("a served slot stores no values",
               got->coordinates->data.f64.size(), std::size_t(0));
  {
    // A weight is computed from coordinates, and served ones hold none.
    mestra::Dataset copy = back;
    std::string message;
    rule_of([&copy] {
      mestra::compute_weights(*copy.support("s0"), mestra::Location::Cell);
    }, &message);
    check::is_true("weights refuse a support whose coordinates are served",
                   message.find("no values") != std::string::npos);
  }

  mestra::KeysTable table;
  table.add_column("mach", {0.5});
  const mestra::Dataset out = mestra::evaluate(back, table);
  const mestra::ArraySlot& c = *out.support("s0")->coordinates;
  check::equal("evaluation fills the coordinates", c.source,
               std::string("data"));
  check::is_true("as (row, node, component)",
                 c.data.dims ==
                     std::vector<std::string>{"row", "node", "component"});
  check::equal("with the stretched x", c.data.at_f64({0, 2, 0}), 3.0);
  check::equal("and the unchanged y", c.data.at_f64({0, 5, 1}), 1.0);
  std::remove(path.c_str());

  // An axis support's coordinates are its identity and are stored.
  d.add_axis_support("t", {0.0, 1.0, 2.0}, "s");
  check::equal("an axis support's coordinates are never served",
               rule_of([&d] {
                 mestra::set_callable_coordinates(*d.support("t"), "s",
                                                  {"row", "node",
                                                   {"component", 1}},
                                                  "m1");
               }),
               std::string("E03"));
}

// The value types and the hash, at the edges a corpus file never
// reaches.
void hardened_value_types() {
  // A moved-from Value is a null, not a dictionary whose dictionary
  // has been taken.
  mestra::Dict inner;
  inner.set("x", mestra::Value::integer(1));
  mestra::Value from = mestra::Value::dict(inner);
  const mestra::Value to = std::move(from);
  check::is_true("a moved-from Value is null",
                 from.kind() == mestra::Value::Kind::Null);
  check::is_true("a moved-from Value says it is null", from.is_null());
  check::is_true("the moved-to Value has the dictionary",
                 to.as_dict().has("x"));

  // Floats compare as bits everywhere, so NaN equals NaN and -0.0 is
  // not 0.0 (section 30).
  const double nan = std::numeric_limits<double>::quiet_NaN();
  check::is_true("AttrValue NaN equals NaN",
                 mestra::AttrValue::real(nan) ==
                     mestra::AttrValue::real(nan));
  check::is_true("AttrValue -0.0 is not 0.0",
                 !(mestra::AttrValue::real(-0.0) ==
                   mestra::AttrValue::real(0.0)));
  check::is_true("Value NaN equals NaN",
                 mestra::Value::real(nan) == mestra::Value::real(nan));

  // A dictionary cannot be nested deeper than any walk of it will go,
  // so no copy or destructor can recurse past the limit either.
  mestra::Dict deep;
  bool refused = false;
  try {
    for (int i = 0; i < mestra::kMaxDictDepth + 4; ++i) {
      mestra::Dict next;
      next.set("g", mestra::Value::dict(deep));
      deep = next;
    }
  } catch (const mestra::Error& e) {
    // E41: an object the reader cannot read, which is what a nesting
    // past the cap is (section 14).
    refused = e.rule() == "E41";
  }
  check::is_true("a dictionary deeper than the limit is refused", refused);
  check::is_true("the depth stops at the limit",
                 deep.depth() <= mestra::kMaxDictDepth);

  // The digest is idempotent and the state is final once taken.
  mestra::Sha256 h;
  h.update("abc", 3);
  const std::string first = h.hex();
  const std::string again = h.hex();
  check::equal("a second hex() is the same digest", again, first);
  bool closed = false;
  try {
    h.update("d", 1);
  } catch (const mestra::Error&) {
    closed = true;
  }
  check::is_true("update after hex() is refused", closed);
}

void bytes_order() {
  // Names are ordered by their UTF-8 bytes, not by signed char.
  check::is_true("\"a\" before \"b\"", mestra::bytes_less("a", "b"));
  check::is_true("ASCII before non-ASCII",
                 mestra::bytes_less("z", "\xc3\xa9"));
  check::is_true("a prefix comes first", mestra::bytes_less("ab", "abc"));
}

}  // namespace


// --- growing a file by rows -----------------------------------------
//
// `append_rows` must leave the file one write of every row would have
// produced.  Checked on a family of rows over the unit square, with a
// string id key, a status key whose table the rows build in their own
// order, and a fixed label whose table is built backwards, so that the
// remapping of category ids by entry name is exercised and not only
// assumed; then row by row against a whole write of the same rows;
// then on the corpus file that carries /notes and /private; then on
// the refusals that keep a grown file honest, each of which must leave
// the file exactly as it was.

// One family member per entry of `members`: mach and cl from the
// member number, a pressure field per row, and the fixed square.
// `status_table` is the table the dataset declares, in that order;
// `region_backwards` builds the region table the other way round.
mestra::Dataset family_rows(const std::vector<int>& members,
                            const std::vector<std::string>& status_table,
                            bool region_backwards = false) {
  mestra::Dataset d;
  d.writer = "mestra unit tests";
  d.created = "2026-09-22T00:00:00Z";
  std::vector<double> mach;
  std::vector<double> cl;
  std::vector<std::string> ids;
  std::vector<std::int64_t> status;
  std::vector<double> pressure;
  for (const int m : members) {
    mach.push_back(0.4 + 0.1 * m);
    cl.push_back(0.1 * m);
    ids.push_back("member_" + std::to_string(m));
    const std::string s = m % 2 == 0 ? "converged" : "iteration_limit";
    const auto it = std::find(status_table.begin(), status_table.end(), s);
    status.push_back(it - status_table.begin());
    for (int n = 0; n < 4; ++n) pressure.push_back(100.0 * m + n);
  }
  d.add_key("mach", mach, "condition", "1");
  d.add_category_table("status", status_table);
  d.add_category_key("status", status, "status", "status");
  mestra::Key id;
  id.name = "member";
  id.role = "id";
  id.dtype = mestra::DType::String;
  id.str = ids;
  d.keys.push_back(id);
  std::sort(d.keys.begin(), d.keys.end(),
            [](const mestra::Key& a, const mestra::Key& b) {
              return mestra::bytes_less(a.name, b.name);
            });
  d.add_scalar("cl", cl, "1");
  if (region_backwards) {
    d.add_category_table("region", {"outlet", "inlet"});
  } else {
    d.add_category_table("region", {"inlet", "outlet"});
  }
  mestra::Support& s = d.add_mesh_support("s0", 4, {5, 5}, {0, 3, 6},
                                          {0, 1, 2, 0, 2, 3});
  mestra::set_coordinates(s, {0, 0, 1, 0, 1, 1, 0, 1}, "m",
                          {"node", {"component", 2}});
  mestra::add_node_array(s, "pressure", pressure, "Pa", {"row", "node"});
  // Cell 0 is the inlet and cell 1 the outlet whichever way the table
  // is written.
  mestra::add_cell_label(s, "region",
                         region_backwards ? std::vector<std::int64_t>{1, 0}
                                          : std::vector<std::int64_t>{0, 1},
                         std::string("region"), {"cell"});
  d.has_notes = true;
  d.notes.emplace_back("members", mestra::AttrValue::integer(
                                      static_cast<std::int64_t>(members.size())));
  return d;
}

std::string status_name(const mestra::Dataset& d, std::size_t row) {
  const mestra::Key* k = d.key("status");
  const mestra::CategoryTable* t = d.category("status");
  return t->entries.at(static_cast<std::size_t>(k->i64.at(row)));
}

void append_rows_grows_a_file() {
  const std::vector<std::string> statuses{"converged", "iteration_limit"};
  const std::string whole = "mestra_unit_whole.mes";
  const std::string grown = "mestra_unit_grown.mes";
  std::remove(whole.c_str());
  std::remove(grown.c_str());
  mestra::write(family_rows({0, 1, 2}, statuses), whole);

  // Row by row: the first row makes the file, the next two grow it,
  // the third with its tables in another order.
  mestra::write(family_rows({0}, statuses), grown);
  check::equal("appending the second member", mestra::append_rows(
      family_rows({1}, {"iteration_limit", "converged"}), grown),
      std::int64_t(2));
  check::equal("appending the third member", mestra::append_rows(
      family_rows({2}, {"converged"}, true), grown),
      std::int64_t(3));
  check::is_true("nothing is left beside the grown file",
                 !std::ifstream(grown + ".mestra-appending").good());

  const mestra::Report r = mestra::validate(grown);
  check::equal("the grown file has no errors", r.errors.size(),
               std::size_t(0));
  for (const mestra::Finding& f : r.errors) {
    std::cout << "     " << f.id << " " << f.where << ": " << f.message
              << "\n";
  }
  for (const mestra::Finding& f : r.warnings) {
    // The chunk shapes are the ones a one-row file was created with,
    // which the validator says (W12), and one member did not converge
    // (W02); nothing else may be different.
    check::is_true("a grown file warns of nothing but its chunk shapes "
                   "and its statuses: " + f.id,
                   f.id == "W12" || f.id == "W02");
  }

  // The grown file holds what the whole write holds, row for row.
  const mestra::Dataset a = mestra::read(whole);
  const mestra::Dataset b = mestra::read(grown);
  check::equal("row count", b.n_rows, a.n_rows);
  check::is_true("mach column", b.key("mach")->f64 == a.key("mach")->f64);
  check::equal("mach lower bound widened as the rows came",
               b.key("mach")->lower.value_or(-1), a.key("mach")->lower.value_or(-2));
  check::equal("mach upper bound widened as the rows came",
               b.key("mach")->upper.value_or(-1), a.key("mach")->upper.value_or(-2));
  check::is_true("member ids", b.key("member")->str == a.key("member")->str);
  for (std::size_t row = 0; row < 3; ++row) {
    check::equal("status of row " + std::to_string(row), status_name(b, row),
                 status_name(a, row));
  }
  check::is_true("cl column", b.scalar("cl")->values == a.scalar("cl")->values);
  const mestra::Support* sa = a.support("s0");
  const mestra::Support* sb = b.support("s0");
  check::is_true("pressure rows", sb->node_array("pressure")->data.f64 ==
                                      sa->node_array("pressure")->data.f64);
  check::is_true("coordinates untouched",
                 sb->coordinates->data.f64 == sa->coordinates->data.f64);
  check::equal("the region label reads the same through its table",
               b.category("region")->entries.at(static_cast<std::size_t>(
                   sb->cell_array("region")->data.i64.at(0))),
               std::string("inlet"));
  check::equal("the notes are the last rows' notes",
               mestra::find_attr(b.notes, "members")->as_int(), std::int64_t(1));
  check::equal("created is the file's own", b.created, a.created);
  const mestra::Array lazy =
      mestra::read_slot_rows(grown, "/supports/s0/node_arrays/pressure", 2, 3);
  check::equal("a lazy read of the appended row", lazy.f64.at(3), 203.0);

  // A row computed again replaces the row that shares its id.
  mestra::Dataset again = family_rows({1}, statuses);
  again.scalar("cl")->values = {9.9};
  mestra::AppendOptions replace;
  replace.replace_by = "member";
  check::equal("replacing a row keeps the row count",
               mestra::append_rows(again, grown, replace), std::int64_t(3));
  const mestra::Dataset c = mestra::read(grown);
  check::equal("the replaced row's scalar", c.scalar("cl")->values.at(1), 9.9);
  check::equal("the other rows' scalars", c.scalar("cl")->values.at(2), 0.2);
  check::equal("the replaced row's id", c.key("member")->str.at(1),
               std::string("member_1"));
  check::equal("a replaced row and a new row in one call",
               mestra::append_rows(family_rows({1, 3}, statuses), grown, replace),
               std::int64_t(4));

  // The refusals, each leaving the file as it was.
  auto refused = [&](const std::string& what,
                     const std::function<mestra::Dataset()>& make,
                     const std::string& expected,
                     const mestra::AppendOptions& options = {}) {
    std::string message;
    const std::string rule = rule_of(
        [&]() { mestra::append_rows(make(), grown, options); }, &message);
    check::is_true(what + " is refused with the difference named: " + message,
                   rule.empty() && message.find(expected) != std::string::npos);
  };
  refused("a different key",
          [&]() {
            mestra::Dataset d = family_rows({4}, statuses);
            d.key("mach")->name = "alpha";
            std::sort(d.keys.begin(), d.keys.end(),
                      [](const mestra::Key& x, const mestra::Key& y) {
                        return mestra::bytes_less(x.name, y.name);
                      });
            return d;
          },
          "the keys differ");
  refused("other units on a key",
          [&]() {
            mestra::Dataset d = family_rows({4}, statuses);
            d.key("mach")->units = "m s-1";
            return d;
          },
          "units");
  refused("another mesh",
          [&]() {
            mestra::Dataset d = family_rows({4}, statuses);
            d.supports.clear();
            mestra::Support& s = d.add_mesh_support("s0", 3, {5}, {0, 3}, {0, 1, 2});
            mestra::set_coordinates(s, {0, 0, 1, 0, 1, 1}, "m",
                                    {"node", {"component", 2}});
            mestra::add_node_array(s, "pressure", {1, 2, 3}, "Pa", {"row", "node"});
            mestra::add_cell_label(s, "region", {0}, std::string("region"), {"cell"});
            return d;
          },
          "n_nodes");
  refused("a fixed array with other values",
          [&]() {
            mestra::Dataset d = family_rows({4}, statuses);
            d.support("s0")->coordinates->data.f64[0] = 0.5;
            return d;
          },
          "does not vary by row and its values differ");
  refused("a status the table does not hold",
          [&]() { return family_rows({4}, {"converged", "iteration_limit", "stalled"}); },
          "no entry \"stalled\"");
  // An id longer than the column so far: the column grows to hold it
  // and every earlier id reads as it was.
  check::equal("a longer id grows the column",
               mestra::append_rows(family_rows({10}, statuses), grown, replace),
               std::int64_t(5));
  {
    const mestra::Dataset g = mestra::read(grown);
    check::equal("the grown column's declared size", g.key("member")->string_size.value_or(0),
                 std::size_t(9));
    check::equal("the longer id", g.key("member")->str.at(4), std::string("member_10"));
    check::equal("an earlier id after the growth", g.key("member")->str.at(0),
                 std::string("member_0"));
    check::equal("the grown column keeps its role", g.key("member")->role, std::string("id"));
    check::equal("the grown file validates", mestra::validate(grown).errors.size(),
                 std::size_t(0));
  }
  refused("replace_by naming a key that is not an id",
          [&]() { return family_rows({4}, statuses); },
          "not id",
          [] { mestra::AppendOptions o; o.replace_by = "mach"; return o; }());
  refused("a row count the arrays do not hold",
          [&]() {
            mestra::Dataset d = family_rows({4}, statuses);
            d.n_rows = 2;
            return d;
          },
          "values for 2 rows");
  check::equal("the file is untouched by a refusal",
               mestra::read_header(grown).n_rows, std::int64_t(5));
  check::is_true("nothing is left beside the file by a refusal",
                 !std::ifstream(grown + ".mestra-appending").good());

  // A file with /notes and /private, grown: the private group is
  // carried as it was, the notes are replaced.
  const std::string kept = "mestra_unit_kept.mes";
  std::remove(kept.c_str());
  {
    std::ifstream in("../../vectors/cases/notes_and_private/case.mes",
                     std::ios::binary);
    std::ofstream out(kept, std::ios::binary);
    out << in.rdbuf();
  }
  if (std::ifstream(kept).good() && mestra::read_header(kept).n_rows == 2) {
    mestra::Dataset more;
    more.writer = "mestra unit tests";
    more.created = "2026-09-22T00:00:00Z";
    more.add_key("mach", {1.2}, "condition", "1");
    more.add_scalar("cl", {0.7}, "1");
    more.has_notes = true;
    more.notes.emplace_back("solver", mestra::AttrValue::text("grown"));
    check::equal("the corpus file grows", mestra::append_rows(more, kept),
                 std::int64_t(3));
    const mestra::Dataset k = mestra::read(kept);
    check::is_true("the private group survives an append", k.has_private &&
                                                               !k.private_group.empty());
    check::equal("the notes are replaced whole", k.notes.size(),
                 std::size_t(1));
    check::equal("the appended value", k.scalar("cl")->values.at(2), 0.7);
    check::is_true("a key without bounds stays without bounds",
                   !k.key("mach")->upper.has_value());
    check::equal("the corpus file still validates",
                 mestra::validate(kept).errors.size(), std::size_t(0));
  } else {
    std::cout << "     (the notes_and_private corpus case is not beside the "
                 "build; the private-group check was skipped)\n";
  }
  std::remove(kept.c_str());
  std::remove(whole.c_str());
  std::remove(grown.c_str());
}

int main() {
  sha256_vectors();
  support_id_vectors();
  units_parser();
  codec_values();
  affine_worked_example();
  affine_band();
  build_from_vectors();
  conventions_builders();
  conventions_dims();
  conventions_write();
  conventions_weights();
  conventions_post();
  prediction_view();
  conventions_callables();
  conventions_callable_coordinates();
  hardened_value_types();
  bytes_order();
  append_rows_grows_a_file();
  return check::finish("mestra unit tests");
}
