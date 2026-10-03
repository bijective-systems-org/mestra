#!/usr/bin/env python3
"""Run the conformance corpus against the C++ command-line tool.

This is the whole of the JSON in the C++ implementation's test suite:
the tool prints plain text and this script reads expected.json and
compares.  Nothing here is a mestra implementation, and it never looks
at the C++ sources; it drives `mestra-cli` the way any other user
would.

    python3 run_corpus.py --cli ../build/mestra-cli --vectors ../../vectors

It needs h5py only for the read-write comparison of section 30, which
it skips with a note when h5py is not importable.  That comparison is
structural and never byte-for-byte: section 30 says byte identity with
the corpus files must not be tested, because the HDF5 library decides
the superblock, the object header layout and where the global heap
objects the dimension-scale machinery uses are allocated.

The driver also prints what the largest case costs -- a metadata open,
a validation and one lazy slot read of `wide_keys`, each the library
call alone.  Nothing asserts on the seconds; the line is there so that
an open or a validation which became the square of the dataset count
would show.
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

try:
    import h5py
    import numpy
except ImportError:                                   # pragma: no cover
    h5py = None
    numpy = None


# --------------------------------------------------------------- tool

class Tool(object):
    def __init__(self, path):
        self.path = path

    def run(self, *arguments, **options):
        """Runs the tool. `validate` exits 1 on a file it rejects, which
        is not a failure of the tool, so a caller may allow it."""
        allowed = options.pop("allowed", (0,))
        done = subprocess.run([self.path] + list(arguments),
                              capture_output=True, text=True)
        if done.returncode not in allowed:
            raise RuntimeError("mestra-cli %s failed: %s"
                               % (" ".join(arguments), done.stderr.strip()))
        return done.stdout


# ---------------------------------------------------- validator output

# docs/api-conventions.md section 5: every finding is printed as
# "<id> <path>: <message>" and the run ends with the counts.  A fault
# no rule covers carries no identifier and is printed as "! ".
FINDING = re.compile(r"^([EW][0-9]{2}) (\S+): (.*)$")
SUMMARY = re.compile(r"^([0-9]+) error\(s\), ([0-9]+) warning\(s\)$")


def findings_of(text):
    """The identifiers the tool printed, sorted and without
    duplicates, which is the form expected.json compares, and the
    problems with the shape of the output itself."""
    lines = text.splitlines()
    ids = []
    printed = 0
    for line in lines[:-1] if lines else []:
        match = FINDING.match(line)
        if match:
            ids.append(match.group(1))
            printed += 1
        elif line.startswith("! "):
            printed += 1
        else:
            return None, None, "not a finding line: %r" % (line,)
    summary = SUMMARY.match(lines[-1]) if lines else None
    if summary is None:
        return None, None, ("the last line is not "
                            "\"<n> error(s), <m> warning(s)\": %r"
                            % (lines[-1] if lines else "",))
    counted = int(summary.group(1)) + int(summary.group(2))
    if counted != printed:
        return None, None, ("the summary counts %d finding(s) and %d "
                            "were printed" % (counted, printed))
    errors = sorted({i for i in ids if i.startswith("E")})
    warnings = sorted({i for i in ids if i.startswith("W")})
    return errors, warnings, None


# ------------------------------------------------------- float compare

def bits(value):
    """The bit pattern of a float64, so that NaN equals NaN."""
    return struct.pack("<d", value)


def parse_float(text):
    if text == "nan":
        return float("nan")
    if text == "inf":
        return float("inf")
    if text == "-inf":
        return float("-inf")
    return float(text)


# ------------------------------------------------- the dictionary dump

def escape(text):
    out = []
    for byte in text.encode("utf-8"):
        if 0x20 < byte < 0x7F and byte != ord("%"):
            out.append(chr(byte))
        else:
            out.append("%%%02X" % byte)
    return "".join(out) if out else "%"


def shape_text(shape):
    return " ".join([str(len(shape))] + [str(e) for e in shape])


def flatten(value, path, lines):
    """The tagged form of section 30, in the tool's own dump format."""
    tag = value["t"]
    if tag == "dict":
        lines.append("D " + path)
        for key in sorted(value["v"], key=lambda k: k.encode("utf-8")):
            flatten(value["v"][key], path + "/" + key, lines)
    elif tag == "null":
        lines.append("N " + path)
    elif tag == "bool":
        lines.append("B %s %s" % (path, "1" if value["v"] else "0"))
    elif tag == "i64":
        lines.append("I %s %d" % (path, value["v"]))
    elif tag == "f64":
        lines.append("F %s %s" % (path, value["v"]))
    elif tag == "str":
        lines.append("S %s %s" % (path, escape(value["v"])))
    elif tag == "array":
        dtype = value["dtype"]
        if dtype == "float64":
            data = list(value["data"])
        elif dtype == "bool":
            data = ["1" if e else "0" for e in value["data"]]
        else:
            data = [str(int(e)) for e in value["data"]]
        lines.append(" ".join(["A", path, dtype, shape_text(value["shape"])]
                              + data))
    elif tag == "strings":
        lines.append(" ".join(["T", path, shape_text(value["shape"])]
                              + [escape(e) for e in value["data"]]))
    else:
        raise ValueError("unknown tag %r" % (tag,))


def expected_dump(tagged):
    lines = []
    flatten(tagged, ".", lines)
    return "".join(line + "\n" for line in lines)


# ------------------------------------------- structural equality (30)

MACHINERY = {"CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST",
             "DIMENSION_LABELS", "_Netcdf4Dimid", "_Netcdf4Coordinates",
             "_nc3_strict", "_NCProperties"}


def type_signature(dataset):
    kind = dataset.id.get_type()
    if isinstance(kind, h5py.h5t.TypeStringID):
        return ("string", kind.get_size(), int(kind.get_cset()),
                int(kind.get_strpad()))
    return ("numeric", dataset.dtype.str)


def dataset_contents(dataset):
    values = dataset[()]
    if isinstance(values, numpy.ndarray):
        if values.dtype.kind == "O":
            return tuple(bytes(e) if isinstance(e, bytes)
                         else str(e).encode("utf-8")
                         for e in values.ravel())
        return values.tobytes()
    return values


def scale_names(dataset):
    out = []
    for axis in range(len(dataset.shape)):
        names = []
        try:
            for _label, scale in dataset.dims[axis].items():
                names.append(scale.name.rsplit("/", 1)[-1])
        except (RuntimeError, KeyError):
            pass
        out.append(tuple(sorted(names)))
    return tuple(out)


def describe(path):
    """Everything section 30 compares, at every object path."""
    out = {}

    def attributes(obj):
        found = {}
        for name, value in obj.attrs.items():
            if name in MACHINERY:
                continue
            attr = obj.attrs.get_id(name)
            kind = attr.get_type()
            if isinstance(kind, h5py.h5t.TypeStringID):
                signature = ("string", kind.get_size(), int(kind.get_cset()),
                             int(kind.get_strpad()))
                stored = bytes(value) if isinstance(value, bytes) \
                    else str(value).encode("utf-8")
            else:
                signature = ("numeric", numpy.asarray(value).dtype.str)
                stored = numpy.asarray(value).tobytes()
            found[name] = (signature, stored)
        return found

    with h5py.File(path, "r") as handle:
        def visit(name, obj):
            full = "/" + name
            if isinstance(obj, h5py.Group):
                out[full] = ("group", attributes(obj))
            else:
                out[full] = ("dataset", attributes(obj), type_signature(obj),
                             tuple(obj.shape), tuple(obj.maxshape),
                             tuple(obj.chunks) if obj.chunks else None,
                             (obj.compression, obj.compression_opts,
                              bool(obj.shuffle), bool(obj.fletcher32)),
                             dataset_contents(obj), scale_names(obj))
        out["/"] = ("group", attributes(handle["/"]))
        handle.visititems(visit)
    return out


def structural_diff(left, right):
    a = describe(left)
    b = describe(right)
    problems = []
    for path in sorted(set(a) | set(b)):
        if path not in a:
            problems.append("only in the written file: " + path)
        elif path not in b:
            problems.append("only in the original: " + path)
        elif a[path] != b[path]:
            for index, (x, y) in enumerate(zip(a[path], b[path])):
                if x != y:
                    what = ["kind", "attributes", "dtype", "shape",
                            "maxshape", "chunk", "filters", "contents",
                            "dimension scales"][index]
                    problems.append("%s: %s differs" % (path, what))
    return problems


# ------------------------------------------------------------- driver

def run_case(tool, directory, name, totals, problems):
    case = os.path.join(directory, name)
    mes = os.path.join(case, "case.mes")
    with open(os.path.join(case, "expected.json"), encoding="utf-8") as fh:
        expected = json.load(fh)

    def fail(what, got, want):
        problems.append("%s: %s\n     got  %s\n     want %s"
                        % (name, what, got, want))

    # --- the validator ------------------------------------------------
    errors, warnings, malformed = findings_of(
        tool.run("validate", mes, allowed=(0, 1)))
    totals["validator"] += 1
    if malformed is not None:
        fail("validator output", malformed,
             "<id> <path>: <message> lines and a summary")
    elif errors != expected["validator"]["errors"] or \
            warnings != expected["validator"]["warnings"]:
        fail("validator outcome",
             "errors=%s warnings=%s" % (errors, warnings),
             "errors=%s warnings=%s" % (expected["validator"]["errors"],
                                        expected["validator"]["warnings"]))
    else:
        totals["validator_ok"] += 1

    # --- support_id ---------------------------------------------------
    for support, digest in sorted(expected["support_ids"].items()):
        totals["support_id"] += 1
        got = tool.run("support-id", mes, support).strip()
        if got != digest:
            fail("support_id of " + support, got, digest)
        else:
            totals["support_id_ok"] += 1
    # A support the expectation leaves out has no digest (section 24),
    # and the tool must refuse to give one. Listing the supports needs
    # h5py, as the read-write comparison does.
    if h5py is not None:
        with h5py.File(mes, "r") as f:
            groups = f.get("supports")
            present = sorted(
                name for name in groups
                if isinstance(groups.get(name, getlink=True), h5py.HardLink)
                and isinstance(groups[name], h5py.Group)) \
                if isinstance(groups, h5py.Group) else []
        for support in present:
            if support in expected["support_ids"]:
                continue
            totals["support_id"] += 1
            done = subprocess.run([tool.path, "support-id", mes, support],
                                  capture_output=True, text=True)
            if done.returncode == 0:
                fail("support_id of " + support, done.stdout.strip(),
                     "a refusal: the support has no digest")
            else:
                totals["support_id_ok"] += 1

    # --- probes -------------------------------------------------------
    for probe in expected["probes"]:
        totals["probe"] += 1
        got = probe_value(tool, mes, probe)
        if not matches(got, probe["value"]):
            fail("probe %s %s" % (probe["slot"], axes_text(probe)), got,
                 probe["value"])
        else:
            totals["probe_ok"] += 1

    # --- lazy access (section 29) -------------------------------------
    # A probe on a row-dimensioned float64 slot is read again through
    # the row-range reader, which must not touch any other slot.
    for probe in expected["probes"]:
        if "row" not in probe or not probe["slot"].startswith("/supports"):
            continue
        if "instance" in probe:
            continue
        totals["lazy"] += 1
        row = probe["row"]
        lines = tool.run("rows", mes, probe["slot"], str(row),
                         str(row + 1)).splitlines()
        shape = [int(e) for e in lines[1].split()[1:]]
        values = lines[2:]
        if shape[0] != 1:
            fail("lazy read of " + probe["slot"], str(shape), "one row")
            continue
        offset = 0
        axes = [a for a in ("draw", "node", "component") if a in probe]
        for position, axis in enumerate(axes):
            step = 1
            for extent in shape[position + 2:]:
                step *= extent
            offset += probe[axis] * step
        if offset >= len(values) or not matches(values[offset],
                                                probe["value"]):
            fail("lazy read of %s %s" % (probe["slot"], axes_text(probe)),
                 values[offset] if offset < len(values) else "(missing)",
                 probe["value"])
        else:
            totals["lazy_ok"] += 1

    # --- the codec ----------------------------------------------------
    for callable_id, tagged in sorted(expected["codec"].items()):
        want = expected_dump(tagged)
        totals["codec"] += 1
        got = tool.run("dict-dump", mes, callable_id)
        if got != want:
            fail("codec dump of " + callable_id, first_difference(got, want),
                 "(see expected.json)")
        else:
            totals["codec_ok"] += 1
        # Write the dictionary back and read it through the tool again.
        totals["codec_roundtrip"] += 1
        with tempfile.TemporaryDirectory() as temporary:
            out = os.path.join(temporary, "codec.mes")
            tool.run("dict-roundtrip", mes, callable_id, out)
            again = tool.run("dict-dump", out, callable_id)
        if again != want:
            fail("codec round trip of " + callable_id,
                 first_difference(again, want), "(see expected.json)")
        else:
            totals["codec_roundtrip_ok"] += 1

    # --- evaluation ---------------------------------------------------
    for evaluation in expected["evaluation"]:
        totals["evaluation"] += 1
        ok = True
        with tempfile.TemporaryDirectory() as temporary:
            csv = os.path.join(temporary, "keys.csv")
            names = sorted(evaluation["keys"])
            rows = len(evaluation["keys"][names[0]]) if names else 0
            with open(csv, "w", encoding="utf-8") as fh:
                fh.write(",".join(names) + "\n")
                for row in range(rows):
                    fh.write(",".join(evaluation["keys"][n][row]
                                      for n in names) + "\n")
            out = os.path.join(temporary, "evaluated.mes")
            tool.run("evaluate", mes, csv, out)
            # Conventions section 7: evaluating turns every callable
            # slot into a stored slot, so the result has no callable
            # to keep and the /callables group is absent, not present
            # and empty.  The four writers agree on that.
            if h5py is not None:
                with h5py.File(out, "r") as handle:
                    if "callables" in handle:
                        ok = False
                        fail("the evaluated file's /callables group",
                             "present", "absent")
            for probe in evaluation["probes"]:
                got = probe_value(tool, out, probe)
                if not matches(got, probe["value"]):
                    ok = False
                    fail("evaluation of %s at %s %s"
                         % (evaluation["callable"], probe["slot"],
                            axes_text(probe)), got, probe["value"])
        if ok:
            totals["evaluation_ok"] += 1

    # --- read, write, compare (section 30) ----------------------------
    if expected["validator"]["errors"]:
        return
    totals["roundtrip"] += 1
    with tempfile.TemporaryDirectory() as temporary:
        out = os.path.join(temporary, "written.mes")
        tool.run("roundtrip", mes, out)
        if h5py is None:
            totals["roundtrip_skipped"] += 1
            return
        differences = structural_diff(mes, out)
        with open(mes, "rb") as fh:
            original = fh.read()
        with open(out, "rb") as fh:
            written = fh.read()
    if differences:
        fail("read, write and compare", "; ".join(differences[:4]),
             "structurally equal")
    else:
        totals["roundtrip_ok"] += 1
        if original == written:
            totals["byte_identical"] += 1


def axes_text(probe):
    return " ".join("%s=%s" % (k, probe[k]) for k in
                    ("row", "instance", "draw", "node", "component", "index")
                    if k in probe)


def probe_value(tool, mes, probe):
    arguments = ["probe", mes, probe["slot"]]
    for axis in ("row", "instance", "draw", "node", "component", "index"):
        if axis in probe:
            arguments.append("%s=%d" % (axis, probe[axis]))
    if len(arguments) == 3:
        arguments.append("row=0")     # a slot with a single element
    return tool.run(*arguments).strip()


def matches(got, want):
    """A probe compares as a float when either side is a float."""
    if "e" in want or "." in want or want in ("nan", "inf", "-inf"):
        return bits(parse_float(got)) == bits(parse_float(want))
    return got == want


def first_difference(got, want):
    got_lines = got.splitlines()
    want_lines = want.splitlines()
    for index in range(max(len(got_lines), len(want_lines))):
        a = got_lines[index] if index < len(got_lines) else "(nothing)"
        b = want_lines[index] if index < len(want_lines) else "(nothing)"
        if a != b:
            return "line %d: %s (want %s)" % (index + 1, a, b)
    return "(no difference)"


# ------------------------------------------------------------- cost

# One case is large enough for a cost to be a number rather than
# noise: `wide_keys`, whose 4200 row-dimensioned datasets all attach
# to the one `row` scale, and which could not be written at all before
# section 21 fixed how a scale is created.  The three costs section 29
# is about are printed beside the counts, each the library call alone,
# so that a change which makes an open or a validation cost the square
# of the dataset count is seen where the corpus is run rather than in
# a study a year later.  They are reported and never asserted on: the
# machine running this decides the seconds.
COST_CASE = "wide_keys"
COST_SLOT = "/scalars/s2100"


def cost_line(tool, directory):
    mes = os.path.join(directory, COST_CASE, "case.mes")
    if not os.path.exists(mes):
        return "%s: not generated (vectors/generate.py --on-demand)" % COST_CASE
    seconds = {}
    for line in tool.run("cost", mes, COST_SLOT).splitlines():
        parts = line.split()
        if len(parts) == 2:
            seconds[parts[0]] = parts[1]
    return ("%s   open %s s   validate %s s   one lazy slot read %s s"
            % (COST_CASE, seconds.get("open", "?"),
               seconds.get("validate", "?"), seconds.get("rows", "?")))


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True,
                        help="the mestra-cli executable")
    parser.add_argument("--vectors", required=True,
                        help="the vectors/ directory of the corpus")
    parser.add_argument("--case", action="append",
                        help="run only these cases")
    arguments = parser.parse_args(argv[1:])

    directory = os.path.join(arguments.vectors, "cases")
    names = sorted(arguments.case or os.listdir(directory))
    tool = Tool(arguments.cli)

    totals = dict.fromkeys(
        ["validator", "validator_ok", "support_id", "support_id_ok",
         "probe", "probe_ok", "codec", "codec_ok", "codec_roundtrip",
         "codec_roundtrip_ok", "evaluation", "evaluation_ok", "lazy",
         "lazy_ok", "roundtrip",
         "roundtrip_ok", "roundtrip_skipped", "byte_identical"], 0)
    problems = []
    for name in names:
        if not os.path.isdir(os.path.join(directory, name)):
            continue
        try:
            run_case(tool, directory, name, totals, problems)
        except Exception as error:                     # noqa: BLE001
            problems.append("%s: %s" % (name, error))

    print("cases                  %d" % len(names))
    print("validator outcomes     %d of %d"
          % (totals["validator_ok"], totals["validator"]))
    print("support ids            %d of %d"
          % (totals["support_id_ok"], totals["support_id"]))
    print("probes                 %d of %d"
          % (totals["probe_ok"], totals["probe"]))
    print("codec dumps            %d of %d"
          % (totals["codec_ok"], totals["codec"]))
    print("codec round trips      %d of %d"
          % (totals["codec_roundtrip_ok"], totals["codec_roundtrip"]))
    print("evaluations            %d of %d"
          % (totals["evaluation_ok"], totals["evaluation"]))
    print("lazy row reads         %d of %d"
          % (totals["lazy_ok"], totals["lazy"]))
    print("read, write, compare   %d of %d structurally equal"
          % (totals["roundtrip_ok"], totals["roundtrip"]))
    print("                       %d of %d byte identical"
          % (totals["byte_identical"], totals["roundtrip"]))
    if totals["roundtrip_skipped"]:
        print("                       %d skipped: h5py is not importable"
              % totals["roundtrip_skipped"])
    try:
        print("cost                   " + cost_line(tool, directory))
    except Exception as error:                         # noqa: BLE001
        problems.append("cost of %s: %s" % (COST_CASE, error))
    if problems:
        print("\n%d problems:" % len(problems))
        for problem in problems:
            print("  " + problem)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
