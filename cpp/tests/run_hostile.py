#!/usr/bin/env python3
"""Run the hostile corpus: every file under tests/hostile/ through
`validate`, `info` and `read`.

These files are not conformance cases and this script checks nothing
about what they mean.  It checks the one thing a reader of an open
format owes anybody who hands it a file: that the answer is an answer.
For each file and each command:

  - the tool exits, within a timeout: never a hang;
  - it exits with a status and not a signal: never a crash;
  - `validate` comes back with at least one finding, on a line in the
    form "<id> <path>: <message>" or "! <path>: <why>";
  - `info` and `read` either succeed or fail with a message, and say
    which file and why.

    python3 run_hostile.py --cli ../build/mestra-cli [--dir hostile]

tests/hostile/make_hostile.py writes the files and says how each one
was made.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

FINDING = re.compile(r"^([EW][0-9]{2}|!) (\S+): (.*)$")

# Most of these files are here for the one thing above: an answer
# rather than a signal.  Three of them are here for which answer,
# because naming the wrong rule, or none, is its own fault.  `must`
# is what `validate` has to name and `must_not` what it may not,
# each an identifier of section 14.
EXPECTED = {
    "category_above_cap.mes": {
        # Section 29: the eager read of a table above the stated
        # maximum element count is E41.  E10 is what a reader says
        # when it read that table as empty and then found every
        # category id outside it, which is the consequence of the
        # fault and not the fault.
        "must": ["E41"],
        "must_not": ["E10"],
    },
    "shape_enormous.mes": {
        "must": ["E16", "E41"],
        "must_not": [],
    },
    "not_hdf5.mes": {"must": ["E01"], "must_not": ["E41"]},
    "truncated.mes": {"must": ["E01"], "must_not": ["E41"]},
    "support_unknown_dataset.mes": {
        # A dataset this version does not know, in a group it does,
        # is a public object, and section 23's chunking rule holds
        # on it.
        "must": ["E27"],
        "must_not": [],
    },
}

# Long enough that a slow machine under a sanitizer is not mistaken
# for a hang, short enough that a hang is not mistaken for patience.
TIMEOUT_SECONDS = 120


def run(cli, command, path):
    try:
        done = subprocess.run([cli, command, path], capture_output=True,
                              text=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return None, "", ""
    return done.returncode, done.stdout, done.stderr


def check(cli, path, problems):
    name = os.path.basename(path)
    for command in ("validate", "info", "read"):
        code, out, err = run(cli, command, path)
        if code is None:
            problems.append("%s: %s did not finish within %d seconds"
                            % (name, command, TIMEOUT_SECONDS))
            continue
        if code < 0:
            problems.append("%s: %s was killed by signal %d"
                            % (name, command, -code))
            continue
        if code not in (0, 1):
            problems.append("%s: %s exited %d, which is neither a clean "
                            "answer nor a refusal" % (name, command, code))
            continue
        if command == "validate":
            findings = [line for line in out.splitlines()
                        if FINDING.match(line)]
            if not findings:
                problems.append("%s: validate found nothing to say"
                                % (name,))
            expected = EXPECTED.get(name)
            if expected is not None:
                named = set(FINDING.match(line).group(1)
                            for line in findings)
                for rule in expected["must"]:
                    if rule not in named:
                        problems.append(
                            "%s: validate did not name %s; it named %s"
                            % (name, rule, ", ".join(sorted(named))))
                for rule in expected["must_not"]:
                    if rule in named:
                        problems.append(
                            "%s: validate named %s, which is downstream of "
                            "the fault and not the fault" % (name, rule))
        elif code == 1 and not (out.strip() or err.strip()):
            problems.append("%s: %s refused the file and said nothing"
                            % (name, command))


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True,
                        help="the mestra-cli executable")
    parser.add_argument("--dir", default=None,
                        help="the directory of hostile files")
    arguments = parser.parse_args(argv[1:])

    directory = arguments.dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hostile")
    names = sorted(n for n in os.listdir(directory) if n.endswith(".mes"))
    if not names:
        print("no hostile files under %s" % directory)
        return 1

    problems = []
    for name in names:
        check(arguments.cli, os.path.join(directory, name), problems)
    # Two files made here rather than committed: one that is not HDF5 at
    # all, and one cut short. Every reader in every language answers
    # both with E01 -- no `format`, so not a mestra file -- and never
    # with E41, which is for an object inside a file.
    with tempfile.TemporaryDirectory() as made:
        not_hdf5 = os.path.join(made, "not_hdf5.mes")
        with open(not_hdf5, "wb") as fh:
            fh.write(b"garbage")
        with open(os.path.join(directory, names[0]), "rb") as fh:
            whole = fh.read()
        truncated = os.path.join(made, "truncated.mes")
        with open(truncated, "wb") as fh:
            fh.write(whole[: len(whole) // 2])
        for path in (not_hdf5, truncated):
            check(arguments.cli, path, problems)

    print("hostile files        %d" % len(names))
    print("commands run         %d" % (len(names) * 3))
    if problems:
        print("\n%d problems:" % len(problems))
        for problem in problems:
            print("  " + problem)
        return 1
    print("\nevery file answered for, none crashed and none hung")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
