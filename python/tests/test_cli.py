"""The command line: `mestra validate FILE` and `mestra info FILE`.

Section 5 of `docs/api-conventions.md` fixes both shapes, so these
check the lines themselves and not only that something was said.
"""

from __future__ import annotations

from mestra.cli import main
from tests import corpus


def test_version(capsys):
    import pytest

    import mestra
    with pytest.raises(SystemExit) as left:
        main(["--version"])
    assert left.value.code == 0
    assert capsys.readouterr().out == "mestra %s (format %s)\n" % (
        mestra.__version__, mestra.FORMAT)


def test_validate_a_clean_file(capsys):
    status = main(["validate", corpus.case_path("mesh_two_rows")])
    out = capsys.readouterr().out
    assert status == 0
    assert out == "0 error(s), 0 warning(s)\n"


def test_a_finding_is_the_id_the_path_and_the_message(capsys):
    status = main(["validate", corpus.case_path("err_e11")])
    out = capsys.readouterr().out
    assert status == 1
    lines = out.strip().split("\n")
    assert lines[0].startswith("E11 /supports/s0/node_arrays/")
    assert ": a field carries units" in lines[0]
    assert lines[-1] == "1 error(s), 0 warning(s)"


def test_validate_reports_warnings_and_still_passes(capsys):
    status = main(["validate", corpus.case_path("warn_w05")])
    out = capsys.readouterr().out
    assert status == 0
    assert "W05 /supports: " in out
    assert out.strip().split("\n")[-1] == "0 error(s), 1 warning(s)"


def test_validate_quietly(capsys):
    main(["validate", "--quiet", corpus.case_path("warn_w05")])
    out = capsys.readouterr().out
    assert out == "0 error(s), 1 warning(s)\n"


def test_validate_several_files(capsys):
    """One summary counts the run; each file's findings are named."""
    status = main(["validate", corpus.case_path("mesh_two_rows"),
                   corpus.case_path("err_e11")])
    out = capsys.readouterr().out
    assert status == 1
    assert out.count("case.mes\n") == 2
    assert out.strip().split("\n")[-1] == "1 error(s), 0 warning(s)"


def test_validate_a_file_that_is_not_there(capsys):
    assert main(["validate", "/no/such/file.mes"]) == 1
    out = capsys.readouterr().out
    assert "E01" in out
    assert "no file at this path" in out
    assert out.strip().split("\n")[-1] == "1 error(s), 0 warning(s)"


def test_a_per_row_rule_is_reported_once_with_a_count(capsys):
    """Section 5: W02, W03 and W04 report once, with the rows."""
    main(["validate", corpus.case_path("cascade_varying_geometry")])
    out = capsys.readouterr().out
    lines = [line for line in out.split("\n") if line.startswith("W02")]
    assert len(lines) == 1
    assert "at row 3" in lines[0]
    assert "converged, failed, partial" in lines[0]
    assert out.strip().split("\n")[-1] == "0 error(s), 4 warning(s)"


def test_a_leaking_split_names_the_units_the_file_names_them(capsys):
    """P12: the file knows these are wing_a and wing_b."""
    main(["validate", corpus.case_path("warn_w01")])
    out = capsys.readouterr().out
    lines = [line for line in out.split("\n") if line.startswith("W01")]
    assert len(lines) == 1
    assert "wing_a, wing_b" in lines[0]
    assert "generalisation test" in lines[0]


def test_info_is_one_screen(capsys):
    status = main(["info", corpus.case_path("mesh_two_rows")])
    out = capsys.readouterr().out
    assert status == 0
    lines = out.strip().split("\n")
    assert len(lines) <= 24
    assert "mestra/0" in out
    assert "2 row(s), aligned, generalisation unit member" in out
    assert "mach" in out and "condition" in out
    assert "bounds [0.1, 0.9]" in out
    assert "categories member (wing_a, wing_b)" in out
    assert "96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c9" \
        "43936c7" in out
    assert "(row, node, component)" in out
    assert "units Pa" in out
    assert "data" in out


def test_info_gives_every_slot_a_shape_under_named_axes(capsys):
    main(["info", corpus.case_path("mesh_two_rows")])
    out = capsys.readouterr().out
    assert "(group:member, node, component) 2x6x2" in out
    assert "(row, node, component) 2x6x1" in out
    assert "(cell, component) 2x1" in out
    assert "(row) 2" in out


def test_info_says_no_support_when_there_is_none(capsys):
    """P13: aligned with what is a fair question."""
    main(["info", corpus.case_path("scalars_only")])
    out = capsys.readouterr().out
    assert "no support" in out
    assert "aligned" not in out


def test_info_names_the_callables(capsys):
    main(["info", corpus.case_path("affine_zero_rows")])
    out = capsys.readouterr().out
    assert "m1" in out and "affine" in out
    assert "callable:m1" in out
    assert "output pressure" in out


def test_info_on_an_unreadable_file(capsys):
    assert main(["info", corpus.case_path("err_e01")]) == 1
    assert "E01" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out


def test_a_reader_that_stops_early_is_not_a_file_that_cannot_be_read():
    """`mestra info f | head`: the pipe closes before the output is
    written. That says nothing about the file, so there is no
    traceback and no "cannot be read", in either command."""
    import subprocess
    import sys

    run = ("import sys; from mestra.cli import main; "
           "sys.exit(main(sys.argv[1:]))")
    for command in ("info", "validate"):
        child = subprocess.Popen(
            [sys.executable, "-c", run, command,
             corpus.case_path("mesh_two_rows")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        child.stdout.close()        # gone before the child has written
        err = child.stderr.read().decode()
        child.stderr.close()
        child.wait(timeout=120)
        assert "Traceback" not in err, err
        assert "cannot be read" not in err
