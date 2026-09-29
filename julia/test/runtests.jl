# The conformance corpus, run in this implementation's own test suite
# (SPEC.md section 15).  No implementation is the reference: what this
# file checks is the specification and the golden files under
# vectors/, and nothing else.

using Test
using Mestra
using JSON3
using NCDatasets
using HDF5

const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", ".."))
const CASES = joinpath(REPO, "vectors", "cases")
const SCRATCH = mktempdir()

case_names() = sort(readdir(CASES))
case_file(name) = joinpath(CASES, name, "case.mes")
expected(name) = JSON3.read(read(joinpath(CASES, name, "expected.json"),
                                String))

"""Section 30: a float in expected.json is the C format %.17e, and a
comparison parses it to float64 and requires bit equality."""
function parse_expected(s::AbstractString)
    s == "nan" && return NaN
    s == "inf" && return Inf
    s == "-inf" && return -Inf
    return parse(Float64, s)
end

bitequal(a::Float64, b::Float64) =
    reinterpret(UInt64, a) == reinterpret(UInt64, b)

"""One probe, read back through the public reader."""
function probe_value(ds::Mestra.Dataset, p)
    slot = String(p.slot)
    parts = split(slot, '/'; keepempty = false)
    idx = Dict{Symbol,Int}()
    for (json, name) in (("row", :row), ("instance", :instance),
                         ("draw", :draw), ("node", :node),
                         ("component", :component))
        haskey(p, json) && (idx[name] = Int(p[json]) + 1)
    end
    if parts[1] == "keys"
        return Mestra.values(ds, ds.keys[parts[2]])[idx[:row]]
    elseif parts[1] == "scalars"
        return Mestra.values(ds, ds.scalars[parts[2]])[idx[:row]]
    elseif parts[1] == "row_support"
        return ds.row_support[idx[:row]]
    elseif parts[1] == "callables"
        v = ds.callables[parts[2]].dict
        for k in parts[3:end]
            v = v[k]
        end
        haskey(p, "component") && return v[idx[:node], idx[:component]]
        haskey(p, "node") && return v[idx[:node]]
        haskey(p, "index") && return v[Int(p["index"]) + 1]
        return v[]
    elseif parts[1] == "supports"
        sup = ds.supports[Mestra.support_by_name(ds, parts[2])]
        if parts[3] == "cell_types"
            return sup.cell_types[Int(p["cell"]) + 1]
        elseif parts[3] == "cell_offsets"
            return sup.cell_offsets[Int(p["cell_plus_one"]) + 1]
        elseif parts[3] == "cell_connectivity"
            return sup.cell_connectivity[Int(p["index"]) + 1]
        elseif parts[3] == "coordinates"
            return Mestra.at(Mestra.values(ds, sup.coordinates); idx...)
        elseif parts[3] == "node_arrays"
            return Mestra.at(Mestra.values(ds, sup.node_arrays[parts[4]]);
                             idx...)
        elseif parts[3] == "cell_arrays"
            return Mestra.at(Mestra.values(ds, sup.cell_arrays[parts[4]]);
                             idx...)
        end
    end
    error("a probe path this test does not know: $(slot)")
end

function check_probe(ds, p)
    got = probe_value(ds, p)
    want = String(p.value)
    if got isa AbstractFloat
        return bitequal(Float64(got), parse_expected(want))
    end
    return string(Int(got)) == want
end

# --------------------------------------- the tagged form of section 30

cflat(a::AbstractArray) =
    ndims(a) <= 1 ? collect(a) :
    vec(permutedims(collect(a), ntuple(i -> ndims(a) - i + 1, ndims(a))))

function tagged_ok(got, want)
    t = String(want["t"])
    if t == "dict"
        got isa AbstractDict || return false
        Set(String.(keys(got))) == Set(String.(keys(want["v"]))) || return false
        return all(tagged_ok(got[String(k)], v) for (k, v) in want["v"])
    elseif t == "null"
        return got === nothing
    elseif t == "bool"
        return got isa Bool && got == want["v"]
    elseif t == "i64"
        return got isa Integer && !(got isa Bool) && Int64(got) == want["v"]
    elseif t == "f64"
        return got isa AbstractFloat &&
               bitequal(Float64(got), parse_expected(String(want["v"])))
    elseif t == "str"
        return got isa AbstractString && String(got) == String(want["v"])
    elseif t == "strings"
        got isa AbstractArray || return false
        collect(size(got)) == Int.(want["shape"]) || return false
        return all(String(a) == String(b)
                   for (a, b) in zip(cflat(got), want["data"]))
    elseif t == "array"
        got isa AbstractArray || return false
        collect(size(got)) == Int.(want["shape"]) || return false
        dtype = String(want["dtype"])
        flat = cflat(got)
        if dtype == "float64"
            eltype(got) === Float64 || return false
            return all(bitequal(Float64(a), parse_expected(String(b)))
                       for (a, b) in zip(flat, want["data"]))
        elseif dtype == "int64"
            eltype(got) === Int64 || return false
        elseif dtype == "int32"
            eltype(got) === Int32 || return false
        elseif dtype == "bool"
            eltype(got) === Bool || return false
        else
            return false
        end
        return all(a == b for (a, b) in zip(flat, want["data"]))
    end
    return false
end

# ---------------------------------------------------------- the tests

function refusal(f)
    try
        f()
        return nothing
    catch e
        e isa Mestra.MestraError || rethrow()
        return e
    end
end

@testset "Mestra" begin

@testset "units parser (W10)" begin
    for u in ("1", "m", "m2", "s", "K", "Pa", "W", "W m-2", "m2 s-1",
              "degree", "kg m-3", "m s-1", "J kg-1 K-1", "1e-3 m",
              "kg/(m s2)", "m^3", "%")
        @test Mestra.parse_units(u)
    end
    for u in ("kg/(m s", "", "m//s", "()", "m (")
        @test !Mestra.parse_units(u)
    end
end

@testset "support_id worked examples (section 24)" begin
    @test Mestra.support_id(6; cell_types = UInt8[9, 9],
                            cell_offsets = Int64[0, 4, 8],
                            cell_connectivity =
                                Int64[0, 1, 4, 3, 1, 2, 5, 4]) ==
          "96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c943936c7"
    @test Mestra.support_id(4;
                            axis_coordinates = [0.0, 0.5, 1.0, 1.5]) ==
          "57467fe7370808bdb0ad01b95d963f59e8bc6762f90f96049453ae33bb05a54c"
    @test Mestra.support_id(0) ==
          "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
end

@testset "corpus: validator outcomes" begin
    for name in case_names()
        e = expected(name)
        r = Mestra.validate(case_file(name))
        @test r.errors == sort(String.(e.validator.errors))
        @test r.warnings == sort(String.(e.validator.warnings))
    end
end

@testset "corpus: support ids" begin
    for name in case_names()
        e = expected(name)
        isempty(e.support_ids) && continue
        # the corpus carries files that break a structural rule on
        # purpose, and probing one is a non-strict read
        ds = Mestra.read(case_file(name); strict = false)
        for (sname, want) in e.support_ids
            i = Mestra.support_by_name(ds, String(sname))
            @test i !== nothing
            @test Mestra.support_id(ds.supports[i]) == String(want)
        end
    end
end

@testset "corpus: probes" begin
    n = 0
    for name in case_names()
        e = expected(name)
        isempty(e.probes) && continue
        # the corpus carries files that break a structural rule on
        # purpose, and probing one is a non-strict read
        ds = Mestra.read(case_file(name); strict = false)
        for p in e.probes
            @test check_probe(ds, p)
            n += 1
        end
    end
    @test n > 0
    @info "probes checked" n
end

@testset "corpus: codec round trip" begin
    n = 0
    for name in case_names()
        e = expected(name)
        isempty(e.codec) && continue
        # the corpus carries files that break a structural rule on
        # purpose, and probing one is a non-strict read
        ds = Mestra.read(case_file(name); strict = false)
        for (id, want) in e.codec
            @test tagged_ok(ds.callables[String(id)].dict, want)
            n += 1
        end
    end
    @test n > 0
    @info "codec round trips checked" n
end

@testset "corpus: evaluation" begin
    n = 0
    for name in case_names()
        e = expected(name)
        isempty(e.evaluation) && continue
        # the corpus carries files that break a structural rule on
        # purpose, and probing one is a non-strict read
        ds = Mestra.read(case_file(name); strict = false)
        for ev in e.evaluation
            table = Dict{String,Vector{Float64}}(
                String(k) => [parse_expected(String(x)) for x in v]
                for (k, v) in ev.keys)
            out = Mestra.evaluate(ds, table)
            for p in ev.probes
                @test check_probe(out, p)
                n += 1
            end
        end
    end
    @test n > 0
    @info "evaluation probes checked" n
end

@testset "corpus: read, write, compare (section 30)" begin
    equal = 0
    identical = 0
    total = 0
    for name in case_names()
        isempty(expected(name).validator.errors) || continue
        total += 1
        src = case_file(name)
        dst = joinpath(SCRATCH, name * ".mes")
        ds = Mestra.read(src; lazy = false)
        Mestra.write(ds, dst)
        diff = Mestra.structural_diff(src, dst)
        isempty(diff) || @info "structural difference" name diff
        @test isempty(diff)
        isempty(diff) && (equal += 1)
        read(src) == read(dst) && (identical += 1)
        # what was written must still validate the same way
        @test Mestra.validate(dst).errors == String[]
    end
    @info "read-write-compare" total equal identical
end

@testset "corpus: a lazy read touches no array" begin
    ds = Mestra.read(case_file("mesh_two_rows"))
    @test ds.nrows == 2
    @test !Mestra.materialised(ds["pressure"])
    @test ds.supports[1].support_id[1:8] == "96df395d"
    # one slot, one row range, and nothing else
    v = Mestra.rows(ds, ds["pressure"], 2:2)
    @test size(v) == (1, 6, 1)
    @test Mestra.at(v; row = 1, node = 4, component = 1) == 204.0
    @test !Mestra.materialised(ds["pressure"])
    big = Mestra.read(case_file("cascade_varying_geometry"))
    w = Mestra.rows(big, big["mach"], 2:3)
    @test size(w) == (1, 6, 2)
    full = Mestra.permute(Mestra.values(big, big["mach"]),
                          (:row, :node, :component))
    @test Mestra.permute(w, (:row, :node, :component))[1, :, 1] ==
          full[2, :, 1]
    # and reaching for the empty field says what to call instead,
    # rather than handing back `nothing` for the next line to trip on
    e = try
        ds.keys["mach"].values[2]
    catch err
        err
    end
    @test e isa Mestra.MestraError
    @test occursin("Mestra.values", sprint(showerror, e))
    @test occursin("/keys/mach", sprint(showerror, e))
    e2 = try
        ds["pressure"].data
    catch err
        err
    end
    @test e2 isa Mestra.MestraError
    @test occursin("Mestra.values", sprint(showerror, e2))
    eager = Mestra.read(case_file("mesh_two_rows"); lazy = false)
    @test Mestra.materialised(eager.keys["mach"])
    @test eager.keys["mach"].values[2] == 0.8
end

@testset "axis order and permuting by name" begin
    ds = Mestra.read(case_file("mesh_two_rows"); lazy = false)
    v = Mestra.values(ds, ds["pressure"])
    # Julia is column major and the file is C order, so the axes are
    # reversed: (component, node, row) for a (row, node, component)
    # array.  The names come with it, so nothing counts axes.
    @test Mestra.dimnames(v) == (:component, :node, :row)
    @test size(v) == (1, 6, 2)
    p = Mestra.permute(v, (:row, :node, :component))
    @test Mestra.dimnames(p) == (:row, :node, :component)
    @test p[2, 4, 1] == 204.0
    @test Mestra.at(v; row = 2, node = 4, component = 1) == 204.0
    # docs/example.md: 105 is what a reader that took the axes by
    # position would return
    @test p[2, 4, 1] != 105.0
    c = Mestra.values(ds, ds.supports[1].coordinates)
    @test Mestra.dimnames(c) == (:component, :node, Symbol("group:member"))
    @test Mestra.at(c; instance = 2, node = 3, component = 1) == 3.0
    @test_throws Mestra.MestraError Mestra.permute(v, (:row, :node))
    @test_throws Mestra.MestraError Mestra.at(v; row = 1, cheese = 1)
end

@testset "dimension names come from the link, not from NAME" begin
    ds = Mestra.read(case_file("mesh_two_rows"))
    @test ds["pressure"].ldims == [:row, :node, :component]
    @test ds.supports[1].coordinates.ldims ==
          [Symbol("group:member"), :node, :component]
    # NAME is the same sentence in every scale; a reader that took the
    # dimension name from it would find them all called the same thing
    HDF5.h5open(case_file("mesh_two_rows"), "r") do f
        a = read(attributes(f["row"])["NAME"])
        b = read(attributes(f["component_1"])["NAME"])
        @test startswith(a, "This is a netCDF dimension")
        @test a[1:53] == b[1:53]
    end
end

@testset "the affine callable (section 27)" begin
    ds = Mestra.read(case_file("affine_zero_rows"))
    c = Mestra.callable(ds, "m1")
    @test c isa Mestra.Affine
    @test c.keys == ["mach", "alpha"]
    @test sprint(show, c) == "affine(mach, alpha -> cl, pressure)"
    out = c(Dict("mach" => [0.5], "alpha" => [4.0]))
    @test bitequal(out["cl"].mean[1], 1.45)
    @test !Mestra.has_uncertainty(out["cl"])
    want = [0.5, 1.1, 3.7, 4.3, 6.9, 7.5]
    got = vec(out["pressure"].mean)
    @test all(bitequal(a, b) for (a, b) in zip(got, want))
    # b added last, no fused multiply-add: accumulating b first differs
    # in the last place
    @test bitequal(out["cl"].mean[1], 2.0 * 0.5 + 0.1 * 4.0 + 0.05)
    # the other two accepted keys tables of section 26
    out2 = c((mach = [0.5], alpha = [4.0]))
    @test bitequal(out2["cl"].mean[1], 1.45)
    out3 = c(([4.0 0.5], ["alpha", "mach"]))
    @test bitequal(out3["cl"].mean[1], 1.45)
    @test_throws Mestra.MestraError c(Dict("mach" => [0.5]))
end

@testset "a prediction is a mean and at most a band (section 10)" begin
    plain = Mestra.Prediction([1.0, 2.0])
    @test plain.uncertainty === nothing && plain.level === nothing
    banded = Mestra.Prediction([1.0, 2.0]; uncertainty = [0.1, 0.2],
                               level = 0.95, method = "m")
    @test banded.level == 0.95 && Mestra.has_uncertainty(banded)
    @test_throws Mestra.MestraError Mestra.Prediction(
        [1.0, 2.0]; uncertainty = [0.1, 0.2], method = "m")
    @test_throws Mestra.MestraError Mestra.Prediction(
        [1.0, 2.0]; uncertainty = [0.1, 0.2], level = 1.96, method = "m")
    @test_throws Mestra.MestraError Mestra.Prediction(
        [1.0, 2.0]; uncertainty = [0.1], level = 0.95, method = "m")
    @test_throws Mestra.MestraError Mestra.Prediction(
        [1.0, 2.0]; uncertainty = [-0.1, 0.2], level = 0.95, method = "m")
    @test_throws Mestra.MestraError Mestra.Prediction([1.0, 2.0]; level = 0.95)

    # section 27: a constant band on an affine output, the same in
    # every row, all three of uncertainty, level and method or none
    c = Mestra.affine(["mach"], Dict(
        "cl" => (A = [2.0], b = [0.05], shape = Int64[],
                 uncertainty = [0.02], level = 0.95, method = "constant band"),
        "p" => (A = reshape([1.0, 2.0], 2, 1), b = [0.0, 0.1],
                shape = Int64[2, 1], uncertainty = [0.5, 0.6], level = 0.68,
                method = "constant band")))
    out = c(Dict("mach" => [0.5, 0.7]))
    @test out["cl"].uncertainty == [0.02, 0.02]
    @test out["cl"].level == 0.95 && out["cl"].method == "constant band"
    @test size(out["p"].uncertainty) == (1, 2, 2)
    @test out["p"].uncertainty[1, 2, 2] == 0.6
    @test bitequal(out["cl"].mean[1], 1.05)
    back = Mestra.from_dict(Mestra.Affine, Mestra.to_dict(c))
    @test Mestra.to_dict(back)["outputs"]["cl"]["level"] == 0.95
    @test Mestra.to_dict(back)["outputs"]["p"]["method"] == "constant band"
    @test_throws Mestra.MestraError Mestra.affine(["mach"], Dict(
        "cl" => (A = [2.0], b = [0.05], shape = Int64[], level = 0.95)))
end

@testset "a band slot takes the uncertainty and its level (section 10)" begin
    ds = Mestra.read(case_file("affine_band"))
    out = Mestra.evaluate(ds, Dict("mach" => [0.1, 0.9],
                                   "alpha" => [-2.0, 10.0]))
    band = out.scalars["cl_band"]
    @test band.source == "data" && band.statistic == "band"
    @test band.level == 0.95 && band.method == "constant band"
    @test Mestra.values(out, band) == [0.02, 0.02]
    pb = out["pressure_band"]
    @test pb.level == 0.68
    v = Mestra.permute(Mestra.values(out, pb), (:row, :node, :component))
    @test v[2, 6, 1] == 0.3
    @test out.scalars["cl"].level === nothing
    path = joinpath(SCRATCH, "banded.mes")
    Mestra.write(out, path)
    r = Mestra.validate(path)
    @test r.errors == String[] && r.warnings == String[]
    back = Mestra.read(path)
    @test back.scalars["cl_band"].level == 0.95
    @test back.scalars["cl_band"].method == "constant band"

    # a band slot served by an output with no uncertainty cannot be
    # filled: the file would hold a band with no level
    m = Mestra.Dataset()
    Mestra.add_key!(m, "mach", Float64[]; role = :condition, units = "1",
                    bounds = (0.1, 0.9))
    Mestra.add_callable!(m, "m1", Mestra.affine(["mach"], Dict(
        "cl" => (A = [2.0], b = [0.05], shape = Int64[]))))
    Mestra.add_callable_scalar!(m, "cl"; units = "1", callable = "m1",
                                output = "cl")
    Mestra.add_callable_scalar!(m, "cl_band"; units = "1", callable = "m1",
                                output = "cl", statistic = "band", of = "cl")
    e = refusal(() -> Mestra.evaluate(m, Dict("mach" => [0.5])))
    @test e !== nothing && e.rule == "E12"
end

@testset "prediction: one question of a stored slot and of a served one" begin
    ds = Mestra.read(case_file("band_stored"))
    r = Mestra.prediction(ds, ds["pressure"])
    @test size(r.mean) == (1, 6, 2)
    @test r.uncertainty[1, 4, 2] == 0.8
    @test r.level == 0.95
    @test startswith(r.method, "half-width of a 95 % interval")
    @test_throws Mestra.MestraError Mestra.prediction(ds, ds["pressure_band"])
    @test_throws Mestra.MestraError Mestra.prediction(
        ds, ds["pressure"]; keys = Dict("mach" => [0.5]))
    two = Mestra.read(case_file("mesh_two_rows"))
    plain = Mestra.prediction(two, two.scalars["cl"])
    @test plain.mean == [0.25, 0.55] && plain.uncertainty === nothing
    m = Mestra.read(case_file("affine_band"))
    served = Mestra.prediction(m, m.scalars["cl"];
                               keys = Dict("mach" => [0.5], "alpha" => [4.0]))
    @test bitequal(served.mean[1], 1.45)
    @test served.uncertainty == [0.02] && served.level == 0.95
    @test_throws Mestra.MestraError Mestra.prediction(m, m.scalars["cl"])
    rows = Mestra.read(case_file("affine_with_rows"))
    @test size(Mestra.prediction(rows, rows["pressure"]).mean) == (1, 6, 2)
    @test size(Mestra.prediction(rows, "pressure").mean) == (1, 6, 2)
    draws = Mestra.read(case_file("draws_and_summaries"))
    @test_throws Mestra.MestraError Mestra.prediction(draws, "pressure_std")
end

@testset "a band states its level and method at build time (E12)" begin
    ds = Mestra.Dataset()
    Mestra.add_key!(ds, "mach", [0.4, 0.8]; role = :condition, units = "1")
    Mestra.add_scalar!(ds, "cl", [0.25, 0.55]; units = "1")
    e = refusal(() -> Mestra.add_scalar!(ds, "cl_band", [0.1, 0.1];
                                         units = "1", statistic = "band"))
    @test e !== nothing && e.rule == "E12"
    e = refusal(() -> Mestra.add_scalar!(ds, "cl_band", [0.1, 0.1];
                                         units = "1", statistic = "band",
                                         of = "cl", level = 1.96,
                                         method = "m"))
    @test e !== nothing && e.rule == "E12"
    e = refusal(() -> Mestra.add_scalar!(ds, "cl_mean", [0.1, 0.1];
                                         units = "1", level = 0.95))
    @test e !== nothing && e.rule == "E12"
    Mestra.add_scalar!(ds, "cl_band", [0.1, 0.1]; units = "1",
                       statistic = "band", of = "cl", level = 0.95,
                       method = "m")
    @test ds.scalars["cl_band"].level == 0.95
    Mestra.add_callable!(ds, "m1", Mestra.affine(["mach"], Dict(
        "cl" => (A = [2.0], b = [0.05], shape = Int64[]))))
    e = refusal(() -> Mestra.add_callable_scalar!(
        ds, "cl_draws"; units = "1", callable = "m1", output = "cl",
        statistic = "draw"))
    @test e !== nothing && e.rule == "E12"
    s = Mestra.add_callable_scalar!(ds, "cl_served"; units = "1",
                                    callable = "m1", output = "cl",
                                    statistic = "band", of = "cl")
    @test s.level === nothing
end

@testset "evaluate produces a dataset that can be written" begin
    ds = Mestra.read(case_file("affine_zero_rows"))
    out = Mestra.evaluate(ds, Dict("mach" => [0.5], "alpha" => [4.0]))
    @test out.nrows == 1
    @test out.scalars["cl"].source == "data"
    @test out.supports[1].node_arrays["pressure"].source == "data"
    path = joinpath(SCRATCH, "evaluated.mes")
    Mestra.write(out, path)
    r = Mestra.validate(path)
    @test r.errors == String[]
    back = Mestra.read(path; lazy = false)
    @test bitequal(Mestra.values(back, back.scalars["cl"])[1], 1.45)
    v = Mestra.permute(Mestra.values(back, back["pressure"]),
                       (:row, :node, :component))
    @test all(bitequal(v[1, i, 1], w)
              for (i, w) in pairs([0.5, 1.1, 3.7, 4.3, 6.9, 7.5]))
    # the support is the same one, and a tool knows it from one
    # attribute
    @test back.supports[1].support_id == ds.supports[1].support_id
end

@testset "the dictionary codec (sections 17 and 25)" begin
    d = Dict{String,Any}(
        "an_int" => Int64(42),
        "a_float" => 42.0,
        "a_bool" => true,
        "a_string" => "skål",
        "nothing_at_all" => nothing,
        "numbers" => [1.0, 2.0, 3.0],
        "ints" => Int32[1, 2, 3],
        "flags" => [true, false],
        "matrix" => [1.0 2.0; 3.0 4.0; 5.0 6.0],
        "names" => ["mach", "alpha"],
        "empty" => Float64[],
        "nested" => Dict{String,Any}("type" => "allowed here",
                                     "repr" => "and here"))
    path = joinpath(SCRATCH, "codec.mes")
    HDF5.h5open(path, "w") do f
        g = Mestra.create_group(f, "g")
        Mestra.write_dict(g, d)
    end
    back = HDF5.h5open(path, "r") do f
        Mestra.read_dict(f["g"])
    end
    @test Set(keys(back)) == Set(keys(d))
    @test back["an_int"] === Int64(42)
    @test back["a_float"] === 42.0
    @test back["a_bool"] === true
    @test back["a_string"] == "skål"
    @test back["nothing_at_all"] === nothing
    @test back["numbers"] == [1.0, 2.0, 3.0]
    @test eltype(back["ints"]) === Int32
    @test back["flags"] == [true, false]
    @test eltype(back["flags"]) === Bool
    @test back["matrix"] == [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test back["names"] == ["mach", "alpha"]
    @test size(back["empty"]) == (0,) && eltype(back["empty"]) === Float64
    @test back["nested"]["type"] == "allowed here"
    # a zero-dimensional array is written as the number it holds
    HDF5.h5open(joinpath(SCRATCH, "zerod.mes"), "w") do f
        g = Mestra.create_group(f, "g")
        Mestra.write_dict(g, Dict{String,Any}("t" => fill(1.5)))
    end
    z = HDF5.h5open(joinpath(SCRATCH, "zerod.mes"), "r") do f
        Mestra.read_dict(f["g"])
    end
    @test z["t"] === 1.5
    # what the codec must refuse
    HDF5.h5open(joinpath(SCRATCH, "refuse.mes"), "w") do f
        g = Mestra.create_group(f, "g")
        @test_throws Mestra.MestraError Mestra.write_dict(g,
            Dict{String,Any}("x" => Float32[1.0]))
        @test_throws Mestra.MestraError Mestra.write_dict(g,
            Dict{String,Any}("mestra_x" => 1))
        @test_throws Mestra.MestraError Mestra.write_dict(g,
            Dict{String,Any}("x" => "a\0b"))
    end
end

@testset "building a dataset from arrays" begin
    ds = Mestra.Dataset(writer = "mestra.jl test 0",
                        created = "2026-09-19T00:00:00Z")
    Mestra.add_category_table!(ds, "member", ["wing_a", "wing_b"])
    Mestra.add_key!(ds, "mach", [0.4, 0.8]; role = :condition, units = "1")
    Mestra.add_key!(ds, "member", [0, 1]; role = :group,
                    category = "member")
    Mestra.add_scalar!(ds, "cl", [0.25, 0.55]; units = "1")
    coords = cat([0.0 0.0; 1.0 0.0; 2.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 1.0],
                 [0.0 0.0; 1.5 0.0; 3.0 0.0; 0.0 1.0; 1.5 1.0; 3.0 1.0];
                 dims = 3)
    coords = permutedims(coords, (3, 1, 2))     # (instance, node, comp)
    s = Mestra.add_mesh_support!(ds, "s0";
            coordinates = coords,
            dims = (Symbol("group:member"), :node, :component),
            cell_types = UInt8[9, 9], cell_offsets = Int64[0, 4, 8],
            cell_connectivity = Int64[0, 1, 4, 3, 1, 2, 5, 4])
    # a (row, node) matrix: the component axis is added for you
    Mestra.add_node_array!(ds, s, "pressure",
                           [101.0 102 103 104 105 106;
                            201.0 202 203 204 205 206]; units = "Pa")
    Mestra.add_cell_array!(ds, s, "region", [0, 1];
                           role = :label, dims = (:cell,),
                           category = "region")
    Mestra.add_category_table!(ds, "region", ["inlet", "outlet"])
    # the bounds were filled in from the data
    @test ds.keys["mach"].lower == 0.4
    @test ds.keys["mach"].upper == 0.8
    # and the support id from the cells
    @test s.support_id ==
          "96df395d80ef548444562292de441525ba0b5c8ad00a8dadff19a19c943936c7"
    path = joinpath(SCRATCH, "built.mes")
    Mestra.write(ds, path)
    r = Mestra.validate(path)
    @test r.errors == String[]
    @test r.warnings == String[]
    back = Mestra.read(path; lazy = false)
    @test back.nrows == 2
    @test Mestra.at(Mestra.values(back, back["pressure"]);
                    row = 2, node = 4, component = 1) == 204.0
    @test Mestra.at(Mestra.values(back, back.supports[1].coordinates);
                    instance = 2, node = 3, component = 1) == 3.0
end

"""A dataset with one support of six nodes and two quads, to build
bad arrays on."""
function six_node_dataset()
    ds = Mestra.Dataset(writer = "mestra.jl test 0",
                        created = "2026-09-19T00:00:00Z")
    Mestra.add_category_table!(ds, "member", ["wing_a", "wing_b"])
    Mestra.add_key!(ds, "mach", [0.4, 0.8]; role = :condition, units = "1")
    Mestra.add_key!(ds, "member", [0, 1]; role = :group,
                    category = "member")
    s = Mestra.add_mesh_support!(ds, "s0";
            coordinates = [0.0 0.0; 1.0 0.0; 2.0 0.0;
                           0.0 1.0; 1.0 1.0; 2.0 1.0],
            dims = (:node, :component),
            cell_types = UInt8[9, 9], cell_offsets = Int64[0, 4, 8],
            cell_connectivity = Int64[0, 1, 4, 3, 1, 2, 5, 4])
    return (ds, s)
end

"""The MestraError a call raises, or nothing."""

@testset "the builder follows dims, and refuses what disagrees" begin
    ds, s = six_node_dataset()
    # `dims` decides `varies`, and a `varies` that says otherwise is
    # refused at build time rather than written as an invalid file
    e = refusal(() -> Mestra.add_node_array!(ds, s, "p",
            [101.0 102 103 104 105 106; 201.0 202 203 204 205 206];
            units = "Pa", dims = (:row, :node), varies = "none"))
    @test e !== nothing && e.rule == "E04"
    @test occursin("varies", e.msg) && occursin("dims", e.msg)
    @test occursin("/supports/s0/node_arrays/p", sprint(showerror, e))
    # :instance does not say which group it is
    e = refusal(() -> Mestra.add_node_array!(ds, s, "p", rand(2, 6, 1);
            units = "Pa", dims = (:instance, :node, :component)))
    @test e !== nothing && e.rule == "E04"
    @test occursin("group:", e.msg)
    # and a group key the dataset does not declare is refused
    e = refusal(() -> Mestra.add_node_array!(ds, s, "p", rand(2, 6, 1);
            units = "Pa", dims = (:instance, :node, :component),
            varies = "group:nosuch"))
    @test e !== nothing && e.rule == "E04"
    @test occursin("nosuch", e.msg)
    # a square array is two readings and the builder will not choose
    e = refusal(() -> Mestra.add_node_array!(ds, s, "p", rand(6, 6);
                                             units = "Pa"))
    @test e !== nothing && e.rule == "E04"
    @test occursin("(node, component)", e.msg) &&
          occursin("(row, node)", e.msg) && occursin("dims", e.msg)
    # `components` follows from the component axis
    e = refusal(() -> Mestra.add_node_array!(ds, s, "p", rand(2, 6, 3);
            units = "Pa", dims = (:row, :node, :component),
            components = 2))
    @test e !== nothing && e.rule == "E31"
    # the ones that agree are built, and `varies` may be said again
    a = Mestra.add_node_array!(ds, s, "p", rand(2, 6); units = "Pa",
                               dims = (:row, :node), varies = "row")
    @test a.varies == "row" && a.components == 1
    b = Mestra.add_node_array!(ds, s, "q", rand(2, 6, 2); units = "Pa",
                               dims = (:instance, :node, :component),
                               varies = "group:member")
    @test b.varies == "group:member" && b.components == 2
    c = Mestra.add_node_array!(ds, s, "t", collect(range(0, 1, 6));
                               units = "1")
    @test c.varies == "none" && c.components == 1
end

@testset "bounds, and the unit of generalisation" begin
    ds, _ = six_node_dataset()
    # the observed finite range unless the caller says otherwise
    @test ds.keys["mach"].lower == 0.4 && ds.keys["mach"].upper == 0.8
    Mestra.add_key!(ds, "alpha", [1.0, 3.0]; role = :condition,
                    units = "degree", lower = -2.0, upper = 10.0)
    @test ds.keys["alpha"].lower == -2.0 && ds.keys["alpha"].upper == 10.0
    Mestra.add_key!(ds, "beta", [1.0, 3.0]; role = :condition,
                    units = "degree", lower = nothing, upper = nothing)
    @test ds.keys["beta"].lower === nothing
    e = refusal(() -> Mestra.add_key!(ds, "gamma", [1.0, 3.0];
                                      role = :condition, units = "degree",
                                      lower = 0.0))
    @test e !== nothing && e.rule == "E19"
    # a non-finite value is not a bound
    Mestra.add_scalar!(ds, "cl", [0.25, NaN]; units = "1")
    Mestra.add_key!(ds, "delta", [1.0, NaN]; role = :condition, units = "1")
    @test ds.keys["delta"].lower == 1.0 && ds.keys["delta"].upper == 1.0

    # the unit of generalisation is a dataset property
    @test ds.generalisation_group == "member"       # the first group key
    Mestra.add_category_table!(ds, "batch", ["b0", "b1"])
    Mestra.add_key!(ds, "batch", [0, 1]; role = :group, category = "batch")
    @test ds.generalisation_group == "member"
    Mestra.set_generalisation_group!(ds, "batch")
    @test ds.generalisation_group == "batch"
    e = refusal(() -> Mestra.set_generalisation_group!(ds, "mach"))
    @test e !== nothing && e.rule == "E02"
    @test refusal(() -> Mestra.set_generalisation_group!(ds, "nope")) !==
          nothing
end

@testset "building a model file with a callable slot" begin
    ds = Mestra.Dataset(writer = "mestra.jl test 0",
                        created = "2026-09-19T00:00:00Z")
    Mestra.add_key!(ds, "mach", Float64[]; role = :condition, units = "1",
                    bounds = (0.1, 0.9))
    Mestra.add_key!(ds, "alpha", Float64[]; role = :condition,
                    units = "degree", bounds = (-2.0, 10.0))
    s = Mestra.add_mesh_support!(ds, "s0";
            coordinates = [0.0 0.0; 1.0 0.0; 2.0 0.0;
                           0.0 1.0; 1.0 1.0; 2.0 1.0],
            dims = (:node, :component),
            cell_types = UInt8[9, 9], cell_offsets = Int64[0, 4, 8],
            cell_connectivity = Int64[0, 1, 4, 3, 1, 2, 5, 4])
    c = Mestra.affine(["mach", "alpha"], Dict(
        "cl" => (A = [2.0 0.1], b = [0.05], shape = Int64[]),
        "pressure" => (A = [1.0 0.0; 2.0 0.0; 3.0 0.5;
                            4.0 0.5; 5.0 1.0; 6.0 1.0],
                       b = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                       shape = Int64[6, 1])))
    Mestra.add_callable!(ds, "m1", c)
    # `callable` names it, and `id` is the older spelling of the same
    # argument; neither is refused with the rule it breaks
    Mestra.add_callable_scalar!(ds, "cl"; units = "1", callable = "m1",
                                output = "cl")
    Mestra.add_callable_slot!(ds, s, "pressure"; units = "Pa",
                              components = 1, id = "m1",
                              output = "pressure")
    e = refusal(() -> Mestra.add_callable_scalar!(ds, "cd"; units = "1",
                                                  output = "cd"))
    @test e !== nothing && e.rule == "E14"
    path = joinpath(SCRATCH, "model.mes")
    Mestra.write(ds, path)
    @test Mestra.validate(path).errors == String[]
    back = Mestra.read(path)
    @test back.nrows == 0
    @test back.scalars["cl"].source == "callable:m1"
    out = Mestra.evaluate(back, Dict("mach" => [0.5], "alpha" => [4.0]))
    @test bitequal(Mestra.values(out, out.scalars["cl"])[1], 1.45)
end

@testset "a callable may serve a mesh support's coordinates" begin
    ds = Mestra.Dataset(writer = "mestra.jl test 0",
                        created = "2026-09-19T00:00:00Z")
    Mestra.add_key!(ds, "mach", Float64[]; role = :condition, units = "1",
                    bounds = (0.1, 0.9))
    cells = (cell_types = UInt8[9, 9], cell_offsets = Int64[0, 4, 8],
             cell_connectivity = Int64[0, 1, 4, 3, 1, 2, 5, 4])
    # No coordinates and no node count: nothing to build a support on.
    e = refusal(() -> Mestra.add_mesh_support!(ds, "s0"; cells...))
    @test e !== nothing && e.rule == "E03"
    s = Mestra.add_mesh_support!(ds, "s0"; n_nodes = 6, cells...)
    @test s.coordinates === nothing
    base = [0.0 0.0; 1.0 0.0; 2.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 1.0]
    # x = x0 (1 + mach): the mesh stretches along x with mach; A is
    # over the flattened (node, component) order of the output.
    stretch = zeros(12, 1)
    stretch[1:2:11, 1] .= base[:, 1]
    c = Mestra.affine(["mach"], Dict(
        "coordinates" => (A = stretch, b = vec(permutedims(base)),
                          shape = Int64[6, 2])))
    # Which callable fills the slot is required; that it exists is
    # what `write` checks, as for every callable slot in this language.
    e = refusal(() -> Mestra.set_callable_coordinates!(ds, s; units = "m",
                                                       components = 2))
    @test e !== nothing && e.rule == "E14"
    Mestra.add_callable!(ds, "m1", c)
    slot = Mestra.set_callable_coordinates!(ds, s; units = "m",
                                            components = 2, callable = "m1")
    @test slot.source == "callable:m1"
    @test slot.output == "coordinates"
    @test slot.varies == "row"
    e = refusal(() -> Mestra.set_callable_coordinates!(ds, s; units = "m",
                                                       components = 2,
                                                       callable = "m1"))
    @test e !== nothing && e.rule == "E03"
    path = joinpath(SCRATCH, "geometry.mes")
    Mestra.write(ds, path)
    @test Mestra.validate(path).errors == String[]
    back = Mestra.read(path)
    @test Mestra.is_callable_slot(back.supports[1].coordinates)
    e = refusal(() -> Mestra.compute_weights(back, "s0", :cell))
    @test e !== nothing && occursin("evaluate the file first", e.msg)
    out = Mestra.evaluate(back, Dict("mach" => [0.5]))
    got = Mestra.values(out, out.supports[1].coordinates)
    @test !Mestra.is_callable_slot(out.supports[1].coordinates)
    @test bitequal(Mestra.at(got; row = 1, node = 3, component = 1), 3.0)
    @test bitequal(Mestra.at(got; row = 1, node = 3, component = 2), 0.0)
    # An axis support's coordinates are its identity and are stored.
    t = Mestra.add_axis_support!(ds, "t"; coordinates = [0.0, 1.0, 2.0],
                                 units = "s")
    e = refusal(() -> Mestra.set_callable_coordinates!(ds, t; units = "s",
                                                       components = 1,
                                                       callable = "m1"))
    @test e !== nothing && e.rule == "E03"
end

@testset "notes and private are carried, never interpreted" begin
    ds = Mestra.read(case_file("mesh_two_rows"); lazy = false)
    Mestra.set_notes!(ds, Dict("solver" => "a solver, version 3",
                               "licence" => "CC-BY-4.0",
                               "iterations" => 1200,
                               "residual" => 1.0e-6,
                               "converged" => true))
    Mestra.set_private!(ds, Dict("our_own_record" => "kept, not read"))
    @test_throws Mestra.MestraError Mestra.set_notes!(ds,
        Dict("not a legal name" => 1))
    path = joinpath(SCRATCH, "noted.mes")
    Mestra.write(ds, path)
    r = Mestra.validate(path)
    @test r.errors == String[]
    @test r.warnings == String[]
    back = Mestra.read(path)
    @test back.notes !== nothing
    got = Dict(a.name => a.value for a in back.notes.attrs)
    @test got["solver"] == "a solver, version 3"
    @test got["iterations"] === Int64(1200)
    @test got["residual"] === 1.0e-6
    @test got["converged"] === true
    @test back.private !== nothing
    # a round trip keeps them unchanged
    again = joinpath(SCRATCH, "noted2.mes")
    Mestra.write(Mestra.read(path; lazy = false), again)
    @test isempty(Mestra.structural_diff(path, again))
end

@testset "the hostile subset is refused with the ids it breaks (section 30)" begin
    # Section 30, of the hostile subset: "Opening the file for its
    # metadata alone, and any operation that reads a slot, must refuse
    # with the same ids rather than return something."  So for each of
    # the fifteen: the validator reports at least the required ids,
    # and both the metadata open and the read refuse naming them.
    hostile = joinpath(REPO, "vectors", "hostile")
    dirs = [d for d in sort(readdir(hostile))
            if isfile(joinpath(hostile, d, "case.mes"))]
    @test length(dirs) == 15
    for d in dirs
        p = joinpath(hostile, d, "case.mes")
        want = String.(JSON3.read(read(joinpath(hostile, d,
                                               "expected.json"),
                                       String)).required_errors)
        @test !isempty(want)
        r = Mestra.validate(p)
        @test issubset(Set(want), Set(r.errors))
        for lazy in (true, false)
            e = refusal(() -> Mestra.read(p; lazy = lazy))
            @test e !== nothing
            said = sprint(showerror, e)
            for id in want
                @test occursin(id, said)
            end
        end
        # `strict = false` is the documented way past the refusal, for
        # a file you are inspecting rather than trusting, and it still
        # opens every one of them
        @test Mestra.read(p; strict = false) isa Mestra.Dataset
    end
end

@testset "a boolean attribute is 0 or 1 and nothing else (section 18)" begin
    # Section 18: "value 0 for false and 1 for true.  No other value
    # is legal."  E19 covers "a boolean that is not int8 or whose
    # value is not 0 or 1".
    src = case_file("mesh_two_rows")
    path = joinpath(SCRATCH, "aligned_is_two.mes")
    cp(src, path; force = true)
    chmod(path, 0o644)
    HDF5.h5open(path, "r+") do f
        HDF5.delete_attribute(f, "aligned")
        HDF5.write_attribute(f, "aligned", Int8(2))
    end
    r = Mestra.validate(path)
    @test r.errors == ["E19"]
    # E28 and E37 are both about the claim the file makes, and it has
    # made none, so E19 says the whole of what is wrong
    @test !("E28" in r.errors) && !("E37" in r.errors)
    e = refusal(() -> Mestra.read(path))
    @test e !== nothing && e.rule == "E19"
    # and a non-strict read never gives 2 the meaning that would
    # suppress the /row_support requirement of E28
    ds = Mestra.read(path; strict = false)
    @test any(f -> f.rule == "E19" && f.path == "/", ds.findings)
    # a legal 0 and a legal 1 are still read as the booleans they are
    for (byte, want) in ((Int8(0), false), (Int8(1), true))
        q = joinpath(SCRATCH, "aligned_$(byte).mes")
        cp(src, q; force = true)
        chmod(q, 0o644)
        HDF5.h5open(q, "r+") do f
            HDF5.delete_attribute(f, "aligned")
            HDF5.write_attribute(f, "aligned", byte)
        end
        @test Mestra.read(q; strict = false).aligned == want
    end
end

@testset "a conforming file carrying /private is accepted (sections 12, 14, 29)" begin
    # Section 14, of the byte-level rules of sections 18 to 25: "They
    # are checked on the public objects only.  `/private` is not
    # checked".  Section 29 forbids a reader to interpret it at all.
    # A producer's private records are in whatever representation it
    # chose, so everything below would be an error in the public tree
    # and none of it may be one here.
    src = case_file("mesh_two_rows")
    path = joinpath(SCRATCH, "with_private.mes")
    cp(src, path; force = true)
    chmod(path, 0o644)
    HDF5.h5open(path, "r+") do f
        p = HDF5.create_group(f, "private")
        # a string attribute stored the way section 18 forbids
        HDF5.attributes(p)["note"] = "a producer's own record"
        # a dimension scale of its own, and a dataset on it
        p["epoch"] = Float32[0, 1, 2, 3]
        HDF5.API.h5ds_set_scale(p["epoch"], "an epoch of our own")
        p["residual"] = Float32[1, 2, 3, 4]
        HDF5.API.h5ds_attach_scale(p["residual"], p["epoch"], 0)
        # a float32 dataset with no scale on it at all, which is E20
        # and E25 in the public tree
        p["history"] = Float32[1 2 3; 4 5 6]
        # and a group inside the group, with a dataset of its own
        g = HDF5.create_group(p, "stamps")
        g["when"] = Int32[1, 2, 3]
    end
    r = Mestra.validate(path)
    @test r.errors == String[]
    @test r.warnings == String[]
    @test all(f -> !startswith(f.path, "/private"), r.findings)
    # and a strict read opens it and keeps the group without reading
    # anything in it as a slot
    ds = Mestra.read(path)
    @test ds.private !== nothing
    @test isempty(ds.findings)
    @test !haskey(ds.supports[1].node_arrays, "residual")
    # the same file with the same objects in the public tree is
    # rejected, so the test is about /private and not about the file
    public = joinpath(SCRATCH, "with_public_junk.mes")
    cp(src, public; force = true)
    chmod(public, 0o644)
    HDF5.h5open(public, "r+") do f
        f["supports"]["s0"]["node_arrays"]["history"] =
            Float32[1 2 3; 4 5 6]
    end
    bad = Mestra.validate(public)
    @test "E25" in bad.errors
    @test "E20" in bad.errors
end

"""The HDF5 object header version of every object in a file.

Section 30 asks a golden file to be byte reproducible, and
`vectors/README.md` rejects the newer object header layout for the
corpus because its root header records four timestamps.  The version
is not something section 30's structural equality compares, so it is
read here directly."""
function header_versions(path::AbstractString)
    out = Dict{String,Int}()
    HDF5.h5open(String(path), "r") do f
        out["/"] = Int(HDF5.API.h5o_get_native_info(f).hdr.version)
        Mestra.walk_objects(f) do p, obj
            out[p] = Int(HDF5.API.h5o_get_native_info(obj).hdr.version)
        end
    end
    return out
end

"""The paths of every dimension scale in a file, by this package's own
walk and never by asking the library for a scale's path."""
function scale_paths(path::AbstractString)
    HDF5.h5open(String(path), "r") do f
        sort([p for (_, p, _, _) in Mestra.collect_scales(f)])
    end
end

@testset "the writer writes the corpus's object header layout" begin
    # libhdf5 2.0 changed the default low libver bound from
    # `earliest` to `v18`, so a writer that takes the default writes
    # version-2 object headers.  The root one then records when the
    # file was written, which costs byte reproducibility, and every
    # reader pays for the layout as well.  `Mestra.WRITER_LIBVER` is
    # the one call that decides it.
    #
    # Section 21 makes one exception, and it is the whole of decision
    # 52: a dimension scale is created with attribute creation order
    # tracked and indexed, which gives that one object a version 2
    # header so that its REFERENCE_LIST can live in the file's heap.
    # So every scale is version 2, every other object is version 1,
    # and the corpus says the same thing object for object.
    for name in ("mesh_two_rows", "affine_with_rows", "scalars_only",
                 "two_supports_unaligned", "labels_tables",
                 "cascade_varying_geometry", "compressed_field",
                 "notes_and_private")
        src = case_file(name)
        dst = joinpath(SCRATCH, "hdr_" * name * ".mes")
        Mestra.write(Mestra.read(src; lazy = false), dst)
        @test isempty(Mestra.structural_diff(src, dst))
        want = header_versions(src)
        got = header_versions(dst)
        @test sort(collect(keys(got))) == sort(collect(keys(want)))
        @test got == want
        # /private is copied and never interpreted, so a scale a
        # producer keeps in there is not one section 21 rules on
        scales = Set(p for p in scale_paths(dst)
                     if !startswith(p, "/private"))
        @test !isempty(scales)
        @test all(p -> got[p] == 2, scales)
        @test all(p -> got[p] == 1,
                  [p for p in keys(got)
                   if !(p in scales) && !startswith(p, "/private")])
    end
    # and two writes a second apart are the same bytes, which is what
    # `?Mestra.write` claims and what section 30 asks of a generator
    ds = Mestra.read(case_file("mesh_two_rows"); lazy = false)
    a = joinpath(SCRATCH, "twice_a.mes")
    b = joinpath(SCRATCH, "twice_b.mes")
    Mestra.write(ds, a)
    sleep(1.1)
    Mestra.write(ds, b)
    @test read(a) == read(b)
end

@testset "every scale is created as section 21 requires (E42)" begin
    # The one rule of section 16 that changes the bytes of every
    # golden file.  A scale created with the
    # library's defaults takes at most 4085 attachments, and the
    # 4086th fails after deleting the REFERENCE_LIST it was extending,
    # leaving a file every reader and every validator accepts.  So the
    # property is asserted on the written file and not inferred from
    # the fact that a write succeeded.
    for name in ("mesh_two_rows", "two_supports_unaligned",
                 "affine_zero_rows", "support_kind_none")
        dst = joinpath(SCRATCH, "e42_" * name * ".mes")
        Mestra.write(Mestra.read(case_file(name); lazy = false), dst)
        HDF5.h5open(dst, "r") do f
            n = 0
            for (_, p, d, _) in Mestra.collect_scales(f)
                startswith(p, "/private") && continue
                n += 1
                @test Mestra.scale_attr_order(d) ==
                      Mestra.SCALE_ATTR_ORDER
                @test Mestra.scale_order_tracked(d)
                dcpl = HDF5.get_create_properties(d)
                @test HDF5.API.h5p_get_obj_track_times(dcpl) == false
            end
            @test n > 0
        end
        @test Mestra.validate(dst).errors == String[]
    end
    # and the corpus case that breaks it on purpose, plus the one that
    # would not exist without the rule
    @test Mestra.validate(case_file("err_e42")).errors == ["E42"]
    @test Mestra.validate(case_file("wide_keys")).errors == String[]
    # a scale a producer keeps under /private is not section 21's
    # business: the byte-level rules are checked on the public objects
    @test Mestra.validate(case_file("notes_and_private")).errors ==
          String[]
end

@testset "only `row` is unlimited (E43)" begin
    @test Mestra.validate(case_file("err_e43")).errors == ["E43"]
    # the five cases that must survive the rule: section 25 requires a
    # zero-length dictionary axis to be unlimited, and there is no
    # other legal way to write one
    for name in ("affine_zero_rows", "affine_with_rows",
                 "callable_two_slots")
        @test !("E43" in Mestra.validate(case_file(name)).errors)
    end
    HDF5.h5open(case_file("affine_zero_rows"), "r") do f
        unlimited = String[]
        for (n, p, d, _) in Mestra.collect_scales(f)
            _, cmax = Mestra.disk_shape(d)
            isempty(cmax) || cmax[1] != -1 || push!(unlimited, p)
        end
        @test "/row" in unlimited
        @test any(p -> occursin("/callables/", p), unlimited)
        @test all(p -> p == "/row" || occursin("/callables/", p), unlimited)
    end
    # a writer never makes one on its own: what it writes carries no
    # unlimited dimension but `row`
    dst = joinpath(SCRATCH, "e43_draws.mes")
    Mestra.write(Mestra.read(case_file("draws_and_summaries");
                             lazy = false), dst)
    @test Mestra.validate(dst).errors == String[]
end

@testset "an axis is named from DIMENSION_LIST, never from REFERENCE_LIST" begin
    # Section 21: REFERENCE_LIST is informational and a scale whose
    # one is missing, short or stale is not an error.  It is not a
    # hypothetical either: the 4086th attachment to a scale created
    # without that property deletes the REFERENCE_LIST it was
    # extending, and this reader called the axis of such a file
    # `unknown` while every other reader still called it `row`.  The fixture is
    # `vectors/cases/mesh_two_rows/case.mes` with the `row` scale's
    # REFERENCE_LIST deleted and `component_1`'s truncated to one
    # entry; see julia/test/scales/make_scales.py.
    lost = joinpath(HERE, "scales", "lost_reference_list.mes")
    @test isfile(lost)
    HDF5.h5open(lost, "r") do f
        @test !haskey(HDF5.attributes(f["row"]), "REFERENCE_LIST")
        short = read(HDF5.attributes(f["component_1"])["REFERENCE_LIST"])
        @test length(short) == 1
    end
    # no rule in section 14 is about it
    r = Mestra.validate(lost)
    @test r.errors == String[]
    @test r.warnings == String[]
    # and every axis is still named, by the link name of the scale its
    # DIMENSION_LIST points at
    ds = Mestra.read(lost)
    @test ds.nrows == 2
    @test ds["pressure"].ldims == [:row, :node, :component]
    @test ds.scalars["cl"].ldims == [:row]
    @test ds.supports[1].coordinates.ldims ==
          [Symbol("group:member"), :node, :component]
    @test ds.supports[1].cell_arrays["region"].ldims == [:cell, :component]
    @test all(s -> !(:unknown in s.ldims), Mestra.all_slots(ds))
    io = IOBuffer()
    Mestra.info(lost; io = io)
    text = String(take!(io))
    @test !occursin("unknown", text)
    @test occursin("(row, node, component) 2x6x1", text)
    # reading by name still works, and gives the same numbers the
    # corpus case does
    v = Mestra.values(ds, ds["pressure"])
    @test Mestra.at(v; row = 2, node = 4, component = 1) == 204.0
    # and so does a round trip: what comes back is the corpus file,
    # REFERENCE_LIST and all, because a writer builds it again
    dst = joinpath(SCRATCH, "lost_reference_list_out.mes")
    Mestra.write(Mestra.read(lost; lazy = false), dst)
    @test isempty(Mestra.structural_diff(case_file("mesh_two_rows"), dst))
    HDF5.h5open(dst, "r") do f
        @test haskey(HDF5.attributes(f["row"]), "REFERENCE_LIST")
    end
end

@testset "4200 row-dimensioned datasets cost seconds, not minutes" begin
    # `wide_keys` is the case that would not exist without decision
    # 52: 100 keys and 4100 scalars on one `row` scale, past the 4085
    # attachments a scale carried before it.  What is asserted here is
    # that the open, the validate and one lazy row range are all
    # linear-ish in the dataset count rather than quadratic in it, so
    # the bound is generous: it is there to catch a reader that went
    # back to asking the library which scale is attached, which cost
    # 516 s at four thousand datasets.
    src = case_file("wide_keys")
    @test isfile(src)   # generated on demand; see vectors/README.md
    open_s = @elapsed ds = Mestra.read(src)
    @test ds.nrows == 2
    @test length(ds.keys) == 100
    @test length(ds.scalars) == 4100
    validate_s = @elapsed r = Mestra.validate(src)
    @test r.errors == String[]
    @test r.warnings == String[]
    range_s = @elapsed v = Mestra.rows(ds, ds.scalars["s2100"], 2:2)
    @test size(v) == (1,)
    @test v[1] == 4200.25
    @info "wide_keys" open_s validate_s range_s
    @test open_s < 30
    @test validate_s < 60
    @test range_s < 10
    # and the metadata open reads no array, whatever the file's width
    @test !Mestra.materialised(ds.scalars["s2100"])
end

@testset "an evaluated file carries no /callables (conventions 7)" begin
    # Evaluating turns every callable slot into a stored slot, so the
    # result has no callable to keep: the group is absent, not present
    # and empty, which is what the other three writers leave.
    for name in ("affine_zero_rows", "affine_with_rows")
        ds = Mestra.read(case_file(name))
        out = Mestra.evaluate(ds, Dict("mach" => [0.5, 0.6],
                                       "alpha" => [4.0, 2.0]))
        @test isempty(out.callables)
        path = joinpath(SCRATCH, "nocallables_" * name * ".mes")
        Mestra.write(out, path)
        @test Mestra.validate(path).errors == String[]
        HDF5.h5open(path, "r") do f
            @test !haskey(f, "callables")
        end
        back = Mestra.read(path)
        @test isempty(back.callables)
        @test !("callables" in back.container_groups)
    end
end

@testset "the open and the read name the same rule (conventions 7)" begin
    # An open reads attributes, dataspaces, link types and
    # dimension-scale structure, and may read a category table in
    # full; it never reads a slot.  The nine structural rules are
    # decided from exactly that, so whether the caller asked for a
    # lazy read or an eager one cannot change which rule refuses the
    # file, on the corpus or on the hostile subset.
    hostile = joinpath(REPO, "vectors", "hostile")
    files = vcat([case_file(n) for n in case_names()],
                 [joinpath(hostile, d, "case.mes")
                  for d in sort(readdir(hostile))
                  if isfile(joinpath(hostile, d, "case.mes"))])
    for p in files
        lazy = refusal(() -> Mestra.read(p))
        eager = refusal(() -> Mestra.read(p; lazy = false))
        @test (lazy === nothing) == (eager === nothing)
        lazy === nothing && continue
        @test lazy.rule == eager.rule
        @test lazy.path == eager.path
    end
    # a category table is read in full by the open, so an entry that
    # is not valid UTF-8 is E26 from the open and not only from the
    # read (section 25: a reader that cannot recover the bytes of a
    # string must say so rather than return something else)
    bad = joinpath(hostile, "string_invalid_utf8", "case.mes")
    e = refusal(() -> Mestra.read(bad))
    @test e !== nothing && e.rule == "E26"
    ds = Mestra.read(bad; strict = false)
    @test any(f -> f.rule == "E26", ds.findings)
    @test !haskey(ds.categories, "region")
end

@testset "gzip and shuffle, the two portable filters (section 23)" begin
    ds = Mestra.Dataset(writer = "mestra.jl test 0",
                        created = "2026-09-19T00:00:00Z")
    Mestra.add_key!(ds, "mach", collect(range(0.1, 0.9, length = 32));
                    role = :condition, units = "1")
    Mestra.add_scalar!(ds, "cl", collect(1.0:32.0); units = "1",
                       deflate = 4, shuffle = true)
    s = Mestra.add_mesh_support!(ds, "s0";
            coordinates = [0.0 0.0; 1.0 0.0; 2.0 0.0;
                           0.0 1.0; 1.0 1.0; 2.0 1.0],
            dims = (:node, :component),
            cell_types = UInt8[9, 9], cell_offsets = Int64[0, 4, 8],
            cell_connectivity = Int64[0, 1, 4, 3, 1, 2, 5, 4])
    Mestra.add_node_array!(ds, s, "pressure",
                           repeat(collect(1.0:32.0), 1, 6); units = "Pa",
                           deflate = 9)
    path = joinpath(SCRATCH, "compressed.mes")
    Mestra.write(ds, path)
    r = Mestra.validate(path)
    @test r.errors == String[]
    @test r.warnings == String[]
    back = Mestra.read(path; lazy = false)
    @test back.scalars["cl"].deflate == 4
    @test back.scalars["cl"].shuffle
    @test back["pressure"].deflate == 9
    @test Mestra.at(Mestra.values(back, back["pressure"]);
                    row = 7, node = 3, component = 1) == 7.0
    @test isempty(Mestra.structural_diff(path, path))
end

@testset "a strict read refuses a broken file, and reads no array" begin
    structural = Set(Mestra.STRUCTURAL_RULES)
    refused = 0
    for name in case_names()
        want = intersect(Set(String.(expected(name).validator.errors)),
                         structural)
        e = refusal(() -> Mestra.read(case_file(name)))
        if isempty(want)
            # nothing structural to refuse it with, so it opens, whatever
            # else it breaks: a missing unit or a leaking split is what a
            # user opens a file to find out
            @test e === nothing
        else
            @test e !== nothing && e.rule in want
            refused += 1
        end
        # and a non-strict read opens every one of them
        @test Mestra.read(case_file(name); strict = false) isa Mestra.Dataset
    end
    @test refused >= 6            # E01, E16, E19, E25, E26, E29, E30
    # the refusal names the rule, the object and the way past it
    e = refusal(() -> Mestra.read(case_file("err_e30")))
    @test e.rule == "E30" && occursin("strict = false", e.msg)

    # deciding those rules reads no array element: a strict read of a
    # good file goes through with room for eight elements, which is
    # fewer than any array in it
    ds = Mestra.read(case_file("mesh_two_rows"); max_elements = 8)
    @test ds.nrows == 2
    @test !Mestra.materialised(ds["pressure"])
    @test_throws Mestra.MestraError Mestra.values(ds, ds["pressure"])
    # and the structural pass is the nine rules and no others
    for name in case_names()
        r = Mestra.validate(case_file(name); structural = true)
        @test all(f -> f.rule in structural, r.findings)
        @test issubset(Set(f.rule for f in r.findings),
                       Set(vcat(String.(expected(name).validator.errors),
                                String.(expected(name).validator.warnings))))
    end
end

@testset "write validates before it writes (section 2)" begin
    ds, s = six_node_dataset()
    Mestra.add_node_array!(ds, s, "pressure", rand(2, 6); units = "Pa")
    good = joinpath(SCRATCH, "checked.mes")
    Mestra.write(ds, good)
    @test Mestra.validate(good).errors == String[]
    @test !isfile(good * ".mestra-check")
    # a slot the builder never saw: setting `varies` on a built slot
    # changes nothing about its shape, so the file this would write is
    # one the validator rejects, and the refusal carries the findings
    ds["pressure"].varies = "none"
    bad = joinpath(SCRATCH, "refused.mes")
    e = refusal(() -> Mestra.write(ds, bad))
    @test e !== nothing
    @test occursin("/supports/s0/node_arrays/pressure", e.msg)
    @test occursin("check = false", e.msg)
    @test !isfile(bad)                       # nothing was written
    @test !isfile(bad * ".mestra-check")     # and nothing was left behind
    # `check = false` writes it, which is how the corpus's own broken
    # files get made, and the refusal named the first rule it breaks
    @test Mestra.write(ds, bad; check = false) == bad
    broken = Mestra.validate(bad)
    @test !isempty(broken.errors) && e.rule in broken.errors
    # Try the refused write against the existing file as well.
    before = Base.read(good)
    @test refusal(() -> Mestra.write(ds, good)) !== nothing
    @test Base.read(good) == before
    @test Mestra.validate(good).errors == String[]
end

@testset "failed publication preserves the destination" begin
    mktempdir() do directory
        ds, _ = six_node_dataset()
        target = joinpath(directory, "result.mes")
        Mestra.write(ds, target)
        before = Base.read(target)
        @test_throws Base.IOError Mestra.publish_file(target * ".missing", target)
        @test Base.read(target) == before
        Mestra.write(ds, target)
        @test Mestra.validate(target).errors == String[]
        @test readdir(directory) == ["result.mes"]
        if !Sys.iswindows()
            control = joinpath(directory, "normal-creation")
            touch(control)
            @test filemode(target) == filemode(control)
            rm(control)
            chmod(target, 0o640)
            Mestra.write(ds, target)
            @test filemode(target) & 0o777 == 0o640
        end

        # Never remove a directory or move the staged file inside it.
        blocked = joinpath(directory, "directory.mes")
        mkdir(blocked)
        @test_throws Base.IOError Mestra.write(ds, blocked)
        @test isdir(blocked) && isempty(readdir(blocked))
        @test Set(readdir(directory)) == Set(["result.mes", "directory.mes"])
    end
end

@testset "one finding per rule per object, and a report to read" begin
    # W02, W03 and W04 could each fire on every row; each fires once,
    # with the count and the first three rows (section 5)
    r = Mestra.validate(case_file("warn_w02"))
    w02 = [f for f in r.findings if f.rule == "W02"]
    @test length(w02) == 1
    @test occursin("converged", w02[1].message)
    @test occursin(r"\d+ rows?", w02[1].message)
    w03 = [f for f in Mestra.validate(case_file("warn_w03")).findings
           if f.rule == "W03"]
    @test !isempty(w03)
    @test all(f -> occursin("missing floating-point data", f.message), w03)
    @test length(unique((f.rule, f.path) for f in w03)) == length(w03)
    bounds = Mestra.validate(case_file("warn_w04"))
    w04 = [f for f in bounds.findings if f.rule == "W04"]
    @test length(w04) == 1
    @test occursin("outside the declared bounds", w04[1].message)
    @test occursin(r"row", w04[1].message)
    # no rule says the same thing twice about one object, anywhere
    for name in case_names()
        rep = Mestra.validate(case_file(name))
        @test length(unique((f.rule, f.path) for f in rep.findings)) ==
              length(rep.findings)
    end
    # W01 names the unit it leaked, and why that matters
    leak = Mestra.validate(case_file("warn_w01"))
    w01 = [f for f in leak.findings if f.rule == "W01"]
    @test length(w01) == 1
    @test occursin("generalisation test", w01[1].message)

    # the printed report: `<id> <path>: <message>` and a summary line
    io = IOBuffer()
    Mestra.report(Mestra.validate(case_file("warn_w01")); io = io)
    text = String(take!(io))
    @test occursin("W01 /keys/split: ", text)
    @test occursin("0 error(s), 1 warning(s)", split(text, '\n')[end - 1])
    io = IOBuffer()
    Mestra.report(case_file("mesh_two_rows"); io = io)
    @test strip(String(take!(io))) == "0 error(s), 0 warning(s)"
end

@testset "info prints what the file declares (section 5)" begin
    io = IOBuffer()
    Mestra.info(case_file("mesh_two_rows"); io = io)
    text = String(take!(io))
    @test occursin("mestra/0, 2 row(s), aligned", text)
    # every key with its role, units, bounds and category
    @test occursin("mach", text) && occursin("condition", text)
    @test occursin("units 1", text) && occursin("bounds [", text)
    @test occursin("category member", text)
    # the support with its kind, counts and id
    @test occursin("mesh", text) && occursin("6 node(s), 2 cell(s)", text)
    @test occursin("96df395d", text)
    # every slot with named axes, shape, units and source
    @test occursin("/supports/s0/node_arrays/pressure", text)
    @test occursin("(row, node, component) 2x6x1", text)
    @test occursin("units Pa", text)
    @test occursin("data", text)
    # a callable slot names its callable and its output
    io = IOBuffer()
    Mestra.info(case_file("affine_zero_rows"); io = io)
    model = String(take!(io))
    @test occursin("callable m1 -> ", model)
    # a file with no support says so rather than claiming alignment
    io = IOBuffer()
    Mestra.info(case_file("scalars_only"); io = io)
    @test occursin("no support", String(take!(io)))
end

@testset "mistakes name the rule they break" begin
    ds = Mestra.Dataset()
    @test_throws Mestra.MestraError Mestra.add_key!(ds, "x", [1.0];
                                                    role = :nonsense)
    e = try
        Mestra.add_key!(ds, "mestra_x", [1.0]; role = :condition,
                        units = "1")
        Mestra.write(ds, joinpath(SCRATCH, "bad.mes"))
        nothing
    catch err
        err
    end
    @test e isa Mestra.MestraError && e.rule == "E33"
    @test occursin("E33", sprint(showerror, e))
    ds2 = Mestra.Dataset()
    Mestra.add_key!(ds2, "m", [1.0, 2.0]; role = :condition, units = "1")
    @test_throws Mestra.MestraError Mestra.add_key!(ds2, "g", [0];
                                                    role = :group)
end

@testset "what is written is a netCDF-4 file with the right dimensions" begin
    for (name, want) in (("mesh_two_rows",
                          Dict("row" => 2, "component_1" => 1,
                               "component_2" => 2, "group_member" => 2,
                               "category_member" => 2,
                               "category_region" => 2)),
                         ("draws_and_summaries",
                          Dict("row" => 2, "component_1" => 1,
                               "component_2" => 2, "draw_3" => 3)))
        path = joinpath(SCRATCH, name * ".mes")
        NCDatasets.NCDataset(path, "r") do nc
            for (dim, len) in want
                @test haskey(nc.dim, dim)
                @test nc.dim[dim] == len
            end
        end
    end
    # the dimension names of one variable, in the file's own order
    NCDatasets.NCDataset(joinpath(SCRATCH, "mesh_two_rows.mes"), "r") do nc
        g = nc.group["supports"].group["s0"].group["node_arrays"]
        @test NCDatasets.dimnames(g["pressure"]) ==
              ("component_1", "node", "row")
    end
end

"""Two rows of pressure on six nodes and two unit quads, with a region
label: the dataset the weight and integration tests work on."""
function weighted_dataset()
    w = Mestra.Dataset(writer = "mestra.jl test 0",
                       created = "2026-09-19T00:00:00Z")
    Mestra.add_category_table!(w, "region", ["inlet", "outlet"])
    Mestra.add_key!(w, "mach", [0.4, 0.8]; role = :condition, units = "1")
    ws = Mestra.add_mesh_support!(w, "s0";
             coordinates = [0.0 0.0; 1.0 0.0; 2.0 0.0;
                            0.0 1.0; 1.0 1.0; 2.0 1.0],
             dims = (:node, :component),
             cell_types = UInt8[9, 9], cell_offsets = Int64[0, 4, 8],
             cell_connectivity = Int64[0, 1, 4, 3, 1, 2, 5, 4])
    Mestra.add_node_array!(w, ws, "pressure",
                           [1.0 2 3 4 5 6; 10.0 20 30 40 50 60];
                           units = "Pa")
    Mestra.add_node_array!(w, ws, "region", Int32[0, 0, 0, 1, 1, 1];
                           role = :label, dims = (:node,),
                           category = "region")
    return (w, ws)
end

@testset "weights computed from the connectivity (section 3)" begin
    w, ws = weighted_dataset()
    # two unit quads: each cell has area 1, and each node the share of
    # the cells it belongs to
    cells = Mestra.compute_weights!(w, ws, :cell)
    @test cells.role === :weight && cells.recomputed === true
    @test cells.units == "m2" && cells.name == "weight"
    @test vec(Mestra.compute_weights(w, ws, :cell)) == [1.0, 1.0]
    nodes = Mestra.compute_weights!(w, ws, :node; name = "node_weight")
    @test vec(Mestra.compute_weights(w, ws, :node)) ==
          [0.25, 0.5, 0.25, 0.25, 0.5, 0.25]
    @test sum(Mestra.compute_weights(w, ws, :node)) == 2.0
    # and what is computed can be written and read back
    wpath = joinpath(SCRATCH, "weighted.mes")
    Mestra.write(w, wpath)
    r = Mestra.validate(wpath)
    @test r.errors == String[] && r.warnings == String[]

    # the cell measures themselves, one cell at a time
    unit = Mestra.Dataset(writer = "mestra.jl test 0",
                          created = "2026-09-19T00:00:00Z")
    Mestra.add_key!(unit, "x", [1.0]; role = :condition, units = "1")
    # a line, a triangle, a tetrahedron, a hexahedron, a wedge, a
    # pyramid, each on its own support of known measure
    cases = [("line", UInt8[3], [0.0 0.0 0.0; 2.0 0.0 0.0], 2, 2.0, "m"),
             ("tri", UInt8[5], [0.0 0.0 0.0; 3.0 0.0 0.0; 0.0 4.0 0.0],
              3, 6.0, "m2"),
             ("tet", UInt8[10], [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0;
                                 0.0 0.0 1.0], 4, 1 / 6, "m3"),
             ("hex", UInt8[12], [0.0 0.0 0.0; 1.0 0.0 0.0; 1.0 1.0 0.0;
                                 0.0 1.0 0.0; 0.0 0.0 2.0; 1.0 0.0 2.0;
                                 1.0 1.0 2.0; 0.0 1.0 2.0], 8, 2.0, "m3"),
             ("wedge", UInt8[13], [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0;
                                   0.0 0.0 1.0; 1.0 0.0 1.0;
                                   0.0 1.0 1.0], 6, 0.5, "m3"),
             ("pyr", UInt8[14], [0.0 0.0 0.0; 1.0 0.0 0.0; 1.0 1.0 0.0;
                                 0.0 1.0 0.0; 0.5 0.5 3.0], 5, 1.0, "m3")]
    for (name, types, coords, n, want, units) in cases
        s = Mestra.add_mesh_support!(unit, name; coordinates = coords,
                dims = (:node, :component), cell_types = types,
                cell_offsets = Int64[0, n],
                cell_connectivity = Int64.(collect(0:(n - 1))))
        got = Mestra.compute_weights(unit, s, :cell)
        @test isapprox(got[1, 1], want; rtol = 1e-12)
        slot = Mestra.compute_weights!(unit, s, :cell)
        @test slot.units == units
    end
    # a cell type with a curved geometry is refused, and says so
    curved = Mestra.Dataset(writer = "t", created = "2026-09-19T00:00:00Z")
    Mestra.add_key!(curved, "x", [1.0]; role = :condition, units = "1")
    cs = Mestra.add_mesh_support!(curved, "s0";
             coordinates = [0.0 0.0; 1.0 0.0; 0.5 0.1],
             dims = (:node, :component), cell_types = UInt8[21],
             cell_offsets = Int64[0, 3],
             cell_connectivity = Int64[0, 1, 2])
    e = refusal(() -> Mestra.compute_weights(curved, cs, :cell))
    @test e !== nothing && e.rule == "E21" && occursin("21", e.msg)

    # an axis support has no cells, and its nodes share the intervals
    ax = Mestra.Dataset(writer = "t", created = "2026-09-19T00:00:00Z")
    Mestra.add_key!(ax, "x", [1.0]; role = :condition, units = "1")
    axs = Mestra.add_axis_support!(ax, "f"; coordinates = [0.0, 1.0, 3.0],
                                   units = "Hz")
    @test vec(Mestra.compute_weights(ax, axs, :node)) == [0.5, 1.5, 1.0]
    @test refusal(() -> Mestra.compute_weights(ax, axs, :cell)) !== nothing
end

@testset "post-processing written against the format" begin
    ds = Mestra.read(case_file("mesh_two_rows"); lazy = false)
    st = Mestra.field_statistics(ds, "pressure")
    @test length(st) == 2
    @test st[1].row == 1 && st[1].mean == 103.5
    @test st[2].row == 2 && st[2].min == 201.0 && st[2].max == 206.0
    # with no label there is no grouping column at all
    @test !haskey(st[1], :region) && !haskey(st[1], :by)
    byregion = Mestra.field_statistics(ds, "region", by = "region")
    @test Set(x.region for x in byregion) == Set(["inlet", "outlet"])
    # and the column is named after the label, whatever it is called:
    # J4, where every label's column was called `region`
    lt = Mestra.read(case_file("labels_tables"); lazy = false)
    face = Mestra.field_statistics(lt, "cad_face_id", by = "cad_face_id")
    @test haskey(face[1], :cad_face_id) && !haskey(face[1], :region)
    @test length(unique(x.cad_face_id for x in face)) > 1
    @test all(x -> x.cad_face_id isa String, face)
    # a scalar is not a field, and says so in this package's words
    sc = Mestra.read(case_file("scalars_only"); lazy = false)
    e = refusal(() -> Mestra.field_statistics(sc, "cl"))
    @test e !== nothing && occursin("scalar", e.msg)
    @test occursin("Mestra.values", e.msg)

    # integration uses the file's own weight array by default
    w, ws = weighted_dataset()
    Mestra.compute_weights!(w, ws, :node)
    total = Mestra.integrate(w, "pressure")
    @test size(total) == (2, 1)
    @test total[1, 1] ≈ 0.25 * 1 + 0.5 * 2 + 0.25 * 3 +
                        0.25 * 4 + 0.5 * 5 + 0.25 * 6
    inlet = Mestra.integrate(w, "pressure", by = "region", region = "inlet")
    @test inlet[1, 1] ≈ 0.25 * 1 + 0.5 * 2 + 0.25 * 3
    # naming one has the same answer
    @test Mestra.integrate(w, "pressure", weight = "weight") == total
    # and with no weight array in the file one is computed, with a word
    # about it rather than an error from a missing keyword
    bare, _ = weighted_dataset()
    out = @test_logs (:info,) match_mode = :any Mestra.integrate(bare,
                                                                 "pressure")
    @test out == total

    # a time series at a node, along one trajectory
    t = Mestra.read(case_file("transient_fixed_mesh"); lazy = false)
    times, xs = Mestra.time_series(t, "u"; node = 3, trajectory = "r000")
    @test times == [0.0, 0.1, 0.3]
    @test xs == [302.0, 312.0, 322.0]
    times2, xs2 = Mestra.time_series(t, "u"; node = 1, trajectory = 1)
    @test times2 == [0.0, 0.25]

    # a split that honours the unit of generalisation
    parts = Mestra.grouped_split(sc, ["train" => 2 / 3, "test" => 1 / 3])
    @test sort(vcat(parts["train"], parts["test"])) == collect(1:6)
    g = Mestra.values(sc, sc.keys["geometry"])
    @test isempty(intersect(Set(g[parts["train"]]), Set(g[parts["test"]])))
    # J1: three units and 80/20 left the test part empty
    eighty = Mestra.grouped_split(sc, ["train" => 0.8, "test" => 0.2])
    @test !isempty(eighty["test"]) && !isempty(eighty["train"])
    @test sort(vcat(eighty["train"], eighty["test"])) == collect(1:6)
    # the seed is the whole of the randomness, and its default is 0
    @test Mestra.grouped_split(sc) == Mestra.grouped_split(sc; seed = 0)
    @test any(Mestra.grouped_split(sc; seed = s) != eighty for s in 1:20)
    # a Dict is taken in name order, so the answer does not depend on it
    @test Mestra.grouped_split(sc, Dict("test" => 0.2, "train" => 0.8)) ==
          Mestra.grouped_split(sc, ["test" => 0.2, "train" => 0.8])
    # more parts than units is refused rather than answered with an
    # empty part
    e = refusal(() -> Mestra.grouped_split(sc, ["a" => 1, "b" => 1,
                                                "c" => 1, "d" => 1]))
    @test e !== nothing && occursin("empty", e.msg)

    # the parts are filled in name order, whatever the fractions' order
    @test Mestra.grouped_split(sc, ["test" => 1, "train" => 2]) ==
          Mestra.grouped_split(sc, ["train" => 2, "test" => 1])

    @test isempty(Mestra.split_leaks(sc))
    leaky = Mestra.read(case_file("warn_w01"); lazy = false)
    leaks = Mestra.split_leaks(leaky)
    # conventions section 4: units by name, parts by name, sorted
    @test Set(keys(leaks)) == Set(["wing_a", "wing_b"])
    @test all(v == sort(v) && length(v) > 1 for v in Base.values(leaks))
    @test all(x isa String for v in Base.values(leaks) for x in v)
    # J10: an empty answer means no leak and nothing else
    e = refusal(() -> Mestra.split_leaks(lt))
    @test e !== nothing
    # a file with no unit of generalisation is refused, not guessed at
    @test_throws Mestra.MestraError Mestra.grouped_split(lt)
end

@testset "the worked example of the grouped split (section 31)" begin
    # The table of section 31, reproduced: five rotors whose category
    # table is in one order and whose unit order is another, seed 0.
    ds = Mestra.Dataset(writer = "mestra.jl test 0")
    Mestra.add_category_table!(ds, "rotor", ["rotor_b", "rotor_a", "rotor_d",
                                             "rotor_c", "rotor_e"])
    Mestra.add_key!(ds, "rotor", [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
                    role = :group, category = "rotor")
    u = Mestra.split_units(ds, ds.keys["rotor"], 0)
    # the units are in name order and not in table order, and one
    # splitmix64 draw is taken for each, in that order
    @test u.names == ["rotor_a", "rotor_b", "rotor_c", "rotor_d", "rotor_e"]
    @test u.ids == Int32[1, 0, 3, 2, 4]
    @test u.draws == [0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4,
                      0x06c45d188009454f, 0xf88bb8a8724c81ec,
                      0x1b39896a51a8749b]
    # sorted by the draw: rotor_c, rotor_e, rotor_b, rotor_a, rotor_d
    @test u.dealt == Int32[3, 4, 0, 1, 2]
    parts = Mestra.grouped_split(ds, ["train" => 0.8, "test" => 0.2];
                                 seed = 0)
    @test parts["test"] == [7, 8]                       # rotor_c
    @test parts["train"] == [1, 2, 3, 4, 5, 6, 9, 10]   # the other four
    # the seed is the whole of it, and the state starts at the seed
    @test Mestra.splitmix64(UInt64(0))[2] == 0xe220a8397b1dcdaf
    # a table written in another order splits the same way
    other = Mestra.Dataset(writer = "mestra.jl test 0")
    Mestra.add_category_table!(other, "rotor",
                               ["rotor_a", "rotor_b", "rotor_c",
                                "rotor_d", "rotor_e"])
    Mestra.add_key!(other, "rotor", [1, 1, 0, 0, 3, 3, 2, 2, 4, 4];
                    role = :group, category = "rotor")
    @test Mestra.grouped_split(other, ["train" => 0.8, "test" => 0.2]) ==
          parts
    # a fraction of nothing still leaves its part one unit, which is
    # two rows here, and a negative fraction is refused
    @test length(Mestra.grouped_split(ds, ["train" => 1.0,
                                           "test" => 0.0])["test"]) == 2
    @test refusal(() -> Mestra.grouped_split(ds, ["a" => -1, "b" => 2])) !==
          nothing
end

"""Run run_hostile.jl over a list of files in a process of its own,
killed if it overruns, and parse what it printed."""
function drive_hostile(files; limit = 300.0, tag = "hostile")
    logfile = joinpath(SCRATCH, tag * ".log")
    runner = joinpath(HERE, "hostile", "run_hostile.jl")
    warm = case_file("mesh_two_rows")
    cmd = `$(Base.julia_cmd()) --startup-file=no
           --project=$(Base.active_project()) $runner --warmup $warm $files`
    proc = run(pipeline(cmd; stdout = logfile, stderr = devnull);
               wait = false)
    finished = timedwait(() -> !process_running(proc), limit; pollint = 0.25)
    if finished !== :ok
        kill(proc, Base.SIGKILL)
        wait(proc)
    end
    text = isfile(logfile) ? read(logfile, String) : ""
    got = Dict{String,NamedTuple}()
    for l in split(text, '\n')
        parts = split(l, '|')
        length(parts) == 7 || continue
        got[String(parts[1])] = (status = String(parts[2]),
                                 seconds = parse(Float64, parts[3]),
                                 errors = split(parts[4], ',';
                                                keepempty = false),
                                 warnings = split(parts[5], ',';
                                                  keepempty = false),
                                 readrules = split(parts[6], ',';
                                                   keepempty = false),
                                 note = String(parts[7]))
    end
    return (finished === :ok, got)
end

@testset "the shared hostile subset (vectors/hostile)" begin
    dir = joinpath(REPO, "vectors", "hostile")
    cases = sort([c for c in readdir(dir) if isdir(joinpath(dir, c))])
    @test length(cases) == 15
    present = [c for c in cases if isfile(joinpath(dir, c, "case.mes"))]
    absent = setdiff(cases, present)
    isempty(absent) || @info(
        "the deep files of the shared subset are generated on demand, " *
        "not committed; run `python vectors/generate.py --on-demand` " *
        "to include them", absent)
    files = [joinpath(dir, c, "case.mes") for c in present]
    okrun, got = drive_hostile(files; tag = "shared")
    @test okrun

    for c in present
        want = JSON3.read(read(joinpath(dir, c, "expected.json"), String))
        required = sort(String.(want.required_errors))
        @test haskey(got, c)
        haskey(got, c) || continue
        g = got[c]
        # a clean exit: no crash, no hang, nothing left unread
        @test g.status == "ok"
        g.status == "ok" || @info "shared hostile" c g.note
        # inside the timeout the subset states
        @test g.seconds < Float64(want.timeout_seconds)
        # at least the required ids, and more are allowed
        for id in required
            @test id in g.errors
        end
        # every id a reader can see for itself, a read must refuse
        # with too, rather than return something
        for id in required
            id in ("E01", "E40", "E41") || continue
            @test id in g.readrules
        end
    end
    # the two deep files are the ones that catch a reader resolving a
    # scale's link name by asking the library for its path, which
    # walks the group hierarchy off the stack (section 21)
    if "deep_groups_keys" in present
        @test "E41" in got["deep_groups_keys"].errors
    end
end

@testset "hostile files: a reader is handed untrusted input" begin
    hostile = joinpath(HERE, "hostile")
    deepdir = joinpath(SCRATCH, "deep")
    include(joinpath(hostile, "make_deep.jl"))
    deepfiles = make_deep_files(deepdir)
    files = vcat(sort([joinpath(hostile, f) for f in readdir(hostile)
                       if endswith(f, ".mes")]), deepfiles)
    @test length(files) >= 18

    # A child process, because a test meant to catch a hang cannot
    # catch it from inside the process that is hanging.
    okrun, got = drive_hostile(files; tag = "own")
    @test okrun

    # Every file answered, none of them slowly, none of them any way
    # but a report or a MestraError with a rule a file may be refused
    # with.
    for f in files
        name = splitext(basename(f))[1]
        @test haskey(got, name)
        haskey(got, name) || continue
        @test got[name].status == "ok"
        got[name].status == "ok" || @info "hostile" name got[name].note
        @test got[name].seconds < 20.0
    end

    # what each kind must be answered with
    for name in ("link_dangling", "link_cycle", "link_external")
        @test "E40" in got[name].errors
    end
    for name in ("huge_declared", "huge_strings", "deep_keys",
                 "deep_callables", "unreadable_continues")
        @test "E41" in got[name].errors
    end
    @test "E01" in got["not_hdf5"].errors
    for name in ("attr_array_root", "attr_array_key", "attr_array_slot")
        @test "E19" in got[name].errors
    end
    @test "E29" in got["filter_many_cd"].errors
    @test "E29" in got["filter_unknown"].errors
    @test "E25" in got["scale_twice"].errors
    # An object of the wrong kind is one the reader cannot read as what
    # its place says it is, and every language names it E41.
    @test "E41" in got["kind_confusion"].errors
    @test "E30" ∉ got["kind_confusion"].errors
    @test "E26" in got["bad_utf8"].errors

    # one unreadable object must not hide what comes after it
    r = Mestra.validate(joinpath(hostile, "unreadable_continues.mes"))
    @test any(f -> f.path == "/scalars/a_broken" && f.rule == "E41",
              r.findings)
    @test "W10" in r.warnings          # y_warns, after the broken one
    @test "E36" in r.errors            # z_errors, after that
    @test "E39" in r.errors            # and a key after both

    # a claim of a trillion elements costs nothing to refuse.  The
    # file also says it in a scalar of a length no row count matches,
    # which is E16, so a strict read refuses it on the structure
    # alone; this is about what it costs to carry on with it.
    huge = joinpath(hostile, "huge_declared.mes")
    Mestra.validate(huge)              # warm
    @test (@allocated Mestra.validate(huge)) < 64_000_000
    @test refusal(() -> Mestra.read(huge)).rule == "E16"
    Mestra.read(huge; strict = false)
    @test (@allocated Mestra.read(huge; strict = false)) < 16_000_000
    ds = Mestra.read(huge; strict = false)
    # the whole slot is refused
    @test_throws Mestra.MestraError Mestra.values(ds, ds.scalars["huge"])
    e = try
        Mestra.values(ds, ds.scalars["huge"])
    catch err
        err
    end
    @test e.rule == "E41"
    # one row of it is a lazy read and costs one row
    one = Mestra.rows(ds, ds.scalars["huge"], 1:1)
    @test size(one) == (1,)
    @test (@allocated Mestra.rows(ds, ds.scalars["huge"], 1:1)) < 1_000_000
    # a range that is not is refused before anything is asked for
    @test_throws Mestra.MestraError Mestra.rows(ds, ds.scalars["huge"],
                                                1:(1 << 40))
    # a chunk this reader would not materialise is refused even for
    # one row, because the library reads whole chunks
    @test_throws Mestra.MestraError Mestra.rows(ds,
        ds.scalars["huge_chunk"], 1:1)
    # an eager read reports rather than throws, so that one refused
    # slot does not lose the file
    tight = Mestra.read(huge; lazy = false, max_elements = 10,
                        strict = false)
    @test any(f -> f.rule == "E41", tight.findings)
    @test !Mestra.materialised(tight.scalars["huge"])

    # a link this reader will not follow is reported by a non-strict
    # read and refused by a strict one; neither follows it
    @test refusal(() -> Mestra.read(joinpath(hostile,
              "link_external.mes"))).rule == "E40"
    ds2 = Mestra.read(joinpath(hostile, "link_external.mes");
                      strict = false)
    @test any(f -> f.rule == "E40", ds2.findings)
    @test !haskey(ds2.scalars, "elsewhere")
    @test !haskey(ds2.scalars, "neighbour")

    # every recursive routine is capped
    @test !Mestra.parse_units("(" ^ 10_000 * "m" * ")" ^ 10_000)
    @test !Mestra.parse_units("m" ^ 100_000)
    @test Mestra.parse_units("((m))")
    @test Mestra.MAX_DEPTH <= 64
    @test Mestra.element_count([1 << 40, 1 << 40]) == typemax(Int)
    @test Mestra.element_count([10, 0, 10]) == 0
end

@testset "the two files of docs/example.md, checked as it asks" begin
    # docs/example.md, "Checking an implementation against these two"
    one = Mestra.read(joinpath(REPO, "docs", "examples",
                               "mesh_two_rows.mes"); lazy = false)
    two = Mestra.read(joinpath(REPO, "docs", "examples",
                               "affine_zero_rows.mes"))
    @test Mestra.support_id(one.supports[1]) == one.supports[1].support_id
    @test Mestra.support_id(two.supports[1]) == two.supports[1].support_id
    @test Mestra.at(Mestra.values(one, one["pressure"]);
                    row = 2, node = 4, component = 1) == 204.0
    @test one["pressure"].ldims == [:row, :node, :component]
    @test one.supports[1].support_id == two.supports[1].support_id
    d = two.callables["m1"].dict
    @test d["keys"] == ["mach", "alpha"]
    @test d["outputs"]["cl"]["shape"] == Int64[]
    @test eltype(d["outputs"]["cl"]["shape"]) === Int64
    out = Mestra.callable(two, "m1")(Dict("mach" => [0.5], "alpha" => [4.0]))
    @test bitequal(out["cl"].mean[1], 1.45)
end

# -------------------------------------------------- docs/examples

"""The lines of a README's "Expected output" block, which is the whole
of what its example prints."""
function expected_output(readme::AbstractString)
    lines = split(Base.read(readme, String), '\n')
    at = findfirst(i -> strip(lines[i]) == "Expected output" &&
                        startswith(lines[i + 1], "---"),
                   1:(length(lines) - 1))
    at === nothing && error("$(readme) has no `Expected output` section")
    out = String[]
    for line in lines[(at + 2):end]
        if isempty(strip(line))
            isempty(out) && continue
            break
        end
        startswith(line, "    ") || break
        push!(out, String(rstrip(line[5:end])))
    end
    isempty(out) && error("$(readme): the `Expected output` section is empty")
    return out
end

# Every number these examples print matches the README's block byte for
# byte.  These lines cannot, because the block is Python's output and
# Julia spells the same value differently: `true` for `True`, a symbol
# for the name of an axis, `["a"]` for `['a']`, `String[]` for an empty
# list of strings, and a refusal that names the path as well as the
# rule.  The README's line is the key and this language's is the value,
# so that every difference between the two is in one place.
julia_spelling = Dict(
    "aligned: True nodes: 6 cells: 2" =>
        "aligned: true nodes: 6 cells: 2",
    "valid: True" => "valid: true",
    "2 rows, aligned: True" => "2 rows, aligned: true",
    "ok: True" => "ok: true",
    "pressure ('row', 'node', 'component') Pa" =>
        "pressure (:row, :node, :component) Pa",
    "pressure ('row', 'node', 'component') Pa row" =>
        "pressure (:row, :node, :component) Pa row",
    "pressure ('row', 'draw', 'node', 'component') draw" =>
        "pressure (:row, :draw, :node, :component) draw",
    "keys: [('mach', 'condition'), ('member', 'group')]" =>
        "keys: [(\"mach\", :condition), (\"member\", :group)]",
    "scalars: ['cl']" => "scalars: [\"cl\"]",
    "  node arrays: ['pressure']" => "  node arrays: [\"pressure\"]",
    "  cell arrays: ['region']" => "  cell arrays: [\"region\"]",
    "region is a label over ['inlet', 'outlet']" =>
        "region is a label over [\"inlet\", \"outlet\"]",
    "rows: 0 callables: ['m1']" => "rows: 0 callables: [\"m1\"]",
    "the split in the file leaks: ['wing_b']" =>
        "the split in the file leaks: [\"wing_b\"]",
    "train rows [0, 1, 2, 3] members ['wing_a', 'wing_b']" =>
        "train rows [0, 1, 2, 3] members [\"wing_a\", \"wing_b\"]",
    "test rows [4, 5] members ['wing_c']" =>
        "test rows [4, 5] members [\"wing_c\"]",
    "errors: [] warnings: ['W01']" =>
        "errors: String[] warnings: [\"W01\"]",
    "pressure: [0.5 1.1 3.7 4.3 6.9 7.5]" =>
        "pressure: [0.5, 1.1, 3.7, 4.3, 6.9, 7.5]",
    "draw 0 of row 0: [100. 101. 102. 103. 104. 105.]" =>
        "draw 0 of row 0: [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]",
    "refused: E11: cl: a scalar carries units; pass units= (\"1\" for a " *
    "dimensionless one)" =>
        "refused: mestra: E11 /scalars/cl: a scalar carries units; pass " *
        "units = \"1\" for a dimensionless one")

@testset "the examples of docs/examples print what their READMEs say" begin
    examples = joinpath(REPO, "docs", "examples")
    dirs = sort([joinpath(examples, d) for d in readdir(examples)
                 if isfile(joinpath(examples, d, "julia.jl"))])
    @test length(dirs) == 7
    # no line of the table above is stale: every one of them is a line
    # some README still asks for
    asked = Set(l for dir in dirs
                for l in expected_output(joinpath(dir, "README.md")))
    @test isempty(setdiff(Base.keys(julia_spelling), asked))
    for dir in dirs
        @testset "$(basename(dir))" begin
            script = joinpath(dir, "julia.jl")
            body = [l for l in eachline(script)
                    if !isempty(strip(l)) && !startswith(strip(l), "#")]
            @test length(body) < 30
            # Each one runs in a directory of its own, because several
            # write a file into the working directory.
            wd = mktempdir()
            log = joinpath(wd, "printed")
            cmd = `$(Base.julia_cmd()) --startup-file=no
                   --project=$(Base.active_project()) $script`
            run(pipeline(setenv(cmd; dir = wd); stdout = log))
            printed = [String(rstrip(l)) for l in eachline(log)]
            want = [get(julia_spelling, l, l)
                    for l in expected_output(joinpath(dir, "README.md"))]
            @test printed == want
        end
    end
end

end # testset Mestra
