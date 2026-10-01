# The data model of SPEC.md sections 2 to 12, as Julia types.
#
# One array is held the way HDF5.jl hands it over: the axes reversed
# from the order the file states, so that the array's linear memory is
# the file's own byte order.  Every slot carries the logical name of
# each axis, and `values(ds, slot)` returns a DimArray carrying those
# names in the Julia order, so nothing downstream counts axes.

"""
    MestraError(rule, path, msg)
    MestraError(rule, msg)

An error this package raises.  It says the rule of section 14 the
caller broke where there is one, then the path of the object it is
about where there is one, then what to do about it, which for a
builder means which argument to change (`docs/api-conventions.md`
section 6).
"""
struct MestraError <: Exception
    rule::Union{String,Nothing}
    path::Union{String,Nothing}
    msg::String
end

MestraError(rule::Union{String,Nothing}, msg::AbstractString) =
    MestraError(rule, nothing, String(msg))

function Base.showerror(io::IO, e::MestraError)
    print(io, "mestra: ")
    if e.rule !== nothing
        print(io, e.rule)
        print(io, e.path === nothing ? ": " : " ")
    end
    e.path === nothing || print(io, e.path, ": ")
    print(io, e.msg)
end

"""One thing a reader or the validator found, with the rule of
section 14 it belongs to and the path it was found at."""
struct Finding
    rule::String
    path::String
    message::String
end

# `<id> <path>: <message>`, which is the one line every language
# prints (`docs/api-conventions.md` section 5).
Base.show(io::IO, f::Finding) =
    print(io, f.rule, " ", one_line(f.path), ": ", one_line(f.message))

"""A path or a message as part of one printed line: a control
character, which an illegal name can hold, is written as \\xHH so that
the finding stays on its line."""
one_line(s::AbstractString) =
    join((c < ' ' || c == '\x7f') ? "\\x" * string(UInt32(c), base = 16, pad = 2) :
         string(c) for c in s)

const KEY_ROLES = (:design, :condition, :time, :categorical, :group,
                   :split, :id, :status)
const ARRAY_ROLES = (:coordinates, :field, :label, :weight, :normal,
                     :derived)
const STATISTICS = ("value", "mean", "band", "std", "quantile", "draw")
const SPLIT_CATEGORIES = ("train", "validation", "test", "holdout")

"""
    KeyColumn

One per-row column, with its role and the bounds that are the domain
the file is valid over (section 3).
"""
mutable struct KeyColumn
    name::String
    role::Union{Symbol,Nothing}
    units::Union{String,Nothing}
    lower::Union{Float64,Nothing}
    upper::Union{Float64,Nothing}
    category::Union{String,Nothing}
    trajectory_group::Union{String,Nothing}
    parent::Union{String,Nothing}
    eltype::DataType                  # Float64, Int32, Int64 or String
    strsize::Int                      # bytes, for a string id column
    chunk::Union{Nothing,Vector{Int}} # as stored, C order
    values::Union{Nothing,AbstractVector}
    path::String
end

KeyColumn(name, role; units = nothing, lower = nothing, upper = nothing,
          category = nothing, trajectory_group = nothing, parent = nothing,
          eltype = Float64, strsize = 1, chunk = nothing, values = nothing,
          path = "") =
    KeyColumn(String(name), role, units, lower, upper, category,
              trajectory_group, parent, eltype, strsize, chunk, values, path)

"""
    Slot

A scalar or an array slot.  It holds data or names a callable; which
one is decided by `source` and never by the row count (section 22).
`ldims` is the logical name of each axis in the order the *file* has
them, which is the order every shape in the specification is written
in; the Julia array reverses it.
"""
mutable struct Slot
    name::String
    path::String
    location::Symbol                  # :scalar, :node or :cell
    support::Union{String,Nothing}
    source::String                    # "data" or "callable:<id>"
    output::Union{String,Nothing}
    role::Union{Symbol,Nothing}
    varies::Union{String,Nothing}
    units::Union{String,Nothing}
    components::Union{Int,Nothing}
    statistic::Union{String,Nothing}
    of::Union{String,Nothing}
    quantile::Union{Float64,Nothing}
    level::Union{Float64,Nothing}     # on a band: its coverage
    method::Union{String,Nothing}     # on a band: how it was made
    category::Union{String,Nothing}
    recomputed::Union{Bool,Nothing}
    derived_from::Union{String,Nothing}
    recipe::Union{String,Nothing}
    reference::Union{String,Nothing}
    ldims::Vector{Symbol}             # logical names, file order
    dshape::Vector{Int}               # shape, file order
    eltype::DataType
    chunk::Union{Nothing,Vector{Int}} # as stored, file order
    deflate::Union{Nothing,Int}
    shuffle::Bool
    data::Union{Nothing,AbstractArray}   # Julia order: reverse of file
end

function Slot(name, location; support = nothing, source = "data",
              output = nothing, role = nothing, varies = nothing,
              units = nothing, components = nothing, statistic = nothing,
              of = nothing, quantile = nothing, level = nothing,
              method = nothing, category = nothing,
              recomputed = nothing, derived_from = nothing, recipe = nothing,
              reference = nothing, ldims = Symbol[], dshape = Int[],
              eltype = Float64, chunk = nothing, deflate = nothing,
              shuffle = false, data = nothing, path = "")
    Slot(String(name), path, location, support, source, output, role,
         varies, units, components, statistic, of, quantile, level, method,
         category,
         recomputed, derived_from, recipe, reference, Symbol[ldims...],
         Int[dshape...], eltype, chunk, deflate, shuffle, data)
end

is_callable_slot(s::Slot) = startswith(s.source, "callable:")
callable_id(s::Slot) = is_callable_slot(s) ? s.source[10:end] : nothing

# ------------------------------------------- reaching for the numbers
#
# Opening a file reads no array (section 29), so a key's `values` and
# a slot's `data` are empty until something asks for them.  Reaching
# for the empty field used to hand back `nothing` and the next index
# raised a Julia error that never mentioned this package; it now says
# what to call instead.  Inside the package the field itself is
# `raw_values` and `raw_data`, which are allowed to be `nothing`.

raw_values(k::KeyColumn) = getfield(k, :values)
raw_data(s::Slot) = getfield(s, :data)

"""
    materialised(key | slot) -> Bool

Whether the numbers are in memory already.  `Mestra.values` reads them
when they are not, and `Mestra.materialise!(ds)` reads all of them.
"""
materialised(k::KeyColumn) = raw_values(k) !== nothing
materialised(s::Slot) = raw_data(s) !== nothing

function Base.getproperty(k::KeyColumn, name::Symbol)
    if name === :values && getfield(k, :values) === nothing
        n = getfield(k, :name)
        throw(MestraError(nothing, getfield(k, :path),
            "this key has not been read: ask for its values with " *
            "`Mestra.values(ds, ds.keys[\"$(n)\"])`, or open the file " *
            "with `Mestra.read(path; lazy = false)`"))
    end
    return getfield(k, name)
end

function Base.getproperty(s::Slot, name::Symbol)
    if name === :data && getfield(s, :data) === nothing
        n = getfield(s, :name)
        throw(MestraError(nothing, getfield(s, :path),
            is_callable_slot(s) ?
            "this slot is served by callable `$(callable_id(s))` and " *
            "holds no data: `Mestra.evaluate(ds, table)` produces it" :
            "this slot has not been read: ask for its values with " *
            "`Mestra.values(ds, ds[\"$(n)\"])`, or open the file with " *
            "`Mestra.read(path; lazy = false)`"))
    end
    return getfield(s, name)
end

"""The logical name of each axis in the Julia order."""
julia_dims(s::Slot) = Tuple(reverse(s.ldims))
julia_size(s::Slot) = Tuple(reverse(s.dshape))

"""
    Support

A mesh, a one-dimensional axis, or none (section 6).  `support_id` is
the content hash of section 24, which is how two files agree that they
are about the same thing.
"""
mutable struct Support
    name::String
    kind::String                      # "mesh", "axis" or "none"
    n_nodes::Int
    n_cells::Int
    support_id::String
    coordinates::Union{Nothing,Slot}
    cell_types::Union{Nothing,Vector{UInt8}}
    cell_offsets::Union{Nothing,Vector{Int64}}
    cell_connectivity::Union{Nothing,Vector{Int64}}
    node_arrays::Dict{String,Slot}
    cell_arrays::Dict{String,Slot}
end

Support(name, kind; n_nodes = 0, n_cells = 0, support_id = "",
        coordinates = nothing, cell_types = nothing, cell_offsets = nothing,
        cell_connectivity = nothing) =
    Support(String(name), String(kind), n_nodes, n_cells, support_id,
            coordinates, cell_types, cell_offsets, cell_connectivity,
            Dict{String,Slot}(), Dict{String,Slot}())

"""A category table: the entries and the byte width they are stored
in (section 19)."""
mutable struct CategoryTable
    name::String
    entries::Vector{String}
    strsize::Int
end

CategoryTable(name, entries) =
    CategoryTable(String(name), String.(entries),
                  maximum(vcat([ncodeunits(e) for e in entries], 1)))

"""A callable as the file carries it: a public `type` and a dictionary
that is opaque to a reader that does not own the type (section 10)."""
mutable struct CallableRef
    id::String
    type::Union{String,Nothing}
    repr::Union{String,Nothing}
    dict::Dict{String,Any}
end

CallableRef(id, type; repr = nothing, dict = Dict{String,Any}()) =
    CallableRef(String(id), type, repr, dict)

# An object this reader does not own, kept so that a round trip does
# not lose it: /notes, /private, and anything section 28 adds later.
"""One dataset of a group this reader does not own, copied rather than
interpreted (sections 12 and 29).

The last four fields are what makes the copy structurally equal to
what it came from (section 30): its filters, whether it is a dimension
scale and the NAME it carries as one, whether its creation property
list tracked attribute creation order, and, for each axis, the scale
attached to it named by its path within the copied subtree.  A
producer's private part may hold a dimension scale of its own, and a
round trip that dropped the attachment would write a different file
back."""
struct RawDatasetCopy
    name::String
    ti::TypeInfo
    cdims::Vector{Int}
    cmax::Vector{Int}
    chunk::Union{Nothing,Vector{Int}}
    raw::Vector{UInt8}
    attrs::Vector{RawAttr}
    deflate::Union{Nothing,Int}
    shuffle::Bool
    scale_name::Union{Nothing,String}
    attr_order::Bool
    attached::Vector{Union{Nothing,String}}
end

RawDatasetCopy(name, ti, cdims, cmax, chunk, raw, attrs) =
    RawDatasetCopy(name, ti, cdims, cmax, chunk, raw, attrs, nothing,
                   false, nothing, false,
                   Union{Nothing,String}[nothing for _ in cdims])

struct RawGroupCopy
    name::String
    attrs::Vector{RawAttr}
    datasets::Vector{RawDatasetCopy}
    groups::Vector{RawGroupCopy}
end

"""
    Dataset

One file's worth of the model.  Build one with `Dataset(...)` and the
`add_*!` functions, read one with `Mestra.read`, write one with
`Mestra.write`.
"""
mutable struct Dataset
    format::String
    writer::String
    created::String
    aligned::Bool
    generalisation_group::Union{String,Nothing}
    nrows::Int
    keys::Dict{String,KeyColumn}
    scalars::Dict{String,Slot}
    categories::Dict{String,CategoryTable}
    supports::Vector{Support}
    row_support::Union{Nothing,Vector{Int32}}
    callables::Dict{String,CallableRef}
    notes::Union{Nothing,RawGroupCopy}
    private::Union{Nothing,RawGroupCopy}
    extra_root_attrs::Vector{RawAttr}
    extra_root_groups::Vector{RawGroupCopy}
    # The container groups the file carried, so that a round trip
    # keeps an empty /scalars that a producer chose to write.
    container_groups::Set{String}
    # What the reader met and would not follow or could not read: a
    # link that is not a hard link (E40) and an object whose read
    # failed or was refused (E41).  A file is untrusted input, so the
    # reader reports rather than throws wherever it can carry on.
    findings::Vector{Finding}
    max_elements::Int
    path::Union{Nothing,String}
    lazy::Bool
end

"""
    Dataset(; writer, created, nrows = 0)

An empty dataset.  `format` is "mestra/0", `aligned` follows from the
supports when the file is written, and `created` defaults to now, in
UTC, as section 11 requires.
"""
function Dataset(; writer::AbstractString = "mestra.jl 0",
                 created::AbstractString = utc_now(),
                 nrows::Integer = 0,
                 generalisation_group = nothing)
    Dataset("mestra/0", String(writer), String(created), true,
            generalisation_group, Int(nrows), Dict{String,KeyColumn}(),
            Dict{String,Slot}(), Dict{String,CategoryTable}(), Support[],
            nothing, Dict{String,CallableRef}(), nothing, nothing,
            RawAttr[], RawGroupCopy[], Set{String}(), Finding[],
            DEFAULT_MAX_ELEMENTS, nothing, false)
end

function utc_now()
    t = round(Int, time())
    # A plain ISO 8601 UTC timestamp, without pulling in Dates.
    days, secs = fldmod(t, 86400)
    h, rem = fldmod(secs, 3600)
    mi, s = fldmod(rem, 60)
    y, mo, d = civil_from_days(days)
    return @sprintf("%04d-%02d-%02dT%02d:%02d:%02dZ", y, mo, d, h, mi, s)
end

# Howard Hinnant's civil_from_days, for a timestamp with no dependency.
function civil_from_days(z::Int)
    z += 719468
    era = fld(z, 146097)
    doe = z - era * 146097
    yoe = div(doe - div(doe, 1460) + div(doe, 36524) - div(doe, 146096), 365)
    y = yoe + era * 400
    doy = doe - (365 * yoe + div(yoe, 4) - div(yoe, 100))
    mp = div(5 * doy + 2, 153)
    d = doy - div(153 * mp + 2, 5) + 1
    m = mp + (mp < 10 ? 3 : -9)
    return (y + (m <= 2 ? 1 : 0), m, d)
end

"""The file's support order: the group names sorted by their UTF-8
bytes, which is the one ordering every language produces identically
(section 22)."""
support_order(ds::Dataset) = sort(ds.supports, by = s -> codeunits(s.name))

"""The file's key order: the names under /keys sorted by their UTF-8
bytes (section 26)."""
key_order(ds::Dataset) = sort(collect(Base.keys(ds.keys)), by = codeunits)

"""The position of a support in `ds.supports`, or nothing."""
support_by_name(ds::Dataset, name) =
    findfirst(s -> s.name == name, ds.supports)

"""
    ds[name]

The key, scalar or array called `name`, wherever it sits.  Array names
are unique across the supports of every file this reader has seen, but
nothing requires them to be, so reach for the support itself when two
supports name an array the same thing.
"""
function Base.getindex(ds::Dataset, name::AbstractString)
    haskey(ds.keys, name) && return ds.keys[name]
    haskey(ds.scalars, name) && return ds.scalars[name]
    for s in ds.supports
        haskey(s.node_arrays, name) && return s.node_arrays[name]
        haskey(s.cell_arrays, name) && return s.cell_arrays[name]
    end
    throw(MestraError(nothing, "no key, scalar or array called $(name)"))
end

"""Every array slot in the file, node arrays and cell arrays alike."""
function array_slots(ds::Dataset)
    out = Slot[]
    for s in support_order(ds)
        s.coordinates === nothing || push!(out, s.coordinates)
        for n in sort(collect(Base.keys(s.node_arrays)), by = codeunits)
            push!(out, s.node_arrays[n])
        end
        for n in sort(collect(Base.keys(s.cell_arrays)), by = codeunits)
            push!(out, s.cell_arrays[n])
        end
    end
    return out
end

"""Every slot, arrays and scalars."""
function all_slots(ds::Dataset)
    out = Slot[]
    for n in sort(collect(Base.keys(ds.scalars)), by = codeunits)
        push!(out, ds.scalars[n])
    end
    append!(out, array_slots(ds))
    return out
end

function Base.show(io::IO, ::MIME"text/plain", ds::Dataset)
    println(io, "mestra Dataset (", ds.format, "), ", ds.nrows, " rows, ",
            ds.aligned ? "aligned" : "unaligned")
    println(io, "  keys     : ", join(key_order(ds), ", "))
    println(io, "  scalars  : ",
            join(sort(collect(Base.keys(ds.scalars)), by = codeunits), ", "))
    for s in support_order(ds)
        println(io, "  support ", s.name, " (", s.kind, ", ", s.n_nodes,
                " nodes, ", s.n_cells, " cells) ", s.support_id[1:min(8, end)])
        isempty(s.node_arrays) ||
            println(io, "    node arrays: ",
                    join(sort(collect(Base.keys(s.node_arrays))), ", "))
        isempty(s.cell_arrays) ||
            println(io, "    cell arrays: ",
                    join(sort(collect(Base.keys(s.cell_arrays))), ", "))
    end
    isempty(ds.callables) ||
        println(io, "  callables: ",
                join([c.id * " (" * string(c.type) * ")"
                      for c in values(ds.callables)], ", "))
end
