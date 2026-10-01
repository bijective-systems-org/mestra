# The writer.
#
# Everything here is sections 18 to 25 and 30 written out: fixed-length
# NUL-padded UTF-8 strings, dimension scales created and attached with
# the H5DS API, chunking along an unlimited `row`, no filter but gzip
# and shuffle, no fill value, and object time tracking off so that two
# runs produce the same bytes.

"""The HDF5 format bounds every file is created with.

One call decides which object header version the library writes, and
it is the file access property list's `libver_bounds`, not any of the
per-object properties.  libhdf5 2.0 changed the default low bound from
`H5F_LIBVER_EARLIEST` to `H5F_LIBVER_V18`, so a writer that takes the
default on a 2.x library writes version-2 object headers where a 1.x
one wrote version 1.  Two things follow, and both are costs this
format does not want to pay:

  - the root object header then records four timestamps, so two runs
    of this writer a second apart produce different bytes, and section
    30 asks a golden file to be byte reproducible.  `obj_track_times`
    is off on every dataset and group this writer creates, but the
    root group's header is the library's and the property list it was
    created from is this one;
  - every reader pays for it.  `vectors/README.md` rejects the layout
    for the corpus, and a C++ validator takes 4.98 s on a
    thousand-key file in it against 0.68 s in the default one.

Asking for the earliest low bound puts this writer back on the layout
the corpus files have.  The high bound stays `:latest`, so nothing
this format uses is refused for being too new.
"""
const WRITER_LIBVER = (:earliest, :latest)

"""Section 23: the default number of rows in a chunk."""
function default_chunk_rows(itemsize::Int, rest::Vector{Int}, nrows::Int)
    nrows == 0 && return 1
    b = itemsize
    for e in rest
        b *= max(1, e)
    end
    c = fld(1048576, b)
    c < 1 && (c = 1)
    c > nrows && (c = nrows)
    return c
end

itemsize_of(T::DataType) = T === String ? 1 : sizeof(T)

function hdf5_type(T::DataType, strsize::Int = 1)
    T === Float64 && return le(HDF5.API.H5T_IEEE_F64LE)
    T === Int64 && return le(HDF5.API.H5T_STD_I64LE)
    T === Int32 && return le(HDF5.API.H5T_STD_I32LE)
    T === Int8 && return le(HDF5.API.H5T_STD_I8LE)
    T === UInt8 && return le(HDF5.API.H5T_STD_U8LE)
    T === String && return fixed_string_type(strsize)
    throw(MestraError("E20", "no on-disk type for $(T)"))
end

"""The raw C-order bytes of an array held the way this package holds
one: the Julia axes reversed from the file's, so the array's linear
memory is already the file's byte order."""
raw_bytes(a::AbstractArray) = collect(reinterpret(UInt8, vec(collect(a))))

function string_records(values::AbstractVector{<:AbstractString}, n::Int)
    raw = UInt8[]
    for s in values
        b = Vector{UInt8}(codeunits(s))
        length(b) <= n || throw(MestraError("E26",
            "string \"$(s)\" does not fit in $(n) bytes"))
        append!(raw, b)
        append!(raw, zeros(UInt8, n - length(b)))
    end
    return raw
end

# ------------------------------------------------------- scale tables

function component_lengths(ds::Dataset)
    out = Set{Int}()
    for s in array_slots(ds)
        is_callable_slot(s) && continue
        isempty(s.dshape) && continue
        push!(out, s.dshape[end])
    end
    return sort(collect(out))
end

function draw_lengths(ds::Dataset)
    out = Set{Int}()
    for s in all_slots(ds)
        is_callable_slot(s) && continue
        s.statistic == "draw" || continue
        length(s.dshape) >= 3 && push!(out, s.dshape[end - 2])
    end
    return sort(collect(out))
end

group_keys(ds::Dataset) =
    sort([k for (k, v) in ds.keys if v.role === :group], by = codeunits)

function n_categories(ds::Dataset, table::Union{Nothing,String})
    table === nothing && return 0
    haskey(ds.categories, table) || throw(MestraError("E39",
        "no category table called $(table)"))
    return length(ds.categories[table].entries)
end

"""The number of rows referencing each support, in support order."""
function rows_per_support(ds::Dataset)
    order = support_order(ds)
    counts = zeros(Int, length(order))
    if ds.row_support === nothing
        length(order) == 1 && (counts[1] = ds.nrows)
    else
        for v in ds.row_support
            i = Int(v) + 1
            1 <= i <= length(counts) && (counts[i] += 1)
        end
    end
    return counts
end

"""The disk name of each axis of a slot, in the file's own order
(sections 19 and 21)."""
function canonical_disk_dims(s::Slot)
    s.location === :scalar && return ["row"]
    names = String[]
    v = something(s.varies, "none")
    if v == "row"
        push!(names, "row")
    elseif startswith(v, "group:")
        push!(names, "group_" * v[7:end])
    end
    if s.statistic == "draw"
        n = length(s.dshape) >= 3 ? s.dshape[end - 2] : 1
        push!(names, "draw_$(n)")
    end
    push!(names, s.location === :cell ? "cell" : "node")
    push!(names, "component_$(isempty(s.dshape) ? 1 : s.dshape[end])")
    return names
end

# ------------------------------------------------------------- write

"""
    Mestra.write(ds, path; check = true)

Write a dataset as a conforming `.mes` file.  Anything the dataset
cannot legally be written as raises a `MestraError` naming the rule of
section 14 that it breaks.

What is written is then validated, and a file with any error is
refused: the findings are in the message, nothing is left at `path`,
and a file that was already there is untouched.  `check = false`
writes it anyway, which is for making a file that breaks a rule on
purpose.  This is `docs/api-conventions.md` section 2, and it is why a
writer cannot hand you a file its own validator rejects.

What comes out is a conforming netCDF-4 file: fixed-length NUL-padded
UTF-8 strings, dimension scales created and attached with the H5DS
API, chunking along an unlimited `row`, no filter but gzip and
shuffle, and no fill value, so `ncdump -h` lists `row`, `node`,
`component_1` and the rest by name.

Two runs of it produce the same bytes.  That takes object time
tracking off on every object this writer creates, and the libver
bounds `Mestra.WRITER_LIBVER`, whose docstring is why; the one
exception is each dimension scale, which section 21 requires to be
created with attribute creation order tracked and indexed, and which
`Mestra.make_dcpl` explains.  A group this format does not own --
`/notes`, `/private`, or one a later version adds -- is copied out
again exactly as it came in, dtypes, filters, chunking and any
dimension scale of the producer's own included, because section 29
forbids interpreting it and a round trip that widened a dtype would be
interpreting it.
"""
function write(ds::Dataset, path::AbstractString; check::Bool = true)
    prepare_for_write!(ds)
    check_destination(String(path))
    if ds.path !== nothing && any_unread(ds) &&
       abspath(String(path)) == abspath(ds.path)
        throw(MestraError(nothing, String(path),
            "this dataset still reads from $(ds.path), so writing over " *
            "it would destroy what is being read; call " *
            "`Mestra.materialise!(ds)` first or write somewhere else"))
    end
    # A checked write goes to a file beside the target and is moved
    # into place once it has validated, so that a refusal leaves
    # whatever was at `path` alone.
    target = String(path)
    mode = check && isfile(target) ? filemode(target) & 0o7777 : nothing
    directory = check ? mktempdir(dirname(abspath(target)); prefix = ".mestra-",
                                   cleanup = false) : nothing
    out = check ? joinpath(directory, "data.mes") : target
    src = nothing
    try
        src = ds.path !== nothing && any_unread(ds) ?
              HDF5.h5open(ds.path, "r") : nothing
        HDF5.h5open(out, "w"; libver_bounds = WRITER_LIBVER) do f
            write_file(f, ds, src)
        end
        if src !== nothing
            close(src)
            src = nothing
        end
        if check
            r = validate(out)
            if !isempty(r.errors)
                bad = [f for f in r.findings if startswith(f.rule, "E")]
                throw(MestraError(first(bad).rule, target,
                    "this dataset does not validate, so nothing was written. " *
                    "Fix what the findings name, or pass `check = false` to " *
                    "write it anyway:\n" *
                    join(["  " * sprint(show, f) for f in bad], "\n")))
            end
            mode === nothing || chmod(out, mode)
            publish_file(out, target)
        end
    finally
        src === nothing || close(src)
        directory === nothing || rm(directory; recursive = true, force = true)
    end
    return target
end

"""The staging and the publish both happen in the destination's
directory (`docs/compatibility.md`, "Replacing files"), so a
destination that cannot take a file is refused here, naming `path`,
before anything is validated, staged or created.  An I/O failure after
this point is the operating system's error, and leaves an existing
destination as it was."""
function check_destination(path::String)
    isempty(path) && throw(MestraError(nothing,
        "`path` is empty; pass the name of the file to write"))
    isdir(path) && throw(MestraError(nothing, path,
        "`path` names a directory, which a file cannot replace; pass a " *
        "file name"))
    parent = dirname(abspath(path))
    isdir(parent) || throw(MestraError(nothing, path,
        "the directory $(parent) does not exist; the file is staged and " *
        "published there, so create it first"))
    (Sys.iswritable(parent) && Sys.isexecutable(parent)) ||
        throw(MestraError(nothing, path,
            "the directory $(parent) is not writable; the file is staged " *
            "and published there"))
    return path
end

function publish_file(source::AbstractString, target::AbstractString)
    # Use the same libuv primitive as Base's rename, without mv's
    # copy/delete fallback (or older Julia rename's fallback). Both paths
    # are on the same filesystem. A failed rename must leave the target.
    err = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), source, target)
    err < 0 && Base.uv_error("rename($(repr(source)), $(repr(target)))", err)
    return target
end

any_unread(ds::Dataset) =
    any(s -> !is_callable_slot(s) && raw_data(s) === nothing, all_slots(ds)) ||
    any(k -> raw_values(k) === nothing, Base.values(ds.keys))

slot_data(ds::Dataset, s::Slot, src) =
    raw_data(s) !== nothing ? raw_data(s) :
    src === nothing ? throw(MestraError(nothing,
        "slot $(s.name) holds no data")) : HDF5.read(src[s.path])

key_values(ds::Dataset, k::KeyColumn, src) =
    raw_values(k) !== nothing ? raw_values(k) :
    src === nothing ? throw(MestraError(nothing,
        "key $(k.name) holds no data")) : read_column(src[k.path])

"""Fill in what a writer can decide for itself: the alignment flag,
the support ids, and the shape of anything built from arrays."""
function prepare_for_write!(ds::Dataset)
    ds.aligned = length(ds.supports) <= 1
    if ds.aligned
        ds.row_support = nothing
    elseif ds.row_support === nothing
        throw(MestraError("E28", "/row_support",
            "a file with more than one support says which support each " *
            "row is on; call `set_row_support!(ds, indices)`"))
    end
    for s in ds.supports
        isempty(s.support_id) && (s.support_id = support_id(s))
    end
    for k in Base.values(ds.keys)
        legal_name(k.name) || throw(MestraError("E33", k.path,
            "`$(k.name)` is not a legal netCDF-4 name; rename the key"))
        reserved(k.name) && throw(MestraError("E33", k.path,
            "`$(k.name)` begins with the reserved prefix `mestra_`; " *
            "rename the key"))
    end
    for s in all_slots(ds)
        legal_name(s.name) || throw(MestraError("E33", s.path,
            "`$(s.name)` is not a legal netCDF-4 name; rename the slot"))
        reserved(s.name) && throw(MestraError("E33", s.path,
            "`$(s.name)` begins with the reserved prefix `mestra_`; " *
            "rename the slot"))
        s.source == "data" || startswith(s.source, "callable:") ||
            throw(MestraError("E36", s.path,
                "`source` is `data` or `callable:<id>`, not " *
                "`$(s.source)`; call `set_callable!(slot, id, output)`"))
        if is_callable_slot(s)
            haskey(ds.callables, callable_id(s)) || throw(MestraError("E14",
                s.path,
                "this slot names callable `$(callable_id(s))`, which the " *
                "dataset does not hold; `add_callable!(ds, id, c)` first"))
        end
        if s.location !== :scalar && !is_callable_slot(s)
            s.components === nothing && (s.components = s.dshape[end])
            s.components == s.dshape[end] || throw(MestraError("E31", s.path,
                "`components` says $(s.components) over a component " *
                "dimension of $(s.dshape[end])"))
        end
    end
    return ds
end

function write_file(f::HDF5.File, ds::Dataset, src)
    write_string_attr(f, "created", ds.created)
    write_string_attr(f, "format", ds.format)
    write_string_attr(f, "writer", ds.writer)
    write_bool_attr(f, "aligned", ds.aligned)
    ds.generalisation_group === nothing ||
        write_string_attr(f, "generalisation_group", ds.generalisation_group)
    for a in ds.extra_root_attrs
        write_raw_attr(f, a)
    end

    scales = Dict{String,HDF5.Dataset}()
    scales["row"] = create_scale(f, "row", ds.nrows; unlimited = true)
    for n in component_lengths(ds)
        scales["component_$(n)"] = create_scale(f, "component_$(n)", n)
    end
    for n in draw_lengths(ds)
        scales["draw_$(n)"] = create_scale(f, "draw_$(n)", n)
    end
    for k in group_keys(ds)
        n = n_categories(ds, ds.keys[k].category)
        scales["group_$(k)"] = create_scale(f, "group_$(k)", n)
    end
    for t in sort(collect(Base.keys(ds.categories)), by = codeunits)
        scales["category_$(t)"] =
            create_scale(f, "category_$(t)", length(ds.categories[t].entries))
    end

    if !isempty(ds.categories) || "categories" in ds.container_groups
        g = create_group(f, "categories")
        for t in sort(collect(Base.keys(ds.categories)), by = codeunits)
            tab = ds.categories[t]
            n = max(1, tab.strsize)
            d = create_raw_dataset(g, t, fixed_string_type(n),
                                   [length(tab.entries)],
                                   [length(tab.entries)],
                                   string_records(tab.entries, n))
            attach_scale!(d, scales["category_$(t)"], 0)
        end
    end

    if !isempty(ds.keys) || "keys" in ds.container_groups
        g = create_group(f, "keys")
        for name in key_order(ds)
            write_key(g, ds, ds.keys[name], scales, src)
        end
    end

    if !isempty(ds.scalars) || "scalars" in ds.container_groups
        g = create_group(f, "scalars")
        for name in sort(collect(Base.keys(ds.scalars)), by = codeunits)
            write_slot(g, ds, ds.scalars[name], scales, nothing, src)
        end
    end

    if ds.row_support !== nothing
        d = create_raw_dataset(f, "row_support", hdf5_type(Int32),
                               [ds.nrows], [-1],
                               raw_bytes(Int32.(ds.row_support));
                               chunk = [default_chunk_rows(4, Int[], ds.nrows)])
        attach_scale!(d, scales["row"], 0)
    end

    order = support_order(ds)
    if !isempty(order) || "supports" in ds.container_groups
        g = create_group(f, "supports")
        counts = rows_per_support(ds)
        for (i, s) in pairs(order)
            write_support(g, ds, s, scales, counts[i], src)
        end
    end

    if !isempty(ds.callables) || "callables" in ds.container_groups
        g = create_group(f, "callables")
        for id in sort(collect(Base.keys(ds.callables)), by = codeunits)
            c = ds.callables[id]
            legal_name(id) && !reserved(id) || throw(MestraError("E33",
                "callable id $(id) is not a legal producer-chosen name"))
            sub = create_group(g, id)
            c.type === nothing && throw(MestraError("E15",
                "callable $(id) has no `type`"))
            write_string_attr(sub, "type", c.type)
            c.repr === nothing || write_string_attr(sub, "repr", c.repr)
            for k in ("type", "repr")
                haskey(c.dict, k) && throw(MestraError("E32",
                    "a callable dictionary may not have a top-level `$(k)`"))
            end
            write_dict(sub, c.dict)
        end
    end

    ds.notes === nothing || restore_group(f, ds.notes)
    ds.private === nothing || restore_group(f, ds.private)
    for g in ds.extra_root_groups
        restore_group(f, g)
    end
    return nothing
end

function write_key(g, ds::Dataset, k::KeyColumn, scales, src)
    vals = key_values(ds, k, src)
    n = length(vals)
    if k.eltype === String
        size = max(1, maximum(vcat([ncodeunits(s) for s in vals],
                                   [k.strsize])))
        dt = fixed_string_type(size)
        raw = string_records(vals, size)
        item = size
    else
        dt = hdf5_type(k.eltype)
        raw = raw_bytes(convert(Vector{k.eltype}, vals))
        item = itemsize_of(k.eltype)
    end
    chunk = k.chunk === nothing ?
            [default_chunk_rows(item, Int[], ds.nrows)] : k.chunk
    d = create_raw_dataset(g, k.name, dt, [n], [-1], raw; chunk = chunk)
    attach_scale!(d, scales["row"], 0)
    k.role === nothing || write_string_attr(d, "role", String(k.role))
    k.units === nothing || write_string_attr(d, "units", k.units)
    k.lower === nothing || write_float_attr(d, "lower", k.lower)
    k.upper === nothing || write_float_attr(d, "upper", k.upper)
    k.category === nothing || write_string_attr(d, "category", k.category)
    k.trajectory_group === nothing ||
        write_string_attr(d, "trajectory_group", k.trajectory_group)
    k.parent === nothing || write_string_attr(d, "parent", k.parent)
    return d
end

function slot_attrs(obj, s::Slot)
    s.role === nothing || write_string_attr(obj, "role", String(s.role))
    s.varies === nothing || write_string_attr(obj, "varies", s.varies)
    s.units === nothing || write_string_attr(obj, "units", s.units)
    s.components === nothing || write_int_attr(obj, "components", s.components)
    write_string_attr(obj, "source", s.source)
    s.output === nothing || write_string_attr(obj, "output", s.output)
    s.statistic === nothing ||
        write_string_attr(obj, "statistic", s.statistic)
    s.of === nothing || write_string_attr(obj, "of", s.of)
    s.quantile === nothing || write_float_attr(obj, "quantile", s.quantile)
    s.level === nothing || write_float_attr(obj, "level", s.level)
    s.method === nothing || write_string_attr(obj, "method", s.method)
    s.category === nothing || write_string_attr(obj, "category", s.category)
    s.recomputed === nothing ||
        write_bool_attr(obj, "recomputed", s.recomputed)
    s.derived_from === nothing ||
        write_string_attr(obj, "derived_from", s.derived_from)
    s.recipe === nothing || write_string_attr(obj, "recipe", s.recipe)
    s.reference === nothing || write_string_attr(obj, "reference", s.reference)
    return nothing
end

function write_slot(parent, ds::Dataset, s::Slot, scales, local_scales, src)
    if is_callable_slot(s)
        # Section 19: a slot served by a callable is an empty group
        # carrying the slot's attributes and no data.
        g = create_group(parent, s.name)
        slot_attrs(g, s)
        return g
    end
    data = slot_data(ds, s, src)
    if s.location === :scalar
        dt = hdf5_type(s.eltype)
        chunk = s.chunk === nothing ?
                [default_chunk_rows(itemsize_of(s.eltype), Int[], ds.nrows)] :
                s.chunk
        d = create_raw_dataset(parent, s.name, dt, [length(data)], [-1],
                               raw_bytes(data); chunk = chunk,
                               deflate = s.deflate, shuffle = s.shuffle)
        attach_scale!(d, scales["row"], 0)
        slot_attrs(d, s)
        return d
    end
    names = canonical_disk_dims(s)
    cdims = collect(s.dshape)
    rowaxis = findfirst(==("row"), names)
    cmax = copy(cdims)
    chunk = nothing
    if rowaxis == 1
        cmax[1] = -1
        if s.chunk === nothing
            rowscale = local_scales !== nothing && haskey(local_scales, "row") ?
                       local_scales["row"] : scales["row"]
            nrows = disk_shape(rowscale)[1][1]
            chunk = vcat(default_chunk_rows(itemsize_of(s.eltype),
                                            cdims[2:end], nrows),
                         cdims[2:end])
        else
            chunk = copy(s.chunk)
        end
    end
    dt = hdf5_type(s.eltype)
    d = create_raw_dataset(parent, s.name, dt, cdims, cmax, raw_bytes(data);
                           chunk = chunk, deflate = s.deflate,
                           shuffle = s.shuffle)
    for (axis, nm) in pairs(names)
        sc = local_scales !== nothing && haskey(local_scales, nm) ?
             local_scales[nm] : get(scales, nm, nothing)
        sc === nothing && throw(MestraError("E25",
            "no dimension scale called $(nm) for slot $(s.name)"))
        attach_scale!(d, sc, axis - 1)
    end
    slot_attrs(d, s)
    return d
end

function write_support(parent, ds::Dataset, s::Support, scales,
                       local_rows::Int, src)
    g = create_group(parent, s.name)
    local_scales = Dict{String,HDF5.Dataset}()
    if s.kind != "none"
        local_scales["node"] = create_scale(g, "node", s.n_nodes)
    end
    if s.n_cells > 0
        local_scales["cell"] = create_scale(g, "cell", s.n_cells)
        local_scales["cell_plus_one"] =
            create_scale(g, "cell_plus_one", s.n_cells + 1)
        local_scales["index"] =
            create_scale(g, "index", length(s.cell_connectivity))
        d = create_raw_dataset(g, "cell_types", hdf5_type(UInt8),
                               [s.n_cells], [s.n_cells],
                               collect(UInt8.(s.cell_types)))
        attach_scale!(d, local_scales["cell"], 0)
        d = create_raw_dataset(g, "cell_offsets", hdf5_type(Int64),
                               [s.n_cells + 1], [s.n_cells + 1],
                               raw_bytes(Int64.(s.cell_offsets)))
        attach_scale!(d, local_scales["cell_plus_one"], 0)
        d = create_raw_dataset(g, "cell_connectivity", hdf5_type(Int64),
                               [length(s.cell_connectivity)],
                               [length(s.cell_connectivity)],
                               raw_bytes(Int64.(s.cell_connectivity)))
        attach_scale!(d, local_scales["index"], 0)
    end
    write_string_attr(g, "kind", s.kind)
    write_int_attr(g, "n_nodes", s.n_nodes)
    write_int_attr(g, "n_cells", s.n_cells)
    write_string_attr(g, "support_id", s.support_id)

    # Section 21: a support-local `row`, only in an unaligned file and
    # only where the support carries an array that varies along it.  A
    # slot a callable serves is such an array though it holds no data.
    if !ds.aligned
        varies_row = any(x -> x.varies == "row", support_slots(s))
        varies_row && (local_scales["row"] =
            create_scale(g, "row", local_rows; unlimited = true))
    end

    s.coordinates === nothing ||
        write_slot(g, ds, s.coordinates, scales, local_scales, src)
    if !isempty(s.node_arrays)
        na = create_group(g, "node_arrays")
        for n in sort(collect(Base.keys(s.node_arrays)), by = codeunits)
            write_slot(na, ds, s.node_arrays[n], scales, local_scales, src)
        end
    end
    if !isempty(s.cell_arrays)
        ca = create_group(g, "cell_arrays")
        for n in sort(collect(Base.keys(s.cell_arrays)), by = codeunits)
            write_slot(ca, ds, s.cell_arrays[n], scales, local_scales, src)
        end
    end
    for x in s.extra_groups
        restore_group(g, x)
    end
    return g
end

function support_slots(s::Support)
    out = Slot[]
    s.coordinates === nothing || push!(out, s.coordinates)
    append!(out, Base.values(s.node_arrays))
    append!(out, Base.values(s.cell_arrays))
    return out
end

"""Write back a group this format does not own -- `/notes`, `/private`
or a group a later version added -- exactly as it was read.

Sections 12 and 29: a producer's private part is copied and never
interpreted, so its dtypes, its filters, its chunking and any
dimension scale of its own come back unchanged, and section 30's
structural equality holds across a round trip.  The datasets are all
made first and the scales attached afterwards, because a scale may sit
in a subgroup of the dataset that uses it or the other way about."""
function restore_group(parent, g::RawGroupCopy)
    made = Dict{String,HDF5.Dataset}()
    pending = Tuple{HDF5.Dataset,String,Int}[]
    h = restore_into!(parent, g, "", made, pending)
    for (d, rel, axis) in pending
        sc = startswith(rel, "/") ? outside_scale(parent, rel) :
             get(made, rel, nothing)
        sc === nothing && continue
        attach_scale!(d, sc, axis)
    end
    return h
end

"""A dimension scale outside a copied group, by its path in the file
being written, or nothing when the rewrite did not write one there."""
function outside_scale(parent, path::String)
    f = HDF5.file(parent)
    haskey(f, path) || return nothing
    obj = f[path]
    return obj isa HDF5.Dataset && is_scale(obj) ? obj : nothing
end

function restore_into!(parent, g::RawGroupCopy, base::String, made, pending)
    h = create_group(parent, g.name)
    for a in g.attrs
        write_raw_attr(h, a)
    end
    for d in g.datasets
        rel = isempty(base) ? d.name : base * "/" * d.name
        ds_ = create_raw_dataset(h, d.name, raw_datatype(d.ti), d.cdims,
                                 d.cmax, d.raw; chunk = d.chunk,
                                 deflate = d.deflate, shuffle = d.shuffle,
                                 attr_order = d.attr_order)
        made[rel] = ds_
        if d.scale_name !== nothing
            # H5DSset_scale writes CLASS and NAME together, and a
            # scale that carried no NAME is copied back with none
            # rather than with an empty one.
            isempty(d.scale_name) ?
                write_string_attr(ds_, "CLASS", "DIMENSION_SCALE") :
                HDF5.API.h5ds_set_scale(ds_, d.scale_name)
        end
        for (axis, target) in pairs(d.attached)
            target === nothing && continue
            push!(pending, (ds_, target, axis - 1))
        end
        for a in d.attrs
            write_raw_attr(ds_, a)
        end
    end
    for sub in g.groups
        restore_into!(h, sub, isempty(base) ? sub.name : base * "/" * sub.name,
                      made, pending)
    end
    return h
end
