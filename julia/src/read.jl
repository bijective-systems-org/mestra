# The reader.
#
# Section 29: opening a file reads no array.  Everything below the
# attribute and dataspace level is read only when it is asked for,
# and a slot can be read for a range of rows on its own.
#
# A file is untrusted input.  The reader opens a child only through a
# hard link, checks what kind of object it got before using it, checks
# every declared shape before sizing anything from it, and records what
# it would not follow (E40) or could not read (E41) in `ds.findings`
# rather than failing the whole open.

const ROOT_ATTRS = Set(["format", "writer", "created", "aligned",
                        "generalisation_group"])
const ROOT_GROUPS = Set(["keys", "scalars", "categories", "supports",
                         "callables", "notes", "private"])

"""Every dimension scale in the file, by link name and by the object
reference a dataset's DIMENSION_LIST holds.  The map is built once per
open, during this package's own bounded walk, and every attached scale
is resolved through it (section 21), so resolving one costs a lookup
and not a scan of the scale's REFERENCE_LIST."""
struct ScaleIndex
    all::Vector{Pair{String,HDF5.Dataset}}
    paths::Vector{String}
    byref::Dict{HDF5.Reference,Int}
end

function ScaleIndex(f::HDF5.File)
    found = collect_scales(f)
    all = Pair{String,HDF5.Dataset}[name => d for (name, _, d, _) in found]
    paths = String[p for (_, p, _, _) in found]
    byref = Dict{HDF5.Reference,Int}()
    for (i, (_, _, _, ref)) in enumerate(found)
        ref === nothing && continue
        haskey(byref, ref) || (byref[ref] = i)
    end
    return ScaleIndex(all, paths, byref)
end

"""For each C-order axis of `d`: how many dimension scales are
attached, the link name of the one scale when exactly one is and this
file holds it, that scale's dataset, and its path.  A count of -1 says
the library would not give the dataset's DIMENSION_LIST at all.

Every one of those comes from the dataset's own DIMENSION_LIST and the
map of section 21, and none of it from the scale's REFERENCE_LIST,
which section 21 calls informational: a scale whose REFERENCE_LIST is
missing, short or stale still names the axes that point at it."""
function axis_scales(d::HDF5.Dataset, idx::ScaleIndex)
    T = Tuple{Int,Union{String,Nothing},Union{Nothing,HDF5.Dataset},
              Union{String,Nothing}}
    cdims, _ = try
        disk_shape(d)
    catch
        return T[]
    end
    dl = dimension_list(d)
    bad = dl === :unreadable
    out = T[]
    for axis in 0:(length(cdims) - 1)
        if bad
            push!(out, (-1, nothing, nothing, nothing))
            continue
        end
        refs = axis_refs(dl, axis)
        if length(refs) == 1
            i = get(idx.byref, refs[1], 0)
            push!(out, i == 0 ? (1, nothing, nothing, nothing) :
                       (1, idx.all[i].first, idx.all[i].second, idx.paths[i]))
        else
            push!(out, (length(refs), nothing, nothing, nothing))
        end
    end
    return out
end

"""The link name of the scale on each C-order axis of `d`, or nothing
where no single scale this file holds is attached."""
axis_scale_names(d::HDF5.Dataset, idx::ScaleIndex) =
    Union{String,Nothing}[t[2] for t in axis_scales(d, idx)]

function slot_ldims(d::HDF5.Dataset, idx::ScaleIndex)
    return Symbol[n === nothing ? :unknown : logical_dim(n)
                  for n in axis_scale_names(d, idx)]
end

sattr(a::Dict{String,RawAttr}, name) =
    haskey(a, name) && a[name].value isa AbstractString ?
    String(a[name].value) : nothing
fattr(a::Dict{String,RawAttr}, name) =
    haskey(a, name) && a[name].value isa Real && !(a[name].value isa Bool) ?
    Float64(a[name].value) : nothing
iattr(a::Dict{String,RawAttr}, name) =
    haskey(a, name) && a[name].value isa Real && !(a[name].value isa Bool) ?
    Int(a[name].value) : nothing
# Section 18 gives a boolean the values 0 and 1 and no other, so an
# int8 holding anything else decodes to nothing and this returns
# nothing too: the reader has no value, rather than a value it made up.
battr(a::Dict{String,RawAttr}, name) =
    haskey(a, name) && a[name].value isa Bool ? a[name].value : nothing

"""True when `name` is there and is not a boolean this format defines,
which is E19 and which the reader says rather than choosing a meaning
for it."""
illegal_bool(a::Dict{String,RawAttr}, name) =
    haskey(a, name) && !(a[name].value isa Bool)

"""
    Mestra.read(path; lazy = true, strict = true) -> Dataset

Open a `.mes` file.  With `lazy = true`, which is the default, this
reads attributes and dataspaces only, so it reports the row count, the
keys with their roles and bounds, the supports with their ids and
every slot with its attributes without touching an array.  Ask for an
array with `Mestra.values(ds, slot)`, or for a range of rows with
`Mestra.rows(ds, slot, 1:10)`.

With `lazy = false` every stored array is read at once.

A strict read, which is the default, refuses a file that breaks one of
the structural rules of `docs/api-conventions.md` section 2 --
`Mestra.STRUCTURAL_RULES`, which is E01, E16, E19, E25, E26, E29, E30,
E40 and E41 -- with a `MestraError` carrying the first it found and
naming every other one in its message.  Those are what a file is made
of rather than what it means, so deciding them costs attributes,
dataspaces, link types, dimension-scale structure and the file's
category tables, and not one array element: a strict read still opens
a file that declares a trillion numbers it does not hold.  A slot is
never read to decide them, so this read and `lazy = false` refuse the
same file with the same rule (`docs/api-conventions.md` section 7).

A semantic fault never stops a read, strict or not: a missing unit, a
bad split, a support id that does not match its cells are all things a
user opens a file to find out, and `Mestra.validate` and `Mestra.info`
are how they find out.  `strict = false` returns the dataset for a
structurally broken file too, with what the reader would not follow or
could not read listed in `ds.findings`, each a `Finding` with its
rule, its path and a sentence.

A `.mes` file is somebody else's, and every number in it about itself
is that somebody's claim, so this reader checks each one before it
acts on it.  It follows a hard link and nothing else, reporting a soft
or external link as E40 and never opening it; it reports an object
that is not the kind the format requires, or that it could not read at
all, as E41, and carries on past it, so that one bad object never
hides what comes after it; it refuses a read above `max_elements`
(`Mestra.DEFAULT_MAX_ELEMENTS`) or `Mestra.MAX_READ_BYTES[]`, chunk
included, so a file may declare a trillion elements and cost what one
row costs; and it walks nothing deeper than `Mestra.MAX_DEPTH`.  A
lazy read and a row range are not subject to the element cap, because
they never materialise the whole dataset, so a file may be readable
one way and E41 the other, which is what section 29 says.
"""
function read(path::AbstractString; lazy::Bool = true, strict::Bool = true,
              max_elements::Integer = DEFAULT_MAX_ELEMENTS)
    isfile(String(path)) || throw(MestraError("E01",
        "no file at $(path)"))
    if strict
        r = validate(String(path); max_elements = max_elements,
                     structural = true)
        isempty(r.findings) || refuse(r.findings, path)
    end
    f = try
        HDF5.h5open(String(path), "r")
    catch e
        throw(MestraError("E01",
            "$(path) is not a file this reader can open as HDF5: " *
            first(sprint(showerror, e), 200)))
    end
    try
        return read_dataset(f, String(path), lazy, Int(max_elements))
    finally
        close(f)
    end
end

"""How many structural findings a refusal names one by one before it
says how many more there are."""
const REFUSAL_FINDINGS = 8

"""How a strict read refuses: the first structural rule it found, the
object it was about, what it says, and then every other structural
rule the file breaks.

Section 30 asks a reader to refuse a hostile file "with the same ids
rather than return something", and the ids a file must produce are not
always the one a walk of it reaches first: `wrong_object_kinds` is
required to name E41 and the first thing wrong with it is a key stored
as a group (E30).  So the refusal names them all."""
function refuse(findings::Vector{Finding}, path)
    head = first(findings)
    rules = sort(unique([f.rule for f in findings]))
    base = basename(String(path))
    said = if length(findings) == 1
        "$(base) breaks a rule of section 14 that says what a file is " *
        "made of"
    else
        listed = ["$(f.rule) $(f.path)"
                  for f in findings[2:min(end, REFUSAL_FINDINGS + 1)]]
        more = length(findings) - 1 - length(listed)
        "$(base) breaks $(length(rules)) rule(s) of section 14 that say " *
        "what a file is made of; it also breaks " * join(listed, ", ") *
        (more > 0 ? ", and $(more) more" : "")
    end
    throw(MestraError(head.rule, head.path,
        head.message * ". " * said *
        "; `Mestra.report(\"$(path)\")` says what else it breaks, and " *
        "`Mestra.read(path; strict = false)` opens it anyway"))
end

refuse(f::Finding, path) = refuse([f], path)

note!(ds::Dataset, rule, path, msg) =
    push!(ds.findings, Finding(rule, String(path), String(msg)))

"""Report every child of `g` this reader will not follow (E40), and
return the names it will."""
function hard_children!(ds::Dataset, g, path::AbstractString)
    out = String[]
    for (name, kind) in child_links(g)
        if kind === :hard
            push!(out, name)
        else
            note!(ds, "E40", "$(path)/$(name)",
                  "a $(kind) link; this reader follows hard links only")
        end
    end
    return out
end

function read_dataset(f::HDF5.File, path::String, lazy::Bool,
                      max_elements::Int = DEFAULT_MAX_ELEMENTS)
    idx = ScaleIndex(f)
    root = own_attrs(f)
    ds = Dataset(writer = something(sattr(root, "writer"), ""),
                 created = something(sattr(root, "created"), ""))
    ds.format = something(sattr(root, "format"), "")
    ds.aligned = something(battr(root, "aligned"), true)
    illegal_bool(root, "aligned") && note!(ds, "E19", "/",
        "`aligned` is not a boolean of section 18, whose only values " *
        "are 0 and 1; this reader has no alignment claim from this file")
    ds.generalisation_group = sattr(root, "generalisation_group")
    ds.path = path
    ds.lazy = lazy
    ds.max_elements = max_elements
    for (name, a) in root
        name in ROOT_ATTRS || push!(ds.extra_root_attrs, a)
    end
    sort!(ds.extra_root_attrs, by = a -> a.name)

    rowscale = hard_child(f, "row")
    ds.nrows = rowscale isa HDF5.Dataset && !isempty(disk_shape(rowscale)[1]) ?
               disk_shape(rowscale)[1][1] : 0

    rootnames = hard_children!(ds, f, "")
    for g in ROOT_GROUPS
        g in rootnames && hard_child(f, g) isa HDF5.Group &&
            push!(ds.container_groups, g)
    end

    cats = "categories" in rootnames ? hard_child(f, "categories") : nothing
    if cats isa HDF5.Group
        for name in hard_children!(ds, cats, "/categories")
            d = hard_child(cats, name)
            if !(d isa HDF5.Dataset)
                note!(ds, "E41", "/categories/$(name)",
                      "a category table that is not a dataset")
                continue
            end
            try
                ti, recs = read_string_records(d;
                                               max_elements = max_elements)
                # Section 25: "A reader that cannot recover the bytes
                # of a string must say so rather than return something
                # else."  An entry that is not valid UTF-8 is E26, and
                # the table is not handed back at all, because a Julia
                # String built from those bytes is something else.
                if !all(check_string_bytes, recs)
                    note!(ds, "E26", "/categories/$(name)",
                          "an entry is not valid UTF-8 or holds a NUL " *
                          "byte before its trailing padding; this reader " *
                          "will not hand back a string it cannot recover")
                else
                    ds.categories[name] = CategoryTable(
                        name, [String(strip_nul(r)) for r in recs], ti.size)
                end
            catch e
                note!(ds, rule_of(e), "/categories/$(name)", message_of(e))
            end
        end
    end

    kg = "keys" in rootnames ? hard_child(f, "keys") : nothing
    if kg isa HDF5.Group
        for name in hard_children!(ds, kg, "/keys")
            obj = hard_child(kg, name)
            if !(obj isa HDF5.Dataset)
                note!(ds, "E41", "/keys/$(name)",
                      "a key that is not a dataset; section 19 says a key " *
                      "is a dataset over `row`")
                continue
            end
            try
                ds.keys[name] = read_key(ds, obj, name, idx, lazy)
            catch e
                note!(ds, rule_of(e), "/keys/$(name)", message_of(e))
            end
        end
    end

    sg = "scalars" in rootnames ? hard_child(f, "scalars") : nothing
    if sg isa HDF5.Group
        for name in hard_children!(ds, sg, "/scalars")
            obj = hard_child(sg, name)
            obj === nothing && continue
            try
                ds.scalars[name] = read_slot(ds, obj, name, :scalar,
                                             nothing, idx, lazy)
            catch e
                note!(ds, rule_of(e), "/scalars/$(name)", message_of(e))
            end
        end
    end

    rs = "row_support" in rootnames ? hard_child(f, "row_support") : nothing
    if rs isa HDF5.Dataset
        try
            ds.row_support = Int32.(vec(safe_read(rs;
                                                  max_elements = max_elements)))
        catch e
            note!(ds, rule_of(e), "/row_support", message_of(e))
        end
    end

    sup = "supports" in rootnames ? hard_child(f, "supports") : nothing
    if sup isa HDF5.Group
        for name in hard_children!(ds, sup, "/supports")
            obj = hard_child(sup, name)
            if !(obj isa HDF5.Group)
                note!(ds, "E41", "/supports/$(name)",
                      "a support that is not a group")
                continue
            end
            try
                push!(ds.supports, read_support(ds, obj, name, idx, lazy))
            catch e
                note!(ds, rule_of(e), "/supports/$(name)", message_of(e))
            end
        end
    end

    cg = "callables" in rootnames ? hard_child(f, "callables") : nothing
    if cg isa HDF5.Group
        for id in hard_children!(ds, cg, "/callables")
            g = hard_child(cg, id)
            if !(g isa HDF5.Group)
                note!(ds, "E41", "/callables/$(id)",
                      "a callable that is not a group")
                continue
            end
            a = own_attrs(g)
            dict = try
                read_dict(ds, g, "/callables/$(id)"; toplevel = true,
                          max_elements = max_elements)
            catch e
                note!(ds, rule_of(e), "/callables/$(id)", message_of(e))
                Dict{String,Any}()
            end
            ds.callables[id] = CallableRef(id, sattr(a, "type");
                                           repr = sattr(a, "repr"),
                                           dict = dict)
        end
    end

    for (name, field) in (("notes", :notes), ("private", :private))
        name in rootnames || continue
        g = hard_child(f, name)
        g isa HDF5.Group || continue
        try
            setfield!(ds, field,
                      snapshot_group(ds, g, name, "/" * name, 0, idx))
        catch e
            note!(ds, rule_of(e), "/" * name, message_of(e))
        end
    end
    for name in rootnames
        name in ROOT_GROUPS && continue
        obj = hard_child(f, name)
        obj isa HDF5.Group || continue
        try
            push!(ds.extra_root_groups,
                  snapshot_group(ds, obj, name, "/" * name, 0, idx))
        catch e
            note!(ds, rule_of(e), "/" * name, message_of(e))
        end
    end
    return ds
end

rule_of(e) = e isa MestraError && e.rule !== nothing ? e.rule : "E41"
message_of(e) = e isa MestraError ? e.msg :
                first(sprint(showerror, e), 200)

function read_key(ds::Dataset, d, name::String, idx::ScaleIndex,
                  lazy::Bool)
    a = own_attrs(d)
    role = sattr(a, "role")
    ti = type_info(HDF5.datatype(d))
    T = ti.class === :string ? String : julia_eltype(ti)
    _, chunk, _ = dataset_layout(d)
    k = KeyColumn(name, role === nothing ? nothing : Symbol(role);
                  units = sattr(a, "units"), lower = fattr(a, "lower"),
                  upper = fattr(a, "upper"), category = sattr(a, "category"),
                  trajectory_group = sattr(a, "trajectory_group"),
                  parent = sattr(a, "parent"), eltype = T,
                  strsize = ti.class === :string ? ti.size : 1,
                  chunk = chunk, path = "/keys/" * name)
    if !lazy
        try
            k.values = read_column(d; max_elements = ds.max_elements)
        catch e
            note!(ds, rule_of(e), k.path, message_of(e))
        end
    end
    return k
end

function read_column(d::HDF5.Dataset;
                     max_elements::Integer = DEFAULT_MAX_ELEMENTS)
    return vec(safe_read(d; max_elements = max_elements))
end

function read_slot(ds::Dataset, obj, name::String, location::Symbol,
                   support::Union{Nothing,String}, idx::ScaleIndex,
                   lazy::Bool)
    a = own_attrs(obj)
    path = try
        HDF5.name(obj)
    catch
        "?"
    end
    role = sattr(a, "role")
    s = Slot(name, location; support = support,
             source = something(sattr(a, "source"), ""),
             output = sattr(a, "output"),
             role = role === nothing ? nothing : Symbol(role),
             varies = sattr(a, "varies"), units = sattr(a, "units"),
             components = iattr(a, "components"),
             statistic = sattr(a, "statistic"), of = sattr(a, "of"),
             quantile = fattr(a, "quantile"), level = fattr(a, "level"),
             method = sattr(a, "method"), category = sattr(a, "category"),
             recomputed = haskey(a, "recomputed") ? battr(a, "recomputed") :
                          nothing,
             derived_from = sattr(a, "derived_from"),
             recipe = sattr(a, "recipe"), reference = sattr(a, "reference"),
             path = path)
    if obj isa HDF5.Dataset
        cdims, _ = disk_shape(obj)
        _, chunk, filters = dataset_layout(obj)
        s.dshape = cdims
        s.ldims = slot_ldims(obj, idx)
        ti = type_info(HDF5.datatype(obj))
        s.eltype = ti.class === :string ? String : uint8_eltype(ti)
        s.chunk = chunk
        for (fid, cd) in filters
            fid == 1 && (s.deflate = isempty(cd) ? 1 : cd[1])
            fid == 2 && (s.shuffle = true)
        end
        if !lazy
            try
                s.data = safe_read(obj; max_elements = ds.max_elements)
            catch e
                note!(ds, rule_of(e), path, message_of(e))
            end
        end
    end
    return s
end

function read_support(ds::Dataset, g, name::String, idx::ScaleIndex,
                      lazy::Bool)
    a = own_attrs(g)
    s = Support(name, something(sattr(a, "kind"), "");
                n_nodes = something(iattr(a, "n_nodes"), 0),
                n_cells = something(iattr(a, "n_cells"), 0),
                support_id = something(sattr(a, "support_id"), ""))
    base = "/supports/" * name
    children = hard_children!(ds, g, base)
    for (cname, T, setter) in (("cell_types", UInt8,
                                (v) -> (s.cell_types = v)),
                               ("cell_offsets", Int64,
                                (v) -> (s.cell_offsets = v)),
                               ("cell_connectivity", Int64,
                                (v) -> (s.cell_connectivity = v)))
        cname in children || continue
        d = hard_child(g, cname)
        if !(d isa HDF5.Dataset)
            note!(ds, "E41", "$(base)/$(cname)", "not a dataset")
            continue
        end
        try
            setter(T.(vec(safe_read(d; max_elements = ds.max_elements))))
        catch e
            note!(ds, rule_of(e), "$(base)/$(cname)", message_of(e))
        end
    end
    if "coordinates" in children
        c = hard_child(g, "coordinates")
        if c === nothing
            note!(ds, "E41", "$(base)/coordinates", "could not be opened")
        else
            s.coordinates = read_slot(ds, c, "coordinates", :node, name,
                                      idx, lazy)
            # An axis support's identity includes its coordinates, so
            # they are read even when the rest is lazy (section 24).
            if s.kind == "axis" && raw_data(s.coordinates) === nothing &&
               c isa HDF5.Dataset
                try
                    s.coordinates.data =
                        safe_read(c; max_elements = ds.max_elements)
                catch e
                    note!(ds, rule_of(e), "$(base)/coordinates",
                          message_of(e))
                end
            end
        end
    end
    for (sub, loc, store) in (("node_arrays", :node, s.node_arrays),
                              ("cell_arrays", :cell, s.cell_arrays))
        sub in children || continue
        sg = hard_child(g, sub)
        sg isa HDF5.Group || continue
        for n in hard_children!(ds, sg, "$(base)/$(sub)")
            obj = hard_child(sg, n)
            obj === nothing && continue
            try
                store[n] = read_slot(ds, obj, n, loc, name, idx, lazy)
            catch e
                note!(ds, rule_of(e), "$(base)/$(sub)/$(n)", message_of(e))
            end
        end
    end
    # A group this version does not know is copied, never interpreted,
    # so that a rewrite puts it back (section 28).
    for n in children
        n in SUPPORT_GROUPS && continue
        obj = hard_child(g, n)
        obj isa HDF5.Group || continue
        try
            push!(s.extra_groups,
                  snapshot_group(ds, obj, n, "$(base)/$(n)", 0, idx))
        catch e
            note!(ds, rule_of(e), "$(base)/$(n)", message_of(e))
        end
    end
    return s
end

const SUPPORT_GROUPS = ("node_arrays", "cell_arrays", "coordinates")

# An object this reader does not own is copied, never interpreted.
# The depth is capped because the file chooses it.
#
# `root` is the path the copy hangs from, so that a dimension scale a
# producer keeps inside its own group can be attached again by a path
# within the copy.  A scale outside the copied subtree is not carried:
# nothing this format defines attaches across one.
function snapshot_group(ds::Dataset, g::HDF5.Group, name::String,
                        path::String, depth::Int,
                        idx::Union{Nothing,ScaleIndex} = nothing,
                        root::String = path)
    attrs = RawAttr[a for a in raw_attrs(g)
                    if !(a.name in MACHINERY_ATTRS) && a.readable]
    dsets = RawDatasetCopy[]
    groups = RawGroupCopy[]
    if depth >= MAX_DEPTH
        note!(ds, "E41", path,
              "groups nested deeper than $(MAX_DEPTH); this reader " *
              "stops here rather than following a file's own depth")
        return RawGroupCopy(name, attrs, dsets, groups)
    end
    for n in hard_children!(ds, g, path)
        obj = hard_child(g, n)
        obj === nothing && continue
        if obj isa HDF5.Group
            push!(groups, snapshot_group(ds, obj, n, "$(path)/$(n)",
                                         depth + 1, idx, root))
        else
            try
                ti, raw, _ = read_raw_dataset(obj;
                                              max_elements = ds.max_elements)
                cdims, cmax = disk_shape(obj)
                _, chunk, filters = dataset_layout(obj)
                deflate = nothing
                shuffle = false
                for (fid, cd) in filters
                    fid == 1 && (deflate = isempty(cd) ? 1 : cd[1])
                    fid == 2 && (shuffle = true)
                end
                push!(dsets, RawDatasetCopy(n, ti, cdims, cmax, chunk, raw,
                    RawAttr[a for a in raw_attrs(obj)
                            if !(a.name in MACHINERY_ATTRS) && a.readable],
                    deflate, shuffle, copied_scale_name(obj),
                    scale_order_tracked(obj),
                    copied_attachments(obj, idx, root, length(cdims))))
            catch e
                note!(ds, rule_of(e), "$(path)/$(n)", message_of(e))
            end
        end
    end
    return RawGroupCopy(name, attrs, dsets, groups)
end

"""The NAME a copied dataset carries as a dimension scale, or nothing
when it is not one.  A scale with no NAME is copied as a scale with an
empty one, which is what it was."""
function copied_scale_name(d::HDF5.Dataset)
    is_scale(d) || return nothing
    a = try
        haskey(HDF5.attributes(d), "NAME") ? read_raw_attr(d, "NAME") : nothing
    catch
        nothing
    end
    return a !== nothing && a.value isa AbstractString ? String(a.value) : ""
end

"""For each C-order axis of a copied dataset, the path within the
copied subtree of the scale attached to it, or nothing."""
function copied_attachments(d::HDF5.Dataset, idx::Union{Nothing,ScaleIndex},
                            root::String, naxes::Int)
    out = Union{Nothing,String}[nothing for _ in 1:naxes]
    idx === nothing && return out
    prefix = root * "/"
    for (axis, t) in pairs(axis_scales(d, idx))
        axis <= naxes || break
        p = t[4]
        p === nothing && continue
        # A scale inside the copied subtree is named by its path within
        # it; one outside -- a support's `node`, the file's `row` -- by
        # its path in the file, which the rewrite has written too.
        out[axis] = startswith(p, prefix) ? p[(length(prefix) + 1):end] :
                    String(p)
    end
    return out
end

# ------------------------------------------------------ reading values

"""
    values(ds, slot) -> DimArray

The whole of one slot, with the name of each axis in the Julia order.
Julia is column major and this format is stored in C order, so the
axes arrive reversed: a field the file holds as (row, node, component)
comes back as (component, node, row).  Ask for the order you want with
`permute(v, (:row, :node, :component))`, never by axis position.
"""
function values(ds::Dataset, s::Slot;
                max_elements::Integer = ds.max_elements)
    raw_data(s) === nothing || return DimArray(raw_data(s), julia_dims(s))
    is_callable_slot(s) && throw(MestraError(nothing,
        "slot $(s.name) is served by callable $(callable_id(s)) and " *
        "holds no data; evaluate the dataset first"))
    ds.path === nothing && throw(MestraError(nothing,
        "slot $(s.name) holds no data and this dataset has no file"))
    a = HDF5.h5open(ds.path, "r") do f
        d = open_path(f, s.path)
        safe_read(d; max_elements = max_elements)
    end
    return DimArray(a, julia_dims(s))
end

"""Open an object by its path, following hard links only.  A path with
a soft or external link in it is refused (E40) rather than followed."""
function open_path(f::HDF5.File, path::AbstractString)
    here = f
    for part in split(String(path), '/'; keepempty = false)
        kind = link_type(here, part)
        kind === :hard || throw(MestraError(kind === :missing ? "E41" : "E40",
            "$(path): `$(part)` is a $(kind) link"))
        here = try
            here[String(part)]
        catch e
            throw(MestraError("E41", "$(path): " *
                              first(sprint(showerror, e), 200)))
        end
    end
    return here
end

"""
    values(ds, name::AbstractString) -> DimArray

The slot called `name`, found as `ds[name]` finds it.
"""
values(ds::Dataset, name::AbstractString) = values(ds, ds[name])

"""
    values(ds, key::KeyColumn) -> Vector

One key column.
"""
function values(ds::Dataset, k::KeyColumn;
                max_elements::Integer = ds.max_elements)
    raw_values(k) === nothing || return raw_values(k)
    ds.path === nothing && throw(MestraError(nothing,
        "key $(k.name) holds no data and this dataset has no file"))
    return HDF5.h5open(ds.path, "r") do f
        read_column(open_path(f, k.path); max_elements = max_elements)
    end
end

"""
    rows(ds, slot, range) -> DimArray

One slot for a range of rows, reading no other slot and no row outside
the range (section 29).  `range` is one based.  For a row-varying
array in an unaligned file the range indexes that support's own rows,
in the file's row order, which section 22 says is not the file's row
number.
"""
function rows(ds::Dataset, s::Slot, range::AbstractUnitRange;
              max_elements::Integer = ds.max_elements)
    is_callable_slot(s) && throw(MestraError(nothing,
        "slot $(s.name) is served by a callable and holds no data"))
    jdims = julia_dims(s)
    axis = findfirst(==(:row), jdims)
    axis === nothing && throw(MestraError(nothing,
        "slot $(s.name) has no row dimension"))
    jsize = julia_size(s)
    isempty(range) || (first(range) >= 1 && last(range) <= jsize[axis]) ||
        throw(MestraError(nothing,
            "rows $(range) outside the $(jsize[axis]) this slot holds"))
    # Size the answer from the range and the shape, both checked, and
    # never from what the file claims the whole slot holds.
    want = element_count([i == axis ? length(range) : jsize[i]
                          for i in 1:length(jsize)])
    want > max_elements && throw(MestraError("E41",
        "that range is $(want) elements, more than the $(max_elements) " *
        "this reader will materialise"))
    if raw_data(s) !== nothing
        sel = ntuple(i -> i == axis ? range : Colon(), length(jdims))
        return DimArray(raw_data(s)[sel...], jdims)
    end
    ds.path === nothing && throw(MestraError(nothing,
        "slot $(s.name) holds no data and this dataset has no file"))
    a = HDF5.h5open(ds.path, "r") do f
        d = open_path(f, s.path)
        d isa HDF5.Dataset || throw(MestraError("E41",
            "$(s.path) is not a dataset"))
        # The library reads whole chunks, so a chunk this reader
        # would not materialise costs the same for one row.
        check_chunk(d; max_elements = max_elements)
        sel = ntuple(i -> i == axis ? range : Colon(), length(jdims))
        try
            d[sel...]
        catch e
            throw(MestraError("E41", "$(s.path): " *
                              first(sprint(showerror, e), 200)))
        end
    end
    return DimArray(a, jdims)
end

"""
    rows(ds, name::AbstractString, range) -> DimArray

The slot called `name`, for a range of rows.
"""
rows(ds::Dataset, name::AbstractString, range::AbstractUnitRange;
     kwargs...) = rows(ds, ds[name], range; kwargs...)

"""
    materialise!(ds) -> Dataset

Read every stored array into memory.  A dataset read with
`lazy = false` is already materialised.
"""
function materialise!(ds::Dataset)
    ds.path === nothing && return ds
    HDF5.h5open(ds.path, "r") do f
        for (_, k) in ds.keys
            raw_values(k) === nothing || continue
            try
                k.values = read_column(open_path(f, k.path);
                                       max_elements = ds.max_elements)
            catch e
                note!(ds, rule_of(e), k.path, message_of(e))
            end
        end
        for s in all_slots(ds)
            is_callable_slot(s) && continue
            raw_data(s) === nothing || continue
            try
                d = open_path(f, s.path)
                d isa HDF5.Dataset || continue
                s.data = safe_read(d; max_elements = ds.max_elements)
            catch e
                note!(ds, rule_of(e), s.path, message_of(e))
            end
        end
    end
    ds.lazy = false
    return ds
end
