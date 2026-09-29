# Building a dataset from arrays.
#
# A handful of calls with sensible defaults: the dimensions follow
# from the names the caller gives the axes, the bounds follow from the
# data unless the caller says otherwise, and a support fills in its own
# id.  Anything that cannot legally be written raises a MestraError
# naming the rule of section 14 it breaks.

"""Turn an array the caller holds into the orientation this package
keeps, given the name of each of its axes.  The file's order is
(row | group | nothing, [draw], node | cell, component); the stored
orientation reverses it, so the array's linear memory is the file's
byte order."""
function to_stored(data::AbstractArray, dims, want::Vector{Symbol}, path)
    dims = Symbol[Symbol(d) for d in dims]
    length(dims) == ndims(data) || throw(MestraError("E04", path,
        "`dims` names $(length(dims)) axes for an array of " *
        "$(ndims(data)); give one name per axis"))
    a = data
    if !(:component in dims) && :component in want
        a = reshape(a, size(a)..., 1)
        dims = vcat(dims, :component)
    end
    Set(dims) == Set(want) || throw(MestraError("E04", path,
        "`dims` names the axes $(Tuple(dims)) where this slot needs " *
        "$(Tuple(want))"))
    perm = [findfirst(==(d), dims) for d in reverse(want)]
    return permutedims(collect(a), perm)
end

"""E33 at build time: a name the file could not hold.  The writer
checks again for a name changed after building; this is so that the
call that made the mistake is the one that is refused."""
function check_name(name::AbstractString, path)
    legal_name(name) || throw(MestraError("E33", path,
        "`$(name)` is not a legal netCDF-4 name: letters, digits, `_`, " *
        "`-`, `.` and `+` only, and no `/`; pass another `name`"))
    reserved(name) && throw(MestraError("E33", path,
        "names beginning with `$(RESERVED_PREFIX)` are reserved for the " *
        "container; pass another `name`"))
    return String(name)
end

"""`values` as one rectangular array of `T`, or E20 naming the slot
and the argument.  What `convert` says about a ragged array, text
where a number belongs or a value `T` cannot hold is a `MethodError`
or an `InexactError` that names no argument, and a caller building a
file should be told which one and what to pass.  `T` of `nothing`
keeps the array's own element type, checking only that every element
is a number."""
function numbers(values, T::Union{Nothing,Type}, path;
                 argument::AbstractString = "values")
    a = values isa AbstractArray ? values : collect(values)
    accepted = T === nothing ? Real :
               T <: AbstractString ? AbstractString :
               T <: Integer ? Integer : Real
    what = accepted === AbstractString ? "strings" :
           accepted === Integer ? "integers" : "numbers"
    i = Base.eltype(a) <: accepted ? nothing :
        findfirst(x -> !(x isa accepted), a)
    if i !== nothing
        x = a[i]
        throw(MestraError("E20", path, x isa AbstractArray ?
            "`$(argument)` must be one rectangular array of $(what), and " *
            "this is a $(typeof(a)), an array of arrays; pass one array " *
            "with one axis per dimension and the same shape in every row" :
            "`$(argument)` must be a rectangular array of $(what), and " *
            "these hold a $(typeof(x)) ($(repr(x))); pass $(what)"))
    end
    T === nothing && return a
    try
        return convert(Array{T}, a)
    catch e
        e isa InexactError || rethrow()
        throw(MestraError("E20", path,
            "`$(argument)` must be $(what) that $(T) can hold, and " *
            "$(repr(e.val)) is not; pass `eltype`, or convert the array"))
    end
end

"""E19 at build time: `units` is text (section 18), and the writer
never has to reject its own file over a number where a unit belongs."""
function check_units(units, path)
    (units === nothing || units isa AbstractString) ||
        throw(MestraError("E19", path,
            "`units` is text, and this $(typeof(units)) is not; pass the " *
            "unit as a string, \"1\" for a dimensionless one"))
    return units
end

"""The path an array slot will have, which a refusal names before the
slot exists."""
array_path(sup::Support, name::AbstractString, location::Symbol) =
    "/supports/$(sup.name)/" *
    (name == "coordinates" ? "coordinates" :
     (location === :cell ? "cell_arrays/" : "node_arrays/") * String(name))

"""Name the axes of an array whose axes the caller did not name.  A
square array says nothing about which reading is meant, so it is
refused here (E04) rather than written the wrong way round."""
function default_dims(data::AbstractArray, nsite::Int, location::Symbol,
                      path)
    axis = location === :cell ? :cell : :node
    nd = ndims(data)
    nd == 1 && return Symbol[axis]
    if nd == 2
        r, c = size(data)
        r == nsite && c == nsite && throw(MestraError("E04", path,
            "an array of shape ($(r), $(c)) on a support of $(nsite) " *
            "$(axis)s is either ($(axis), component) or (row, $(axis)); " *
            "say which with `dims`"))
        c == nsite && return Symbol[:row, axis]
        r == nsite && return Symbol[axis, :component]
        throw(MestraError("E05", path,
            "an array of shape ($(r), $(c)) has no axis of $(nsite) " *
            "$(axis)s; name the axes with `dims`"))
    end
    nd == 3 && return Symbol[:row, axis, :component]
    throw(MestraError("E04", path,
        "name the axes of a $(nd)-axis array with `dims`"))
end

"""
    add_category_table!(ds, name, entries)

A category table.  Category ids are the zero-based positions of the
entries (section 21), so the first entry is id 0.
"""
function add_category_table!(ds::Dataset, name::AbstractString, entries)
    check_name(name, "/categories/" * String(name))
    ds.categories[String(name)] = CategoryTable(name, collect(entries))
    push!(ds.container_groups, "categories")
    return ds.categories[String(name)]
end

"""
    add_key!(ds, name, values; role, units, lower, upper, category, ...)

A per-row key column: the name and the values, then the role and the
units, which is the order every language has (`docs/api-conventions.md`
section 1).  A design, condition or time key takes `units`; a
categorical, group, split or status key takes `category`, naming a
table added with `add_category_table!`.

Bounds are the domain of validity the file declares.  Left alone they
are the observed finite range of the data, so that the same arrays
produce the same file in every language and W04 and W08 are decidable
on every file; pass `lower` and `upper` together for a wider domain,
or both as `nothing` for a file that declares none.

The row count follows from the first key or scalar added.
"""
function add_key!(ds::Dataset, name::AbstractString, values;
                  role::Symbol, units = nothing, lower = :auto,
                  upper = :auto, bounds = :unset, category = nothing,
                  trajectory_group = nothing, parent = nothing,
                  eltype = nothing)
    path = "/keys/" * String(name)
    check_name(name, path)
    role in KEY_ROLES || throw(MestraError("E02", path,
        "`$(role)` is not a key role of section 3; pass `role` as one of " *
        join(string.(KEY_ROLES), ", ")))
    check_units(units, path)
    # `bounds = (lo, hi)`, `bounds = nothing` and `bounds = :auto` are
    # the older spelling of the same three choices.
    if bounds !== :unset
        lower, upper = bounds === nothing ? (nothing, nothing) :
                       bounds === :auto ? (:auto, :auto) :
                       (bounds[1], bounds[2])
    end
    given = values isa AbstractArray ? values : collect(values)
    T = eltype === nothing ? default_key_eltype(role, given) : eltype
    vals = numbers(given, T, path)
    ndims(vals) == 1 || throw(MestraError("E04", path,
        "a key column has one dimension, row, and these values have " *
        "$(ndims(vals)); pass one value per row"))
    if isempty(ds.keys) && isempty(ds.scalars)
        ds.nrows = length(vals)
    end
    length(vals) == ds.nrows || throw(MestraError("E16", path,
        "$(length(vals)) values in a dataset of $(ds.nrows) rows; pass " *
        "one value per row, or build this key before the others"))
    lo, hi = key_bounds(role, vals, lower, upper, path)
    if role in (:categorical, :group, :split, :status) && category === nothing
        throw(MestraError("E39", path,
            "a $(role) key requires `category` naming a table added with " *
            "`add_category_table!(ds, name, entries)` (section 19)"))
    end
    if role in (:design, :condition, :time) && units === nothing
        throw(MestraError("E39", path,
            "a $(role) key requires `units`, a UDUNITS string such as " *
            "\"m s-1\" or \"1\" for a dimensionless one"))
    end
    if !(role in (:design, :condition, :time)) && units !== nothing
        throw(MestraError("E39", path,
            "a $(role) key carries no units, because its values are " *
            (role === :id ? "row identifiers" : "category ids") *
            "; drop `units`"))
    end
    k = KeyColumn(name, role; units = units, lower = lo, upper = hi,
                  category = category, trajectory_group = trajectory_group,
                  parent = parent, eltype = T,
                  strsize = T === String ?
                            maximum(vcat([ncodeunits(s) for s in vals], 1)) : 1,
                  values = vals, path = "/keys/" * String(name))
    ds.keys[String(name)] = k
    push!(ds.container_groups, "keys")
    # Sugar: the first group key added is the unit of generalisation
    # until `set_generalisation_group!` says otherwise, because a file
    # with a group key and no unit is E39.
    role === :group && ds.generalisation_group === nothing &&
        (ds.generalisation_group = String(name))
    return k
end

"""
    set_generalisation_group!(ds, name)

Name the key that is the unit of generalisation (section 7): the thing
a split must keep whole and a model is asked to generalise over, such
as the member of a family or the trajectory.  It is a key of role
group, and a file that declares a group key declares one of these
(E39).

`grouped_split` and `split_leaks` both refuse without it, and the
first group key added takes the job until this says otherwise, which
is what to call when a file has more than one group key.
"""
function set_generalisation_group!(ds::Dataset, name::AbstractString)
    k = get(ds.keys, String(name), nothing)
    k === nothing && throw(MestraError(nothing, "/",
        "`$(name)` is not a key of this dataset; add it with " *
        "`add_key!(ds, \"$(name)\", values; role = :group, " *
        "category = ...)` first"))
    k.role === :group || throw(MestraError("E02", k.path,
        "the unit of generalisation is a key of role group, and " *
        "`$(name)` has role $(k.role); name a group key"))
    ds.generalisation_group = String(name)
    return ds
end

"""The declared bounds of a key: the observed finite range when the
caller states none, which is what `docs/api-conventions.md` section 1
asks of every language."""
function key_bounds(role::Symbol, vals, lower, upper, path)
    if lower === :auto && upper === :auto
        role in (:design, :condition, :time) || return (nothing, nothing)
        finite = Float64[x for x in vals if x isa Real && isfinite(x)]
        isempty(finite) && return (nothing, nothing)
        return (minimum(finite), maximum(finite))
    end
    lower === nothing && upper === nothing && return (nothing, nothing)
    (lower === :auto || upper === :auto || lower === nothing ||
     upper === nothing) && throw(MestraError("E19", path,
        "pass `lower` and `upper` together, or neither"))
    lo = finite_bound("lower", lower, path)
    hi = finite_bound("upper", upper, path)
    lo <= hi || throw(MestraError("E19", path,
        "`lower` is $(lo) and `upper` is $(hi); the lower bound comes first"))
    return (lo, hi)
end

"""E19 at build time: a bound the caller declares is a finite number
(section 18), named by the argument it came in as."""
function finite_bound(argument::AbstractString, value, path)
    (value isa Real && isfinite(value)) || throw(MestraError("E19", path,
        "`$(argument)` is a finite float64 bound, and $(repr(value)) is " *
        "not; pass a number, or leave both bounds out for the observed " *
        "range"))
    return Float64(value)
end

default_key_eltype(role::Symbol, vals) =
    role in (:design, :condition, :time) ? Float64 :
    role === :id ? (isempty(vals) || vals[1] isa AbstractString ?
                    String : Int64) : Int32

"""
    add_scalar!(ds, name, values; units, ...)

A per-row quantity of interest: the name and the values, then the
units, which every scalar carries (E11) and which the builder refuses
to guess at.

    add_scalar!(ds, "cl", [0.21, 0.25]; units = "1")
"""
function add_scalar!(ds::Dataset, name::AbstractString, values;
                     units = nothing, statistic = nothing,
                     of = nothing, quantile = nothing, level = nothing,
                     method = nothing, deflate = nothing,
                     shuffle::Bool = false)
    path = "/scalars/" * String(name)
    check_name(name, path)
    units === nothing && throw(MestraError("E11", path,
        "a scalar carries units; pass units = \"1\" for a dimensionless " *
        "one"))
    check_units(units, path)
    check_statistic(statistic, of, quantile, level, method, false, path)
    vals = numbers(values, Float64, path)
    ndims(vals) == 1 || throw(MestraError("E04", path,
        "a scalar has one dimension, row, and these values have " *
        "$(ndims(vals)); pass one value per row"))
    if isempty(ds.keys) && isempty(ds.scalars)
        ds.nrows = length(vals)
    end
    length(vals) == ds.nrows || throw(MestraError("E16",
        "/scalars/" * String(name),
        "$(length(vals)) values in a dataset of $(ds.nrows) rows; pass " *
        "one value per row"))
    s = Slot(name, :scalar; units = units, source = "data",
             statistic = statistic, of = of, quantile = quantile,
             level = level, method = method,
             ldims = [:row], dshape = [length(vals)], eltype = Float64,
             deflate = deflate, shuffle = shuffle,
             data = vals, path = "/scalars/" * String(name))
    ds.scalars[String(name)] = s
    push!(ds.container_groups, "scalars")
    return s
end

"""
    add_mesh_support!(ds, name; coordinates, cell_types, cell_offsets,
                      cell_connectivity, units = "m", dims, varies)

A mesh support.  `coordinates` is (node, component) by default, and
`dims` names its axes when it is anything else -- (:row, :node,
:component) for a moving mesh, (:instance, :node, :component) with
`varies = "group:member"` for a parametric family.  The support id is
computed from the cells (section 24).

With no `coordinates`, `n_nodes` says how many nodes the cells are on
and the coordinates come later from `set_callable_coordinates!`; a
file written before they do is refused with E03.
"""
function add_mesh_support!(ds::Dataset, name::AbstractString;
                           coordinates = nothing, cell_types, cell_offsets,
                           cell_connectivity, units::AbstractString = "m",
                           dims = nothing, varies = nothing,
                           n_nodes = nothing)
    check_name(name, "/supports/" * String(name))
    types = UInt8.(collect(cell_types))
    offsets = Int64.(collect(cell_offsets))
    conn = Int64.(collect(cell_connectivity))
    if coordinates === nothing
        n_nodes === nothing && throw(MestraError("E03", "/supports/" * name,
            "a mesh support has exactly one coordinates array; pass " *
            "`coordinates`, or `n_nodes` and then set_callable_coordinates! " *
            "for coordinates a callable serves"))
        s = Support(name, "mesh"; n_nodes = Int(n_nodes),
                    n_cells = length(types), cell_types = types,
                    cell_offsets = offsets, cell_connectivity = conn)
        push!(ds.supports, s)
        push!(ds.container_groups, "supports")
        s.support_id = support_id(s)
        return s
    end
    coordinates = numbers(coordinates, Float64, "/supports/$(name)/coordinates";
                          argument = "coordinates")
    dims === nothing && (dims = ndims(coordinates) == 2 ?
                                [:node, :component] :
                                [:row, :node, :component])
    n_nodes = size(coordinates, findfirst(==(:node), Symbol.(dims)))
    s = Support(name, "mesh"; n_nodes = n_nodes, n_cells = length(types),
                cell_types = types, cell_offsets = offsets,
                cell_connectivity = conn)
    s.coordinates = make_array_slot(ds, s, "coordinates", coordinates, :node;
                                    role = :coordinates, units = units,
                                    dims = dims, varies = varies)
    push!(ds.supports, s)
    push!(ds.container_groups, "supports")
    s.support_id = support_id(s)
    return s
end

"""
    add_axis_support!(ds, name; coordinates, units)

A one-dimensional axis: nodes along one coordinate, no cells.  Its
coordinate is part of its identity, so it must not vary (E35).
"""
function add_axis_support!(ds::Dataset, name::AbstractString;
                           coordinates, units::AbstractString)
    check_name(name, "/supports/" * String(name))
    c = numbers(coordinates, Float64, "/supports/$(name)/coordinates";
                argument = "coordinates")
    ndims(c) == 1 && (c = reshape(c, length(c), 1))
    s = Support(name, "axis"; n_nodes = size(c, 1), n_cells = 0)
    s.coordinates = make_array_slot(ds, s, "coordinates", c, :node;
                                    role = :coordinates, units = units,
                                    dims = [:node, :component],
                                    varies = "none")
    push!(ds.supports, s)
    push!(ds.container_groups, "supports")
    s.support_id = support_id(s)
    return s
end

"""
    add_none_support!(ds, name)

A support of kind none: no nodes, no cells and no coordinates.  It
exists so that a slot may say it lives on no support.
"""
function add_none_support!(ds::Dataset, name::AbstractString)
    check_name(name, "/supports/" * String(name))
    s = Support(name, "none"; n_nodes = 0, n_cells = 0)
    push!(ds.supports, s)
    push!(ds.container_groups, "supports")
    s.support_id = support_id(s)
    return s
end

"""What the array varies along, which `dims` decides: a leading `row`
axis is "row", a leading `group:<k>` axis is that group, and anything
else is "none" (section 5).  `varies` need not be passed at all; when
it is and it says something other than what `dims` names, that is E04
and is refused here, because one of the two is wrong and a writer
cannot tell which."""
function agreed_varies(ds::Dataset, dims::Vector{Symbol}, varies, path)
    lead = first(dims)
    implied = lead === :row ? "row" :
              lead === :instance ? nothing :
              startswith(String(lead), "group:") ? String(lead) : "none"
    if varies === nothing
        implied === nothing && throw(MestraError("E04", path,
            "`dims` names the leading axis :instance, which does not say " *
            "which group it is; add `varies = \"group:<key>\"`, or name " *
            "the axis `Symbol(\"group:<key>\")` in `dims`"))
        varies = implied
    else
        varies = String(varies)
        if implied === nothing
            startswith(varies, "group:") || throw(MestraError("E04", path,
                "`dims` names the leading axis :instance, so `varies` is " *
                "\"group:<key>\" and not \"$(varies)\"; change one of them"))
        elseif varies != implied
            throw(MestraError("E04", path,
                "`varies` says \"$(varies)\" and `dims` names the leading " *
                "axis $(lead), which is \"$(implied)\"; change one of them"))
        end
    end
    if startswith(varies, "group:")
        g = varies[7:end]
        k = get(ds.keys, g, nothing)
        (k !== nothing && k.role === :group) || throw(MestraError("E04", path,
            "`varies` names the group key `$(g)`, which this dataset does " *
            "not declare; add it with `add_key!(ds, \"$(g)\", values; " *
            "role = :group, category = ...)` before this array"))
        dims[1] = Symbol(varies)
    end
    return varies
end

function make_array_slot(ds::Dataset, sup::Support, name::AbstractString,
                         data, location::Symbol; role::Symbol = :field,
                         units = nothing, dims = nothing, varies = nothing,
                         components = nothing, statistic = nothing,
                         of = nothing, quantile = nothing, level = nothing,
                         method = nothing, category = nothing,
                         recomputed = nothing, derived_from = nothing,
                         recipe = nothing, reference = nothing,
                         eltype = nothing, deflate = nothing,
                         shuffle = false)
    path = array_path(sup, name, location)
    name == "coordinates" || check_name(name, path)
    check_statistic(statistic, of, quantile, level, method, false, path)
    role in ARRAY_ROLES || throw(MestraError("E02", path,
        "`$(role)` is not an array role of section 3; pass `role` as one " *
        "of " * join(string.(ARRAY_ROLES), ", ")))
    role === :field && units === nothing &&
        throw(MestraError("E11", path,
            "a field carries units; pass units = \"1\" for a " *
            "dimensionless one"))
    check_units(units, path)
    data = numbers(data, eltype, path;
                   argument = name == "coordinates" ? "coordinates" : "values")
    nsite = location === :cell ? sup.n_cells : sup.n_nodes
    dims === nothing && (dims = default_dims(data, nsite, location, path))
    dims = Symbol[Symbol(d) for d in dims]
    varies = agreed_varies(ds, dims, varies, path)
    want = Symbol[]
    varies == "row" && push!(want, :row)
    startswith(varies, "group:") && push!(want, Symbol(varies))
    statistic == "draw" && push!(want, :draw)
    push!(want, location === :cell ? :cell : :node)
    push!(want, :component)
    stored = to_stored(data, dims, want, path)
    T = eltype === nothing ? Base.eltype(stored) : eltype
    T === Float64 || T === Int32 || T === Int64 ||
        throw(MestraError("E20", path,
            "an array may not be stored as $(T); pass `eltype` as " *
            "Float64, Int32 or Int64, or convert the array"))
    dshape = collect(reverse(size(stored)))
    components === nothing || Int(components) == dshape[end] ||
        throw(MestraError("E31", path,
            "`components` says $(components) where the component axis of " *
            "this array has $(dshape[end]); drop `components`, which " *
            "follows from `dims`"))
    return Slot(name, location; support = sup.name, role = role,
                varies = varies, units = units,
                components = dshape[end],
                statistic = statistic, of = of, quantile = quantile,
                level = level, method = method,
                category = category, recomputed = recomputed,
                derived_from = derived_from, recipe = recipe,
                reference = reference,
                ldims = [logical_dim(disk_dim(w, dshape[i]))
                         for (i, w) in pairs(want)],
                dshape = dshape, eltype = T, deflate = deflate,
                shuffle = shuffle,
                data = convert(Array{T}, stored), path = path)
end

"""
    add_node_array!(ds, support, name, values; units, dims, role = :field)

A field-like quantity on the nodes of a support: the support, the
name and the values, then the units and the axes, which is the order
every language has (`docs/api-conventions.md` section 1).

`dims` names the axes of `values` in the order your array has them,
and everything else follows from it: `varies` is "row" for a leading
`:row` axis, the group for a leading `Symbol("group:<key>")` axis, and
"none" otherwise, and `components` is the length of the component
axis, which is added for you with length one when your array has none,
because the component dimension is always present (section 19).
Passing a `varies` that says something else is refused as E04.

Left out, `dims` is read off the shape: (:node,) for a vector,
(:row, :node) or (:node, :component) for a matrix, whichever agrees
with the support, and (:row, :node, :component) for a three-axis
array.  A square matrix on a support of as many nodes as it has
columns says nothing about which of the two is meant, so it is
refused rather than guessed at.
"""
function add_node_array!(ds::Dataset, sup::Support, name::AbstractString,
                         data; kwargs...)
    s = make_array_slot(ds, sup, name, data, :node; kwargs...)
    sup.node_arrays[String(name)] = s
    return s
end

"""
    add_cell_array!(ds, support, name, values; units, dims, role = :field)

The same, on the cells.  `dims` names the cell axis `:cell`.
"""
function add_cell_array!(ds::Dataset, sup::Support, name::AbstractString,
                         data; kwargs...)
    s = make_array_slot(ds, sup, name, data, :cell; kwargs...)
    sup.cell_arrays[String(name)] = s
    return s
end

"""
    add_callable!(ds, id, c::Callable; repr = string(c))

Store a callable under `/callables/<id>` and make it available to
slots.  Fill a slot from it with `set_callable!`.
"""
function add_callable!(ds::Dataset, id::AbstractString, c::Callable;
                       type::Union{Nothing,AbstractString} = nothing,
                       repr::Union{Nothing,AbstractString} = nothing)
    check_name(id, "/callables/" * String(id))
    ty = type === nothing ? registered_name(typeof(c)) : String(type)
    ty === nothing && throw(MestraError("E15", "/callables/" * String(id),
        "no `type` string is registered for $(typeof(c)); call " *
        "`register_callable!(\"<type>\", $(typeof(c)))`, or pass `type`"))
    ref = CallableRef(id, ty;
                      repr = repr === nothing ? sprint(show, c) : repr,
                      dict = to_dict(c))
    ds.callables[String(id)] = ref
    push!(ds.container_groups, "callables")
    return ref
end

function registered_name(T::Type)
    for (name, registered) in CALLABLE_REGISTRY
        registered === T && return name
    end
    return nothing
end

"""Which callable fills a slot, from `callable` or from the older
spelling `id`, one of which a caller must give."""
function callable_name(callable, id, path)
    callable === nothing && id === nothing && throw(MestraError("E14", path,
        "say which callable fills this slot with `callable`, naming one " *
        "added with `add_callable!(ds, id, c)`"))
    (callable === nothing || id === nothing ||
     String(callable) == String(id)) ||
        throw(MestraError("E14", path,
            "`callable` says `$(callable)` and `id` says `$(id)`; they are " *
            "two spellings of one argument, so pass `callable` alone"))
    return String(callable === nothing ? id : callable)
end

"""
    set_callable!(slot, id, output)

Make a slot say that a callable fills it: the slot keeps its
attributes and holds no data (section 10).
"""
function set_callable!(s::Slot, id::AbstractString, output::AbstractString)
    s.source = "callable:" * String(id)
    s.output = String(output)
    s.data = nothing
    return s
end

"""
    add_callable_slot!(ds, support, name; units, components, callable,
                       output, location = :node, role = :field,
                       varies = "row")

A slot with no data at all, served by a callable.  This is how a model
file declares what it produces before anything is evaluated.  The
arguments are the array builders' own, plus `callable`, naming a
callable added with `add_callable!`, and `output`, naming which of its
outputs fills this slot.  `id` is accepted for `callable`.
"""
function add_callable_slot!(ds::Dataset, sup::Support, name::AbstractString;
                            location::Symbol = :node, role::Symbol = :field,
                            units::AbstractString, components::Integer,
                            callable = nothing, id = nothing,
                            output::AbstractString,
                            varies::AbstractString = "row",
                            statistic = nothing, of = nothing)
    check_name(name, array_path(sup, name, location))
    id = callable_name(callable, id, array_path(sup, name, location))
    check_statistic(statistic, of, nothing, nothing, nothing, true,
                    array_path(sup, name, location))
    s = Slot(name, location; support = sup.name, role = role, varies = varies,
             units = units, components = components,
             statistic = statistic, of = of,
             source = "callable:" * String(id), output = String(output),
             path = "/supports/$(sup.name)/" *
                    (location === :cell ? "cell_arrays/" : "node_arrays/") *
                    String(name))
    (location === :cell ? sup.cell_arrays : sup.node_arrays)[String(name)] = s
    return s
end

"""
    set_callable_coordinates!(ds, sup; units, components, callable, output)

Coordinates a callable serves rather than data: the coordinates of a
mesh support with the values left out and the callable and its output
added, the way `add_callable_slot!` is `add_node_array!` without its
values -- a model of the geometry itself.  `output` defaults to
`"coordinates"`, `varies` to `"row"`, and `id` is accepted for
`callable`.  An axis support's coordinates are part of its identity
(section 24) and are always stored.
"""
function set_callable_coordinates!(ds::Dataset, sup::Support;
                                   units::AbstractString, components::Integer,
                                   callable = nothing, id = nothing,
                                   output::AbstractString = "coordinates",
                                   varies::AbstractString = "row")
    path = "/supports/$(sup.name)/coordinates"
    sup.kind == "mesh" || throw(MestraError("E03", path,
        "a callable may serve the coordinates of a mesh support; this " *
        "support is of kind $(sup.kind), whose coordinates are stored"))
    sup.coordinates === nothing || throw(MestraError("E03", path,
        "this support already has its one coordinates array"))
    components >= 1 || throw(MestraError("E31", path,
        "a callable slot stores no values, so it declares its width; " *
        "pass `components`"))
    id = callable_name(callable, id, path)
    s = Slot("coordinates", :node; support = sup.name, role = :coordinates,
             varies = varies, units = units, components = components,
             source = "callable:" * String(id), output = String(output),
             path = path)
    sup.coordinates = s
    return s
end

"""
    add_callable_scalar!(ds, name; units, callable, output)

A scalar slot served by a callable.  `id` is accepted for `callable`.
"""
function add_callable_scalar!(ds::Dataset, name::AbstractString;
                              units::AbstractString, callable = nothing,
                              id = nothing, output::AbstractString,
                              statistic = nothing, of = nothing)
    check_name(name, "/scalars/" * String(name))
    id = callable_name(callable, id, "/scalars/" * String(name))
    check_statistic(statistic, of, nothing, nothing, nothing, true,
                    "/scalars/" * String(name))
    s = Slot(name, :scalar; units = units, statistic = statistic, of = of,
             source = "callable:" * String(id), output = String(output),
             path = "/scalars/" * String(name))
    ds.scalars[String(name)] = s
    push!(ds.container_groups, "scalars")
    return s
end

"""E02 and E12 at build time: a statistic with what it needs (section
9), and on a slot a callable serves, one a callable produces (section
10)."""
function check_statistic(statistic, of, quantile, level, method,
                         served::Bool, path)
    if statistic != "band" && (level !== nothing || method !== nothing)
        throw(MestraError("E12", path,
            "`level` and `method` belong to a band; pass statistic = " *
            "\"band\" with `of` naming the base quantity"))
    end
    statistic === nothing && return
    statistic in STATISTICS || throw(MestraError("E02", path,
        "`$(statistic)` is not a statistic of section 9; pass one of " *
        join(STATISTICS, ", ")))
    statistic == "quantile" && quantile === nothing &&
        throw(MestraError("E12", path,
            "a quantile statistic carries its quantile; pass `quantile`"))
    !(statistic in ("value", "draw")) && of === nothing &&
        throw(MestraError("E12", path,
            "a $(statistic) names the quantity it is a statistic of; pass " *
            "`of`"))
    served && statistic in ("std", "quantile", "draw") &&
        throw(MestraError("E12", path,
            "a callable returns a mean and at most a band (section 10), " *
            "so a slot it serves is value, mean or band; store a " *
            "$(statistic) as data"))
    if statistic == "band" && !served
        level === nothing && throw(MestraError("E12", path,
            "a band states the coverage it claims; pass `level` in " *
            "(0, 1), for example 0.95"))
        0 < level < 1 || throw(MestraError("E12", path,
            "`level` is the coverage a band claims, a fraction in (0, 1), " *
            "and $(level) is not; a 1.96-sigma Gaussian band is 0.95"))
        (method === nothing || isempty(method)) &&
            throw(MestraError("E12", path,
                "a band says how it was made; pass `method` with one " *
                "sentence"))
    end
    return
end

"""
    set_row_support!(ds, indices)

Which support each row sits on, as zero-based positions in the file's
support order (section 22).  A file with more than one support needs
one; a file with at most one must not have one.
"""
function set_row_support!(ds::Dataset, indices)
    ds.row_support = Int32.(collect(indices))
    return ds
end

# ------------------------------------------------- notes and private

"""An attribute built for writing, in the encoding of section 18."""
function raw_attr(name::AbstractString, value)
    legal_name(name) || throw(MestraError("E33",
        "attribute name `$(name)` is not a legal netCDF-4 name"))
    reserved(name) && throw(MestraError("E33",
        "attribute name `$(name)` begins with the reserved prefix"))
    if value isa Bool
        ti = TypeInfo(:int, 1, true, true, false, 0, 0)
        return RawAttr(String(name), ti, UInt8[value ? 0x01 : 0x00], 1,
                       true, true, value)
    elseif value isa Integer
        ti = TypeInfo(:int, 8, true, true, false, 0, 0)
        return RawAttr(String(name), ti,
                       collect(reinterpret(UInt8, [Int64(value)])), 1,
                       true, true, Int64(value))
    elseif value isa Real
        ti = TypeInfo(:float, 8, true, true, false, 0, 0)
        return RawAttr(String(name), ti,
                       collect(reinterpret(UInt8, [Float64(value)])), 1,
                       true, true, Float64(value))
    elseif value isa AbstractString
        bytes = Vector{UInt8}(codeunits(String(value)))
        isempty(bytes) && (bytes = UInt8[0x00])
        ti = TypeInfo(:string, length(bytes), false, true, false,
                      Int(HDF5.API.H5T_CSET_UTF8),
                      Int(HDF5.API.H5T_STR_NULLPAD))
        return RawAttr(String(name), ti, bytes, 1, true, true,
                       String(value))
    end
    throw(MestraError(nothing,
        "an attribute of a type this format has no encoding for: " *
        "$(typeof(value))"))
end

function attr_group(name::AbstractString, d::AbstractDict)
    attrs = RawAttr[raw_attr(k, d[k])
                    for k in sort(collect(keys(d)), by = codeunits)]
    return RawGroupCopy(String(name), attrs, RawDatasetCopy[],
                        RawGroupCopy[])
end

"""
    set_notes!(ds, attributes)

Free-form attributes under `/notes`: a solver name, a dataset licence,
a comment.  Nothing in the format depends on them and no tool may
require them (section 11).  Pass an empty dictionary to drop the
group.
"""
function set_notes!(ds::Dataset, attributes::AbstractDict)
    ds.notes = attr_group("notes", attributes)
    return ds
end

"""
    set_private!(ds, attributes)

A producer's own records under `/private`.  A reader treats it as
opaque and must not interpret it, and a writer must not put any public
information only there (E18), so this is for lineage, validation and
history and for nothing the format already carries.
"""
function set_private!(ds::Dataset, attributes::AbstractDict)
    ds.private = attr_group("private", attributes)
    return ds
end
