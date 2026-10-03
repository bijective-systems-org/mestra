# The dictionary codec of sections 17 and 25.
#
# A callable's dictionary is a nested Dict{String,Any} whose leaves
# are Bool, Int64, Float64, String, `nothing`, a Vector{String}, or a
# numeric array.  An array is held in the shape the file states, which
# is C order; this file is the only place that converts between that
# and the Julia memory layout.

const NULL_SENTINEL = UInt8[0x00, 0x6e, 0x75, 0x6c, 0x6c]  # "\0null"
const RESERVED_PREFIX = "mestra_"

reserved(name::AbstractString) = startswith(name, RESERVED_PREFIX)

"""Raw C-order bytes to a Julia array of the stated C-order shape."""
function c_to_julia(raw::Vector{UInt8}, ::Type{T},
                    cdims::Vector{Int}) where {T}
    n = prod(cdims)
    flat = n == 0 ? T[] : collect(reinterpret(T, raw))[1:n]
    rev = reshape(flat, reverse(cdims)...)
    nd = length(cdims)
    nd <= 1 && return Array(rev)
    return permutedims(rev, ntuple(i -> nd - i + 1, nd))
end

"""A Julia array of a C-order shape back to raw C-order bytes."""
function julia_to_c(a::AbstractArray{T}) where {T}
    nd = ndims(a)
    rev = nd <= 1 ? a : permutedims(a, ntuple(i -> nd - i + 1, nd))
    return collect(reinterpret(UInt8, vec(collect(rev))))
end

# ------------------------------------------------------------ reading

"""
    read_dict(group; toplevel = false) -> Dict{String,Any}

Reconstruct a dictionary from an HDF5 group by the codec of sections
17 and 25.  Every member and every attribute whose name begins with
`mestra_` is the container's and is skipped, as are the machinery
names of section 18; at the top level `type` and `repr` are the
callable's own attributes and are skipped too.
"""
function read_dict(ds::Dataset, g::HDF5.Group, path::AbstractString;
                   toplevel::Bool = false, depth::Int = 0,
                   max_elements::Integer = DEFAULT_MAX_ELEMENTS)
    out = Dict{String,Any}()
    if depth >= MAX_DEPTH
        note!(ds, "E41", path,
              "a dictionary nested deeper than $(MAX_DEPTH); this reader " *
              "stops here rather than following a file's own depth")
        return out
    end
    for name in attr_names(g)
        name in MACHINERY_ATTRS && continue
        reserved(name) && continue
        toplevel && (name == "type" || name == "repr") && continue
        a = read_raw_attr(g, name)
        if !a.readable || !a.scalar
            note!(ds, "E41", "$(path)@$(name)",
                  "an attribute this reader would not read: " *
                  (a.scalar ? "unreadable" : "not a scalar"))
            continue
        end
        try
            out[name] = decode_dict_attr(a)
        catch e
            note!(ds, rule_of(e), "$(path)@$(name)", message_of(e))
        end
    end
    for (name, kind) in child_links(g)
        reserved(name) && continue
        if kind !== :hard
            note!(ds, "E40", "$(path)/$(name)",
                  "a $(kind) link; this reader follows hard links only")
            continue
        end
        obj = hard_child(g, name)
        obj === nothing && continue
        if obj isa HDF5.Group
            out[name] = read_dict(ds, obj, "$(path)/$(name)";
                                  depth = depth + 1,
                                  max_elements = max_elements)
        else
            is_scale(obj) && continue
            try
                out[name] = read_dict_dataset(obj;
                                              max_elements = max_elements)
            catch e
                note!(ds, rule_of(e), "$(path)/$(name)", message_of(e))
            end
        end
    end
    return out
end

"""
    read_dict(group) -> Dict{String,Any}

The same, for a group in hand and with nothing to report findings to.
"""
read_dict(g::HDF5.Group; kwargs...) =
    read_dict(Dataset(), g, try HDF5.name(g) catch; "?" end; kwargs...)

function decode_dict_attr(a::RawAttr)
    if a.ti.class === :string
        a.ti.vlen && throw(MestraError("E19",
            "a variable-length string attribute is never legal"))
        a.raw == NULL_SENTINEL && return nothing
        return a.value
    elseif a.ti.class === :int
        a.ti.size == 1 && return a.value === true
        return a.value
    elseif a.ti.class === :float
        return a.value
    end
    throw(MestraError("E32", "an attribute of a type the codec cannot hold"))
end

function read_dict_dataset(d::HDF5.Dataset;
                           max_elements::Integer = DEFAULT_MAX_ELEMENTS)
    ti, raw, _ = read_raw_dataset(d; max_elements = max_elements)
    cdims, _ = disk_shape(d)
    if ti.class === :string
        ti.vlen && throw(MestraError("E19",
            "a variable-length string dataset is never legal"))
        n = prod(cdims)
        out = Vector{String}(undef, n)
        for i in 1:n
            rec = raw[((i - 1) * ti.size + 1):(i * ti.size)]
            out[i] = String(strip_nul(rec))
        end
        return length(cdims) <= 1 ? out :
               permutedims(reshape(out, reverse(cdims)...),
                           ntuple(i -> length(cdims) - i + 1, length(cdims)))
    end
    T = julia_eltype(ti)
    if ti.class === :int && ti.size == 1
        vals = c_to_julia(raw, Int8, cdims)
        return map(v -> v != 0, vals)
    end
    T in (Int32, Int64, Float64) ||
        throw(MestraError("E32",
            "a dataset dtype the codec does not allow: $(T)"))
    ti.little || throw(MestraError("E32",
        "a big-endian dataset; section 25 stores a numeric array " *
        "little-endian"))
    return c_to_julia(raw, T, cdims)
end

# ------------------------------------------------------------ writing

"""
    write_dict(group, dict)

Write a dictionary into an HDF5 group by the codec.  Keys are visited
in ascending order of their UTF-8 bytes (section 25), so that two
writers given one dictionary produce one file.
"""
function write_dict(g, dict::AbstractDict)
    for key in sort(collect(keys(dict)), by = codeunits)
        key isa AbstractString || throw(MestraError("E32",
            "a dictionary key that is not a string"))
        reserved(key) && throw(MestraError("E33",
            "a dictionary key may not begin with `$(RESERVED_PREFIX)`: $(key)"))
        legal_name(key) || throw(MestraError("E33",
            "a dictionary key that is not a legal netCDF-4 name: $(key)"))
        write_dict_value(g, String(key), dict[key])
    end
    return nothing
end

function write_dict_value(g, name::String, v)
    if v === nothing
        write_raw_string_attr(g, name, copy(NULL_SENTINEL))
    elseif v isa AbstractDict
        sub = create_group(g, name)
        write_dict(sub, v)
    elseif v isa Bool
        write_bool_attr(g, name, v)
    elseif v isa Integer
        write_int_attr(g, name, Int64(v))
    elseif v isa AbstractFloat
        write_float_attr(g, name, Float64(v))
    elseif v isa AbstractString
        occursin('\0', v) && throw(MestraError("E32",
            "a string with an embedded NUL is not representable"))
        write_string_attr(g, name, v)
    elseif v isa AbstractArray && ndims(v) == 0
        # Section 25: a zero-dimensional array is written as the
        # number it holds, never as a zero-dimensional dataset.
        write_dict_value(g, name, v[])
    elseif v isa AbstractArray{<:AbstractString}
        write_dict_strings(g, name, v)
    elseif v isa AbstractArray
        write_dict_array(g, name, v)
    else
        throw(MestraError("E32",
            "a value the codec cannot represent: $(typeof(v))"))
    end
    return nothing
end

function dict_scales!(g, name::String, cdims::Vector{Int}, d::HDF5.Dataset)
    for (axis, len) in pairs(cdims)
        s = create_scale(g, "$(RESERVED_PREFIX)$(name)_d$(axis - 1)", len;
                         unlimited = len == 0)
        attach_scale!(d, s, axis - 1)
    end
end

function write_dict_array(g, name::String, v::AbstractArray)
    T = eltype(v)
    if T === Bool
        dt = le(HDF5.API.H5T_STD_I8LE)
        raw = julia_to_c(map(x -> Int8(x ? 1 : 0), v))
    elseif T === Int32
        dt = le(HDF5.API.H5T_STD_I32LE)
        raw = julia_to_c(v)
    elseif T <: Integer
        dt = le(HDF5.API.H5T_STD_I64LE)
        raw = julia_to_c(Int64.(v))
    elseif T === Float64
        dt = le(HDF5.API.H5T_IEEE_F64LE)
        raw = julia_to_c(v)
    else
        throw(MestraError("E32",
            "a dtype the codec does not allow: $(T). Only int8 " *
            "(boolean), int32, int64, float64 and fixed-length UTF-8 " *
            "strings are representable"))
    end
    cdims = collect(size(v))
    empty = any(==(0), cdims)
    # Section 25 makes each zero-length axis unlimited, and E43 allows
    # that axis and no other.
    cmax = [n == 0 ? -1 : n for n in cdims]
    chunk = empty ? fill(1, length(cdims)) : nothing
    d = create_raw_dataset(g, name, dt, cdims, cmax, raw; chunk = chunk)
    dict_scales!(g, name, cdims, d)
    return d
end

function write_dict_strings(g, name::String, v::AbstractArray)
    ndims(v) == 1 || throw(MestraError("E32",
        "a list of strings must be one-dimensional"))
    for s in v
        occursin('\0', s) && throw(MestraError("E32",
            "a string with an embedded NUL is not representable"))
    end
    n = maximum(vcat([ncodeunits(s) for s in v], 1))
    raw = UInt8[]
    for s in v
        b = Vector{UInt8}(codeunits(s))
        append!(raw, b)
        append!(raw, zeros(UInt8, n - length(b)))
    end
    dt = fixed_string_type(n)
    cdims = [length(v)]
    empty = length(v) == 0
    d = create_raw_dataset(g, name, dt, cdims, empty ? [-1] : cdims, raw;
                           chunk = empty ? [1] : nothing)
    dict_scales!(g, name, cdims, d)
    return d
end

"""Section 18: a legal netCDF-4 name."""
function legal_name(name::AbstractString)
    isempty(name) && return false
    occursin('/', name) && return false
    occursin('\0', name) && return false
    (startswith(name, ' ') || endswith(name, ' ')) && return false
    # Section 18: the first character is a letter, a digit or an
    # underscore, as netCDF-C requires.
    first = name[firstindex(name)]
    (isletter(first) || isdigit(first) || first == '_') || return false
    for c in name
        ok = isletter(c) || isdigit(c) || c == '_' || c == '-' ||
             c == '.' || c == '+'
        ok || return false
    end
    return true
end
