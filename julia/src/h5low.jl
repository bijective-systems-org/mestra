# Low-level HDF5 helpers.
#
# Everything this file does is fixed by SPEC.md sections 18 to 23: the
# attribute encodings, the fixed-length UTF-8 strings, the dimension
# scales, the chunk shapes and the filters.  Nothing above this file
# is allowed to reach for the raw API.
#
# A file is untrusted input.  Every number a file states about itself
# -- how many elements a dataset has, how many values a filter
# declares, how deep the groups go -- is a claim by whoever wrote it,
# and this file is where each claim is checked before anything is
# sized from it.  Nothing above here allocates from a shape it has not
# seen checked, opens a link without asking what kind it is, or
# recurses on a structure the file controls the depth of.

"""The most elements any single read will materialise.  Above this a
read is refused with E41 rather than attempted: a file may declare a
trillion elements and hold none.  `Mestra.read`, `values` and `rows`
take a `max_elements` keyword to change it for one call."""
const DEFAULT_MAX_ELEMENTS = 1 << 31

"""The most bytes any single read will materialise, whatever the
element count allows.  A chunk of four hundred million float64 is
under the element cap and is still three gigabytes, so the two caps
are both needed.  Raise it, deliberately, for a file you trust:

    Mestra.MAX_READ_BYTES[] = 8 * (1 << 30)
"""
const MAX_READ_BYTES = Ref(1 << 30)

"""The most bytes one attribute may hold.  Every attribute this format
names is a scalar, so nothing legal is near this."""
const MAX_ATTR_BYTES = 1 << 24

"""The deepest this reader will walk into a file.  Nothing this format
defines nests beyond about six, and a callable's dictionary has no
reason to; beyond the cap the subtree is reported as E41 and not
read."""
const MAX_DEPTH = 64

"""The most objects one walk will visit, so that a file cannot hold a
reader in a loop of its own making."""
const MAX_OBJECTS = 1 << 20

"""The most client-data values this reader will take from one filter
declaration."""
const MAX_FILTER_CD = 256

"""
    element_count(dims) -> Int

The product of the extents, saturating at `typemax(Int)` instead of
wrapping.  A file that declares (2^40, 2^40) would otherwise overflow
into a small positive number and be read.
"""
function element_count(dims)
    any(d -> d == 0, dims) && return 0
    n = 1
    for d in dims
        d < 0 && return typemax(Int)
        big = Int128(n) * Int128(d)
        big > typemax(Int) && return typemax(Int)
        n = Int(big)
    end
    return n
end

const MACHINERY_ATTRS = Set([
    "CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST",
    "DIMENSION_LABELS", "_Netcdf4Dimid", "_Netcdf4Coordinates",
    "_nc3_strict", "_NCProperties",
])

# Section 21: the NAME attribute of a dimension scale with no
# coordinate variable.  53 characters, then the length in ten columns.
const DIM_SENTENCE = "This is a netCDF dimension but not a netCDF variable."

dim_scale_name(length::Integer) = @sprintf("%s%10d", DIM_SENTENCE, length)

# --------------------------------------------------------- datatypes

"""Fixed-length UTF-8, NUL-padded (sections 18 and 19)."""
function fixed_string_type(nbytes::Integer)
    t = HDF5.API.h5t_copy(HDF5.API.H5T_C_S1)
    HDF5.API.h5t_set_size(t, max(1, Int(nbytes)))
    HDF5.API.h5t_set_cset(t, HDF5.API.H5T_CSET_UTF8)
    HDF5.API.h5t_set_strpad(t, HDF5.API.H5T_STR_NULLPAD)
    return HDF5.Datatype(t)
end

le(t) = HDF5.Datatype(HDF5.API.h5t_copy(t))

scalar_space() = HDF5.Dataspace(HDF5.API.h5s_create(HDF5.API.H5S_SCALAR))

"""The number of elements a dataspace holds.  HDF5.jl does not wrap
H5Sget_simple_extent_npoints, so this is the product of the extents,
with a scalar dataspace counting one."""
function space_npoints(sp::HDF5.Dataspace)
    n = HDF5.API.h5s_get_simple_extent_ndims(sp)
    n == 0 && return 1
    dims, _ = HDF5.API.h5s_get_simple_extent_dims(sp)
    return element_count(Int.(dims))
end

"""True when the dataspace is the scalar one section 18 requires of
every attribute."""
is_scalar_space(sp::HDF5.Dataspace) =
    HDF5.API.h5s_get_simple_extent_ndims(sp) == 0

# ------------------------------------------------------------- links
#
# `keys(group)` lists link names, and indexing a group follows the
# link.  A soft link may point at nothing or in a circle, and an
# external link opens another file, which this reader must never do
# on a file's say-so.  So the link is asked what kind it is before
# anything opens it.

# HDF5.jl wraps H5Lget_info, which is the 1.10 name; libhdf5 1.12
# renamed it H5Lget_info2 and 2.0 dropped the old symbol, so there is
# no wrapper to call on a current library.  This is the one place in
# the package that reaches past HDF5.jl to the C API with a ccall, and
# H5Lget_info2 is the one function it calls.  Only the first field of
# H5L_info2_t is read, the link type, which is first in the struct in
# every version of it; the buffer is far larger than the struct so
# that the rest of the layout does not have to be guessed.
const _LINK_INFO_BYTES = 128

function _h5l_get_info2(parent_id, name::String)
    buf = zeros(UInt8, _LINK_INFO_BYTES)
    st = ccall((:H5Lget_info2, HDF5.API.libhdf5), HDF5.API.herr_t,
               (HDF5.API.hid_t, Cstring, Ptr{UInt8}, HDF5.API.hid_t),
               parent_id, name, buf, HDF5.API.H5P_DEFAULT)
    st < 0 && return nothing
    return Int(reinterpret(Int32, buf[1:4])[1])
end

"""
    link_type(parent, name) -> Symbol

`:hard`, `:soft`, `:external`, `:missing` or `:other`.  Nothing here
follows the link: this is what is asked before anything is opened.
"""
function link_type(parent, name::AbstractString)
    try
        HDF5.API.h5l_exists(parent, String(name), HDF5.API.H5P_DEFAULT) ||
            return :missing
        t = _h5l_get_info2(parent.id, String(name))
        t === nothing && return :other
        # The values are the C enum H5L_type_t.  HDF5.jl's own
        # H5L_TYPE_EXTERNAL is 2, which is not what the library
        # returns, so the literals are used and anything unrecognised
        # is treated as not a hard link, which is the safe answer.
        t == 0 && return :hard          # H5L_TYPE_HARD
        t == 1 && return :soft          # H5L_TYPE_SOFT
        t == 64 && return :external     # H5L_TYPE_EXTERNAL
        return :other
    catch
        return :other
    end
end

"""Every child of a group as (name, link type), in name order, with a
cap so that a group cannot hold a reader forever."""
function child_links(g)
    out = Tuple{String,Symbol}[]
    # Ask how many links there are before asking for their names: a
    # group may declare a billion, and listing them all to throw most
    # of the list away is the allocation this is here to avoid.
    n = try
        Int(HDF5.API.h5g_get_num_objs(g))
    catch
        -1
    end
    names = if 0 <= n <= MAX_OBJECTS
        try
            collect(keys(g))
        catch
            String[]
        end
    elseif n > MAX_OBJECTS
        [try
             HDF5.API.h5l_get_name_by_idx(g, ".", HDF5.API.H5_INDEX_NAME,
                                          HDF5.API.H5_ITER_INC, i - 1,
                                          HDF5.API.H5P_DEFAULT)
         catch
             ""
         end for i in 1:MAX_OBJECTS]
    else
        try
            collect(keys(g))
        catch
            String[]
        end
    end
    for nm in names
        isempty(nm) && continue
        push!(out, (String(nm), link_type(g, nm)))
    end
    return out
end

"""The names of the children reached by a hard link, in name order.
A soft, external or broken link is not one, and is left to the
validator to report as E40."""
hard_link_names(g) = String[n for (n, t) in child_links(g) if t === :hard]

"""
    hard_child(parent, name) -> object or nothing

Open a child only when the link to it is a hard link and the open
succeeds.  This is the only way anything above this file opens an
object it found by walking.
"""
function hard_child(parent, name::AbstractString)
    link_type(parent, String(name)) === :hard || return nothing
    try
        return parent[String(name)]
    catch
        return nothing
    end
end

"""What an HDF5 datatype is, as sections 18 and 19 name the kinds."""
struct TypeInfo
    class::Symbol      # :string, :int, :float, :other
    size::Int          # bytes
    signed::Bool
    little::Bool
    vlen::Bool
    cset::Int
    strpad::Int
end

function type_info(t::HDF5.Datatype)
    cls = HDF5.API.h5t_get_class(t)
    size = Int(HDF5.API.h5t_get_size(t))
    if cls == HDF5.API.H5T_STRING
        vlen = HDF5.API.h5t_is_variable_str(t)
        return TypeInfo(:string, size, false, true, vlen,
                        Int(HDF5.API.h5t_get_cset(t)),
                        Int(HDF5.API.h5t_get_strpad(t)))
    elseif cls == HDF5.API.H5T_INTEGER
        sign = HDF5.API.h5t_get_sign(t) == HDF5.API.H5T_SGN_2
        order = HDF5.API.h5t_get_order(t)
        return TypeInfo(:int, size, sign, order == HDF5.API.H5T_ORDER_LE,
                        false, 0, 0)
    elseif cls == HDF5.API.H5T_FLOAT
        order = HDF5.API.h5t_get_order(t)
        return TypeInfo(:float, size, true, order == HDF5.API.H5T_ORDER_LE,
                        false, 0, 0)
    else
        return TypeInfo(:other, size, false, true, false, 0, 0)
    end
end

"""The Julia element type an on-disk numeric type maps to."""
function julia_eltype(ti::TypeInfo)
    ti.class === :int && ti.size == 1 && return Int8
    ti.class === :int && ti.size == 2 && return (ti.signed ? Int16 : UInt16)
    ti.class === :int && ti.size == 4 && return (ti.signed ? Int32 : UInt32)
    ti.class === :int && ti.size == 8 && return (ti.signed ? Int64 : UInt64)
    ti.class === :int && ti.size == 1 && return Int8
    ti.class === :float && ti.size == 4 && return Float32
    ti.class === :float && ti.size == 8 && return Float64
    ti.class === :string && return String
    return Nothing
end

function uint8_eltype(ti::TypeInfo)
    ti.class === :int && ti.size == 1 && !ti.signed && return UInt8
    return julia_eltype(ti)
end

"""
    raw_datatype(ti) -> HDF5.Datatype

The on-disk datatype a `TypeInfo` describes, rebuilt exactly: byte
order, sign, size, character set and padding all as they were.  This
is how a group this format does not own is copied back out (sections
12 and 29).  The public tree never needs it, because section 19 names
one dtype per role; a producer's private part may hold any of them,
float32 included, and a copy that widened one would not be a copy."""
function raw_datatype(ti::TypeInfo)
    if ti.class === :string
        ti.vlen && throw(MestraError("E41",
            "a variable-length string is not one this writer can copy"))
        t = HDF5.API.h5t_copy(HDF5.API.H5T_C_S1)
        HDF5.API.h5t_set_size(t, max(1, ti.size))
        HDF5.API.h5t_set_cset(t, ti.cset)
        HDF5.API.h5t_set_strpad(t, ti.strpad)
        return HDF5.Datatype(t)
    elseif ti.class === :int
        base = ti.size == 1 ? (ti.signed ?
                               (HDF5.API.H5T_STD_I8LE, HDF5.API.H5T_STD_I8LE) :
                               (HDF5.API.H5T_STD_U8LE, HDF5.API.H5T_STD_U8LE)) :
               ti.size == 2 ? (ti.signed ?
                               (HDF5.API.H5T_STD_I16LE, HDF5.API.H5T_STD_I16BE) :
                               (HDF5.API.H5T_STD_U16LE, HDF5.API.H5T_STD_U16BE)) :
               ti.size == 4 ? (ti.signed ?
                               (HDF5.API.H5T_STD_I32LE, HDF5.API.H5T_STD_I32BE) :
                               (HDF5.API.H5T_STD_U32LE, HDF5.API.H5T_STD_U32BE)) :
               ti.size == 8 ? (ti.signed ?
                               (HDF5.API.H5T_STD_I64LE, HDF5.API.H5T_STD_I64BE) :
                               (HDF5.API.H5T_STD_U64LE, HDF5.API.H5T_STD_U64BE)) :
               nothing
        base === nothing && throw(MestraError("E41",
            "an integer of $(ti.size) bytes is not one this writer can copy"))
        return le(ti.little ? base[1] : base[2])
    elseif ti.class === :float
        ti.size == 4 && return le(ti.little ? HDF5.API.H5T_IEEE_F32LE :
                                  HDF5.API.H5T_IEEE_F32BE)
        ti.size == 8 && return le(ti.little ? HDF5.API.H5T_IEEE_F64LE :
                                  HDF5.API.H5T_IEEE_F64BE)
        throw(MestraError("E41",
            "a float of $(ti.size) bytes is not one this writer can copy"))
    end
    throw(MestraError("E41", "a datatype this writer cannot copy"))
end

# -------------------------------------------------------- attributes

"""An attribute exactly as it is stored, so that E19 and E26 can be
decided and so that an unknown attribute can be copied unchanged."""
struct RawAttr
    name::String
    ti::TypeInfo
    raw::Vector{UInt8}       # the stored bytes, for a scalar attribute
    npoints::Int
    scalar::Bool             # the dataspace is scalar, as section 18 asks
    readable::Bool           # the value was read; false when refused
    value::Any               # decoded: Bool, Int64, Float64, String, ...
end

function strip_nul(raw::AbstractVector{UInt8})
    last = length(raw)
    while last > 0 && raw[last] == 0x00
        last -= 1
    end
    return raw[1:last]
end

function decode_raw(ti::TypeInfo, raw::Vector{UInt8}, npoints::Int)
    if ti.class === :string
        if ti.vlen
            return nothing          # never legal; E19 reports it
        end
        if npoints == 1
            return String(strip_nul(raw))
        end
        out = String[]
        for i in 1:npoints
            chunk = raw[((i - 1) * ti.size + 1):(i * ti.size)]
            push!(out, String(strip_nul(chunk)))
        end
        return out
    elseif ti.class === :int
        T = julia_eltype(ti)
        T === Nothing && return nothing
        vals = in_byte_order(reinterpret(T, raw), ti)
        if ti.size == 1
            # Section 18: an int8 attribute is a boolean, "value 0 for
            # false and 1 for true.  No other value is legal."  There
            # is no value to decode from any other byte, so nothing is
            # decoded and E19 says why; a reader that mapped 2 to true
            # would give a file a meaning the format does not define.
            legal(x) = x == 0 || x == 1
            if npoints == 1
                return legal(vals[1]) ? (vals[1] != 0) : nothing
            end
            return all(legal, vals) ? [v != 0 for v in vals] : nothing
        end
        return npoints == 1 ? Int64(vals[1]) : Int64.(collect(vals))
    elseif ti.class === :float
        T = julia_eltype(ti)
        T === Nothing && return nothing
        vals = in_byte_order(reinterpret(T, raw), ti)
        return npoints == 1 ? Float64(vals[1]) : Float64.(collect(vals))
    end
    return nothing
end

"""The values of `raw` bytes as the numbers the file means.  The bytes
are read in the file's own byte order, so a big-endian attribute is
swapped here; E19 still refuses it, and the message it prints then
quotes the value the file holds rather than a byte-swapped one."""
in_byte_order(vals, ti::TypeInfo) =
    (ti.little || ti.size == 1) ? vals : bswap.(vals)

"""Read one attribute exactly as stored."""
function read_raw_attr(obj, name::AbstractString)
    a = try
        HDF5.open_attribute(obj, name)
    catch
        return RawAttr(String(name), TypeInfo(:other, 0, false, true,
                                              false, 0, 0),
                       UInt8[], 0, false, false, nothing)
    end
    try
        t = HDF5.datatype(a)
        ti = type_info(t)
        sp = HDF5.dataspace(a)
        scalar = is_scalar_space(sp)
        npoints = space_npoints(sp)
        # A variable-length attribute is never legal (E19) and its
        # buffer is pointers, not bytes, so it is never read here.
        if ti.vlen
            return RawAttr(String(name), ti, UInt8[], npoints, scalar,
                           false, nothing)
        end
        # Size the buffer from the file's claim only after checking it.
        if ti.size <= 0 || npoints < 0 ||
           Int128(ti.size) * Int128(max(npoints, 1)) > MAX_ATTR_BYTES
            return RawAttr(String(name), ti, UInt8[], npoints, scalar,
                           false, nothing)
        end
        raw = Vector{UInt8}(undef, ti.size * max(npoints, 1))
        try
            HDF5.API.h5a_read(a, t, raw)
        catch
            return RawAttr(String(name), ti, UInt8[], npoints, scalar,
                           false, nothing)
        end
        # Only a scalar attribute carries a value this format names;
        # an array one is E19 and its elements are not interpreted.
        value = scalar ? decode_raw(ti, raw, npoints) : nothing
        return RawAttr(String(name), ti, raw, npoints, scalar, true, value)
    catch
        return RawAttr(String(name), TypeInfo(:other, 0, false, true,
                                              false, 0, 0),
                       UInt8[], 0, false, false, nothing)
    finally
        close(a)
    end
end

function attr_names(obj)
    try
        return collect(keys(HDF5.attributes(obj)))
    catch
        return String[]
    end
end

function raw_attrs(obj)
    out = RawAttr[]
    for name in attr_names(obj)
        push!(out, read_raw_attr(obj, name))
    end
    return out
end

"""Every attribute of `obj` that this format owns, by name."""
function own_attrs(obj)
    d = Dict{String,RawAttr}()
    for name in attr_names(obj)
        name in MACHINERY_ATTRS && continue
        d[name] = read_raw_attr(obj, name)
    end
    return d
end


# writing

function write_string_attr(obj, name::AbstractString, value::AbstractString)
    write_raw_string_attr(obj, name, Vector{UInt8}(codeunits(value)))
end

function write_raw_string_attr(obj, name::AbstractString,
                               bytes::Vector{UInt8})
    n = max(1, length(bytes))
    padded = vcat(bytes, zeros(UInt8, n - length(bytes)))
    dt = fixed_string_type(n)
    sp = scalar_space()
    a = HDF5.create_attribute(obj, String(name), dt, sp)
    try
        HDF5.API.h5a_write(a, dt, padded)
    finally
        close(a)
    end
    return nothing
end

function write_scalar_attr(obj, name::AbstractString, value, dt::HDF5.Datatype)
    sp = scalar_space()
    a = HDF5.create_attribute(obj, String(name), dt, sp)
    try
        HDF5.API.h5a_write(a, dt, [value])
    finally
        close(a)
    end
    return nothing
end

write_bool_attr(obj, name, v::Bool) =
    write_scalar_attr(obj, name, Int8(v ? 1 : 0), le(HDF5.API.H5T_STD_I8LE))
write_int_attr(obj, name, v::Integer) =
    write_scalar_attr(obj, name, Int64(v), le(HDF5.API.H5T_STD_I64LE))
write_float_attr(obj, name, v::Real) =
    write_scalar_attr(obj, name, Float64(v), le(HDF5.API.H5T_IEEE_F64LE))

"""Copy an attribute back out exactly as it came in."""
function write_raw_attr(obj, a::RawAttr)
    if a.ti.class === :string && !a.ti.vlen && a.npoints == 1
        write_raw_string_attr(obj, a.name, a.raw)
    elseif a.ti.class === :int && a.ti.size == 1
        write_bool_attr(obj, a.name, a.value === true)
    elseif a.ti.class === :int
        write_int_attr(obj, a.name, a.value)
    elseif a.ti.class === :float
        write_float_attr(obj, a.name, a.value)
    else
        error("cannot copy attribute $(a.name)")
    end
end

# ---------------------------------------------------------- datasets

"""The on-disk shape of a dataset, in C order, with its maximum."""
function disk_shape(d::HDF5.Dataset)
    sp = HDF5.dataspace(d)
    if HDF5.API.h5s_get_simple_extent_ndims(sp) == 0
        return (Int[], Int[])
    end
    dims, maxdims = HDF5.API.h5s_get_simple_extent_dims(sp)
    return (Int.(dims), [m == HDF5.API.H5S_UNLIMITED ? -1 : Int(m)
                         for m in maxdims])
end

function dataset_layout(d::HDF5.Dataset)
    dcpl = HDF5.get_create_properties(d)
    layout = HDF5.API.h5p_get_layout(dcpl)
    chunk = nothing
    if layout == HDF5.API.H5D_CHUNKED
        try
            c, _ = HDF5.API.h5p_get_chunk(dcpl)
            chunk = Int.(c)
        catch
            chunk = nothing
        end
    end
    nf = try
        Int(HDF5.API.h5p_get_nfilters(dcpl))
    catch
        0
    end
    filters = Tuple{Int,Vector{Int}}[]
    for i in 0:(nf - 1)
        flags = Ref{Cuint}()
        # The library writes at most as many values as the buffer
        # holds and reports how many the filter declares, which may
        # be more.  Taking the reported count as the length of the
        # buffer is how a reader reads past the end of it.
        cd = Vector{Cuint}(undef, MAX_FILTER_CD)
        nelem = Ref{Csize_t}(length(cd))
        namebuf = Vector{UInt8}(undef, 256)
        fid = try
            HDF5.API.h5p_get_filter(dcpl, i, flags, nelem, cd,
                                    length(namebuf), namebuf, C_NULL)
        catch
            continue
        end
        got = min(Int(nelem[]), length(cd))
        got = max(got, 0)
        push!(filters, (Int(fid), Int.(cd[1:got])))
    end
    return (layout == HDF5.API.H5D_CHUNKED ? :chunked :
            layout == HDF5.API.H5D_CONTIGUOUS ? :contiguous : :other,
            chunk, filters)
end

"""
    check_readable(d; max_elements) -> Int

The number of elements `d` holds, having checked that reading it will
not ask for more than `max_elements` of them and that its chunk is no
larger either, since the library reads a whole chunk at a time.
Throws `MestraError("E41")` rather than letting a claim size a
buffer.
"""
function check_readable(d::HDF5.Dataset;
                        max_elements::Integer = DEFAULT_MAX_ELEMENTS,
                        max_bytes::Integer = MAX_READ_BYTES[])
    cdims, _ = disk_shape(d)
    n = element_count(cdims)
    path = try
        HDF5.name(d)
    catch
        "?"
    end
    n > max_elements && throw(MestraError("E41",
        "$(path) declares $(n) elements, more than the $(max_elements) " *
        "this reader will materialise; raise `max_elements` if the " *
        "file is trusted"))
    width = try
        Int(HDF5.API.h5t_get_size(HDF5.datatype(d)))
    catch
        1
    end
    bytes = Int128(n) * Int128(max(width, 1))
    bytes > max_bytes && throw(MestraError("E41",
        "$(path) would take $(bytes) bytes, more than the $(max_bytes) " *
        "this reader will materialise; raise `Mestra.MAX_READ_BYTES[]` " *
        "if the file is trusted"))
    check_chunk(d; max_elements = max_elements, max_bytes = max_bytes)
    return n
end

"""
    check_chunk(d; max_elements, max_bytes)

The library reads a whole chunk at a time, so a chunk this reader
would not materialise is a dataset it will not read, however small the
part asked for.  This is what makes a one-row lazy read of a hostile
file cost one row.
"""
function check_chunk(d::HDF5.Dataset;
                     max_elements::Integer = DEFAULT_MAX_ELEMENTS,
                     max_bytes::Integer = MAX_READ_BYTES[])
    _, chunk, _ = dataset_layout(d)
    chunk === nothing && return nothing
    path = try
        HDF5.name(d)
    catch
        "?"
    end
    width = try
        Int(HDF5.API.h5t_get_size(HDF5.datatype(d)))
    catch
        1
    end
    c = element_count(chunk)
    c > max_elements && throw(MestraError("E41",
        "$(path) has a chunk of $(c) elements, more than the " *
        "$(max_elements) this reader will materialise"))
    bytes = Int128(c) * Int128(max(width, 1))
    bytes > max_bytes && throw(MestraError("E41",
        "$(path) has a chunk of $(bytes) bytes, more than the " *
        "$(max_bytes) this reader will materialise"))
    return nothing
end

"""Raw bytes of a whole dataset, in the file's own datatype."""
function read_raw_dataset(d::HDF5.Dataset;
                          max_elements::Integer = DEFAULT_MAX_ELEMENTS)
    t = HDF5.datatype(d)
    ti = type_info(t)
    ti.vlen && throw(MestraError("E41",
        "a variable-length dataset is not one this reader will read"))
    n = check_readable(d; max_elements = max_elements)
    n == 0 && return ti, UInt8[], 0
    ti.size > 0 || throw(MestraError("E41", "a datatype of no size"))
    raw = Vector{UInt8}(undef, ti.size * n)
    try
        HDF5.API.h5d_read(d, t, HDF5.API.H5S_ALL, HDF5.API.H5S_ALL,
                          HDF5.API.H5P_DEFAULT, raw)
    catch e
        throw(MestraError("E41",
            "the library could not read this dataset: " *
            first(sprint(showerror, e), 200)))
    end
    return ti, raw, n
end

"""A fixed-length string dataset, as the raw record bytes."""
function read_string_records(d::HDF5.Dataset;
                             max_elements::Integer = DEFAULT_MAX_ELEMENTS)
    ti, raw, n = read_raw_dataset(d; max_elements = max_elements)
    recs = Vector{Vector{UInt8}}(undef, n)
    for i in 1:n
        recs[i] = raw[((i - 1) * ti.size + 1):(i * ti.size)]
    end
    return ti, recs
end

"""A fixed-length string dataset as Julia strings.

Section 25: "A reader that cannot recover the bytes of a string must
say so rather than return something else."  A record that is not valid
UTF-8, or that holds a NUL byte before its trailing padding, is E26
here and not a Julia `String` built over the bytes anyway."""
function read_strings(d::HDF5.Dataset;
                      max_elements::Integer = DEFAULT_MAX_ELEMENTS)
    recs = read_string_records(d; max_elements = max_elements)[2]
    all(check_string_bytes, recs) || throw(MestraError("E26",
        "a string here is not valid UTF-8 or holds a NUL byte before " *
        "its trailing padding; this reader will not hand back a string " *
        "it cannot recover"))
    return [String(strip_nul(r)) for r in recs]
end

"""
    safe_read(d; max_elements) -> Array

A whole dataset as a Julia array, checked first.  Everything above
this file that materialises a dataset comes through here or through
`read_raw_dataset`, so no HDF5.jl high-level read is ever reached with
a shape this package has not looked at.
"""
function safe_read(d::HDF5.Dataset;
                   max_elements::Integer = DEFAULT_MAX_ELEMENTS)
    ti = type_info(HDF5.datatype(d))
    ti.class === :string &&
        return read_strings(d; max_elements = max_elements)
    check_readable(d; max_elements = max_elements)
    try
        return HDF5.read(d)
    catch e
        throw(MestraError("E41",
            "the library could not read $(HDF5.name(d)): " *
            first(sprint(showerror, e), 200)))
    end
end

"""Create a dataset with object times off, as section 30 requires.

`attr_order` is section 21's rule for a dimension scale and for
nothing else: attribute creation order tracked and indexed, which is
what gives the scale a version 2 object header and so lets its
REFERENCE_LIST live in the file's heap rather than in an object header
message, where an attribute may not exceed 64 KiB.

It is not decoration.  That list grows sixteen bytes an attachment,
and every key, every scalar, every row-varying array and
`/row_support` attaches to the one `row` scale, so a scale created
with the library's defaults takes 4085 attachments and the 4086th
fails *after* deleting the list it was extending -- leaving a file
every reader and every validator accepts.  `vectors/cases/wide_keys`
is the corpus case that could not be written without the rule, and
`vectors/cases/err_e42` the one that breaks it on purpose."""
function make_dcpl(; chunk = nothing, deflate = nothing, shuffle = false,
                   attr_order::Bool = false)
    dcpl = HDF5.DatasetCreateProperties()
    # Section 30: object time tracking off, so that two runs of a
    # writer produce the same bytes.  HDF5.jl initialises the property
    # list on the first set, so this must come before the raw calls.
    dcpl.obj_track_times = false
    if attr_order
        HDF5.API.h5p_set_attr_creation_order(dcpl, SCALE_ATTR_ORDER)
    end
    if chunk !== nothing
        # The raw API is C order throughout, which is the order every
        # shape in the specification is written in.
        HDF5.API.h5p_set_chunk(dcpl, length(chunk),
                               HDF5.API.hsize_t[chunk...])
    end
    if shuffle
        HDF5.API.h5p_set_shuffle(dcpl)
    end
    if deflate !== nothing
        HDF5.API.h5p_set_deflate(dcpl, deflate)
    end
    return dcpl
end

"""A dataspace from a C-order shape, with unlimited marked by -1."""
function make_space(cdims::Vector{Int}, cmax::Vector{Int})
    jd = HDF5.API.hsize_t[cdims...]
    jm = HDF5.API.hsize_t[m < 0 ? HDF5.API.H5S_UNLIMITED :
                          HDF5.API.hsize_t(m) for m in cmax]
    return HDF5.Dataspace(HDF5.API.h5s_create_simple(length(jd), jd, jm))
end

"""Create a dataset and write raw bytes into it.  `cdims` is the
C-order shape, which is what every shape in the specification is."""
function create_raw_dataset(parent, name::AbstractString,
                            dt::HDF5.Datatype, cdims::Vector{Int},
                            cmax::Vector{Int}, raw::Vector{UInt8};
                            chunk = nothing, deflate = nothing,
                            shuffle = false, attr_order::Bool = false)
    dcpl = make_dcpl(chunk = chunk === nothing ? nothing : Vector{Int}(chunk),
                     deflate = deflate, shuffle = shuffle,
                     attr_order = attr_order)
    sp = make_space(cdims, cmax)
    d = HDF5.create_dataset(parent, String(name), dt, sp; dcpl = dcpl)
    if prod(cdims) > 0 && !isempty(raw)
        HDF5.API.h5d_write(d, dt, HDF5.API.H5S_ALL, HDF5.API.H5S_ALL,
                           HDF5.API.H5P_DEFAULT, raw)
    end
    return d
end

"""A group with object times off."""
function create_group(parent, name::AbstractString)
    gcpl = HDF5.GroupCreateProperties()
    gcpl.obj_track_times = false
    lcpl = HDF5._link_properties(String(name))
    return HDF5.create_group(parent, String(name), lcpl, gcpl)
end

# ---------------------------------------------------- dimension scales

"""The creation-order flags section 21 requires on the dataset
creation property list of every dimension scale, and E42 checks for.
It is what netCDF-C sets on every object it creates."""
const SCALE_ATTR_ORDER = HDF5.API.H5P_CRT_ORDER_TRACKED |
                         HDF5.API.H5P_CRT_ORDER_INDEXED

"""
    scale_attr_order(d) -> Union{Nothing,UInt32}

The attribute creation-order flags `d` was created with, or `nothing`
when the library would not say.  This is the one property list this
format requires to be other than the default (sections 21 and 23),
and it is what E42 is decided from.
"""
function scale_attr_order(d::HDF5.Dataset)
    try
        dcpl = HDF5.get_create_properties(d)
        return UInt32(HDF5.API.h5p_get_attr_creation_order(dcpl))
    catch
        return nothing
    end
end

"""True when `d` was created with attribute creation order tracked and
indexed, which is what section 21 requires of a dimension scale and
what E42 reports the absence of."""
scale_order_tracked(d::HDF5.Dataset) =
    (f = scale_attr_order(d); f !== nothing && (f & SCALE_ATTR_ORDER) ==
                                               SCALE_ATTR_ORDER)

"""Create a dimension scale exactly as netCDF-C writes one (21).

The creation property list is the point of it: attribute creation
order tracked and indexed, so that the scale gets a version 2 object
header and its REFERENCE_LIST can grow in the file's heap past the
64 KiB an object header message may hold, and object time tracking
off, because a version 2 object header records four timestamps unless
it is told not to and a file that records when it was written is not
byte reproducible.  Without the first, a scale takes at most 4085
attachments and the 4086th destroys the REFERENCE_LIST it was
extending (E42).  This is the only property list in this format that
is not the library's default."""
function create_scale(parent, name::AbstractString, length_::Integer;
                      unlimited::Bool = false)
    len = Int(length_)
    cdims = [len]
    cmax = [unlimited ? -1 : len]
    chunk = unlimited ? [1] : [max(1, len)]
    d = create_raw_dataset(parent, name, le(HDF5.API.H5T_IEEE_F32BE),
                           cdims, cmax, UInt8[]; chunk = chunk,
                           attr_order = true)
    HDF5.API.h5ds_set_scale(d, dim_scale_name(len))
    return d
end

function is_scale(d::HDF5.Dataset)
    try
        return HDF5.API.h5ds_is_scale(d)
    catch
        return false
    end
end

"""Attach `scale` to C-order axis `axis` (zero based) of `d`."""
attach_scale!(d::HDF5.Dataset, scale::HDF5.Dataset, axis::Integer) =
    HDF5.API.h5ds_attach_scale(d, scale, axis)

"""The DIMENSION_LIST attribute of `d`: for each C-order axis, the
object references of the scales attached to that axis.  `nothing`
means the attribute is not there, so nothing is attached anywhere;
`:unreadable` means it is there and the library would not give it.

This attribute is how section 21 says to resolve an attached scale --
from the dataset's own record, through a map built during this
package's own bounded walk.  The alternative the HDF5.jl wrapper
offers, asking H5DSis_attached of each scale in turn, reads the
scale's REFERENCE_LIST every time, and that list holds one entry per
dataset attached to it: on a file with n datasets on the `row`
dimension the walk then costs n^2.  Nothing here asks the library for
the path of a scale object, because that search walks the group
hierarchy and runs off the stack on a deeply nested file.
"""
function dimension_list(d::HDF5.Dataset)
    have = try
        haskey(HDF5.attributes(d), "DIMENSION_LIST")
    catch
        return :unreadable
    end
    have || return nothing
    try
        a = HDF5.open_attribute(d, "DIMENSION_LIST")
        try
            v = HDF5.read(a)
            return v isa Vector{Vector{HDF5.Reference}} ? v : :unreadable
        finally
            close(a)
        end
    catch
        return :unreadable
    end
end

"""The references on C-order axis `axis` of a `dimension_list` result."""
function axis_refs(dl, axis::Integer)
    dl isa Vector{Vector{HDF5.Reference}} || return HDF5.Reference[]
    i = Int(axis) + 1
    (1 <= i <= length(dl)) || return HDF5.Reference[]
    return dl[i]
end

"""Every dimension scale in the file, as (link name, path, dataset,
object reference), nearest group first so that a support-local `row`
is found before the file one.  The reference is what a dataset's
DIMENSION_LIST holds, so it is the key a scale is found by; the path
is this walk's own and is never asked of the library, which would
search the group hierarchy for it and run off the stack on a deep
file (section 21)."""
function collect_scales(f::HDF5.File)
    out = Tuple{String,String,HDF5.Dataset,Union{Nothing,HDF5.Reference}}[]
    # An explicit stack, not the call stack: a file chooses how deep
    # its groups go and thirty thousand levels would overflow one.
    stack = Tuple{Any,String,Int}[(f, "", 0)]
    visited = 0
    while !isempty(stack)
        g, base, depth = pop!(stack)
        depth >= MAX_DEPTH && continue
        for (name, kind) in child_links(g)
            kind === :hard || continue
            visited += 1
            visited > MAX_OBJECTS && return out
            obj = hard_child(g, name)
            obj === nothing && continue
            path = base * "/" * String(name)
            if obj isa HDF5.Dataset
                ok = try
                    haskey(HDF5.attributes(obj), "CLASS") && is_scale(obj)
                catch
                    false
                end
                if ok
                    ref = try
                        HDF5.Reference(g, String(name))
                    catch
                        nothing
                    end
                    push!(out, (String(name), path, obj, ref))
                end
            elseif obj isa HDF5.Group
                push!(stack, (obj, path, depth + 1))
            end
        end
    end
    return out
end
