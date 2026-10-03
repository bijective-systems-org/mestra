# Section 24: support_id.

"""
    support_id(n_nodes; cell_types, cell_offsets, cell_connectivity,
               axis_coordinates) -> String

The SHA-256 digest of section 24, in lower-case hexadecimal: the node
count as one little-endian int64, then the cell types as uint8, the
offsets and the connectivity as little-endian int64, and, for an
`axis` support only, the coordinates as little-endian float64, all in
storage order with nothing between them.

Coordinates of a mesh support are not hashed, because they may vary
between rows while the support does not.
"""
function support_id(n_nodes::Integer;
                    cell_types = UInt8[],
                    cell_offsets = Int64[],
                    cell_connectivity = Int64[],
                    axis_coordinates = nothing)
    ctx = SHA.SHA256_CTX()
    SHA.update!(ctx, reinterpret(UInt8, [htol(Int64(n_nodes))]))
    isempty(cell_types) ||
        SHA.update!(ctx, collect(UInt8.(cell_types)))
    isempty(cell_offsets) ||
        SHA.update!(ctx, reinterpret(UInt8,
                                     htol.(Int64.(collect(cell_offsets)))))
    isempty(cell_connectivity) ||
        SHA.update!(ctx, reinterpret(UInt8,
                                     htol.(Int64.(collect(cell_connectivity)))))
    if axis_coordinates !== nothing
        v = Float64.(vec(axis_coordinates))
        isempty(v) || SHA.update!(ctx, reinterpret(UInt8, htol.(v)))
    end
    return bytes2hex(SHA.digest!(ctx))
end

"""
    support_id(s::Support) -> String

The digest computed from what the support actually stores.  An `axis`
support hashes its coordinates; section 24 says to hash the stored
bytes as they are even when those coordinates wrongly vary, so that
such a file breaks E35 and nothing else.

The declared kind decides which arrays are hashed, so a support that
declares none (E39), or one section 6 does not have (E02), has no
digest and this throws.
"""
function support_id(s::Support)
    isempty(s.kind) && throw(MestraError("E39",
        "support $(s.name) does not say its kind, and the kind decides " *
        "which of its arrays section 24 hashes, so it has no support_id"))
    s.kind in ("mesh", "axis", "none") || throw(MestraError("E02",
        "support $(s.name) is of kind \"$(s.kind)\", which section 6 " *
        "does not have, so it has no support_id"))
    # A support whose arrays the reader refused or could not read has
    # no digest: computing one over what is missing would answer a
    # question the file did not.
    if s.kind == "mesh" && s.n_cells > 0 &&
       (s.cell_types === nothing || s.cell_offsets === nothing ||
        s.cell_connectivity === nothing)
        throw(MestraError("E41",
            "support $(s.name) declares $(s.n_cells) cells whose arrays " *
            "this reader could not read, so it has no support_id"))
    end
    if s.kind == "axis" &&
       (s.coordinates === nothing || raw_data(s.coordinates) === nothing)
        throw(MestraError("E41",
            "support $(s.name) is an axis whose coordinates this reader " *
            "could not read, so it has no support_id"))
    end
    coords = nothing
    if s.kind == "axis" && s.coordinates !== nothing &&
       raw_data(s.coordinates) !== nothing
        # The Julia array's linear memory is the file's own byte
        # order, so no permutation is needed here.
        coords = vec(raw_data(s.coordinates))
    end
    # Section 24: an `axis` or `none` support has no cell arrays, so
    # steps 2 to 4 contribute no bytes at all for it, even where a file
    # wrongly carries a cell dataset (which is E38 and nothing else).
    mesh = s.kind == "mesh"
    return support_id(s.n_nodes;
                      cell_types = mesh && s.cell_types !== nothing ?
                                   s.cell_types : UInt8[],
                      cell_offsets = mesh && s.cell_offsets !== nothing ?
                                     s.cell_offsets : Int64[],
                      cell_connectivity = mesh &&
                                          s.cell_connectivity !== nothing ?
                                          s.cell_connectivity : Int64[],
                      axis_coordinates = coords)
end
