function hex = supportId(support)
%MESTRA.SUPPORTID  The content hash that identifies a support.
%
%   HEX = MESTRA.SUPPORTID(SUPPORT) returns the 64-character lower-case
%   hexadecimal SHA-256 of the byte string specification section 24
%   defines: the node count as one little-endian int64, then
%   `cell_types` as uint8, then `cell_offsets` and `cell_connectivity`
%   as little-endian int64, and, for a support of kind `axis` only, the
%   coordinates as little-endian float64 in storage order.
%
%   Coordinates of a mesh support are not hashed, because they may vary
%   between rows while the support does not.  A support of kind `axis`
%   or `none` has no cell arrays, so those three steps contribute no
%   bytes at all and are not replaced by anything.
%
%   SUPPORT is a support record of a mestra.Dataset.  The id is the
%   same in every file that holds the same mesh, which is what makes
%   "same support" an attribute comparison and not an array
%   comparison.
%
%   The declared kind decides which arrays are hashed, so a support
%   that declares none has no digest and this raises mestra:E39, and
%   one whose kind section 6 does not have raises mestra:E02.
%
%   The digest is computed in MATLAB by mestra.internal.Sha and not by
%   any outside library, so the package needs nothing but base MATLAB.
%
%   Example
%
%       s = d.support('s0');
%       isequal(mestra.supportId(s), s.supportId)     % E08 is this test
%
%   See also mestra.Dataset, mestra.validate.

    if isempty(support.kind)
        error('mestra:E39', ...
              ['E39: support %s does not say its kind, and the kind ' ...
               'decides which of its arrays section 24 hashes, so it ' ...
               'has no support_id'], support.name);
    end
    if ~any(strcmp(support.kind, {'mesh', 'axis', 'none'}))
        error('mestra:E02', ...
              ['E02: support %s is of kind "%s", which section 6 does ' ...
               'not have, so it has no support_id'], ...
              support.name, support.kind);
    end
    Sha = mestra.internal.Sha;
    bytes = Sha.int64le(support.nNodes);
    if strcmp(support.kind, 'mesh')
        % A support of kind `axis` or `none` has no cell arrays, so
        % steps 2 to 4 contribute no bytes at all for it, whatever a
        % malformed file happens to store (sections 20 and 24).
        if ~isempty(support.cellTypes)
            bytes = [bytes uint8(support.cellTypes(:)')];
        end
        if ~isempty(support.cellOffsets)
            bytes = [bytes Sha.int64le(support.cellOffsets)];
        end
        if ~isempty(support.cellConnectivity)
            bytes = [bytes Sha.int64le(support.cellConnectivity)];
        end
    end
    if strcmp(support.kind, 'axis') && ~isempty(support.coordinates)
        coords = support.coordinates;
        % The coordinates are hashed in the file's storage order.  A
        % MATLAB array holds the file's axes reversed, so its own
        % column-major order IS the file's C order, element for
        % element, and no permutation is needed here.
        bytes = [bytes Sha.float64le(reshape(coords.values, 1, []))];
    end
    hex = Sha.hex256(bytes);
end
