classdef Codec
%Codec  The dictionary codec of specification sections 17 and 25.
%
%   A callable's dictionary is a containers.Map with char keys.  Its
%   leaves are
%
%       a nested dictionary   a containers.Map
%       a number              a double (float64) or an int64
%       a boolean             a logical
%       a string              a char row vector
%       a null                the MATLAB value missing
%       an array or a list    a mestra.Array
%
%   A number, a boolean, a string and a null are attributes on the
%   enclosing group; an array is a dataset with a dimension scale on
%   each axis named mestra_<dataset>_d<i>.  A 1-by-1 double and a
%   mestra.Array of shape (1) are therefore different values in the
%   file and stay different across a round trip, which is what section
%   25 asks for.
%
%   See also mestra.Array, mestra.Callable.

    properties (Constant)
        % The reserved value of section 18: one NUL byte then "null".
        NULL_BYTES = uint8([0 110 117 108 108]);
    end

    methods (Static)

        function [dict, problems] = read(gid, isTop, depth, eager)
        %read  Rebuild a dictionary from an HDF5 group.
        %   `problems` lists the section 25 violations found, by rule
        %   identifier and reason, so that the validator can report
        %   E32 without a second pass.  A problem that is this
        %   reader's limit rather than the file's mistake is prefixed
        %   "U03 ", because a dictionary nested past what this reader
        %   follows is not a breach of section 25.
        %
        %   The nesting of a dictionary is the file's choice, so the
        %   walk stops at maxDepth levels rather than descending until
        %   the stack gives out, and it never follows a soft or an
        %   external link.
        %
        %   With `eager` false the walk is the same and no dataset's
        %   data is read, because section 7 of docs/api-conventions.md
        %   says an open never reads a dataset inside a dictionary.
        %   Everything the structure decides is still found: the depth
        %   cap, a zero-dimensional dataset, a dtype section 25 does
        %   not allow, a link that is not a hard one, and a top-level
        %   key the container owns.  What the bytes decide -- a string
        %   holding a NUL, and a dataset above the element cap --
        %   waits for the read.
            if nargin < 2, isTop = false; end
            if nargin < 3, depth = 0; end
            if nargin < 4, eager = true; end
            dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
            problems = {};
            H5 = mestra.internal.H5;
            if depth > mestra.internal.Limits.get('maxDepth')
                problems = {sprintf(['U03 a dictionary nested past %d ' ...
                    'levels was not followed'], ...
                    mestra.internal.Limits.get('maxDepth'))};
                return
            end

            for name = H5.publicAttrNames(gid)
                key = name{1};
                if isTop && (strcmp(key, 'type') || strcmp(key, 'repr'))
                    continue    % the container's own, not the dictionary's
                end
                if mestra.internal.Text.reserved(key), continue, end
                try
                    info = H5.attrInfo(gid, key);
                catch
                    problems{end + 1} = sprintf( ...
                        '%s: an attribute that would not be described', ...
                        key); %#ok<AGROW>
                    continue
                end
                if ~info.scalar
                    % Section 25 stores a number, a boolean, a string
                    % and a null as attributes, and every one of them
                    % is one value.
                    problems{end + 1} = sprintf( ...
                        '%s: an attribute that is not a scalar', ...
                        key); %#ok<AGROW>
                    continue
                end
                switch info.type
                    case 'int8'
                        dict(key) = logical(H5.readAttr(gid, key));
                    case 'int64'
                        dict(key) = int64(H5.readAttr(gid, key));
                    case 'float64'
                        dict(key) = double(H5.readAttr(gid, key));
                    case 'string'
                        bytes = H5.readRawStrAttr(gid, key);
                        if isequal(bytes, mestra.internal.Codec.NULL_BYTES)
                            dict(key) = missing;
                        else
                            [ok, why] = ...
                                mestra.internal.Text.checkStringBytes(bytes);
                            if ~ok
                                problems{end + 1} = ...
                                    sprintf('%s: %s', key, why); %#ok<AGROW>
                            end
                            dict(key) = H5.toText(bytes);
                        end
                    case 'vlstring'
                        problems{end + 1} = sprintf( ...
                            '%s: a variable-length string', key); %#ok<AGROW>
                        dict(key) = H5.readAttr(gid, key);
                    otherwise
                        problems{end + 1} = ...
                            sprintf('%s: a dtype section 25 does not allow', ...
                                    key); %#ok<AGROW>
                end
            end

            for name = H5.children(gid)
                key = name{1};
                if mestra.internal.Text.reserved(key), continue, end
                kind = H5.childType(gid, key);
                if any(strcmp(kind, {'soft', 'external'}))
                    % E40: a link in the public tree that is not a
                    % hard link, and a dictionary is public.
                    problems{end + 1} = sprintf( ...
                        'E40 %s: a %s link, which this reader does not follow', ...
                        key, kind); %#ok<AGROW>
                    continue
                end
                if ~any(strcmp(kind, {'group', 'dataset'}))
                    problems{end + 1} = sprintf( ...
                        'U03 %s: a %s, which this reader does not follow', ...
                        key, kind); %#ok<AGROW>
                    continue
                end
                if strcmp(kind, 'group')
                    if isTop && (strcmp(key, 'type') || strcmp(key, 'repr'))
                        problems{end + 1} = sprintf( ...
                            '%s: a top-level key the container owns', ...
                            key); %#ok<AGROW>
                    end
                    sub = H5G.open(gid, key);
                    [dict(key), subProblems] = ...
                        mestra.internal.Codec.read(sub, false, depth + 1, ...
                                                   eager);
                    H5G.close(sub);
                    problems = [problems subProblems]; %#ok<AGROW>
                else
                    did = H5D.open(gid, key);
                    try
                        info = H5.dsetInfo(did);
                    catch
                        H5D.close(did);
                        problems{end + 1} = sprintf( ...
                            '%s: a dataset that would not be described', ...
                            key); %#ok<AGROW>
                        continue
                    end
                    if isempty(info.dims)
                        problems{end + 1} = sprintf( ...
                            '%s: a zero-dimensional dataset', key); %#ok<AGROW>
                    end
                    if ~any(strcmp(info.type, ...
                            {'int8', 'int32', 'int64', 'float64', 'string'}))
                        problems{end + 1} = ...
                            sprintf('%s: dtype %s is not representable', ...
                                    key, info.type); %#ok<AGROW>
                    end
                    if isTop && (strcmp(key, 'type') || strcmp(key, 'repr'))
                        problems{end + 1} = sprintf( ...
                            '%s: a top-level key the container owns', ...
                            key); %#ok<AGROW>
                    end
                    if ~eager
                        H5D.close(did);
                        continue
                    end
                    try
                        data = H5.readData(did, info);
                    catch err
                        H5D.close(did);
                        problems{end + 1} = sprintf('U03 %s: %s', key, ...
                            regexprep(strtrim(err.message), ...
                                      '\s+', ' ')); %#ok<AGROW>
                        continue
                    end
                    if strcmp(info.type, 'string')
                        for i = 1:numel(data)
                            if any(uint8(data{i}) == 0)
                                problems{end + 1} = ...
                                    sprintf('%s: a string with a NUL byte', ...
                                            key); %#ok<AGROW>
                            end
                        end
                        dict(key) = mestra.Array(reshape(data, [], 1), ...
                                                 info.dims);
                    elseif isempty(info.dims)
                        dict(key) = double(data);
                    else
                        elements = mestra.internal.Codec.cOrder( ...
                            data, info.dims);
                        dict(key) = mestra.Array.fromFile(elements, ...
                            info.dims, mestra.internal.H5.baseType(info.type));
                    end
                    H5D.close(did);
                end
            end
        end

        function v = cOrder(data, dims) %#ok<INUSD>
        %cOrder  Flatten data read from HDF5 into C order.
        %   A MATLAB array read from HDF5 holds the file's axes
        %   reversed, so its own column-major order is already the
        %   file's C order, element for element.
            v = data(:)';
        end

        function write(gid, dict, isTop)
        %write  Write a dictionary into an HDF5 group.
        %   Keys are visited in ascending order of their UTF-8 bytes,
        %   so that two writers given the same dictionary produce the
        %   same file (section 25).
            if nargin < 3, isTop = false; end
            H5 = mestra.internal.H5;
            keys = H5.sortByBytes(dict.keys());
            for i = 1:numel(keys)
                key = keys{i};
                value = dict(key);
                if ~mestra.internal.Text.legalName(key)
                    error('mestra:E33', ...
                          'dictionary key "%s" is not a legal name (E33)', key);
                end
                if mestra.internal.Text.reserved(key)
                    error('mestra:E33', ...
                          ['dictionary key "%s" uses the reserved ' ...
                           'prefix (E33)'], key);
                end
                if isTop && (strcmp(key, 'type') || strcmp(key, 'repr'))
                    error('mestra:E32', ...
                          ['a dictionary may not have "%s" at its top ' ...
                           'level (E32)'], key);
                end
                mestra.internal.Codec.writeValue(gid, key, value);
            end
        end

        function writeValue(gid, key, value)
        %writeValue  Write one dictionary entry.
            H5 = mestra.internal.H5;
            if isa(value, 'containers.Map')
                gcpl = H5.plist('H5P_GROUP_CREATE');
                sub = H5G.create(gid, key, 'H5P_DEFAULT', gcpl, 'H5P_DEFAULT');
                H5P.close(gcpl);
                mestra.internal.Codec.write(sub, value);
                H5G.close(sub);
                return
            end
            if isa(value, 'mestra.Array')
                mestra.internal.Codec.writeArray(gid, key, value);
                return
            end
            if isa(value, 'missing')
                H5.writeRawStrAttr(gid, key, mestra.internal.Codec.NULL_BYTES);
                return
            end
            if ischar(value)
                if any(uint8(value) == 0)
                    error('mestra:E32', ...
                          'the string "%s" holds a NUL byte (E32)', key);
                end
                H5.writeStrAttr(gid, key, value);
                return
            end
            if isstring(value) && isscalar(value)
                H5.writeStrAttr(gid, key, char(value));
                return
            end
            if islogical(value) && isscalar(value)
                H5.writeNumAttr(gid, key, int8(value), 'int8');
                return
            end
            if isa(value, 'int64') && isscalar(value)
                H5.writeNumAttr(gid, key, value, 'int64');
                return
            end
            if isa(value, 'double') && isscalar(value)
                H5.writeNumAttr(gid, key, value, 'float64');
                return
            end
            error('mestra:E32', ...
                  ['the value of "%s" is a %s, which section 25 does not ' ...
                   'represent (E32); wrap an array in mestra.Array'], ...
                  key, class(value));
        end

        function writeArray(gid, key, arr)
        %writeArray  A dictionary dataset and its own scales.
            H5 = mestra.internal.H5;
            dtype = arr.dtype();
            if isempty(dtype)
                error('mestra:E32', ...
                      ['the array "%s" holds a %s, which is not ' ...
                       'representable (E32)'], key, class(arr.data));
            end
            dims = arr.shape;
            if isempty(dims)
                error('mestra:E32', ...
                      'the array "%s" is zero-dimensional (E32)', key);
            end
            empty = any(dims == 0);
            if empty
                % Section 25 makes each zero-length axis unlimited, and
                % E43 allows that axis and no other.
                maxdims = dims;
                maxdims(dims == 0) = -1;
                chunk = ones(1, numel(dims));
            else
                maxdims = dims;
                chunk = [];
            end
            strSize = 0;
            data = arr.buffer();
            if strcmp(dtype, 'string')
                strSize = 1;
                for i = 1:numel(data)
                    if any(uint8(data{i}) == 0)
                        error('mestra:E32', ...
                              'a string in "%s" holds a NUL byte (E32)', key);
                    end
                    strSize = max(strSize, ...
                        numel(unicode2native(data{i}, 'UTF-8')));
                end
            end
            did = H5.createDataset(gid, key, dtype, dims, maxdims, chunk, ...
                                   [], strSize);
            H5.writeData(did, dtype, data, strSize);
            for axis = 1:numel(dims)
                scaleName = sprintf('mestra_%s_d%d', key, axis - 1);
                sc = H5.makeScale(gid, scaleName, dims(axis), dims(axis) == 0);
                H5DS.attach_scale(did, sc, axis - 1);
                H5D.close(sc);
            end
            H5D.close(did);
        end

        function out = toTagged(value)
        %toTagged  A dictionary in the tagged JSON form of section 30.
        %   Used by the test suite to compare against expected.json.
            if isa(value, 'containers.Map')
                v = struct();
                keys = value.keys();
                for i = 1:numel(keys)
                    v.(matlab.lang.makeValidName(keys{i})) = ...
                        mestra.internal.Codec.toTagged(value(keys{i}));
                end
                out = struct('t', 'dict', 'v', v);
            elseif isa(value, 'missing')
                out = struct('t', 'null');
            elseif isa(value, 'mestra.Array')
                if strcmp(value.dtype(), 'string')
                    out = struct('t', 'strings', 'shape', value.shape, ...
                                 'data', {value.elements()});
                else
                    el = value.elements();
                    if strcmp(value.dtype(), 'float64')
                        cells = cell(1, numel(el));
                        for i = 1:numel(el)
                            cells{i} = mestra.internal.Text.decimal(el(i));
                        end
                        data = cells;
                    elseif strcmp(value.dtype(), 'int8')
                        data = logical(el);
                    else
                        data = double(el);
                    end
                    dt = value.dtype();
                    if strcmp(dt, 'int8'), dt = 'bool'; end
                    out = struct('t', 'array', 'dtype', dt, ...
                                 'shape', value.shape, 'data', {data});
                end
            elseif ischar(value)
                out = struct('t', 'str', 'v', value);
            elseif islogical(value)
                out = struct('t', 'bool', 'v', logical(value));
            elseif isa(value, 'int64')
                out = struct('t', 'i64', 'v', double(value));
            else
                out = struct('t', 'f64', ...
                             'v', mestra.internal.Text.decimal(double(value)));
            end
        end
    end
end
