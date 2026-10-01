classdef H5
%H5  Low-level HDF5 helpers shared by the reader, writer and validator.
%
%   Everything here uses the low-level HDF5 interface (H5F, H5G, H5D,
%   H5A, H5S, H5T, H5P and H5DS).  The high-level interface cannot
%   write fixed-length null-padded UTF-8 strings, cannot attach
%   dimension scales, and cannot turn object time tracking off, so it
%   is not used anywhere in this package.
%
%   Axis order.  HDF5 numbers the axes of a dataset in C order, with
%   the fastest-varying axis last.  MATLAB is column major, so an
%   array read back from HDF5 arrives with its axes reversed: a file
%   array of shape (row, node, component) becomes a MATLAB array of
%   size (component, node, row).  Dimension vectors returned by
%   dsetInfo and taken by createDataset are in FILE order; array data
%   passed to and returned by readData and createDataset is in MATLAB
%   order.  Nothing here reverses silently.
%
%   See also mestra.read, mestra.write, mestra.validate.

    properties (Constant)
        % Attributes written by the HDF5 dimension scale machinery and
        % by the netCDF-C library.  A reader ignores these wherever
        % they appear (specification section 18).
        MACHINERY = {'CLASS', 'NAME', 'DIMENSION_LIST', ...
                     'REFERENCE_LIST', 'DIMENSION_LABELS', ...
                     '_Netcdf4Dimid', '_Netcdf4Coordinates', ...
                     '_nc3_strict', '_NCProperties'};

        % The sentence every dimension scale carries in its NAME
        % attribute (specification section 21).  The dimension's name
        % is the scale's link name, never this.
        SENTENCE = 'This is a netCDF dimension but not a netCDF variable.';
    end

    methods (Static)

        % ---------------------------------------------------- types

        function tid = strType(nbytes)
        %strType  A fixed-length UTF-8 string type with NUL padding.
            tid = H5T.copy('H5T_C_S1');
            H5T.set_size(tid, max(nbytes, 1));
            H5T.set_cset(tid, H5ML.get_constant_value('H5T_CSET_UTF8'));
            H5T.set_strpad(tid, H5ML.get_constant_value('H5T_STR_NULLPAD'));
        end

        function name = typeName(tid)
        %typeName  A short name for an HDF5 type, or '' when it is one
        %   this format never uses.
            cls = H5T.get_class(tid);
            if cls == H5ML.get_constant_value('H5T_STRING')
                if H5T.is_variable_str(tid)
                    name = 'vlstring';
                else
                    name = 'string';
                end
                return
            end
            sz = H5T.get_size(tid);
            if cls == H5ML.get_constant_value('H5T_FLOAT')
                switch sz
                    case 8, name = 'float64';
                    case 4, name = 'float32';
                    otherwise, name = '';
                end
            elseif cls == H5ML.get_constant_value('H5T_INTEGER')
                signed = H5T.get_sign(tid) ~= ...
                         H5ML.get_constant_value('H5T_SGN_NONE');
                switch sz
                    case 8, base = 'int64';
                    case 4, base = 'int32';
                    case 2, base = 'int16';
                    case 1, base = 'int8';
                    otherwise, base = '';
                end
                if isempty(base)
                    name = '';
                elseif signed
                    name = base;
                else
                    name = ['u' base];
                end
            else
                name = '';
            end
            % Section 19 stores every numeric dataset little-endian and
            % section 18 names little-endian types for every number, so
            % a big-endian one is named apart: comparing class and size
            % alone let both rules pass a file that breaks them.
            if ~isempty(name) && sz > 1 && H5T.get_order(tid) == ...
                    H5ML.get_constant_value('H5T_ORDER_BE')
                name = [name 'be'];
            end
        end

        function name = baseType(name)
        %baseType  A short type name without its big-endian mark, for
        %   the questions byte order does not change: which MATLAB
        %   class holds the value and how many bytes an element takes.
            if numel(name) > 2 && strcmp(name(end - 1:end), 'be') && ...
                    ~strcmp(name, 'vlstring')
                name = name(1:end - 2);
            end
        end

        function tid = typeId(name)
        %typeId  The file type for a short type name: little-endian,
        %   unless the name carries the big-endian mark typeName gives
        %   a type read from a file, which a copy keeps as it found it.
            base = mestra.internal.H5.baseType(name);
            if ~strcmp(base, name)
                tid = mestra.internal.H5.typeId(base);
                H5T.set_order(tid, H5ML.get_constant_value('H5T_ORDER_BE'));
                return
            end
            switch name
                case 'float64', tid = H5T.copy('H5T_IEEE_F64LE');
                case 'float32', tid = H5T.copy('H5T_IEEE_F32LE');
                case 'int64',   tid = H5T.copy('H5T_STD_I64LE');
                case 'int32',   tid = H5T.copy('H5T_STD_I32LE');
                case 'int8',    tid = H5T.copy('H5T_STD_I8LE');
                case 'uint8',   tid = H5T.copy('H5T_STD_U8LE');
                otherwise
                    error('mestra:internal', ...
                          'no file type for "%s"', name);
            end
        end

        function tid = memType(name)
        %memType  The native memory type for a short type name.
            switch mestra.internal.H5.baseType(name)
                case 'float64', tid = H5T.copy('H5T_NATIVE_DOUBLE');
                case 'float32', tid = H5T.copy('H5T_NATIVE_FLOAT');
                case 'int64',   tid = H5T.copy('H5T_NATIVE_LLONG');
                case 'int32',   tid = H5T.copy('H5T_NATIVE_INT');
                case 'int8',    tid = H5T.copy('H5T_NATIVE_SCHAR');
                case 'uint8',   tid = H5T.copy('H5T_NATIVE_UCHAR');
                otherwise
                    error('mestra:internal', ...
                          'no memory type for "%s"', name);
            end
        end

        function v = cast(name, data)
        %cast  Convert a MATLAB array to the class a type name names.
        %   float32 is forbidden in the public part of a file
        %   (section 19), but /private and any group this version does
        %   not know are copied through as they stand, and MATLAB
        %   refuses to write a double into an H5T_IEEE_F32LE dataset,
        %   so the name has to mean `single` here.
            switch mestra.internal.H5.baseType(name)
                case 'float64', v = double(data);
                case 'float32', v = single(data);
                case 'int64',  v = int64(data);
                case 'int32',  v = int32(data);
                case 'int8',   v = int8(data);
                case 'uint8',  v = uint8(data);
                otherwise,     v = data;
            end
        end

        % ----------------------------------------------- properties

        function plist = plist(class)
        %plist  A creation property list with object time tracking off.
        %   MATLAB does not wrap H5Pset_obj_track_times, so the
        %   underlying "object header flags" property is set directly:
        %   clearing bit 0x20 is exactly what that call does.
            plist = H5P.create(class);
            try
                flags = H5P.get(plist, 'object header flags');
                H5P.set(plist, 'object header flags', ...
                        bitand(uint8(flags), uint8(223)));
            catch
                % An HDF5 build without that property keeps the times;
                % nothing else in this package depends on it.
            end
        end

        function n = crtOrderTrackedIndexed()
        %crtOrderTrackedIndexed  H5P_CRT_ORDER_TRACKED | _INDEXED.
        %   The one creation property this format requires, on the
        %   dataset creation property list of every dimension scale
        %   (specification section 21).
            n = bitor(H5ML.get_constant_value('H5P_CRT_ORDER_TRACKED'), ...
                      H5ML.get_constant_value('H5P_CRT_ORDER_INDEXED'));
        end

        function order = attrCreationOrder(dcpl)
        %attrCreationOrder  The attribute creation order flags of a
        %   dataset creation property list, or 0 when the build cannot
        %   be asked.  This is what E42 is decided on.
            order = 0;
            try
                order = double(H5P.get_attr_creation_order(dcpl));
            catch
            end
        end

        % --------------------------------------------------- a pass

        function token = pass()
        %pass  Begin one reading pass over one file.
        %
        %   Within a pass an object's attribute names are listed once
        %   and every later question about them is answered from that
        %   list.  A reader asks nine of those questions of each
        %   dataset -- what it is, its role, its units, its source, its
        %   statistic, whether it has a DIMENSION_LIST -- and listing
        %   the names opens and closes one HDF5 identifier per
        %   attribute, so listing nine times over is most of what an
        %   open of a wide file costs.
        %
        %   The scope is the caller's, and it is what makes this safe:
        %   nothing writes an attribute inside a read, so the list
        %   cannot go stale while the token is alive.
        %
        %       closer = mestra.internal.H5.pass();  %#ok<NASGU>
        %
        %   The token clears the table when it goes out of scope.
        %   Passes nest: an inner one does not clear the outer one's
        %   table, which is what lets a read inside a validation share
        %   it.
            mestra.internal.H5.attrTable(+1);
            token = onCleanup(@() mestra.internal.H5.attrTable(-1));
        end

        function table = attrTable(delta)
        %attrTable  The pass-scoped table of attribute names, keyed by
        %   object address, or [] outside a pass.  pass() is how a
        %   caller opens and closes one; nothing else should call this.
            persistent depth store
            if isempty(depth), depth = 0; end
            if nargin >= 1
                if delta > 0
                    depth = depth + 1;
                    if depth == 1
                        store = containers.Map('KeyType', 'double', ...
                                               'ValueType', 'any');
                    end
                else
                    depth = max(0, depth - 1);
                    if depth == 0, store = []; end
                end
            end
            table = store;
        end

        % ---------------------------------------------- attributes

        function names = attrNames(oid)
        %attrNames  Every attribute name on an object, in name order.
        %   Inside a pass (see pass) the answer for one object is
        %   found once and kept.  The key is the object's address,
        %   checked against the file number it came from, so two files
        %   open at once cannot be confused for one another.
            info = H5O.get_info(oid);
            table = mestra.internal.H5.attrTable();
            % A containers.Map calls itself empty when it holds
            % nothing, so the question here is whether there is a
            % table at all.  A build that cannot be asked where an
            % object is cannot have one either.
            keeping = isa(table, 'containers.Map') && ...
                      isfield(info, 'addr') && isfield(info, 'fileno');
            if keeping
                key = double(info.addr);
                fileno = double(info.fileno);
                if table.isKey(key)
                    kept = table(key);
                    if kept.fileno == fileno
                        names = kept.names;
                        return
                    end
                end
            end
            n = double(info.num_attrs);
            names = cell(1, n);
            for i = 1:n
                aid = H5A.open_by_idx(oid, '.', 'H5_INDEX_NAME', ...
                                      'H5_ITER_INC', i - 1, ...
                                      'H5P_DEFAULT', 'H5P_DEFAULT');
                names{i} = H5A.get_name(aid);
                H5A.close(aid);
            end
            if keeping
                table(key) = struct('fileno', fileno, 'names', {names});
            end
        end

        function names = publicAttrNames(oid)
        %publicAttrNames  Attribute names minus the machinery names.
            names = mestra.internal.H5.attrNames(oid);
            names = names(~ismember(names, mestra.internal.H5.MACHINERY));
        end

        function tf = hasAttr(oid, name)
        %hasAttr  True when the object carries that attribute.
            tf = ismember(name, mestra.internal.H5.attrNames(oid));
        end

        function info = attrInfo(oid, name)
        %attrInfo  How an attribute is encoded.
        %   Fields: type (short type name), size (bytes, strings only),
        %   cset, strpad, scalar (true when the dataspace is scalar).
            aid = H5A.open(oid, name);
            tid = H5A.get_type(aid);
            sid = H5A.get_space(aid);
            info.type = mestra.internal.H5.typeName(tid);
            info.size = H5T.get_size(tid);
            info.cset = -1;
            info.strpad = -1;
            if strcmp(info.type, 'string')
                info.cset = H5T.get_cset(tid);
                info.strpad = H5T.get_strpad(tid);
            end
            info.scalar = H5S.get_simple_extent_type(sid) == ...
                          H5ML.get_constant_value('H5S_SCALAR');
            H5S.close(sid);
            H5T.close(tid);
            H5A.close(aid);
        end

        function [v, ok] = scalarAttr(oid, name)
        %scalarAttr  An attribute, only when its dataspace is scalar.
        %   Section 18 gives every attribute this format names a scalar
        %   dataspace.  A file that puts an array there would hand a
        %   caller an array where it expects one number, and every
        %   comparison downstream would then be an array too, so the
        %   value is refused rather than passed on.  `ok` is false when
        %   the attribute is missing, is not scalar, or cannot be read.
            v = [];
            ok = false;
            try
                if ~mestra.internal.H5.hasAttr(oid, name)
                    return
                end
                [v, ok] = mestra.internal.H5.readIfScalar(oid, name);
            catch
                v = [];
                ok = false;
            end
        end

        function detail = attrDetail(oid, name)
        %attrDetail  How an attribute is encoded and what it holds,
        %   from one open of it.  The fields attrInfo gives, plus
        %   `raw`, the value exactly as H5A.read returns it.
        %
        %   Asking the two questions separately opened the attribute,
        %   its datatype and its dataspace two or three times over,
        %   and both a reader and the validator ask both of every
        %   attribute of every object in the file.
        %
        %   Nothing is read until the dataspace has been seen to be
        %   the scalar section 18 requires, which is what keeps an
        %   attribute declaring a large array from being materialised
        %   by a reader that was only asking what it is (section 29).
        %   `raw` is empty when nothing was read.
            aid = H5A.open(oid, name);
            closer = onCleanup(@() H5A.close(aid)); %#ok<NASGU>
            tid = H5A.get_type(aid);
            sid = H5A.get_space(aid);
            detail.type = mestra.internal.H5.typeName(tid);
            detail.size = H5T.get_size(tid);
            detail.cset = -1;
            detail.strpad = -1;
            if strcmp(detail.type, 'string')
                detail.cset = H5T.get_cset(tid);
                detail.strpad = H5T.get_strpad(tid);
            end
            detail.scalar = H5S.get_simple_extent_type(sid) == ...
                            H5ML.get_constant_value('H5S_SCALAR');
            H5S.close(sid);
            H5T.close(tid);
            detail.raw = [];
            if detail.scalar
                detail.raw = H5A.read(aid);
            end
        end

        function [v, ok] = readIfScalar(oid, name)
        %readIfScalar  An attribute's value, only when its dataspace
        %   is the scalar section 18 requires, from one open.
            v = [];
            ok = false;
            detail = mestra.internal.H5.attrDetail(oid, name);
            if ~detail.scalar, return, end
            if any(strcmp(detail.type, {'string', 'vlstring'}))
                mestra.internal.H5.checkAsciiRead(detail.raw, 0, ...
                    sprintf('the attribute %s', name));
                v = mestra.internal.H5.toText(detail.raw);
            else
                v = detail.raw;
            end
            ok = true;
        end

        function bytes = stringBytes(raw, name)
        %stringBytes  A string attribute's stored bytes, from what
        %   H5A.read gave back.
            if iscell(raw), raw = raw{1}; end
            mestra.internal.H5.checkAsciiRead(raw, 0, ...
                sprintf('the attribute %s', name));
            bytes = uint8(raw(:)');
        end

        function v = readAttr(oid, name)
        %readAttr  An attribute as a MATLAB value.
        %   Strings arrive as char with the trailing NUL padding
        %   stripped, booleans as int8, integers as int64, floats as
        %   double.  Nothing here interprets int8 as logical: the
        %   caller knows which attributes are booleans.  This does not
        %   check the dataspace; scalarAttr is what a reader uses.
            aid = H5A.open(oid, name);
            raw = H5A.read(aid);
            tid = H5A.get_type(aid);
            isStr = H5T.get_class(tid) == ...
                    H5ML.get_constant_value('H5T_STRING');
            H5T.close(tid);
            H5A.close(aid);
            if isStr
                mestra.internal.H5.checkAsciiRead(raw, 0, ...
                    sprintf('the attribute %s', name));
                v = mestra.internal.H5.toText(raw);
            else
                v = raw;
            end
        end

        function s = toText(raw)
        %toText  Bytes from HDF5 to a char row vector.
            if iscell(raw)
                if isempty(raw)
                    s = '';
                else
                    s = mestra.internal.H5.toText(raw{1});
                end
                return
            end
            if isstring(raw)
                s = char(raw(1));
                return
            end
            bytes = uint8(raw(:)');
            last = find(bytes ~= 0, 1, 'last');
            if isempty(last)
                s = '';
            else
                s = native2unicode(bytes(1:last), 'UTF-8');
            end
        end

        function writeStrAttr(oid, name, value)
        %writeStrAttr  A fixed-length UTF-8 attribute, NUL padded.
            bytes = unicode2native(value, 'UTF-8');
            mestra.internal.H5.checkAsciiBytes(bytes, ...
                sprintf('the attribute %s', name));
            n = max(numel(bytes), 1);
            tid = mestra.internal.H5.strType(n);
            sid = H5S.create('H5S_SCALAR');
            aid = H5A.create(oid, name, tid, sid, 'H5P_DEFAULT');
            buf = char(zeros(1, n));
            buf(1:numel(bytes)) = char(double(bytes));
            H5A.write(aid, tid, buf);
            H5A.close(aid); H5S.close(sid); H5T.close(tid);
        end

        function writeRawStrAttr(oid, name, bytes, size, cset, strpad)
        %writeRawStrAttr  A fixed-length string attribute, byte exact.
        %   Used for the null sentinel of section 18, whose first byte
        %   is NUL and which therefore cannot go through unicode2native
        %   and back.  SIZE, CSET and STRPAD, when given, are the type
        %   the attribute had where it was copied from, so that a copy
        %   of a group this reader does not interpret keeps them.
            n = max(numel(bytes), 1);
            tid = mestra.internal.H5.strType(n);
            if nargin > 3
                n = max([size n]);
                H5T.set_size(tid, n);
                H5T.set_cset(tid, cset);
                H5T.set_strpad(tid, strpad);
            end
            sid = H5S.create('H5S_SCALAR');
            aid = H5A.create(oid, name, tid, sid, 'H5P_DEFAULT');
            buf = char(zeros(1, n));
            buf(1:numel(bytes)) = char(double(uint8(bytes)));
            H5A.write(aid, tid, buf);
            H5A.close(aid); H5S.close(sid); H5T.close(tid);
        end

        function bytes = readRawStrAttr(oid, name)
        %readRawStrAttr  A string attribute as its stored bytes.
            aid = H5A.open(oid, name);
            raw = H5A.read(aid);
            H5A.close(aid);
            bytes = mestra.internal.H5.stringBytes(raw, name);
        end

        function writeNumAttr(oid, name, value, typeName)
        %writeNumAttr  A scalar numeric or boolean attribute.
            ftid = mestra.internal.H5.typeId(typeName);
            mtid = mestra.internal.H5.memType(typeName);
            sid = H5S.create('H5S_SCALAR');
            aid = H5A.create(oid, name, ftid, sid, 'H5P_DEFAULT');
            H5A.write(aid, mtid, mestra.internal.H5.cast(typeName, value));
            H5A.close(aid); H5S.close(sid);
            H5T.close(ftid); H5T.close(mtid);
        end

        % ------------------------------------------- the ASCII wall

        function checkAsciiBytes(bytes, where)
        %checkAsciiBytes  Refuse to write a string MATLAB would mangle.
        %   MATLAB's HDF5 interface writes a fixed-length string from a
        %   char array and refuses any code above 127, so a non-ASCII
        %   UTF-8 string cannot be written through it at all.  Saying
        %   so here is better than the library's own message, and far
        %   better than writing something else.
            if any(uint8(bytes) > 127)
                error('mestra:matlabAscii', ...
                      ['%s holds a byte above 127. MATLAB''s HDF5 ' ...
                       'interface writes a fixed-length string only ' ...
                       'from ASCII characters, so this package cannot ' ...
                       'write non-ASCII UTF-8 strings. Keep names, ' ...
                       'category entries and string ids ASCII.'], where);
            end
        end

        function checkAsciiRead(raw, expected, where)
        %checkAsciiRead  Refuse to hand back a string MATLAB mangled.
        %   MATLAB decodes a fixed-length string as text before this
        %   package sees it: with a UTF-8 locale it turns the bytes
        %   into characters, which breaks the fixed blocking, and
        %   otherwise it replaces them.  Either way the bytes are lost,
        %   so an ASCII-only file is the only one that can be trusted.
            if ischar(raw) && any(double(raw(:)) > 127)
                error('mestra:matlabAscii', ...
                      ['%s holds a byte above 127. MATLAB''s HDF5 ' ...
                       'interface decodes a fixed-length string before ' ...
                       'this package sees it, so the stored bytes ' ...
                       'cannot be recovered and nothing is returned ' ...
                       'rather than something wrong.'], where);
            end
            if expected > 0 && numel(raw) ~= expected
                error('mestra:matlabAscii', ...
                      ['%s came back as %d characters where the file ' ...
                       'stores %d bytes; MATLAB re-encoded it.'], ...
                      where, numel(raw), expected);
            end
        end

        % ---------------------------------------------------- links

        function names = children(gid)
        %children  The link names of a group, sorted by UTF-8 bytes.
        %   A group whose links cannot be listed gives an empty list
        %   rather than an error, so that one damaged group does not
        %   end a walk over the rest of the file.
        %
        %   The names come from one H5L.iterate pass and not from one
        %   indexed lookup each.  An indexed lookup into a group whose
        %   links live in the heap rebuilds the group's whole link
        %   table to answer it, so listing a group of n links that way
        %   costs n squared; on a file with four thousand scalars that
        %   is most of the time an open takes.  H5_INDEX_NAME is the
        %   library's own byte order over the names, which is the
        %   order sortByBytes wants, so the sort below has nothing
        %   left to do on a well-formed group and is kept for the one
        %   that is not.
            names = {};
            try
                info = H5G.get_info(gid);
                n = double(info.nlinks);
                if n == 0, return, end
                [~, ~, names] = H5L.iterate(gid, 'H5_INDEX_NAME', ...
                    'H5_ITER_INC', 0, @mestra.internal.H5.collectLink, {});
                names = reshape(names, 1, []);
            catch
                names = mestra.internal.H5.childrenByIndex(gid);
            end
            names = mestra.internal.H5.sortByBytes(names);
        end

        function [status, names] = collectLink(gid, name, names) %#ok<INUSD>
        %collectLink  The H5L.iterate callback children uses.
            names{end + 1} = name;
            status = 0;
        end

        function names = childrenByIndex(gid)
        %childrenByIndex  children, one indexed lookup at a time, for a
        %   build or a group that will not iterate.
            names = {};
            try
                info = H5G.get_info(gid);
                n = double(info.nlinks);
                names = cell(1, n);
                for i = 1:n
                    names{i} = H5L.get_name_by_idx(gid, '.', ...
                        'H5_INDEX_NAME', 'H5_ITER_INC', i - 1, 'H5P_DEFAULT');
                end
            catch
                names = {};
            end
        end

        function out = listGroup(gid)
        %listGroup  Every link of a group, asked about once: its name,
        %   what it really is, and the address a hard link points at.
        %   A 1-by-n struct array with fields name, kind and address,
        %   where kind is what childType reports.
        %
        %   A walk wants all three of those for every child, and
        %   asking them separately is three lookups per link where one
        %   H5L.get_info answers two of them.
            names = mestra.internal.H5.children(gid);
            n = numel(names);
            out = struct('name', {}, 'kind', {}, 'address', {});
            if n == 0, return, end
            % Grown one element at a time this would copy the whole
            % array n times, which on a group of four thousand links
            % costs more than every HDF5 call in the loop.
            out(n).name = '';
            for i = 1:n
                link = mestra.internal.H5.linkInfo(gid, names{i});
                out(i).name = names{i};
                out(i).address = link.address;
                switch link.kind
                    case 'hard'
                        out(i).kind = mestra.internal.H5.objectKind(gid, ...
                                                                    names{i});
                    case 'unknown'
                        out(i).kind = 'unreadable';
                    otherwise
                        out(i).kind = link.kind;
                end
            end
        end

        function link = linkInfo(gid, name)
        %linkInfo  What sort of link a name is and where it points,
        %   without following it.  Fields: kind ('hard', 'soft',
        %   'external' or 'unknown') and address ([] for anything but
        %   a hard link).
        %
        %   H5L.get_info reads the link itself and never the object it
        %   points at, which is the only safe question to ask first: a
        %   soft link may dangle or loop, and an external link names
        %   another file, which this package never opens.  Two names
        %   with one address are one object, which is how a walk
        %   notices that a file's groups form a cycle.
            link = struct('kind', 'unknown', 'address', []);
            try
                info = H5L.get_info(gid, name, 'H5P_DEFAULT');
                switch double(info.type)
                    case 0
                        link.kind = 'hard';
                        if isfield(info, 'address')
                            link.address = double(info.address);
                        end
                    case 1, link.kind = 'soft';
                    case 64, link.kind = 'external';
                end
            catch
            end
        end

        function kind = linkKind(gid, name)
        %linkKind  What sort of link a name is, without following it.
            kind = mestra.internal.H5.linkInfo(gid, name).kind;
        end

        function addr = linkAddress(gid, name)
        %linkAddress  The address a hard link points at, or [].
            addr = mestra.internal.H5.linkInfo(gid, name).address;
        end

        function out = sortByBytes(names)
        %sortByBytes  Sort names by their UTF-8 bytes.
        %   This is the one ordering every language produces
        %   identically (specification section 22).
            if isempty(names), out = names; return, end
            n = numel(names);
            keys = cell(1, n);
            for i = 1:n
                keys{i} = double(unicode2native(names{i}, 'UTF-8'));
            end
            order = 1:n;
            for i = 2:n
                j = i;
                while j > 1 && mestra.internal.H5.byteLess( ...
                        keys{order(j)}, keys{order(j - 1)})
                    t = order(j); order(j) = order(j - 1); order(j - 1) = t;
                    j = j - 1;
                end
            end
            out = names(order);
        end

        function tf = byteLess(a, b)
        %byteLess  Lexicographic comparison of two byte vectors.
            m = min(numel(a), numel(b));
            for k = 1:m
                if a(k) ~= b(k)
                    tf = a(k) < b(k);
                    return
                end
            end
            tf = numel(a) < numel(b);
        end

        function t = childType(gid, name)
        %childType  What a name under a group really is.
        %   One of 'group', 'dataset', 'other', 'soft', 'external' or
        %   'unreadable'.  The link is inspected before the object is
        %   opened, so a soft link is never followed, an external link
        %   never opens another file, and an object that will not open
        %   is reported rather than thrown.
            t = mestra.internal.H5.linkKind(gid, name);
            switch t
                case 'soft'
                    return
                case 'external'
                    return
                case 'unknown'
                    t = 'unreadable';
                    return
            end
            t = mestra.internal.H5.objectKind(gid, name);
        end

        function t = objectKind(gid, name)
        %objectKind  'group', 'dataset', 'other' or 'unreadable' for a
        %   name already known to be a hard link.  childType is what a
        %   caller that does not know that yet asks.
            try
                oid = H5O.open(gid, name, 'H5P_DEFAULT');
            catch
                t = 'unreadable';
                return
            end
            kind = H5I.get_type(oid);
            H5O.close(oid);
            if kind == H5ML.get_constant_value('H5I_GROUP')
                t = 'group';
            elseif kind == H5ML.get_constant_value('H5I_DATASET')
                t = 'dataset';
            else
                t = 'other';
            end
        end

        function gid = openGroup(parent, name)
        %openGroup  Open a hard-linked group, or raise mestra:reader.
            if ~strcmp(mestra.internal.H5.childType(parent, name), 'group')
                error('mestra:reader', ...
                      '"%s" is not a group this reader will open', name);
            end
            try
                gid = H5G.open(parent, name);
            catch err
                error('mestra:reader', ...
                      'the group "%s" would not open: %s', name, ...
                      regexprep(strtrim(err.message), '\s+', ' '));
            end
        end

        function did = openDataset(parent, name)
        %openDataset  Open a hard-linked dataset, or raise mestra:reader.
            if ~strcmp(mestra.internal.H5.childType(parent, name), 'dataset')
                error('mestra:reader', ...
                      '"%s" is not a dataset this reader will open', name);
            end
            try
                did = H5D.open(parent, name);
            catch err
                error('mestra:reader', ...
                      'the dataset "%s" would not open: %s', name, ...
                      regexprep(strtrim(err.message), '\s+', ' '));
            end
        end

        function tf = exists(loc, path)
        %exists  True when a path exists under a location.
            parts = strsplit(path, '/');
            parts = parts(~cellfun(@isempty, parts));
            tf = true;
            here = '';
            for i = 1:numel(parts)
                if isempty(here)
                    here = parts{i};
                else
                    here = [here '/' parts{i}]; %#ok<AGROW>
                end
                if ~H5L.exists(loc, here, 'H5P_DEFAULT')
                    tf = false;
                    return
                end
            end
        end

        % -------------------------------------------------- datasets

        function info = dsetInfo(did)
        %dsetInfo  Shape, storage and type of a dataset, in FILE order.
        %   Fields: type, dims, maxdims (-1 for unlimited), chunk ([]
        %   when contiguous), filters (an n-by-2 matrix of filter id
        %   and first parameter), attrOrder (the attribute creation
        %   order flags E42 is decided on), strSize, cset, strpad,
        %   isScale.
            sid = H5D.get_space(did);
            [~, dims, maxdims] = H5S.get_simple_extent_dims(sid);
            H5S.close(sid);
            info.dims = double(dims);
            info.maxdims = double(maxdims);
            tid = H5D.get_type(did);
            info.type = mestra.internal.H5.typeName(tid);
            info.strSize = H5T.get_size(tid);
            info.cset = -1; info.strpad = -1;
            if strcmp(info.type, 'string')
                info.cset = H5T.get_cset(tid);
                info.strpad = H5T.get_strpad(tid);
            end
            H5T.close(tid);
            info.elements = prod(max(info.dims, 0));
            info.chunk = [];
            info.filters = zeros(0, 2);
            info.attrOrder = 0;
            % A creation property list is the file's word for how the
            % data is stored, including filters this build may not have
            % and client data longer than any reader expects. Every
            % question is asked separately so that one unanswerable one
            % leaves the rest of the description intact.
            try
                dcpl = H5D.get_create_plist(did);
            catch
                dcpl = [];
            end
            if ~isempty(dcpl)
                try
                    if H5P.get_layout(dcpl) == ...
                       H5ML.get_constant_value('H5D_CHUNKED')
                        [~, chunk] = H5P.get_chunk(dcpl);
                        info.chunk = double(chunk);
                    end
                catch
                end
                nf = 0;
                try
                    nf = H5P.get_nfilters(dcpl);
                catch
                end
                rows = zeros(0, 2);
                for i = 1:nf
                    id = -1; p = 0;
                    try
                        [id, ~, cd] = H5P.get_filter(dcpl, i - 1);
                        if ~isempty(cd), p = double(cd(1)); end
                    catch
                        id = -1;
                    end
                    rows(end + 1, :) = [double(id) p]; %#ok<AGROW>
                end
                info.filters = rows;
                info.attrOrder = ...
                    mestra.internal.H5.attrCreationOrder(dcpl);
                try
                    H5P.close(dcpl);
                catch
                end
            end
            info.isScale = false;
            try
                info.isScale = H5DS.is_scale(did) > 0;
            catch
            end
        end

        function map = scaleMap(fid)
        %scaleMap  Every dimension scale in a file, by object address.
        %   Specification section 21.  Asking the library
        %   for the path of a scale attached to an axis makes it search
        %   the group hierarchy for a name that leads there, and on a
        %   file with a deep chain of groups that search runs off the
        %   stack and takes the process with it.  A reader must build
        %   its own map during its own bounded walk and resolve
        %   attached scales through that.
        %
        %   The key is the object's address, which is exactly what the
        %   eight bytes of an H5R_OBJECT reference hold, so resolving a
        %   scale needs no dereference either.  The value carries, for
        %   each scale:
        %
        %     name       the link name, which is the dimension's name
        %     length     because the chunk default of section 23 is
        %                judged against the dimension's length and not
        %                the dataset's own extent (section 23)
        %     hasName    whether it carries a NAME attribute (E25)
        %     path       its HDF5 path, so that a finding can name it
        %                and so that section 14 can leave /private out
        %     unlimited  whether its own extent is unlimited (E43)
        %     order      the attribute creation order flags of its
        %                creation property list (E42)
        %
        %   The last three cost nothing here and would otherwise need a
        %   second walk of the file: this is the one pass that already
        %   sees every scale there is.
            map = containers.Map('KeyType', 'double', 'ValueType', 'any');
            limits = mestra.internal.Limits.get();
            budget = limits.maxObjects;
            pending = {'/'};
            depths = 0;
            while ~isempty(pending) && budget > 0
                here = pending{1};
                depth = depths(1);
                pending(1) = [];
                depths(1) = [];
                if depth > limits.maxDepth, continue, end
                try
                    gid = H5G.open(fid, here);
                catch
                    continue
                end
                for entry = mestra.internal.H5.listGroup(gid)
                    budget = budget - 1;
                    if budget <= 0, break, end
                    if strcmp(here, '/')
                        path = ['/' entry.name];
                    else
                        path = [here '/' entry.name];
                    end
                    if strcmp(entry.kind, 'group')
                        pending{end + 1} = path; %#ok<AGROW>
                        depths(end + 1) = depth + 1; %#ok<AGROW>
                    elseif strcmp(entry.kind, 'dataset')
                        address = entry.address;
                        if isempty(address) || map.isKey(address), continue, end
                        try
                            did = H5D.open(gid, entry.name);
                        catch
                            continue
                        end
                        try
                            % The question that settles it first, and
                            % the reading only for the few datasets
                            % that answer yes: most of the datasets in
                            % a wide file are not scales, and this walk
                            % is what a lazy read of one row range pays
                            % before it can name the slot's axes.
                            if H5DS.is_scale(did) > 0
                                names = mestra.internal.H5.attrNames(did);
                                info = mestra.internal.H5.dsetInfo(did);
                                length_ = 0;
                                unlimited = false;
                                if ~isempty(info.dims)
                                    length_ = info.dims(1);
                                    unlimited = info.maxdims(1) < 0;
                                end
                                map(address) = struct( ...
                                    'name', entry.name, 'length', length_, ...
                                    'hasName', any(strcmp(names, 'NAME')), ...
                                    'path', path, 'unlimited', unlimited, ...
                                    'order', info.attrOrder);
                            end
                        catch
                        end
                        H5D.close(did);
                    end
                    % Anything else is a link this reader never
                    % follows, or an object it cannot open; the rules
                    % for those are E40 and E41 and belong to the
                    % caller's own walk, not to this map.
                end
                H5G.close(gid);
            end
        end

        function found = scaleNames(did, axis, map)
        %scaleNames  The scales attached to one axis, as link names.
        %   `axis` is zero based and in FILE order.  `map` comes from
        %   scaleMap; without it nothing can be resolved and the axis
        %   reads as unattached, which is E25.  The result is a struct
        %   array shaped like a scaleMap record, one per scale.
        %
        %   The name comes from the link and never from the NAME
        %   attribute, which holds the same sentence in every scale in
        %   the file, and never from a path lookup, which section 21
        %   forbids.
            found = mestra.internal.H5.noScales();
            if nargin < 3 || isempty(map), return, end
            if ~mestra.internal.H5.hasAttr(did, 'DIMENSION_LIST')
                return
            end
            try
                aid = H5A.open(did, 'DIMENSION_LIST');
                list = H5A.read(aid);
                H5A.close(aid);
            catch
                return
            end
            if ~iscell(list) || axis + 1 > numel(list), return, end
            refs = uint8(list{axis + 1}(:));
            for k = 1:8:numel(refs) - 7
                address = double(typecast(refs(k:k + 7)', 'uint64'));
                if map.isKey(address)
                    found(end + 1) = map(address); %#ok<AGROW>
                else
                    found(end + 1) = ...
                        mestra.internal.H5.unresolvedScale(); %#ok<AGROW>
                end
            end
        end

        function s = noScales()
        %noScales  An empty scale list of the shape scaleMap fills.
            s = struct('name', {}, 'length', {}, 'hasName', {}, ...
                       'path', {}, 'unlimited', {}, 'order', {});
        end

        function s = unresolvedScale()
        %unresolvedScale  What an axis attached to something this
        %   reader could not resolve to a dimension looks like.  A
        %   length of -1 is what says so; E25 is the rule.
            s = struct('name', '', 'length', -1, 'hasName', false, ...
                       'path', '', 'unlimited', false, 'order', 0);
        end

        function n = numScales(did, axis)
        %numScales  How many scales are attached to one axis.
            n = double(H5DS.get_num_scales(did, axis));
        end

        function data = readData(did, info)
        %readData  A whole dataset, in MATLAB axis order.
        %   Strings come back as a cell array of char row vectors.
            if nargin < 2
                info = mestra.internal.H5.dsetInfo(did);
            end
            count = prod(max(info.dims, 0));
            mestra.internal.Limits.checkElements(count, 'this dataset');
            if strcmp(info.type, 'string')
                mestra.internal.H5.checkStringSize(info.strSize);
                raw = mestra.internal.H5.guardedRead(did);
                mestra.internal.H5.checkAsciiRead(raw, ...
                    info.strSize * count, 'a fixed-length string dataset');
                data = mestra.internal.H5.splitStrings(raw, count);
                return
            end
            if any(info.dims == 0)
                data = zeros([fliplr(info.dims) 1 1]);
                data = mestra.internal.H5.cast(info.type, data);
                return
            end
            data = mestra.internal.H5.guardedRead(did);
        end

        function raw = guardedRead(did, varargin)
        %guardedRead  H5D.read, with any failure named as this reader's.
        %   A filter this build cannot run, a checksum that does not
        %   match, a shape MATLAB refuses to allocate: all of them are
        %   the file's doing, not the caller's mistake, so they carry
        %   this package's own identifier.
            try
                raw = H5D.read(did, varargin{:});
            catch err
                if strcmp(err.identifier, 'mestra:matlabAscii')
                    rethrow(err);
                end
                error('mestra:E41', ...
                      'the data would not read: %s', ...
                      regexprep(strtrim(err.message), '\s+', ' '));
            end
        end

        function checkStringSize(size)
        %checkStringSize  Refuse an absurd fixed-length string width.
            limit = mestra.internal.Limits.get('maxStringSize');
            if size > limit
                error('mestra:E41', ...
                      ['a fixed-length string of %d bytes an element is ' ...
                       'past the %d this reader accepts'], size, limit);
            end
        end

        function out = splitStrings(raw, n)
        %splitStrings  A fixed-length string buffer to a cell array.
            out = cell(1, n);
            if n == 0, return, end
            if iscell(raw)
                for i = 1:n
                    out{i} = mestra.internal.H5.toText(raw{i});
                end
                return
            end
            m = numel(raw) / n;
            buf = reshape(raw, m, n);
            for i = 1:n
                out{i} = mestra.internal.H5.toText(buf(:, i)');
            end
        end

        function out = decodeStrings(did, info)
        %decodeStrings  A fixed-length string dataset, and how it fared.
        %   MATLAB decodes a fixed-length string to text before this
        %   package sees it, so the stored bytes are not always
        %   recoverable.  The verdict says which of three things
        %   happened:
        %
        %     'bytes'     the characters came back one per stored byte,
        %                 so out.bytes is exactly what the file holds
        %                 and every rule about those bytes is decidable
        %     'replaced'  the count is right but characters above 255
        %                 came back, which is MATLAB substituting one
        %                 replacement character per byte it could not
        %                 decode: the bytes were not valid UTF-8
        %     'decoded'   the count changed, so MATLAB decoded valid
        %                 multi-byte UTF-8 and the bytes are gone
            out.verdict = 'bytes';
            out.bytes = zeros(info.strSize, 0, 'uint8');
            n = prod(max(info.dims, 0));
            mestra.internal.H5.checkStringSize(info.strSize);
            mestra.internal.Limits.checkElements( ...
                n * max(info.strSize, 1), 'this string dataset');
            if n == 0, return, end
            buf = mestra.internal.H5.guardedRead(did);
            if iscell(buf)
                out.bytes = zeros(info.strSize, n, 'uint8');
                for i = 1:n
                    b = uint8(buf{i});
                    m = min(numel(b), info.strSize);
                    out.bytes(1:m, i) = b(1:m)';
                end
                return
            end
            codes = double(buf(:));
            if numel(codes) ~= info.strSize * n
                out.verdict = 'decoded';
                return
            end
            if any(codes > 255)
                out.verdict = 'replaced';
                return
            end
            out.bytes = reshape(uint8(codes), info.strSize, n);
        end

        function raw = readRawStrings(did, info)
        %readRawStrings  A fixed-length string dataset as its bytes.
        %   Returns a size-by-count uint8 matrix, padding and all, so
        %   that a validator can see a NUL where it should not be.
            out = mestra.internal.H5.decodeStrings(did, info);
            if ~strcmp(out.verdict, 'bytes')
                error('mestra:matlabAscii', ...
                      ['a fixed-length string dataset came back %s. ' ...
                       'MATLAB''s HDF5 interface decodes a fixed-length ' ...
                       'string before this package sees it, so the ' ...
                       'stored bytes cannot always be recovered.'], ...
                      out.verdict);
            end
            raw = out.bytes;
        end

        function data = readRows(did, info, first, count)
        %readRows  Rows [first, first+count) of a dataset.
        %   `first` is zero based.  Only the chunks that hold those
        %   rows are touched, which is what section 29 asks a reader
        %   to be able to do.  The result is in MATLAB axis order.
            block = info.dims;
            block(1) = count;
            mestra.internal.Limits.checkElements(prod(max(block, 0)), ...
                                                 'this row range');
            if strcmp(info.type, 'string')
                mestra.internal.H5.checkStringSize(info.strSize);
            end
            fileSid = H5D.get_space(did);
            start = zeros(1, numel(info.dims));
            start(1) = first;
            H5S.select_hyperslab(fileSid, 'H5S_SELECT_SET', start, [], ...
                                 ones(1, numel(block)), block);
            memSid = H5S.create_simple(numel(block), block, block);
            if strcmp(info.type, 'string')
                tid = H5D.get_type(did);
                raw = mestra.internal.H5.guardedRead(did, tid, memSid, ...
                                                     fileSid, 'H5P_DEFAULT');
                H5T.close(tid);
                mestra.internal.H5.checkAsciiRead(raw, ...
                    info.strSize * prod(block), ...
                    'a fixed-length string dataset');
                data = mestra.internal.H5.splitStrings(raw, prod(block));
            else
                mtid = mestra.internal.H5.memType(info.type);
                data = mestra.internal.H5.guardedRead(did, mtid, memSid, ...
                                                     fileSid, 'H5P_DEFAULT');
                H5T.close(mtid);
                data = reshape(data, [fliplr(block) 1 1]);
            end
            H5S.close(memSid); H5S.close(fileSid);
        end

        % --------------------------------------------------- writing

        function did = createDataset(gid, name, typeName, dims, maxdims, ...
                                     chunk, filters, strSize, attrOrder)
        %createDataset  Create one dataset, in FILE axis order.
        %   `maxdims` uses -1 for an unlimited extent, `chunk` is []
        %   for contiguous storage, `filters` is an n-by-2 matrix of
        %   filter id and first parameter, `strSize` is the byte size
        %   of a fixed-length string type, and `attrOrder` is the
        %   attribute creation order flags of section 21, which only a
        %   dimension scale needs and which defaults to the library's
        %   own (none).
            if nargin < 7 || isempty(filters), filters = zeros(0, 2); end
            if nargin < 8, strSize = 0; end
            if nargin < 9 || isempty(attrOrder), attrOrder = 0; end
            rank = numel(dims);
            hmax = maxdims;
            hmax(maxdims < 0) = H5ML.get_constant_value('H5S_UNLIMITED');
            sid = H5S.create_simple(rank, dims, hmax);
            dcpl = mestra.internal.H5.plist('H5P_DATASET_CREATE');
            if ~isempty(chunk)
                H5P.set_chunk(dcpl, chunk);
            end
            if attrOrder ~= 0
                H5P.set_attr_creation_order(dcpl, attrOrder);
            end
            for i = 1:size(filters, 1)
                switch filters(i, 1)
                    case 1, H5P.set_deflate(dcpl, filters(i, 2));
                    case 2, H5P.set_shuffle(dcpl);
                    case 3, H5P.set_fletcher32(dcpl);
                    otherwise
                        error('mestra:E29', ...
                              'filter %d is not allowed (E29)', filters(i, 1));
                end
            end
            if strcmp(typeName, 'string')
                tid = mestra.internal.H5.strType(strSize);
            else
                tid = mestra.internal.H5.typeId(typeName);
            end
            did = H5D.create(gid, name, tid, sid, 'H5P_DEFAULT', dcpl, ...
                             'H5P_DEFAULT');
            H5T.close(tid); H5P.close(dcpl); H5S.close(sid);
        end

        function writeData(did, typeName, data, strSize)
        %writeData  Fill a dataset.  `data` is in MATLAB axis order.
            if strcmp(typeName, 'string')
                tid = mestra.internal.H5.strType(strSize);
                n = numel(data);
                buf = char(zeros(max(strSize, 1), max(n, 1)));
                for i = 1:n
                    bytes = unicode2native(data{i}, 'UTF-8');
                    mestra.internal.H5.checkAsciiBytes(bytes, ...
                        'a fixed-length string dataset');
                    buf(1:numel(bytes), i) = char(double(bytes))';
                end
                if n > 0
                    H5D.write(did, tid, 'H5S_ALL', 'H5S_ALL', ...
                              'H5P_DEFAULT', buf(:, 1:n));
                end
                H5T.close(tid);
                return
            end
            if isempty(data), return, end
            mtid = mestra.internal.H5.memType(typeName);
            H5D.write(did, mtid, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', ...
                      mestra.internal.H5.cast(typeName, data));
            H5T.close(mtid);
        end

        function did = makeScale(gid, name, len, unlimited)
        %makeScale  A dimension scale, written as netCDF-C writes one.
        %   One-dimensional, big-endian float32, no value ever stored,
        %   CLASS and NAME as section 21 requires.  An unlimited scale
        %   is chunked with chunk length one; a fixed one is chunked
        %   over its whole length, which is what the corpus carries.
        %
        %   The creation property list is the whole point here.
        %   Attribute creation order tracked and indexed gives the
        %   scale a version 2 object header, which is what lets its
        %   REFERENCE_LIST live in the file's heap instead of in an
        %   object header message that may not exceed 64 KiB.  Without
        %   it a scale takes at most 4085 attachments and the 4086th
        %   fails after it has already deleted the attribute it was
        %   extending.  Object time tracking has to go off in the same
        %   list, which plist already does, because a version 2 header
        %   records four timestamps unless it is told not to and a file
        %   that records when it was written is not byte reproducible.
        %   No other object's property list is touched, so the
        %   superblock and every non-scale object are as they were.
            if unlimited
                maxd = H5ML.get_constant_value('H5S_UNLIMITED');
                chunk = 1;
            else
                maxd = len;
                chunk = max(len, 1);
            end
            sid = H5S.create_simple(1, len, maxd);
            dcpl = mestra.internal.H5.plist('H5P_DATASET_CREATE');
            H5P.set_chunk(dcpl, chunk);
            H5P.set_attr_creation_order(dcpl, ...
                mestra.internal.H5.crtOrderTrackedIndexed());
            did = H5D.create(gid, name, 'H5T_IEEE_F32BE', sid, ...
                             'H5P_DEFAULT', dcpl, 'H5P_DEFAULT');
            H5P.close(dcpl); H5S.close(sid);
            H5DS.set_scale(did, sprintf('%s%10d', ...
                                        mestra.internal.H5.SENTENCE, len));
        end

        % --------------------------------------- whole subtree copies

        function tree = captureTree(gid, map, depth, seen)
        %captureTree  Everything under a group, kept as it stands.
        %   Used for /notes, for /private, which section 29 forbids a
        %   reader to interpret, and for any group this version does
        %   not know (section 28).  A round trip through this package
        %   therefore loses none of them.
        %
        %   The walk is bounded in three ways, because the shape of the
        %   tree is the file's choice and not this reader's: it stops
        %   at maxDepth levels, it stops at a group it has already
        %   visited in this walk, which is how a cycle of hard links
        %   ends, and it never follows a soft or an external link.
        %   Each of those records a note in `tree.stopped` and returns
        %   what it has, rather than descending until the stack gives
        %   out.
            H5 = mestra.internal.H5;
            if nargin < 2, map = []; end
            if nargin < 3, depth = 0; end
            if nargin < 4, seen = []; end
            tree.stopped = {};
            tree.attrs = H5.captureAttrs([]);
            tree.datasets = struct('name', {}, 'info', {}, 'data', {}, ...
                                   'scales', {}, 'label', {}, 'attrs', {}, ...
                                   'scalePaths', {});
            tree.groups = struct('name', {}, 'tree', {});
            if depth > mestra.internal.Limits.get('maxDepth')
                tree.stopped{end + 1} = sprintf( ...
                    'nesting past %d levels was not followed', ...
                    mestra.internal.Limits.get('maxDepth'));
                return
            end
            tree.attrs = H5.captureAttrs(gid);
            for name = H5.children(gid)
                kind = H5.childType(gid, name{1});
                switch kind
                    case 'group'
                        address = H5.linkAddress(gid, name{1});
                        if ~isempty(address) && any(seen == address)
                            tree.stopped{end + 1} = sprintf( ...
                                ['"%s" is another name for a group ' ...
                                 'already visited; it was not ' ...
                                 'followed'], name{1}); %#ok<AGROW>
                            continue
                        end
                        sub = H5G.open(gid, name{1});
                        subTree = H5.captureTree(sub, map, depth + 1, ...
                                                 [seen address]);
                        H5G.close(sub);
                        tree.groups(end + 1) = struct('name', name{1}, ...
                                                      'tree', subTree);
                        for i = 1:numel(subTree.stopped)
                            tree.stopped{end + 1} = ...
                                [name{1} '/' subTree.stopped{i}]; %#ok<AGROW>
                        end
                    case 'dataset'
                        did = H5D.open(gid, name{1});
                        try
                            rec = H5.captureDataset(did, name{1}, map);
                            tree.datasets(end + 1) = rec;
                        catch err
                            tree.stopped{end + 1} = sprintf( ...
                                '"%s" would not read: %s', name{1}, ...
                                regexprep(strtrim(err.message), ...
                                          '\s+', ' ')); %#ok<AGROW>
                        end
                        H5D.close(did);
                    otherwise
                        tree.stopped{end + 1} = sprintf( ...
                            '"%s" is a %s and was not followed', ...
                            name{1}, kind); %#ok<AGROW>
                end
            end
        end

        function rec = captureDataset(did, name, map)
        %captureDataset  One dataset of a captured subtree.
            H5 = mestra.internal.H5;
            if nargin < 3, map = []; end
            info = H5.dsetInfo(did);
            rec.name = name;
            rec.info = info;
            rec.data = [];
            if ~info.isScale
                rec.data = H5.readData(did, info);
            end
            rec.scales = cell(1, numel(info.dims));
            scalePaths = cell(1, numel(info.dims));
            for axis = 1:numel(info.dims)
                found = H5.scaleNames(did, axis - 1, map);
                if isempty(found)
                    rec.scales{axis} = '';
                    scalePaths{axis} = '';
                else
                    rec.scales{axis} = found(1).name;
                    scalePaths{axis} = found(1).path;
                end
            end
            rec.label = '';
            rec.attrs = H5.captureAttrs(did);
            % The scale's path in the file, so that one outside the
            % copied group -- a support's node, the file's row -- is
            % attached again in the rewrite, which writes it too.
            rec.scalePaths = scalePaths;
            if info.isScale && H5.hasAttr(did, 'NAME')
                label = H5.readAttr(did, 'NAME');
                if ischar(label), rec.label = label; end
            end
        end

        function attrs = captureAttrs(oid)
        %captureAttrs  An object's own attributes, kept as they stand:
        %   a string with its stored bytes, its size, character set and
        %   padding, a number with its type.  With OID empty, the empty
        %   list of the right shape.
            attrs = struct('name', {}, 'type', {}, 'bytes', {}, ...
                           'value', {}, 'size', {}, 'cset', {}, ...
                           'strpad', {});
            if isempty(oid), return, end
            H5 = mestra.internal.H5;
            for name = H5.publicAttrNames(oid)
                info = H5.attrInfo(oid, name{1});
                rec.name = name{1};
                rec.type = info.type;
                rec.bytes = [];
                rec.value = [];
                rec.size = [];
                rec.cset = [];
                rec.strpad = [];
                if strcmp(info.type, 'string')
                    rec.bytes = H5.readRawStrAttr(oid, name{1});
                    rec.size = info.size;
                    rec.cset = info.cset;
                    rec.strpad = info.strpad;
                elseif ~isempty(info.type)
                    rec.value = H5.readAttr(oid, name{1});
                end
                attrs(end + 1) = rec; %#ok<AGROW>
            end
        end

        function replayAttrs(oid, attrs)
        %replayAttrs  Write back what captureAttrs took.
            H5 = mestra.internal.H5;
            for i = 1:numel(attrs)
                a = attrs(i);
                if strcmp(a.type, 'string') && ~isempty(a.size)
                    H5.writeRawStrAttr(oid, a.name, a.bytes, a.size, ...
                                       a.cset, a.strpad);
                elseif strcmp(a.type, 'string')
                    H5.writeRawStrAttr(oid, a.name, a.bytes);
                elseif ~isempty(a.type)
                    H5.writeNumAttr(oid, a.name, a.value, a.type);
                end
            end
        end

        function replayTree(gid, tree)
        %replayTree  Write back a subtree captureTree took.
            H5 = mestra.internal.H5;
            H5.replayAttrs(gid, tree.attrs);
            made = containers.Map('KeyType', 'char', 'ValueType', 'any');
            for i = 1:numel(tree.datasets)
                ds = tree.datasets(i);
                order = 0;
                if isfield(ds.info, 'attrOrder'), order = ds.info.attrOrder; end
                if ds.info.isScale
                    % A scale is a scale wherever it is kept, so one
                    % replayed into /notes or /private is created with
                    % the property list section 21 gives it, whatever
                    % the file it came from used.
                    order = mestra.internal.H5.crtOrderTrackedIndexed();
                end
                did = H5.createDataset(gid, ds.name, ds.info.type, ...
                    ds.info.dims, ds.info.maxdims, ds.info.chunk, ...
                    ds.info.filters, ds.info.strSize, order);
                if ds.info.isScale
                    % A scale whose NAME attribute was missing is
                    % written back with the sentence section 21 gives
                    % it, because H5DS.set_scale needs some name and
                    % nothing reads this one anyway.
                    label = ds.label;
                    if isempty(label)
                        label = sprintf('%s%10d', ...
                            mestra.internal.H5.SENTENCE, ...
                            max(ds.info.dims(1), 0));
                    end
                    H5DS.set_scale(did, label);
                else
                    H5.writeData(did, ds.info.type, ds.data, ds.info.strSize);
                end
                H5.replayAttrs(did, ds.attrs);
                made(ds.name) = did;
            end
            for i = 1:numel(tree.groups)
                gcpl = H5.plist('H5P_GROUP_CREATE');
                sub = H5G.create(gid, tree.groups(i).name, 'H5P_DEFAULT', ...
                                 gcpl, 'H5P_DEFAULT');
                H5P.close(gcpl);
                H5.replayTree(sub, tree.groups(i).tree);
                H5G.close(sub);
            end
            % Attached after the subgroups exist, so that a scale kept
            % in one of them is there to attach.
            fid = H5I.get_file_id(gid);
            for i = 1:numel(tree.datasets)
                ds = tree.datasets(i);
                for axis = 1:numel(ds.scales)
                    if ~isempty(ds.scales{axis}) && made.isKey(ds.scales{axis})
                        H5DS.attach_scale(made(ds.name), ...
                                          made(ds.scales{axis}), axis - 1);
                    elseif isfield(ds, 'scalePaths') && ...
                            numel(ds.scalePaths) >= axis && ...
                            ~isempty(ds.scalePaths{axis}) && ...
                            H5.exists(fid, ds.scalePaths{axis})
                        sd = H5D.open(fid, ds.scalePaths{axis});
                        if H5DS.is_scale(sd)
                            H5DS.attach_scale(made(ds.name), sd, axis - 1);
                        end
                        H5D.close(sd);
                    end
                end
            end
            H5F.close(fid);
            for key = made.keys()
                H5D.close(made(key{1}));
            end
        end

    end
end
