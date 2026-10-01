classdef Writer
%Writer  Turn a mestra.Dataset into a conforming file.
%
%   mestra.write is the public way in.  Everything is written through
%   the low-level HDF5 interface: fixed-length null-padded UTF-8
%   strings, dimension scales attached and named as section 21
%   requires, chunking and filters per section 23, an unlimited `row`,
%   and object time tracking off on every object.
%
%   See also mestra.write.

    methods (Static)

        function save(d, path)
        %save  Write a dataset to a file, replacing what is there.
            H5 = mestra.internal.H5;
            if exist(path, 'file') == 2
                delete(path);
            end
            fcpl = H5.plist('H5P_FILE_CREATE');
            fid = H5F.create(path, 'H5F_ACC_TRUNC', fcpl, 'H5P_DEFAULT');
            H5P.close(fcpl);
            closeFile = onCleanup(@() H5F.close(fid));
            root = H5G.open(fid, '/');
            closeRoot = onCleanup(@() H5G.close(root)); %#ok<NASGU>

            % ------------------------------------- root attributes
            H5.writeStrAttr(root, 'created', d.created);
            H5.writeStrAttr(root, 'format', d.format);
            H5.writeStrAttr(root, 'writer', d.writer);
            H5.writeNumAttr(root, 'aligned', int8(d.aligned ~= 0), 'int8');
            if ~isempty(d.generalisationGroup)
                H5.writeStrAttr(root, 'generalisation_group', ...
                                d.generalisationGroup);
            end
            for name = H5.sortByBytes(d.unknownAttrs.keys())
                a = d.unknownAttrs(name{1});
                if strcmp(a.type, 'string')
                    H5.writeRawStrAttr(root, name{1}, a.bytes);
                else
                    H5.writeNumAttr(root, name{1}, a.value, a.type);
                end
            end

            % ---------------------------------------- root scales
            scales = containers.Map('KeyType', 'char', 'ValueType', 'any');
            scales('row') = H5.makeScale(root, 'row', d.nRows, true);
            plan = mestra.internal.Writer.scalePlan(d);
            for name = H5.sortByBytes(plan.keys())
                scales(name{1}) = H5.makeScale(root, name{1}, ...
                                               plan(name{1}), false);
            end

            wanted = unique([d.groupsPresent ...
                             mestra.internal.Writer.neededGroups(d)]);

            % ------------------------------------------ categories
            if ismember('categories', wanted)
                g = mestra.internal.Writer.group(root, 'categories');
                for i = 1:numel(d.categories)
                    c = d.categories(i);
                    size_ = c.strSize;
                    if isempty(size_)
                        size_ = mestra.internal.Writer.stringSize(c.entries);
                    end
                    did = H5.createDataset(g, c.name, 'string', ...
                        numel(c.entries), numel(c.entries), [], [], size_);
                    H5.writeData(did, 'string', c.entries, size_);
                    mestra.internal.Writer.attach(did, ...
                        {['category_' c.name]}, scales, []);
                    H5D.close(did);
                end
                H5G.close(g);
            end

            % ------------------------------------------------ keys
            if ismember('keys', wanted)
                g = mestra.internal.Writer.group(root, 'keys');
                for i = 1:numel(d.keys)
                    mestra.internal.Writer.writeKey(g, d.keys(i), scales);
                end
                H5G.close(g);
            end

            % --------------------------------------------- scalars
            if ismember('scalars', wanted)
                g = mestra.internal.Writer.group(root, 'scalars');
                for i = 1:numel(d.scalars)
                    mestra.internal.Writer.writeScalar(g, d.scalars(i), scales);
                end
                H5G.close(g);
            end

            % ----------------------------------------- row_support
            % Section 22: a file that declares more than one support
            % has /row_support, and with no rows it is empty, not
            % absent.
            if ~isempty(d.rowSupport) || numel(d.supports) > 1
                chunk = mestra.internal.Writer.rowChunk(4, [], ...
                                                        numel(d.rowSupport));
                did = H5.createDataset(root, 'row_support', 'int32', ...
                    numel(d.rowSupport), -1, chunk);
                H5.writeData(did, 'int32', d.rowSupport);
                mestra.internal.Writer.attach(did, {'row'}, scales, []);
                H5D.close(did);
            end

            % -------------------------------------------- supports
            if ismember('supports', wanted)
                g = mestra.internal.Writer.group(root, 'supports');
                for i = 1:numel(d.supports)
                    mestra.internal.Writer.writeSupport(g, d, d.supports(i), ...
                                                        scales);
                end
                H5G.close(g);
            end

            % ------------------------------------------- callables
            if ismember('callables', wanted)
                g = mestra.internal.Writer.group(root, 'callables');
                for i = 1:numel(d.callables)
                    c = d.callables(i);
                    cg = mestra.internal.Writer.group(g, c.id);
                    if ~isempty(c.type)
                        H5.writeStrAttr(cg, 'type', c.type);
                    end
                    if ~isempty(c.repr)
                        H5.writeStrAttr(cg, 'repr', c.repr);
                    end
                    dict = c.dict;
                    if isempty(dict) && ~isempty(c.obj)
                        dict = c.obj.toDict();
                    end
                    mestra.internal.Codec.write(cg, dict, true);
                    H5G.close(cg);
                end
                H5G.close(g);
            end

            % ------------------------------- notes, private, others
            if ~isempty(d.notes)
                g = mestra.internal.Writer.group(root, 'notes');
                H5.replayTree(g, d.notes);
                H5G.close(g);
            end
            if ~isempty(d.privateTree)
                g = mestra.internal.Writer.group(root, 'private');
                H5.replayTree(g, d.privateTree);
                H5G.close(g);
            end
            for i = 1:numel(d.unknownGroups)
                u = d.unknownGroups{i};
                if isempty(u.tree), continue, end
                g = mestra.internal.Writer.group(root, u.name);
                H5.replayTree(g, u.tree);
                H5G.close(g);
            end

            for name = scales.keys()
                H5D.close(scales(name{1}));
            end
        end

        % -------------------------------------------------- pieces

        function g = group(parent, name)
        %group  A group with object time tracking off.
            gcpl = mestra.internal.H5.plist('H5P_GROUP_CREATE');
            g = H5G.create(parent, name, 'H5P_DEFAULT', gcpl, 'H5P_DEFAULT');
            H5P.close(gcpl);
        end

        function names = neededGroups(d)
        %neededGroups  The root groups that have something in them.
            names = {};
            if ~isempty(d.keys), names{end + 1} = 'keys'; end
            if ~isempty(d.scalars), names{end + 1} = 'scalars'; end
            if ~isempty(d.categories), names{end + 1} = 'categories'; end
            if ~isempty(d.supports), names{end + 1} = 'supports'; end
            if ~isempty(d.callables), names{end + 1} = 'callables'; end
        end

        function plan = scalePlan(d)
        %scalePlan  Which root scales the file needs, and how long.
        %   component_<n> and draw_<n> for each length a dataset
        %   actually uses, group_<k> for every group key, and
        %   category_<t> for every category table (section 21).
            plan = containers.Map('KeyType', 'char', 'ValueType', 'any');
            for i = 1:numel(d.categories)
                plan(['category_' d.categories(i).name]) = ...
                    numel(d.categories(i).entries);
            end
            for i = 1:numel(d.keys)
                k = d.keys(i);
                if strcmp(k.role, 'group')
                    n = 0;
                    if ~isempty(k.category)
                        j = find(strcmp({d.categories.name}, k.category), 1);
                        if ~isempty(j)
                            n = numel(d.categories(j).entries);
                        end
                    end
                    plan(['group_' k.name]) = n;
                end
            end
            found = d.slots();
            for i = 1:numel(found)
                slot = found(i).slot;
                if strcmp(found(i).kind, 'scalar'), continue, end
                if isempty(slot.dims), continue, end
                for axis = 1:numel(slot.dims)
                    name = slot.dims{axis};
                    len = size(slot.values, axis);
                    if strcmp(name, 'component')
                        plan(sprintf('component_%d', len)) = len;
                    elseif strcmp(name, 'draw')
                        plan(sprintf('draw_%d', len)) = len;
                    end
                end
            end
        end

        function writeKey(g, k, scales)
        %writeKey  One key column.
            H5 = mestra.internal.H5;
            n = numel(k.values);
            filters = mestra.internal.Writer.filtersOf(k);
            if strcmp(k.dtype, 'string')
                size_ = k.strSize;
                if isempty(size_)
                    size_ = mestra.internal.Writer.stringSize(k.values);
                end
                chunk = k.chunk;
                if isempty(chunk)
                    chunk = mestra.internal.Writer.rowChunk(size_, [], n);
                end
                did = H5.createDataset(g, k.name, 'string', n, -1, chunk, ...
                                       filters, size_);
                H5.writeData(did, 'string', k.values, size_);
            else
                itemsize = mestra.internal.Writer.itemSize(k.dtype);
                chunk = k.chunk;
                if isempty(chunk)
                    chunk = mestra.internal.Writer.rowChunk(itemsize, [], n);
                end
                did = H5.createDataset(g, k.name, k.dtype, n, -1, chunk, ...
                                       filters);
                H5.writeData(did, k.dtype, k.values);
            end
            mestra.internal.Writer.attach(did, {'row'}, scales, []);
            if ~isempty(k.role), H5.writeStrAttr(did, 'role', k.role); end
            if ~isempty(k.units), H5.writeStrAttr(did, 'units', k.units); end
            if ~isempty(k.category)
                H5.writeStrAttr(did, 'category', k.category);
            end
            if ~isempty(k.trajectoryGroup)
                H5.writeStrAttr(did, 'trajectory_group', k.trajectoryGroup);
            end
            if ~isempty(k.parent)
                H5.writeStrAttr(did, 'parent', k.parent);
            end
            if ~isempty(k.lower)
                H5.writeNumAttr(did, 'lower', k.lower, 'float64');
            end
            if ~isempty(k.upper)
                H5.writeNumAttr(did, 'upper', k.upper, 'float64');
            end
            H5D.close(did);
        end

        function writeScalar(g, s, scales)
        %writeScalar  One scalar slot, stored or served by a callable.
            H5 = mestra.internal.H5;
            if ~strcmp(s.source, 'data')
                oid = mestra.internal.Writer.group(g, s.name);
                mestra.internal.Writer.slotAttrs(oid, s, false);
                H5G.close(oid);
                return
            end
            n = numel(s.values);
            chunk = s.chunk;
            if isempty(chunk)
                chunk = mestra.internal.Writer.rowChunk(8, [], n);
            end
            did = H5.createDataset(g, s.name, 'float64', n, -1, chunk, ...
                                   mestra.internal.Writer.filtersOf(s));
            H5.writeData(did, 'float64', s.values);
            mestra.internal.Writer.attach(did, {'row'}, scales, []);
            mestra.internal.Writer.slotAttrs(did, s, false);
            H5D.close(did);
        end

        function writeSupport(g, d, s, scales)
        %writeSupport  One support, its cells, its scales and its arrays.
            H5 = mestra.internal.H5;
            sid = mestra.internal.Writer.group(g, s.name);
            H5.writeStrAttr(sid, 'kind', s.kind);
            H5.writeNumAttr(sid, 'n_nodes', int64(s.nNodes), 'int64');
            H5.writeNumAttr(sid, 'n_cells', int64(s.nCells), 'int64');
            H5.writeStrAttr(sid, 'support_id', s.supportId);

            local = containers.Map('KeyType', 'char', 'ValueType', 'any');
            if ~strcmp(s.kind, 'none')
                local('node') = H5.makeScale(sid, 'node', s.nNodes, false);
            end
            if s.nCells > 0
                local('cell') = H5.makeScale(sid, 'cell', s.nCells, false);
                local('cell_plus_one') = H5.makeScale(sid, 'cell_plus_one', ...
                                                      s.nCells + 1, false);
                local('index') = H5.makeScale(sid, 'index', ...
                    numel(s.cellConnectivity), false);
            end
            localRows = mestra.internal.Writer.localRowCount(d, s);
            if ~isempty(localRows)
                local('row') = H5.makeScale(sid, 'row', localRows, true);
                rowCount = localRows;
            else
                rowCount = d.nRows;
            end

            if s.nCells > 0
                mestra.internal.Writer.plainDataset(sid, 'cell_types', ...
                    'uint8', s.cellTypes, 'cell', scales, local);
                mestra.internal.Writer.plainDataset(sid, 'cell_offsets', ...
                    'int64', s.cellOffsets, 'cell_plus_one', scales, local);
                mestra.internal.Writer.plainDataset(sid, ...
                    'cell_connectivity', 'int64', s.cellConnectivity, ...
                    'index', scales, local);
            end

            if ~isempty(s.coordinates)
                mestra.internal.Writer.writeSlot(sid, s.coordinates, ...
                                                 scales, local, rowCount);
            end
            present = s.groupsPresent;
            if isempty(present), present = {}; end
            if ~isempty(s.nodeArrays) || ismember('node_arrays', present)
                ng = mestra.internal.Writer.group(sid, 'node_arrays');
                for i = 1:numel(s.nodeArrays)
                    mestra.internal.Writer.writeSlot(ng, s.nodeArrays(i), ...
                                                     scales, local, rowCount);
                end
                H5G.close(ng);
            end
            if ~isempty(s.cellArrays) || ismember('cell_arrays', present)
                cg = mestra.internal.Writer.group(sid, 'cell_arrays');
                for i = 1:numel(s.cellArrays)
                    mestra.internal.Writer.writeSlot(cg, s.cellArrays(i), ...
                                                     scales, local, rowCount);
                end
                H5G.close(cg);
            end
            % Groups inside the support this version does not know, put
            % back as they were read (section 28).
            extras = [];
            if isfield(s, 'extraGroups'), extras = s.extraGroups; end
            for i = 1:numel(extras)
                eg = mestra.internal.Writer.group(sid, extras(i).name);
                mestra.internal.H5.replayTree(eg, extras(i).tree);
                H5G.close(eg);
            end

            for name = local.keys()
                H5D.close(local(name{1}));
            end
            H5G.close(sid);
        end

        function n = localRowCount(d, s)
        %localRowCount  The length of a support-local `row` scale, or
        %   [] when the support does not need one.  Section 21: only
        %   in an unaligned file, and only in a support that carries
        %   an array with varies = row.
            n = [];
            if d.aligned, return, end
            slots = [s.nodeArrays s.cellArrays];
            if ~isempty(s.coordinates), slots = [slots s.coordinates]; end
            needs = false;
            for i = 1:numel(slots)
                if strcmp(slots(i).varies, 'row')
                    needs = true;
                end
            end
            if ~needs, return, end
            order = mestra.internal.H5.sortByBytes({d.supports.name});
            which = find(strcmp(order, s.name), 1) - 1;
            n = sum(double(d.rowSupport) == which);
        end

        function plainDataset(sid, name, dtype, values, dimName, scales, local)
        %plainDataset  One of a support's own cell datasets.
            H5 = mestra.internal.H5;
            n = numel(values);
            did = H5.createDataset(sid, name, dtype, n, n, []);
            H5.writeData(did, dtype, values);
            mestra.internal.Writer.attach(did, {dimName}, scales, local);
            H5D.close(did);
        end

        function writeSlot(g, slot, scales, local, rowCount)
        %writeSlot  One array slot, stored or served by a callable.
        %   ROWCOUNT is the length of the `row` dimension the leading
        %   axis is attached to, which is what section 23 measures the
        %   default chunk against, and not the dataset's own leading
        %   extent.
            H5 = mestra.internal.H5;
            if ~strcmp(slot.source, 'data')
                oid = mestra.internal.Writer.group(g, slot.name);
                mestra.internal.Writer.slotAttrs(oid, slot, true);
                H5G.close(oid);
                return
            end
            fileDims = fliplr(slot.dims);
            sz = size(slot.values);
            sz = [sz ones(1, numel(slot.dims) - numel(sz))];
            dims = fliplr(sz(1:numel(slot.dims)));
            rowLeading = ~isempty(fileDims) && strcmp(fileDims{1}, 'row');
            maxdims = dims;
            chunk = slot.chunk;
            filters = mestra.internal.Writer.filtersOf(slot);
            if rowLeading
                maxdims(1) = -1;
                if isempty(chunk)
                    c = mestra.internal.Writer.rowChunk( ...
                        mestra.internal.Writer.itemSize(slot.dtype), ...
                        dims(2:end), rowCount);
                    chunk = [c dims(2:end)];
                end
            elseif isempty(chunk) && ~isempty(filters)
                % Section 23: a slot with no `row` dimension may be
                % contiguous, and a filter needs chunked storage, so a
                % compressed one takes the section's other default --
                % the whole dataset when that is a mebibyte or less,
                % and otherwise the same rule over its leading axis.
                chunk = mestra.internal.Writer.plainChunk( ...
                    mestra.internal.Writer.itemSize(slot.dtype), dims);
            end
            did = H5.createDataset(g, slot.name, slot.dtype, dims, maxdims, ...
                                   chunk, filters);
            H5.writeData(did, slot.dtype, slot.values);
            mestra.internal.Writer.attach(did, fileDims, scales, local);
            mestra.internal.Writer.slotAttrs(did, slot, true);
            H5D.close(did);
        end

        function slotAttrs(oid, slot, isArray)
        %slotAttrs  The attributes section 19 puts on a slot.
            H5 = mestra.internal.H5;
            if isArray
                if ~isempty(slot.role)
                    H5.writeStrAttr(oid, 'role', slot.role);
                end
                if ~isempty(slot.varies)
                    H5.writeStrAttr(oid, 'varies', slot.varies);
                end
            end
            if ~isempty(slot.units)
                H5.writeStrAttr(oid, 'units', slot.units);
            end
            if isArray && ~isempty(slot.components)
                H5.writeNumAttr(oid, 'components', int64(slot.components), ...
                                'int64');
            end
            if ~isempty(slot.source)
                H5.writeStrAttr(oid, 'source', slot.source);
            end
            if ~isempty(slot.output)
                H5.writeStrAttr(oid, 'output', slot.output);
            end
            if ~isempty(slot.statistic)
                H5.writeStrAttr(oid, 'statistic', slot.statistic);
            end
            if ~isempty(slot.of)
                H5.writeStrAttr(oid, 'of', slot.of);
            end
            if ~isempty(slot.quantile)
                H5.writeNumAttr(oid, 'quantile', slot.quantile, 'float64');
            end
            if isfield(slot, 'level') && ~isempty(slot.level)
                H5.writeNumAttr(oid, 'level', slot.level, 'float64');
            end
            if isfield(slot, 'method') && ~isempty(slot.method)
                H5.writeStrAttr(oid, 'method', slot.method);
            end
            if isArray
                if ~isempty(slot.category)
                    H5.writeStrAttr(oid, 'category', slot.category);
                end
                if ~isempty(slot.recomputed)
                    H5.writeNumAttr(oid, 'recomputed', ...
                                    int8(slot.recomputed ~= 0), 'int8');
                end
                if ~isempty(slot.derivedFrom)
                    H5.writeStrAttr(oid, 'derived_from', slot.derivedFrom);
                end
                if ~isempty(slot.recipe)
                    H5.writeStrAttr(oid, 'recipe', slot.recipe);
                end
                if ~isempty(slot.reference)
                    H5.writeStrAttr(oid, 'reference', slot.reference);
                end
            end
        end

        function attach(did, fileDims, scales, local)
        %attach  Attach one dimension scale to each axis, by name.
            info = mestra.internal.H5.dsetInfo(did);
            for axis = 1:numel(fileDims)
                len = 0;
                if axis <= numel(info.dims), len = info.dims(axis); end
                name = mestra.Dataset.diskDim(fileDims{axis}, len);
                sid = [];
                if ~isempty(local) && local.isKey(name)
                    sid = local(name);
                elseif scales.isKey(name)
                    sid = scales(name);
                end
                if isempty(sid)
                    error('mestra:E25', ...
                          'no dimension scale named "%s" (E25)', name);
                end
                H5DS.attach_scale(did, sid, axis - 1);
            end
        end

        function f = filtersOf(rec)
        %filtersOf  The filter pipeline a record carries, or none.
        %   Section 23 makes compression optional, so the default is
        %   no filter; but what a file said about its own layout has
        %   to survive being read and written again, alongside a chunk
        %   shape that is not the default.  A record built before this
        %   field existed has no `filters` at all, which is why the
        %   question is asked rather than assumed.
            f = zeros(0, 2);
            if ~isstruct(rec) || ~isfield(rec, 'filters'), return, end
            if isempty(rec.filters), return, end
            f = mestra.Dataset.filterMatrix(rec.filters);
        end

        function chunk = plainChunk(itemsize, dims)
        %plainChunk  The chunk section 23 gives a dataset with no
        %   `row` dimension that is chunked or compressed: the whole
        %   dataset when that is 1 MiB or less, and otherwise the row
        %   rule applied to its leading axis.  A zero-length axis
        %   takes 1 on every axis, which is the only chunk HDF5 allows
        %   for it.
            dims = double(reshape(dims, 1, []));
            if isempty(dims), chunk = []; return, end
            if any(dims == 0)
                chunk = ones(1, numel(dims));
                return
            end
            if itemsize * prod(dims) <= 1048576
                chunk = dims;
                return
            end
            c = mestra.internal.Writer.rowChunk(itemsize, dims(2:end), ...
                                                dims(1));
            chunk = [c dims(2:end)];
        end

        function c = rowChunk(itemsize, rest, nRows)
        %rowChunk  The default chunk length of section 23.
            if nRows == 0
                c = 1;
                return
            end
            b = itemsize;
            for i = 1:numel(rest)
                b = b * max(1, rest(i));
            end
            c = floor(1048576 / b);
            if c < 1, c = 1; end
            if c > nRows, c = nRows; end
        end

        function n = itemSize(dtype)
        %itemSize  Bytes per element of a dtype.
            switch mestra.internal.H5.baseType(dtype)
                case {'float64', 'int64'}, n = 8;
                case {'float32', 'int32'}, n = 4;
                case {'int8', 'uint8'}, n = 1;
                otherwise, n = 1;
            end
        end

        function n = stringSize(values)
        %stringSize  The largest UTF-8 byte length, at least one.
            n = 1;
            for i = 1:numel(values)
                n = max(n, numel(unicode2native(values{i}, 'UTF-8')));
            end
        end
    end
end
