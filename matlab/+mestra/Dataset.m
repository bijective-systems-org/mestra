classdef Dataset < handle
%mestra.Dataset  One dataset in the open format: keys, scalars,
%   supports, arrays, categories and callables.
%
%   Why a handle class.  A dataset is built up a piece at a time, and
%   a value class would make every builder call a reassignment
%   (d = d.addKey(...)) and would copy every array it already holds.
%   A handle class lets the builder read the way people write it and
%   keeps the arrays in one place.  mestra.read and mestra.evaluate
%   each return a fresh object, so nothing is shared by accident.
%
%   AXIS ORDER.  HDF5 stores an array in C order and MATLAB is column
%   major, so every array this class holds has its axes REVERSED with
%   respect to the file.  The file's logical order is
%
%       (row | group | nothing, [draw], node | cell, component)
%
%   and a MATLAB array therefore comes back as
%
%       (component, node | cell, [draw], row | group)
%
%   Every array carries its dimension names in the same order as its
%   own axes, so you never have to count axes: permute by name.
%
%       a = d.nodeArray('s0', 'pressure');
%       p = mestra.permute(a.values, a.dims, {'row', 'node'});
%
%   Building one
%
%       d = mestra.Dataset();
%       d.addCategoryTable('member', {'wing_a', 'wing_b'});
%       d.addKey('mach', [0.4 0.8], 'condition', '1');
%       d.addKey('member', int32([0 1]), 'group', 'Category', 'member');
%       d.setGeneralisationGroup('member');
%       d.addMeshSupport('s0', coords, uint8([9 9]), int64([0 4 8]), ...
%                        int64([0 1 4 3 1 2 5 4]), 'm', ...
%                        'Dims', {'component', 'node', 'group:member'});
%       d.addNodeArray('s0', 'pressure', p, 'field', 'Pa', ...
%                      'Dims', {'row', 'node'});
%       d.addScalar('cl', [0.25 0.55], '1');
%       mestra.write(d, 'two_rows.mes');
%
%   The argument order is the one every implementation uses
%   (docs/api-conventions.md, section 1): the name, the values, then
%   the role and the units, and a name-value pair for everything
%   else.  'Dims' names the axes of the array you hand over and
%   settles what it varies along and how many components it has.
%
%   Reading one
%
%       d = mestra.read('two_rows.mes');    % everything
%       d = mestra.open('two_rows.mes');    % attributes only, no array
%       p = d.readRows('/supports/s0/node_arrays/pressure', [1 1]);
%
%   See also mestra.read, mestra.open, mestra.write, mestra.validate,
%   mestra.evaluate, mestra.permute.

    properties
        % Where this dataset was read from, or '' when it was built in
        % memory.  Lazy reads need it.
        path = ''

        % Root attributes (section 11).
        format = 'mestra/0'
        writer = 'mestra matlab 0'
        created = ''
        aligned = true
        generalisationGroup = ''

        % The number of rows.  The `row` dimension scale carries it,
        % so a file with no row-dimensioned dataset still states it.
        nRows = 0

        % Struct arrays.  Their fields are listed by emptyKey,
        % emptyScalar, emptySupport and emptySlot below.
        keys
        scalars
        categories
        supports
        callables

        % int32, one per row, zero based, present only in an unaligned
        % file (sections 8 and 22).
        rowSupport = []

        % Free-form and opaque subtrees, kept exactly as they were
        % read so that a round trip does not lose them.
        notes = []
        privateTree = []

        % Groups and root attributes this version does not know.  They
        % are ignored, reported (W11) and written back unchanged.
        unknownGroups
        unknownAttrs

        % Root groups the file carried, so that one that was there and
        % empty is written back.  The writer creates the union of this
        % and the groups that have content.
        groupsPresent = {}

        % What the reader would not follow in this file, one line
        % each: a soft or external link, an object of the wrong kind,
        % an attribute that is not a scalar, a walk that hit a limit.
        % Empty for every conforming file.
        skipped = {}
    end

    methods

        function obj = Dataset()
        %Dataset  An empty dataset with sensible defaults.
            obj.keys = mestra.Dataset.emptyKey();
            obj.scalars = mestra.Dataset.emptyScalar();
            obj.categories = mestra.Dataset.emptyCategory();
            obj.supports = mestra.Dataset.emptySupport();
            obj.callables = mestra.Dataset.emptyCallable();
            obj.unknownGroups = {};
            obj.skipped = {};
            obj.unknownAttrs = containers.Map('KeyType', 'char', ...
                                              'ValueType', 'any');
            obj.created = mestra.Dataset.nowUtc();
        end

        % ------------------------------------------------ accessors

        function names = keyNames(obj)
        %keyNames  Key names in the file's key order, which is the
        %   names sorted by their UTF-8 bytes (section 26).
            names = mestra.internal.H5.sortByBytes({obj.keys.name});
        end

        function k = key(obj, name)
        %key  One key by name.
            k = mestra.Dataset.pick(obj.keys, name, 'key');
        end

        function s = scalar(obj, name)
        %scalar  One scalar slot by name.
            s = mestra.Dataset.pick(obj.scalars, name, 'scalar');
        end

        function s = support(obj, name)
        %support  One support by name.
            s = mestra.Dataset.pick(obj.supports, name, 'support');
        end

        function c = category(obj, name)
        %category  One category table by name.
            c = mestra.Dataset.pick(obj.categories, name, 'category table');
        end

        function a = nodeArray(obj, supportName, name)
        %nodeArray  One node array by support and name.
            s = obj.support(supportName);
            a = mestra.Dataset.pick(s.nodeArrays, name, 'node array');
        end

        function a = cellArray(obj, supportName, name)
        %cellArray  One cell array by support and name.
            s = obj.support(supportName);
            a = mestra.Dataset.pick(s.cellArrays, name, 'cell array');
        end

        function c = callable(obj, id)
        %callable  One callable record by id.
            c = mestra.Dataset.pick(obj.callables, id, 'callable');
        end

        function names = supportNames(obj)
        %supportNames  Support names in the file's support order, which
        %   is the names sorted by their UTF-8 bytes (section 22).
            names = mestra.internal.H5.sortByBytes({obj.supports.name});
        end

        function out = slots(obj)
        %slots  Every slot in the file, as a struct array with fields
        %   path, kind ('scalar', 'coordinates', 'node' or 'cell'),
        %   support and slot.
        %
        %   Collected and joined once.  Appending one entry at a time
        %   copies the whole array every time, and each entry carries
        %   a slot record, so a file with four thousand scalars spent
        %   longer here than it did reading them.
            parts = {};
            for i = 1:numel(obj.scalars)
                parts{end + 1} = struct('path', ...
                    ['/scalars/' obj.scalars(i).name], 'kind', 'scalar', ...
                    'support', '', 'slot', obj.scalars(i)); %#ok<AGROW>
            end
            for i = 1:numel(obj.supports)
                s = obj.supports(i);
                base = ['/supports/' s.name];
                if ~isempty(s.coordinates)
                    parts{end + 1} = struct('path', [base '/coordinates'], ...
                        'kind', 'coordinates', 'support', s.name, ...
                        'slot', s.coordinates); %#ok<AGROW>
                end
                for j = 1:numel(s.nodeArrays)
                    parts{end + 1} = struct('path', ...
                        [base '/node_arrays/' s.nodeArrays(j).name], ...
                        'kind', 'node', 'support', s.name, ...
                        'slot', s.nodeArrays(j)); %#ok<AGROW>
                end
                for j = 1:numel(s.cellArrays)
                    parts{end + 1} = struct('path', ...
                        [base '/cell_arrays/' s.cellArrays(j).name], ...
                        'kind', 'cell', 'support', s.name, ...
                        'slot', s.cellArrays(j)); %#ok<AGROW>
                end
            end
            out = struct('path', {}, 'kind', {}, 'support', {}, 'slot', {});
            if ~isempty(parts), out = [out parts{:}]; end
        end

        function t = keysTable(obj)
        %keysTable  The stored keys as a MATLAB table, in the file's
        %   key order.  This is the shape section 26 gives a keys
        %   table, so it can be handed straight to a callable.
            names = obj.keyNames();
            columns = cell(1, numel(names));
            for i = 1:numel(names)
                k = obj.key(names{i});
                v = k.values;
                if iscell(v)
                    columns{i} = reshape(v, [], 1);
                else
                    columns{i} = double(reshape(v, [], 1));
                end
            end
            if isempty(names)
                t = table.empty(obj.nRows, 0);
            else
                t = table(columns{:}, 'VariableNames', names);
            end
        end

        function s = axisOrder(obj) %#ok<MANU>
        %axisOrder  One sentence stating the order arrays come back in.
            s = ['arrays are returned in MATLAB order, which is the ' ...
                 'reverse of the file''s: (component, node | cell, ' ...
                 '[draw], row | group). Every array carries its ' ...
                 'dimension names in that same order; permute by name ' ...
                 'with mestra.permute.'];
        end

        % ----------------------------------------------- lazy access

        function out = readRows(obj, slotPath, rowRange)
        %readRows  One slot, for a range of rows, reading nothing else.
        %   `slotPath` is the HDF5 path of the slot, for example
        %   '/supports/s0/node_arrays/pressure' or '/scalars/cl'.
        %   `rowRange` is [first last], one based and inclusive; pass
        %   a single number for one row.  The result is a struct with
        %   fields values and dims, in the same axis order as every
        %   other array this class hands back.
        %
        %   Section 29 asks a reader to be able to do this without
        %   reading any other slot and without reading the rows
        %   outside the range; chunking along `row` with the non-row
        %   extents full is what makes that one run of chunks.
            if numel(rowRange) == 1
                rowRange = [rowRange rowRange];
            end
            first = rowRange(1);
            count = rowRange(2) - rowRange(1) + 1;
            if count < 0
                error('mestra:rowRange', 'the row range is empty');
            end
            if isempty(obj.path)
                out = obj.sliceInMemory(slotPath, first, count);
                return
            end
            fid = mestra.internal.Reader.openFile(obj.path);
            cleanup = onCleanup(@() H5F.close(fid));
            closePass = mestra.internal.H5.pass(); %#ok<NASGU>
            did = mestra.Dataset.openSlot(fid, slotPath);
            closeSlot = onCleanup(@() H5D.close(did)); %#ok<NASGU>
            info = mestra.internal.H5.dsetInfo(did);
            map = mestra.internal.H5.scaleMap(fid);
            names = cell(1, numel(info.dims));
            for axis = 1:numel(info.dims)
                found = mestra.internal.H5.scaleNames(did, axis - 1, map);
                if isempty(found) || isempty(found(1).name)
                    names{axis} = sprintf('axis%d', axis - 1);
                else
                    names{axis} = mestra.Dataset.logicalDim(found(1).name);
                end
            end
            if isempty(info.dims) || ~strcmp(names{1}, 'row')
                error('mestra:rowRange', ...
                      'the slot "%s" has no row dimension', slotPath);
            end
            if first < 1 || first - 1 + count > info.dims(1)
                error('mestra:rowRange', ...
                      'rows %d to %d are outside the %d the slot holds', ...
                      first, first + count - 1, info.dims(1));
            end
            values = mestra.internal.H5.readRows(did, info, first - 1, count);
            out.values = values;
            out.dims = fliplr(names);
        end

        % -------------------------------------------------- builders

        function addCategoryTable(obj, name, entries, varargin)
        %addCategoryTable  A category table.  Ids are the positions,
        %   so the first entry is id 0 (section 21).
        %
        %   addCategoryTable(NAME, ENTRIES) is the one way to attach a
        %   category table to a file; the key or the label that uses
        %   it then names it with 'Category'.  There is no inline
        %   alternative, in this language or in any other
        %   (docs/api-conventions.md, section 1).
        %
        %       d.addCategoryTable('member', {'wing_a', 'wing_b'});
        %       d.addKey('member', int32([0 1]), 'group', ...
        %                'Category', 'member');
            p = inputParser();
            p.addParameter('Size', []);
            p.parse(varargin{:});
            if isstring(entries), entries = cellstr(entries); end
            if ischar(entries), entries = {entries}; end
            rec = mestra.Dataset.emptyCategory();
            rec(1).name = name;
            rec(1).entries = reshape(entries, 1, []);
            rec(1).strSize = p.Results.Size;
            obj.categories = mestra.Dataset.append(obj.categories, rec, name);
        end

        function addCategory(obj, varargin) %#ok<INUSD>
        %addCategory  Renamed to addCategoryTable.
            error('mestra:renamed', ...
                  ['addCategory is now addCategoryTable, which is the ' ...
                   'name every implementation uses ' ...
                   '(docs/api-conventions.md, section 1); the ' ...
                   'arguments are unchanged']);
        end

        function addKey(obj, name, values, varargin)
        %addKey  A key column: addKey(NAME, VALUES, ROLE, UNITS).
        %
        %   The order is the one every implementation uses
        %   (docs/api-conventions.md, section 1): the name, the
        %   values, the role, the units.  ROLE and UNITS may be given
        %   positionally or as 'Role' and 'Units'; everything else is
        %   a name-value pair.
        %
        %       d.addKey('mach', [0.4 0.8], 'condition', '1');
        %       d.addKey('member', int32([0 1]), 'group', ...
        %                'Category', 'member');
        %
        %   A key of role design, condition or time requires units
        %   (E39).  A categorical, group, split or status key names
        %   its category table with 'Category' instead (E10).
        %
        %   BOUNDS.  When neither 'Lower' nor 'Upper' is given, a
        %   design, condition or time key records the observed finite
        %   minimum and maximum, so that the same arrays written by
        %   any implementation give the same file and W04 and W08 are
        %   decidable.  A caller who wants a wider domain of validity
        %   passes them.
        %
        %   The dtype follows the role (section 19) unless Dtype says
        %   otherwise.
        %
        %   LAYOUT.  'Chunk' and 'Filters' say how the column is
        %   stored.  Both default to what section 23 prescribes, both
        %   survive a read and a write again, and 'Filters' is an
        %   n-by-2 matrix of HDF5 filter id and parameter in pipeline
        %   order: [2 0] for shuffle, [1 4] for gzip at level 4,
        %   [2 0; 1 4] for the two together.
            p = inputParser();
            p.addParameter('Role', '');
            p.addParameter('Units', '');
            p.addParameter('Lower', []);
            p.addParameter('Upper', []);
            p.addParameter('Category', '');
            p.addParameter('TrajectoryGroup', '');
            p.addParameter('Parent', '');
            p.addParameter('Dtype', '');
            p.addParameter('Chunk', []);
            p.addParameter('Filters', []);
            mestra.Dataset.refuseOldOrder('addKey', name, values, ...
                @mestra.internal.Args.isKeyRole, ...
                'addKey(name, values, role, units)');
            [pos, rest] = mestra.internal.Args.positional(varargin, ...
                {{'', @mestra.internal.Args.isText}, ...
                 {'', @mestra.internal.Args.isText}}, ...
                mestra.internal.Args.parameterNames(p));
            p.parse(rest{:});
            r = p.Results;
            role = mestra.Dataset.oneOf(pos{1}, r.Role, 'Role');
            units = mestra.Dataset.oneOf(pos{2}, r.Units, 'Units');
            path = ['/keys/' name];
            if isempty(role)
                error('mestra:E02', ...
                      ['E02: %s: a key needs a role; give one of %s as ' ...
                       'the third argument'], path, ...
                      strjoin(mestra.internal.Args.keyRoles, ', '));
            end
            if ~mestra.internal.Args.isKeyRole(role)
                error('mestra:E02', ...
                      ['E02: %s: "%s" is not a key role; the roles of ' ...
                       'section 3 are %s'], path, role, ...
                      strjoin(mestra.internal.Args.keyRoles, ', '));
            end
            if any(strcmp(role, {'design', 'condition', 'time'})) && ...
                    isempty(units)
                error('mestra:E39', ...
                      ['E39: %s: a key of role %s requires units; give ' ...
                       'them as the fourth argument, "1" when it is ' ...
                       'dimensionless'], path, role);
            end
            mestra.Dataset.checkUnitsText(units, path);
            mestra.Dataset.checkBound('Lower', r.Lower, path);
            mestra.Dataset.checkBound('Upper', r.Upper, path);
            rec = mestra.Dataset.emptyKey();
            rec(1).name = name;
            rec(1).role = role;
            rec(1).units = units;
            rec(1).lower = r.Lower;
            rec(1).upper = r.Upper;
            rec(1).category = r.Category;
            rec(1).trajectoryGroup = r.TrajectoryGroup;
            rec(1).parent = r.Parent;
            rec(1).chunk = r.Chunk;
            rec(1).filters = mestra.Dataset.filterMatrix(r.Filters);
            if iscell(values) || isstring(values)
                rec(1).values = reshape(cellstr(values), 1, []);
                rec(1).dtype = 'string';
            else
                rec(1).dtype = r.Dtype;
                if isempty(rec(1).dtype)
                    rec(1).dtype = mestra.Dataset.dtypeForRole(role, values);
                end
                rec(1).values = reshape(values, 1, []);
                [rec(1).lower, rec(1).upper] = ...
                    mestra.Dataset.observedBounds(role, rec(1).values, ...
                                                  r.Lower, r.Upper);
            end
            obj.checkCategories(rec(1).category, rec(1).values, path);
            obj.noteRows(numel(rec(1).values));
            obj.keys = mestra.Dataset.append(obj.keys, rec, name);
        end

        function setGeneralisationGroup(obj, name)
        %setGeneralisationGroup  Name the unit of generalisation.
        %
        %   The unit of generalisation is a property of the dataset in
        %   every implementation (docs/api-conventions.md, section 1)
        %   and names a key of role group (section 7).  Exactly one
        %   group key is the unit of generalisation (E03).
        %
        %       d.setGeneralisationGroup('member');
        %
        %   Setting the generalisationGroup property does the same
        %   thing; this is the spelling the documents show.
            if isstring(name), name = char(name); end
            if ~isempty(name)
                i = find(strcmp({obj.keys.name}, name), 1);
                if ~isempty(i) && ~strcmp(obj.keys(i).role, 'group')
                    error('mestra:E03', ...
                          ['E03: /keys/%s: the unit of generalisation ' ...
                           'must be a key of role group and "%s" has ' ...
                           'role %s; name a group key instead'], ...
                          name, name, obj.keys(i).role);
                end
            end
            obj.generalisationGroup = name;
        end

        function addScalar(obj, name, values, varargin)
        %addScalar  A per-row quantity of interest:
        %   addScalar(NAME, VALUES, UNITS).
        %
        %   The order is the one every implementation uses
        %   (docs/api-conventions.md, section 1).  UNITS may be given
        %   positionally or as 'Units', and is required (E11).
        %
        %       d.addScalar('cl', [0.25 0.55], '1');
        %
        %   A scalar served by a callable takes no values:
        %
        %       d.addScalar('cl', [], '1', 'Callable', 'm1', ...
        %                   'Output', 'cl');
            p = inputParser();
            p.addParameter('Units', '');
            p.addParameter('Source', 'data');
            p.addParameter('Callable', '');
            p.addParameter('Output', '');
            p.addParameter('Statistic', '');
            p.addParameter('Of', '');
            p.addParameter('Quantile', []);
            p.addParameter('Level', []);
            p.addParameter('Method', '');
            p.addParameter('Chunk', []);
            p.addParameter('Filters', []);
            [pos, rest] = mestra.internal.Args.positional(varargin, ...
                {{'', @mestra.internal.Args.isText}}, ...
                mestra.internal.Args.parameterNames(p));
            p.parse(rest{:});
            r = p.Results;
            units = mestra.Dataset.oneOf(pos{1}, r.Units, 'Units');
            source = r.Source;
            if ~isempty(r.Callable)
                source = ['callable:' r.Callable];
            end
            mestra.Dataset.checkUnitsText(units, ['/scalars/' name]);
            if isempty(units)
                error('mestra:E11', ...
                      ['E11: /scalars/%s: a scalar requires units; give ' ...
                       'them as the third argument, "1" when it is ' ...
                       'dimensionless'], name);
            end
            rec = mestra.Dataset.emptyScalar();
            rec(1).name = name;
            rec(1).units = units;
            rec(1).source = source;
            rec(1).output = r.Output;
            rec(1).statistic = r.Statistic;
            rec(1).of = r.Of;
            rec(1).quantile = r.Quantile;
            rec(1).level = r.Level;
            rec(1).method = char(r.Method);
            rec(1).chunk = r.Chunk;
            rec(1).filters = mestra.Dataset.filterMatrix(r.Filters);
            rec(1).dtype = 'float64';
            rec(1).values = reshape(double(values), 1, []);
            rec(1).dims = {'row'};
            if ~strcmp(source, 'data')
                rec(1).values = [];
            else
                obj.noteRows(numel(rec(1).values));
            end
            obj.scalars = mestra.Dataset.append(obj.scalars, rec, name);
        end

        function addSupport(obj, name, kind, nNodes, varargin)
        %addSupport  A support of kind 'mesh', 'axis' or 'none'.
        %   addMeshSupport and addAxisSupport are the usual way in.
            p = inputParser();
            p.addParameter('CellTypes', []);
            p.addParameter('CellOffsets', []);
            p.addParameter('CellConnectivity', []);
            p.parse(varargin{:});
            r = p.Results;
            rec = mestra.Dataset.emptySupport();
            rec(1).name = name;
            rec(1).kind = kind;
            rec(1).nNodes = double(nNodes);
            rec(1).cellTypes = uint8(r.CellTypes(:)');
            rec(1).cellOffsets = int64(r.CellOffsets(:)');
            rec(1).cellConnectivity = int64(r.CellConnectivity(:)');
            rec(1).nCells = numel(rec(1).cellTypes);
            rec(1).coordinates = [];
            rec(1).nodeArrays = mestra.Dataset.emptySlot();
            rec(1).cellArrays = mestra.Dataset.emptySlot();
            rec(1).supportId = '';
            rec(1).groupsPresent = {};
            obj.supports = mestra.Dataset.append(obj.supports, rec, name);
            obj.refreshSupportId(name);
            obj.aligned = numel(obj.supports) <= 1;
        end

        function addMeshSupport(obj, name, coordinates, cellTypes, ...
                                cellOffsets, cellConnectivity, varargin)
        %addMeshSupport  A mesh support and its coordinates in one call.
        %   `coordinates` is given in MATLAB axis order, or in the
        %   order named by Dims.  The support id is computed for you,
        %   and Dims settles what the coordinates vary along.
        %
        %       d.addMeshSupport('s0', coords, types, offsets, conn, ...
        %                        'm', 'Dims', {'component', 'node'});
            p = inputParser();
            p.addParameter('Units', '');
            p.addParameter('Varies', '');
            p.addParameter('Dims', {});
            p.addParameter('Components', []);
            [pos, rest] = mestra.internal.Args.positional(varargin, ...
                {{'', @mestra.internal.Args.isText}}, ...
                mestra.internal.Args.parameterNames(p));
            p.parse(rest{:});
            r = p.Results;
            units = mestra.Dataset.oneOf(pos{1}, r.Units, 'Units');
            if isempty(units), units = 'm'; end
            nNodes = mestra.Dataset.countNodes(coordinates, r.Dims);
            obj.addSupport(name, 'mesh', nNodes, ...
                           'CellTypes', cellTypes, ...
                           'CellOffsets', cellOffsets, ...
                           'CellConnectivity', cellConnectivity);
            obj.setCoordinates(name, coordinates, units, ...
                               'Varies', r.Varies, 'Dims', r.Dims, ...
                               'Components', r.Components);
        end

        function addAxisSupport(obj, name, coordinates, varargin)
        %addAxisSupport  An axis support and its coordinates.  The
        %   coordinates of an axis support never vary (E35).
            p = inputParser();
            p.addParameter('Units', '');
            p.addParameter('Dims', {});
            [pos, rest] = mestra.internal.Args.positional(varargin, ...
                {{'', @mestra.internal.Args.isText}}, ...
                mestra.internal.Args.parameterNames(p));
            p.parse(rest{:});
            r = p.Results;
            units = mestra.Dataset.oneOf(pos{1}, r.Units, 'Units');
            if isempty(units), units = '1'; end
            derived = mestra.Dataset.variesFromDims(r.Dims, obj);
            if ~strcmp(derived, 'none')
                error('mestra:E35', ...
                      ['E35: /supports/%s/coordinates: the coordinates ' ...
                       'of an axis support never vary and Dims names a ' ...
                       '%s axis; drop that axis from Dims'], name, derived);
            end
            nNodes = mestra.Dataset.countNodes(coordinates, r.Dims);
            obj.addSupport(name, 'axis', nNodes);
            obj.setCoordinates(name, coordinates, units, ...
                               'Varies', 'none', 'Dims', r.Dims, ...
                               'Components', 1);
        end

        function setCoordinates(obj, supportName, values, varargin)
        %setCoordinates  The one coordinates array of a support:
        %   setCoordinates(SUPPORT, VALUES, UNITS).
            slot = obj.makeSlot('coordinates', values, 'coordinates', ...
                                supportName, 'node', varargin{:});
            i = obj.supportIndex(supportName);
            obj.supports(i).coordinates = slot;
            obj.refreshSupportId(supportName);
        end

        function setCallableCoordinates(obj, supportName, varargin)
        %setCallableCoordinates  Coordinates a callable serves rather
        %   than data: setCallableCoordinates(SUPPORT, UNITS, CALLABLE,
        %   OUTPUT, 'Components', C).
        %
        %   setCoordinates with the values left out and the callable
        %   id and the output name added, the way addCallableSlot is
        %   addNodeArray without its values (docs/api-conventions.md,
        %   section 1): a model of the geometry itself.  The support
        %   was added with its node count (addSupport) and no
        %   coordinates; the slot has no values, so it must declare
        %   'Components' (E31).  'Varies' defaults to 'row'.  An axis
        %   support's coordinates are part of its identity (section
        %   24) and are always stored.
        %
        %       d.addSupport('s0', 'mesh', 6, 'CellTypes', types, ...
        %                    'CellOffsets', offsets, ...
        %                    'CellConnectivity', conn);
        %       d.addCallable('m1', A);
        %       d.setCallableCoordinates('s0', 'm', 'm1', ...
        %                                'coordinates', 'Components', 2);
            p = inputParser();
            p.addParameter('Units', '');
            p.addParameter('Callable', '');
            p.addParameter('Output', 'coordinates');
            p.addParameter('Components', []);
            p.addParameter('Varies', 'row');
            [pos, rest] = mestra.internal.Args.positional(varargin, ...
                {{'', @mestra.internal.Args.isText}, ...
                 {'', @mestra.internal.Args.isText}, ...
                 {'', @mestra.internal.Args.isText}}, ...
                mestra.internal.Args.parameterNames(p));
            p.parse(rest{:});
            r = p.Results;
            units = mestra.Dataset.oneOf(pos{1}, r.Units, 'Units');
            id = mestra.Dataset.oneOf(pos{2}, r.Callable, 'Callable');
            output = r.Output;
            if ~isempty(pos{3}), output = pos{3}; end
            i = obj.supportIndex(supportName);
            path = ['/supports/' supportName '/coordinates'];
            if ~strcmp(obj.supports(i).kind, 'mesh')
                error('mestra:E03', ...
                      ['E03: %s: a callable may serve the coordinates ' ...
                       'of a mesh support; this support is of kind %s, ' ...
                       'whose coordinates are stored'], path, ...
                      obj.supports(i).kind);
            end
            if ~isempty(obj.supports(i).coordinates)
                error('mestra:E03', ...
                      'E03: %s: this support already has its one coordinates array', ...
                      path);
            end
            if isempty(id)
                error('mestra:E14', ...
                      ['E14: %s: a callable slot must name the callable ' ...
                       'that serves it; give its id as the second ' ...
                       'argument'], path);
            end
            if isempty(r.Components)
                error('mestra:E31', ...
                      ['E31: %s: a callable slot stores no values, so it ' ...
                       'declares its width; pass ''Components'''], path);
            end
            slot = obj.makeSlot('coordinates', [], 'coordinates', ...
                                supportName, 'node', 'Units', units, ...
                                'Varies', r.Varies, ...
                                'Components', r.Components, ...
                                'Source', ['callable:' id], ...
                                'Output', output);
            obj.supports(i).coordinates = slot;
        end

        function addNodeArray(obj, supportName, name, values, varargin)
        %addNodeArray  A node array on a support:
        %   addNodeArray(SUPPORT, NAME, VALUES, ROLE, UNITS, 'Dims', ...).
        %
        %   The order is the one every implementation uses
        %   (docs/api-conventions.md, section 1).  ROLE and UNITS may
        %   be given positionally or as 'Role' and 'Units'; ROLE
        %   defaults to 'field', which is what an array usually is.
        %
        %       d.addNodeArray('s0', 'pressure', p, 'field', 'Pa', ...
        %                      'Dims', {'row', 'node'});
        %       d.addNodeArray('s0', 'cad_face_id', ids, 'label', ...
        %                      'Category', 'faces', 'Dims', {'node'});
        %
        %   'Dims' names the axes of the array you hand over, in that
        %   array's own order.  It settles what the array varies
        %   along: a 'row' axis means row, a 'group:<k>' axis means
        %   that group, neither means none.  It also settles the
        %   component count.
            slot = obj.makeSlot(name, values, '', supportName, 'node', ...
                                varargin{:});
            i = obj.supportIndex(supportName);
            obj.supports(i).nodeArrays = ...
                mestra.Dataset.append(obj.supports(i).nodeArrays, slot, name);
        end

        function addCellArray(obj, supportName, name, values, varargin)
        %addCellArray  A cell array on a support, with the same
        %   argument order as addNodeArray.
            slot = obj.makeSlot(name, values, '', supportName, 'cell', ...
                                varargin{:});
            i = obj.supportIndex(supportName);
            obj.supports(i).cellArrays = ...
                mestra.Dataset.append(obj.supports(i).cellArrays, slot, name);
        end

        function addCallableSlot(obj, supportName, name, role, varargin)
        %addCallableSlot  An array slot served by a callable:
        %   addCallableSlot(SUPPORT, NAME, ROLE, UNITS, CALLABLE,
        %   OUTPUT).
        %
        %   The argument order is the array builders' with the values
        %   left out and the callable id and the output name added
        %   (docs/api-conventions.md, section 1).  A callable slot has
        %   no data and no shape, so it must declare its width with
        %   'Components' (E31), and 'Location' says whether it is a
        %   node array or a cell array.
        %
        %       d.addCallable('m1', A);
        %       d.addCallableSlot('s0', 'pressure', 'field', 'Pa', ...
        %                         'm1', 'pressure', 'Components', 1);
            p = inputParser();
            p.addParameter('Units', '');
            p.addParameter('Callable', '');
            p.addParameter('Output', '');
            p.addParameter('Location', 'node');
            p.addParameter('Components', []);
            p.addParameter('Varies', 'row');
            p.addParameter('Category', '');
            p.addParameter('Statistic', '');
            p.addParameter('Of', '');
            p.addParameter('Quantile', []);
            p.addParameter('Level', []);
            p.addParameter('Method', '');
            [pos, rest] = mestra.internal.Args.positional(varargin, ...
                {{'', @mestra.internal.Args.isText}, ...
                 {'', @mestra.internal.Args.isText}, ...
                 {'', @mestra.internal.Args.isText}}, ...
                mestra.internal.Args.parameterNames(p));
            p.parse(rest{:});
            r = p.Results;
            units = mestra.Dataset.oneOf(pos{1}, r.Units, 'Units');
            id = mestra.Dataset.oneOf(pos{2}, r.Callable, 'Callable');
            output = mestra.Dataset.oneOf(pos{3}, r.Output, 'Output');
            if isempty(id)
                error('mestra:E14', ...
                      ['E14: /supports/%s/%s_arrays/%s: a callable slot ' ...
                       'must name the callable that serves it; give its ' ...
                       'id as the fifth argument'], supportName, ...
                      r.Location, name);
            end
            slot = obj.makeSlot(name, [], role, supportName, r.Location, ...
                                'Units', units, 'Varies', r.Varies, ...
                                'Components', r.Components, ...
                                'Source', ['callable:' id], ...
                                'Output', output, 'Category', r.Category, ...
                                'Statistic', r.Statistic, 'Of', r.Of, ...
                                'Quantile', r.Quantile, ...
                                'Level', r.Level, 'Method', r.Method);
            i = obj.supportIndex(supportName);
            if strcmp(r.Location, 'cell')
                obj.supports(i).cellArrays = mestra.Dataset.append( ...
                    obj.supports(i).cellArrays, slot, name);
            else
                obj.supports(i).nodeArrays = mestra.Dataset.append( ...
                    obj.supports(i).nodeArrays, slot, name);
            end
        end

        function addCallable(obj, id, callableOrDict, varargin)
        %addCallable  Store a callable under an id.
            p = inputParser();
            p.addParameter('Type', '');
            p.addParameter('Repr', '');
            p.parse(varargin{:});
            r = p.Results;
            rec = mestra.Dataset.emptyCallable();
            rec(1).id = id;
            if isa(callableOrDict, 'mestra.Callable')
                rec(1).obj = callableOrDict;
                rec(1).dict = callableOrDict.toDict();
                rec(1).type = r.Type;
                if isempty(rec(1).type)
                    rec(1).type = callableOrDict.type();
                end
                rec(1).repr = r.Repr;
                if isempty(rec(1).repr)
                    rec(1).repr = callableOrDict.repr();
                end
            else
                rec(1).obj = [];
                rec(1).dict = callableOrDict;
                rec(1).type = r.Type;
                rec(1).repr = r.Repr;
            end
            if isempty(rec(1).type)
                error('mestra:E15', ...
                      ['E15: /callables/%s: a callable needs a type; ' ...
                       'pass ''Type'', or hand over a mestra.Callable, ' ...
                       'which names its own'], id);
            end
            obj.callables = mestra.Dataset.append(obj.callables, rec, id);
        end

        function setRowSupport(obj, values)
        %setRowSupport  Which support each row is on, zero based.
            obj.rowSupport = int32(reshape(values, 1, []));
            obj.noteRows(numel(obj.rowSupport));
            obj.aligned = numel(obj.supports) <= 1;
        end

        function refreshSupportId(obj, name)
        %refreshSupportId  Recompute one support's content hash.
            i = obj.supportIndex(name);
            obj.supports(i).supportId = mestra.supportId(obj.supports(i));
        end

        function disp(obj)
        %disp  A short summary.
            fprintf('  mestra.Dataset  %s  %d row(s)\n', obj.format, obj.nRows);
            if ~isempty(obj.path)
                fprintf('    read from %s\n', obj.path);
            end
            fprintf(['    aligned %d, %d support(s), %d key(s), ' ...
                     '%d scalar(s)\n'], obj.aligned, ...
                    numel(obj.supports), numel(obj.keys), ...
                    numel(obj.scalars));
            for i = 1:numel(obj.keys)
                k = obj.keys(i);
                fprintf('    key %-14s %-12s', k.name, k.role);
                if ~isempty(k.units), fprintf(' [%s]', k.units); end
                if ~isempty(k.lower) && ~isempty(k.upper)
                    fprintf(' in [%g, %g]', k.lower, k.upper);
                end
                fprintf('\n');
            end
            for i = 1:numel(obj.supports)
                s = obj.supports(i);
                fprintf('    support %-8s %-5s %d node(s) %d cell(s) %s\n', ...
                        s.name, s.kind, s.nNodes, s.nCells, s.supportId(1:8));
            end
            for i = 1:numel(obj.callables)
                fprintf('    callable %-8s %s\n', obj.callables(i).id, ...
                        obj.callables(i).type);
            end
        end
    end

    % ------------------------------------------------------ internals

    methods (Access = private)

        function i = supportIndex(obj, name)
            i = find(strcmp({obj.supports.name}, name), 1);
            if isempty(i)
                error('mestra:noSupport', 'no support "%s"', name);
            end
        end

        function noteRows(obj, n)
            if n > obj.nRows
                obj.nRows = n;
            end
        end

        function slot = makeSlot(obj, name, values, role, supportName, ...
                                 location, varargin)
            p = inputParser();
            p.addParameter('Role', '');
            p.addParameter('Units', '');
            p.addParameter('Varies', '');
            p.addParameter('Components', []);
            p.addParameter('Source', 'data');
            p.addParameter('Callable', '');
            p.addParameter('Output', '');
            p.addParameter('Statistic', '');
            p.addParameter('Of', '');
            p.addParameter('Quantile', []);
            p.addParameter('Level', []);
            p.addParameter('Method', '');
            p.addParameter('Category', '');
            p.addParameter('Recomputed', []);
            p.addParameter('DerivedFrom', '');
            p.addParameter('Recipe', '');
            p.addParameter('Reference', '');
            p.addParameter('Dims', {});
            p.addParameter('Dtype', '');
            p.addParameter('Chunk', []);
            p.addParameter('Filters', []);
            names = mestra.internal.Args.parameterNames(p);
            if isempty(role)
                specs = {{'', @mestra.internal.Args.isArrayRole}, ...
                         {'', @mestra.internal.Args.isText}};
            else
                specs = {{'', @mestra.internal.Args.isText}};
            end
            [pos, rest] = mestra.internal.Args.positional(varargin, specs, ...
                                                          names);
            p.parse(rest{:});
            r = p.Results;
            if isempty(role)
                role = mestra.Dataset.oneOf(pos{1}, r.Role, 'Role');
                units = mestra.Dataset.oneOf(pos{2}, r.Units, 'Units');
                if isempty(role), role = 'field'; end
            else
                units = mestra.Dataset.oneOf(pos{1}, r.Units, 'Units');
            end
            path = mestra.Dataset.slotPath(supportName, location, name);
            if ~mestra.internal.Args.isArrayRole(role)
                error('mestra:E02', ...
                      ['E02: %s: "%s" is not an array role; the roles ' ...
                       'of section 3 are %s'], path, ...
                      mestra.internal.Args.text(role), ...
                      strjoin(mestra.internal.Args.arrayRoles, ', '));
            end
            if any(strcmp(role, {'field', 'derived'})) && isempty(units)
                error('mestra:E11', ...
                      ['E11: %s: an array of role %s requires units; ' ...
                       'give them after the role, "1" when it is ' ...
                       'dimensionless'], path, role);
            end
            if strcmp(role, 'coordinates') && isempty(units)
                error('mestra:E39', ...
                      ['E39: %s: coordinates require units; give them ' ...
                       'after the values'], path);
            end

            slot = mestra.Dataset.emptySlot();
            slot(1).name = name;
            slot(1).role = role;
            slot(1).units = units;
            slot(1).source = r.Source;
            slot(1).output = r.Output;
            slot(1).statistic = r.Statistic;
            slot(1).of = r.Of;
            slot(1).quantile = r.Quantile;
            slot(1).level = r.Level;
            slot(1).method = char(r.Method);
            slot(1).category = r.Category;
            slot(1).recomputed = r.Recomputed;
            slot(1).derivedFrom = r.DerivedFrom;
            slot(1).recipe = r.Recipe;
            slot(1).reference = r.Reference;
            slot(1).location = location;
            slot(1).support = supportName;
            slot(1).chunk = r.Chunk;
            slot(1).filters = mestra.Dataset.filterMatrix(r.Filters);
            if ~isempty(r.Callable)
                slot(1).source = ['callable:' r.Callable];
            end

            % Dims settles what the array varies along.  A 'row' axis
            % means row, a 'group:<k>' axis means that group, neither
            % means none (docs/api-conventions.md, section 1).  This
            % is what keeps a builder from writing a file its own
            % validator rejects: `Varies` no longer has a default that
            % contradicts the array that was handed over.
            given = r.Dims;
            if ischar(given), given = {given}; end
            if isstring(given), given = cellstr(given); end
            given = reshape(given, 1, []);
            varies = mestra.Dataset.oneOf(r.Varies, '', 'Varies');
            if ~isempty(given)
                derived = mestra.Dataset.variesFromDims(given, obj);
                if ~isempty(varies) && ~strcmp(varies, derived)
                    mestra.Dataset.refuseVaries(path, varies, derived, given);
                end
                varies = derived;
            elseif isempty(varies)
                if strcmp(role, 'coordinates')
                    varies = 'none';
                else
                    varies = 'row';
                end
            end
            slot(1).varies = varies;

            if ~strcmp(slot(1).source, 'data')
                slot(1).values = [];
                slot(1).dims = {};
                slot(1).dtype = '';
                slot(1).components = r.Components;
                if isempty(slot(1).components)
                    error('mestra:E31', ...
                          ['E31: %s: a callable slot has no shape of ' ...
                           'its own, so it must declare its width; pass ' ...
                           '''Components'''], path);
                end
                return
            end

            hasDraw = strcmp(r.Statistic, 'draw');
            logicalNames = mestra.Dataset.logicalDims(varies, location, ...
                                                      hasDraw);
            wanted = fliplr(logicalNames);
            % Dims names the axes of the array the caller passes, in
            % that array's own order.  With no Dims the array is taken
            % to be in the order a reader hands one back.
            if isempty(given)
                given = wanted;
            else
                given = mestra.Dataset.canonicalDims(given, varies);
            end
            slot(1).values = mestra.permute(values, given, wanted);
            slot(1).dims = wanted;
            sz = size(slot(1).values);
            sz = [sz ones(1, numel(wanted) - numel(sz))];
            slot(1).shape = fliplr(sz(1:numel(wanted)));
            obj.checkExtent(slot(1), supportName, location, path, given);
            obj.checkCategories(r.Category, slot(1).values, path);
            slot(1).dtype = r.Dtype;
            if isempty(slot(1).dtype)
                slot(1).dtype = mestra.Dataset.dtypeForArrayRole(role, values);
            end
            slot(1).components = r.Components;
            if isempty(slot(1).components)
                slot(1).components = size(slot(1).values, 1);
            end
            if strcmp(varies, 'row')
                obj.noteRows(size(slot(1).values, numel(wanted)));
            end
        end

        function checkCategories(obj, table, values, path)
        %checkCategories  E10 at build time: a value outside the table
        %   it names.  The table has to be there already, which is the
        %   order the documents show; one added later is checked by
        %   the validator when the file is written.
            if isempty(table) || isempty(values) || ~isnumeric(values)
                return
            end
            i = find(strcmp({obj.categories.name}, table), 1);
            if isempty(i), return, end
            n = numel(obj.categories(i).entries);
            v = double(values(:)');
            bad = v(v < 0 | v > n - 1 | v ~= floor(v));
            if isempty(bad), return, end
            error('mestra:E10', ...
                  ['E10: %s: the value %g is outside the category table ' ...
                   '"%s", whose %d entries are ids 0 to %d; add the ' ...
                   'entry or correct the values'], path, bad(1), table, ...
                  n, n - 1);
        end

        function checkExtent(obj, slot, supportName, location, path, given)
        %checkExtent  E05 at build time: an array whose node or cell
        %   count disagrees with its support.
        %
        %   This is the commonest way to hand over an array the wrong
        %   way round, and the message says so rather than leaving it
        %   to the validator after the file is on disk.  The model is
        %   Python's E04 message: state the reading, state what the
        %   support is, and say which argument settles it.
            i = find(strcmp({obj.supports.name}, supportName), 1);
            if isempty(i), return, end
            s = obj.supports(i);
            if strcmp(location, 'cell')
                want = s.nCells;
            else
                want = s.nNodes;
            end
            if want == 0, return, end
            axis = find(strcmp(slot.dims, location), 1);
            if isempty(axis), return, end
            got = size(slot.values, axis);
            if got == want, return, end
            if isempty(given)
                advice = ['name the axes of the array you passed with ' ...
                          'Dims'];
            else
                advice = sprintf(['Dims says the axes are {%s}; check ' ...
                                  'that against the array'], ...
                                 strjoin(given, ', '));
            end
            error('mestra:E05', ...
                  ['E05: %s: an array of shape %s reads as %d %s(s) on ' ...
                   'a support of %d; %s'], path, ...
                  mat2str(size(slot.values)), got, location, want, advice);
        end

        function out = sliceInMemory(obj, slotPath, first, count)
            found = obj.slots();
            i = find(strcmp({found.path}, slotPath), 1);
            if isempty(i)
                error('mestra:noSlot', 'no slot at "%s"', slotPath);
            end
            slot = found(i).slot;
            n = numel(slot.dims);
            if n == 0 || ~strcmp(slot.dims{end}, 'row')
                error('mestra:rowRange', ...
                      'the slot "%s" has no row dimension', slotPath);
            end
            subs = repmat({':'}, 1, n);
            subs{n} = first:(first + count - 1);
            out.values = slot.values(subs{:});
            out.dims = slot.dims;
        end
    end

    methods (Static)

        function s = emptyKey()
            s = struct('name', {}, 'role', {}, 'units', {}, 'lower', {}, ...
                       'upper', {}, 'category', {}, 'trajectoryGroup', {}, ...
                       'parent', {}, 'values', {}, 'dtype', {}, ...
                       'chunk', {}, 'filters', {}, 'strSize', {});
        end

        function s = emptyScalar()
            s = struct('name', {}, 'units', {}, 'source', {}, 'output', {}, ...
                       'statistic', {}, 'of', {}, 'quantile', {}, ...
                       'level', {}, 'method', {}, ...
                       'values', {}, 'dtype', {}, 'dims', {}, 'chunk', {}, ...
                       'filters', {});
        end

        function s = emptyCategory()
            s = struct('name', {}, 'entries', {}, 'strSize', {});
        end

        function s = emptySupport()
            s = struct('name', {}, 'kind', {}, 'nNodes', {}, 'nCells', {}, ...
                       'supportId', {}, 'cellTypes', {}, 'cellOffsets', {}, ...
                       'cellConnectivity', {}, 'coordinates', {}, ...
                       'nodeArrays', {}, 'cellArrays', {}, ...
                       'groupsPresent', {}, 'extraGroups', {});
        end

        function s = emptySlot()
            s = struct('name', {}, 'role', {}, 'varies', {}, 'units', {}, ...
                       'components', {}, 'source', {}, 'output', {}, ...
                       'statistic', {}, 'of', {}, 'quantile', {}, ...
                       'level', {}, 'method', {}, ...
                       'category', {}, 'recomputed', {}, 'derivedFrom', {}, ...
                       'recipe', {}, 'reference', {}, 'values', {}, ...
                       'dims', {}, 'shape', {}, 'dtype', {}, ...
                       'location', {}, 'support', {}, 'chunk', {}, ...
                       'filters', {});
        end

        function s = emptyCallable()
            s = struct('id', {}, 'type', {}, 'repr', {}, 'dict', {}, ...
                       'obj', {});
        end

        function m = filterMatrix(v)
        %filterMatrix  The filter pipeline of section 23, as stored.
        %   An n-by-2 matrix of HDF5 filter id and first client
        %   parameter, in pipeline order: [2 0] is shuffle, [1 4] is
        %   gzip at level 4, and [2 0; 1 4] is shuffle then gzip,
        %   which is the order that makes shuffle worth anything.
        %   This is exactly what a dataset creation property list
        %   reports, so a filter a file declares survives being read
        %   and written again without being translated on the way.
        %   Section 23 allows gzip at levels 1 to 9 and shuffle and
        %   nothing else (E29).
            m = zeros(0, 2);
            if isempty(v), return, end
            if ~isnumeric(v) || ~ismatrix(v) || size(v, 2) ~= 2
                error('mestra:filters', ...
                      ['Filters must be an n-by-2 matrix of HDF5 ' ...
                       'filter id and parameter, for example [2 0; 1 4]']);
            end
            m = double(v);
            for i = 1:size(m, 1)
                switch m(i, 1)
                    case 1
                        if m(i, 2) < 1 || m(i, 2) > 9
                            error('mestra:E29', ...
                                  ['E29: gzip at level %g; section 23 ' ...
                                   'allows levels 1 to 9'], m(i, 2));
                        end
                    case 2
                        % shuffle takes no parameter
                    otherwise
                        error('mestra:E29', ...
                              ['E29: filter %g is not gzip or shuffle, ' ...
                               'which are the only two section 23 ' ...
                               'allows'], m(i, 1));
                end
            end
        end

        function out = append(arr, rec, name)
            % Every producer-chosen name comes through here, so this is
            % where section 18's name rule is held at build time (E33),
            % as every other builder holds it.
            if ~mestra.internal.Text.legalName(name)
                error('mestra:E33', ...
                      ['E33: "%s" is not a legal netCDF-4 name: letters, ' ...
                       'digits, underscore, hyphen, . and + only'], name);
            end
            if strncmp(name, 'mestra_', 7)
                error('mestra:E33', ...
                      ['E33: "%s": names beginning with mestra_ are ' ...
                       'reserved for the container'], name);
            end
            if ~isempty(arr)
                if isfield(arr, 'name')
                    taken = {arr.name};
                else
                    taken = {arr.id};
                end
                if any(strcmp(taken, name))
                    error('mestra:duplicate', '"%s" is already there', name);
                end
            end
            out = [arr rec];
        end

        function checkUnitsText(units, path)
        %checkUnitsText  E19: units, when given, are text.
            if ~isempty(units) && ~mestra.internal.Args.isText(units)
                error('mestra:E19', ...
                      ['E19: %s: units is a string, and this is %s; ' ...
                       'give the unit as text, "1" when it is ' ...
                       'dimensionless'], path, class(units));
            end
        end

        function checkBound(label, value, path)
        %checkBound  E19: a bound the caller declares is finite
        %   (section 18).  Empty means "not given" and is fine.
            if isempty(value), return, end
            if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value)
                error('mestra:E19', ...
                      ['E19: %s: %s is a finite float64 bound, and %s is ' ...
                       'not; give a number, or leave it out for the ' ...
                       'observed one'], path, label, mat2str(value));
            end
        end

        function v = pick(arr, name, what)
            if isempty(arr)
                error('mestra:notFound', 'no %s "%s"', what, name);
            end
            if isfield(arr, 'name')
                i = find(strcmp({arr.name}, name), 1);
            else
                i = find(strcmp({arr.id}, name), 1);
            end
            if isempty(i)
                error('mestra:notFound', 'no %s "%s"', what, name);
            end
            v = arr(i);
        end

        function names = logicalDims(varies, location, hasDraw)
        %logicalDims  The file's logical dimension names for a slot.
            names = {};
            if ~strcmp(varies, 'none')
                names{end + 1} = varies;      % 'row' or 'group:<k>'
            end
            if hasDraw
                names{end + 1} = 'draw';
            end
            names{end + 1} = location;        % 'node' or 'cell'
            names{end + 1} = 'component';
        end

        function out = canonicalDims(given, varies)
        %canonicalDims  Accept 'instance' for a group leading axis.
            out = given;
            for i = 1:numel(out)
                if strcmp(out{i}, 'instance')
                    out{i} = varies;
                end
            end
        end

        function v = variesFromDims(given, obj)
        %variesFromDims  What an array varies along, from the names of
        %   its own axes.  A 'row' axis means row, a 'group:<k>' axis
        %   means that group, neither means none
        %   (docs/api-conventions.md, section 1).  'instance' is the
        %   corpus's name for a group axis and is accepted when the
        %   file declares exactly one group key.
            v = 'none';
            if ischar(given), given = {given}; end
            if isstring(given), given = cellstr(given); end
            for i = 1:numel(given)
                name = given{i};
                if strcmp(name, 'row')
                    v = 'row';
                    return
                elseif strncmp(name, 'group:', 6)
                    v = name;
                    return
                elseif strcmp(name, 'instance')
                    v = mestra.Dataset.theOneGroupKey(obj);
                    return
                end
            end
        end

        function v = theOneGroupKey(obj)
        %theOneGroupKey  'group:<k>' when the dataset declares exactly
        %   one group key, and an error naming the choice otherwise.
            groups = {};
            if ~isempty(obj) && ~isempty(obj.keys)
                groups = {obj.keys(strcmp({obj.keys.role}, 'group')).name};
            end
            if numel(groups) == 1
                v = ['group:' groups{1}];
                return
            end
            if isempty(groups)
                error('mestra:E04', ...
                      ['E04: an axis called "instance" means the group ' ...
                       'key the array varies along, and this dataset ' ...
                       'declares none; name the axis "group:<key>" in ' ...
                       'Dims after adding the key']);
            end
            error('mestra:E04', ...
                  ['E04: an axis called "instance" is ambiguous when ' ...
                   'the file declares more than one group key (%s); ' ...
                   'name the axis "group:<key>" in Dims'], ...
                  strjoin(groups, ', '));
        end

        function refuseVaries(path, given, derived, dims)
        %refuseVaries  A Varies that disagrees with Dims, refused at
        %   build time with the rule the validator would report.
        %
        %   The row count is what goes wrong when one of the two says
        %   `row` and the other does not, so that disagreement is E16;
        %   any other is a leading dimension that disagrees with
        %   `varies`, which is E04.
            if strcmp(given, 'row') || strcmp(derived, 'row')
                id = 'E16';
            else
                id = 'E04';
            end
            error(['mestra:' id], ...
                  ['%s: %s: Dims names the axes %s, so this array ' ...
                   'varies along %s, and Varies says %s; the two ' ...
                   'disagree. Drop Varies and let Dims settle it, or ' ...
                   'name a %s axis in Dims'], id, path, ...
                  ['{' strjoin(dims, ', ') '}'], derived, given, given);
        end

        function p = slotPath(supportName, location, name)
        %slotPath  Where a slot will sit in the file.
            if strcmp(name, 'coordinates')
                p = ['/supports/' supportName '/coordinates'];
            else
                p = ['/supports/' supportName '/' location '_arrays/' name];
            end
        end

        function out = oneOf(positional, named, what)
        %oneOf  One value from a positional argument and its name-value
        %   twin, refusing the two together.
            positional = mestra.internal.Args.text(positional);
            named = mestra.internal.Args.text(named);
            if ~isempty(positional) && ~isempty(named) && ...
                    ~strcmp(positional, named)
                error('mestra:arguments', ...
                      ['%s was given twice, as "%s" positionally and ' ...
                       'as "%s" by name; give it once'], what, ...
                      positional, named);
            end
            if ~isempty(positional)
                out = positional;
            else
                out = named;
            end
        end

        function refuseOldOrder(fname, name, values, isRole, shape) %#ok<INUSL>
        %refuseOldOrder  Catch a call written in the order this
        %   package used before the conventions, and say the new one.
            v = values;
            if isstring(v) && isscalar(v), v = char(v); end
            if ~(ischar(v) && isrow(v)), return, end
            if ~isRole(v), return, end
            error('mestra:arguments', ...
                  ['%s now takes its arguments in the order every ' ...
                   'implementation uses, %s, so the values come before ' ...
                   'the role; "%s" arrived where the values belong'], ...
                  fname, shape, v);
        end

        function [lower, upper] = observedBounds(role, values, lower, upper)
        %observedBounds  The bounds a key records when the caller gives
        %   none: the observed finite minimum and maximum.
        %
        %   Section 1 of docs/api-conventions.md: every implementation
        %   fills them in, so that the same arrays give the same file
        %   and W04 and W08 are decidable on every file.  Bounds are a
        %   domain of validity, which only the continuous roles have:
        %   a categorical, group, split, status or id key is bounded
        %   by its category table (E10) and takes none.
            if ~any(strcmp(role, {'design', 'condition', 'time'}))
                return
            end
            if ~isnumeric(values) || isempty(values), return, end
            finite = double(values(isfinite(double(values))));
            if isempty(finite), return, end
            if isempty(lower), lower = min(finite); end
            if isempty(upper), upper = max(finite); end
        end

        function name = logicalDim(diskName)
        %logicalDim  The logical dimension name for a name on disk
        %   (section 21).
            if strncmp(diskName, 'component_', 10)
                name = 'component';
            elseif strncmp(diskName, 'draw_', 5)
                name = 'draw';
            elseif strncmp(diskName, 'group_', 6)
                name = ['group:' diskName(7:end)];
            elseif strncmp(diskName, 'category_', 9)
                name = 'category';
            else
                name = diskName;   % row, node, cell, cell_plus_one, index
            end
        end

        function name = diskDim(logicalName, length)
        %diskDim  The name on disk for a logical dimension name.
            switch logicalName
                case 'component'
                    name = sprintf('component_%d', length);
                case 'draw'
                    name = sprintf('draw_%d', length);
                otherwise
                    if strncmp(logicalName, 'group:', 6)
                        name = ['group_' logicalName(7:end)];
                    else
                        name = logicalName;
                    end
            end
        end

        function t = dtypeForRole(role, values)
        %dtypeForRole  The dtype table of section 19, for a key.
            switch role
                case {'design', 'condition', 'time'}
                    t = 'float64';
                case {'categorical', 'group', 'split', 'status'}
                    t = 'int32';
                case 'id'
                    t = 'int64';
                otherwise
                    t = 'float64';
            end
            if isa(values, 'int64'), t = 'int64'; end
            if isa(values, 'int32') && ~strcmp(t, 'float64'), t = 'int32'; end
        end

        function t = dtypeForArrayRole(role, values)
        %dtypeForArrayRole  The dtype table of section 19, for an array.
            switch role
                case 'label'
                    if isa(values, 'int64'), t = 'int64'; else, t = 'int32'; end
                otherwise
                    t = 'float64';
            end
        end

        function n = countNodes(coordinates, dims)
        %countNodes  How many nodes a coordinates array describes.
            if isempty(dims)
                % MATLAB order is (component, node, ...), so node is
                % the second axis.
                n = size(coordinates, 2);
            else
                i = find(strcmp(dims, 'node'), 1);
                if isempty(i)
                    error('mestra:dims', ...
                          ['the coordinates Dims must name "node"; it ' ...
                           'names {%s}'], strjoin(dims, ', '));
                end
                n = size(coordinates, i);
            end
        end

        function did = openSlot(fid, slotPath)
        %openSlot  Walk to a slot one hard link at a time.
        %   H5D.open on a whole path would follow a soft or an
        %   external link on the way, so each component is checked
        %   first and nothing but a hard link is entered.
            parts = strsplit(slotPath, '/');
            parts = parts(~cellfun(@isempty, parts));
            if isempty(parts)
                error('mestra:reader', 'no slot path was given');
            end
            open = H5G.open(fid, '/');
            try
                for i = 1:numel(parts) - 1
                    next = mestra.internal.H5.openGroup(open(end), parts{i});
                    open(end + 1) = next; %#ok<AGROW>
                end
                did = mestra.internal.H5.openDataset(open(end), parts{end});
            catch err
                mestra.Dataset.closeAll(open);
                rethrow(err);
            end
            mestra.Dataset.closeAll(open);
        end

        function closeAll(ids)
        %closeAll  Close every group opened on the way to a slot.
            for i = numel(ids):-1:1
                try
                    H5G.close(ids(i));
                catch
                end
            end
        end

        function s = nowUtc()
        %nowUtc  An ISO 8601 UTC timestamp for `created`.
            s = char(datetime('now', 'TimeZone', 'UTC', ...
                              'Format', 'uuuu-MM-dd''T''HH:mm:ss''Z'''));
        end
    end
end
