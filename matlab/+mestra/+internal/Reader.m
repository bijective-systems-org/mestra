classdef Reader
%Reader  Turn a file into a mestra.Dataset.
%
%   mestra.read and mestra.open are the public way in.  Everything
%   here reads through the low-level HDF5 interface and takes every
%   dimension name from a scale's link name, never from its NAME
%   attribute, which holds the same sentence in every scale in the
%   file (specification section 21).
%
%   See also mestra.read, mestra.open.

    properties (Constant)
        ROOT_ATTRS = {'format', 'writer', 'created', 'aligned', ...
                      'generalisation_group'};
        ROOT_GROUPS = {'keys', 'scalars', 'categories', 'supports', ...
                       'callables', 'notes', 'private'};
    end

    methods (Static)

        function d = load(path, eager, strict)
        %load  Read a file.  With `eager` false no array is read.
        %   Every failure below leaves this function with one of the
        %   identifiers section 14 names: mestra:E01 for another major
        %   version, mestra:E40 for a link that is not a hard link,
        %   mestra:E41 for an object that could not be read, or
        %   mestra:reader for a file that would not open at all.  A
        %   file is untrusted input: the caller gets an identifier and
        %   a sentence, never a library stack trace.
        %
        %   With `strict` true, which is the default, a file that made
        %   the reader pass anything over is refused rather than
        %   returned half read, which is what section 30 asks of a
        %   reader handed a hostile file.  With `strict` false the
        %   dataset comes back and `skipped` says what was passed over.
            if nargin < 2, eager = true; end
            if nargin < 3, strict = true; end
            try
                d = mestra.internal.Reader.loadUnguarded(path, eager);
            catch err
                if any(strcmp(err.identifier, {'mestra:E01', 'mestra:E40', ...
                                               'mestra:E41', ...
                                               'mestra:reader', ...
                                               'mestra:noFile'}))
                    rethrow(err);
                end
                error('mestra:reader', ...
                      'the file "%s" could not be read: %s', path, ...
                      regexprep(strtrim(err.message), '\s+', ' '));
            end
            if strict && ~isempty(d.skipped)
                error(mestra.internal.Reader.strictIdentifier(d.skipped), ...
                      ['this file was not read in full. %s. Pass ' ...
                       '''Strict'', false to take what could be read, ' ...
                       'with the rest listed in `skipped`.'], ...
                      strjoin(d.skipped, '; '));
            end
        end

        function d = loadUnguarded(path, eager)
        %loadUnguarded  load, before the identifiers are tidied.
            if exist(path, 'file') ~= 2
                error('mestra:noFile', 'no file at "%s"', path);
            end
            fid = mestra.internal.Reader.openFile(path);
            closeFile = onCleanup(@() H5F.close(fid));
            % One pass over one file: every object's attribute names
            % are listed once and answered from that list afterwards.
            closePass = mestra.internal.H5.pass(); %#ok<NASGU>
            d = mestra.Dataset();
            d.path = path;
            root = H5G.open(fid, '/');
            d.skipped = {};
            H5 = mestra.internal.H5;
            % Section 21: the scale names come from this
            % map, built by a bounded walk of our own, and never from
            % asking the library for a scale object's path.
            scales = H5.scaleMap(fid);
            closeRoot = onCleanup(@() H5G.close(root)); %#ok<NASGU>

            [value, ok] = H5.scalarAttr(root, 'format');
            d.format = '';
            if ok && ischar(value), d.format = value; end
            major = mestra.internal.Reader.majorVersion(d.format);
            if isnan(major) || major ~= 0
                error('mestra:E01', ...
                      ['this reader accepts "mestra/0" and the file ' ...
                       'says "%s"; it will not be read partially (E01)'], ...
                      d.format);
            end
            mestra.internal.Reader.checkAttrs(d, root, '/');
            d.writer = mestra.internal.Reader.str(root, 'writer');
            d.created = mestra.internal.Reader.str(root, 'created');
            aligned = mestra.internal.Reader.num(root, 'aligned');
            if ~isempty(aligned), d.aligned = aligned ~= 0; end
            d.generalisationGroup = mestra.internal.Reader.str( ...
                root, 'generalisation_group');
            for name = H5.publicAttrNames(root)
                if ~ismember(name{1}, mestra.internal.Reader.ROOT_ATTRS)
                    try
                        info = H5.attrInfo(root, name{1});
                        if ~info.scalar || isempty(info.type)
                            % Section 28: an attribute this version
                            % does not know is ignored and reported,
                            % and checkAttrs has already said so under
                            % the rule it breaks.
                            continue
                        elseif strcmp(info.type, 'string')
                            d.unknownAttrs(name{1}) = ...
                                struct('type', info.type, 'bytes', ...
                                       H5.readRawStrAttr(root, name{1}));
                        else
                            d.unknownAttrs(name{1}) = ...
                                struct('type', info.type, 'value', ...
                                       H5.readAttr(root, name{1}));
                        end
                    catch
                        mestra.internal.Reader.note(d, ['/' name{1}], ...
                            'E41', 'an attribute that would not read');
                    end
                end
            end

            d.nRows = mestra.internal.Reader.rowCount(fid);

            if mestra.internal.Reader.hasGroup(fid, 'categories')
                g = H5.openGroup(fid, 'categories');
                % Collected and joined once.  Appending a record to a
                % struct array copies the whole array, so a file with
                % four thousand of them spends more time copying than
                % reading (see the note on `join` below).
                found = {};
                for name = H5.children(g)
                    if ~mestra.internal.Reader.isKind(d, g, name{1}, ...
                            'dataset', ['/categories/' name{1}])
                        continue
                    end
                    did = H5D.open(g, name{1});
                    path = ['/categories/' name{1}];
                    try
                        info = H5.dsetInfo(did);
                        mestra.internal.Reader.checkAttrs(d, did, path);
                        mestra.internal.Reader.inspect(did, info, d, path, ...
                            scales, {['category_' name{1}]});
                        rec = mestra.Dataset.emptyCategory();
                        rec(1).name = name{1};
                        rec(1).entries = mestra.internal.Reader.readValues( ...
                            did, info, d, path);
                        if isempty(rec(1).entries), rec(1).entries = {}; end
                        rec(1).strSize = info.strSize;
                        found{end + 1} = rec; %#ok<AGROW>
                    catch err
                        H5D.close(did);
                        rethrow(err);
                    end
                    H5D.close(did);
                end
                d.categories = mestra.internal.Reader.join(d.categories, ...
                                                           found);
                H5G.close(g);
            end

            if mestra.internal.Reader.hasGroup(fid, 'keys')
                g = H5.openGroup(fid, 'keys');
                found = {};
                for name = H5.children(g)
                    if ~mestra.internal.Reader.isKind(d, g, name{1}, ...
                            'dataset', ['/keys/' name{1}])
                        continue
                    end
                    found{end + 1} = mestra.internal.Reader.readKey( ...
                        g, name{1}, eager, d, scales); %#ok<AGROW>
                end
                d.keys = mestra.internal.Reader.join(d.keys, found);
                H5G.close(g);
            end

            if mestra.internal.Reader.hasGroup(fid, 'scalars')
                g = H5.openGroup(fid, 'scalars');
                found = {};
                for name = H5.children(g)
                    kind = H5.childType(g, name{1});
                    if ~any(strcmp(kind, {'group', 'dataset'}))
                        mestra.internal.Reader.note(d, ...
                            ['/scalars/' name{1}], ...
                            mestra.internal.Reader.kindRule(kind), ...
                            [mestra.internal.Reader.describeKind(kind, ...
                                'dataset') ', not followed']);
                        continue
                    end
                    found{end + 1} = ...
                        mestra.internal.Reader.readScalar(g, name{1}, ...
                                                    eager, d, scales); %#ok<AGROW>
                end
                d.scalars = mestra.internal.Reader.join(d.scalars, found);
                H5G.close(g);
            end

            if mestra.internal.Reader.hasKind(fid, 'row_support', 'dataset')
                did = H5D.open(fid, 'row_support');
                info = H5.dsetInfo(did);
                mestra.internal.Reader.inspect(did, info, d, ...
                                               '/row_support', scales);
                d.rowSupport = int32(H5.readData(did, info));
                d.rowSupport = reshape(d.rowSupport, 1, []);
                H5D.close(did);
            end

            if mestra.internal.Reader.hasGroup(fid, 'supports')
                g = H5.openGroup(fid, 'supports');
                found = {};
                for name = H5.children(g)
                    if ~mestra.internal.Reader.isKind(d, g, name{1}, ...
                            'group', ['/supports/' name{1}])
                        continue
                    end
                    found{end + 1} = ...
                        mestra.internal.Reader.readSupport(g, name{1}, ...
                                                    eager, scales, d); %#ok<AGROW>
                end
                d.supports = mestra.internal.Reader.join(d.supports, found);
                H5G.close(g);
                mestra.internal.Reader.checkRowsPerSupport(d);
            end

            if mestra.internal.Reader.hasGroup(fid, 'callables')
                g = H5.openGroup(fid, 'callables');
                found = {};
                for name = H5.children(g)
                    if ~mestra.internal.Reader.isKind(d, g, name{1}, ...
                            'group', ['/callables/' name{1}])
                        continue
                    end
                    [record, limits] = ...
                        mestra.internal.Reader.readCallable(g, name{1}, ...
                                                            d, eager, scales);
                    found{end + 1} = record; %#ok<AGROW>
                    for i = 1:numel(limits)
                        mestra.internal.Reader.note(d, ...
                            ['/callables/' name{1}], limits{i}{1}, ...
                            limits{i}{2});
                    end
                end
                d.callables = mestra.internal.Reader.join(d.callables, found);
                H5G.close(g);
            end

            for which = {'notes', 'private'}
                if ~mestra.internal.Reader.hasGroup(fid, which{1})
                    continue
                end
                g = H5.openGroup(fid, which{1});
                tree = H5.captureTree(g, scales);
                links = {};
                if strcmp(which{1}, 'notes')
                    % /notes is public, so a link among its members is
                    % E40 (section 14), and the walk's own note of it
                    % is not a second finding; a dataset there is held
                    % to the byte-level rules like any public dataset.
                    mestra.internal.Reader.noteLinks(d, g, '/notes');
                    for name = H5.children(g)
                        if ~strcmp(H5.childType(g, name{1}), 'dataset')
                            continue
                        end
                        did = H5D.open(g, name{1});
                        info = H5.dsetInfo(did);
                        if ~info.isScale
                            mestra.internal.Reader.inspect(did, info, d, ...
                                ['/notes/' name{1}], scales);
                        end
                        H5D.close(did);
                    end
                    for name = H5.children(g)
                        if any(strcmp(H5.childType(g, name{1}), ...
                                      {'soft', 'external'}))
                            links{end + 1} = ['"' name{1} '" is a ']; %#ok<AGROW>
                        end
                    end
                end
                H5G.close(g);
                for i = 1:numel(tree.stopped)
                    if any(cellfun(@(l) startsWith(tree.stopped{i}, l), ...
                                   links))
                        continue
                    end
                    mestra.internal.Reader.note(d, ['/' which{1}], ...
                        'E41', tree.stopped{i});
                end
                if strcmp(which{1}, 'notes')
                    d.notes = tree;
                else
                    d.privateTree = tree;
                end
            end

            for name = H5.children(root)
                if ismember(name{1}, mestra.internal.Reader.ROOT_GROUPS) && ...
                   strcmp(H5.childType(root, name{1}), 'group')
                    d.groupsPresent{end + 1} = name{1};
                end
                if mestra.internal.Reader.knownRootChild(root, name{1})
                    continue
                end
                kind = H5.childType(root, name{1});
                if strcmp(kind, 'group')
                    g = H5G.open(root, name{1});
                    tree = H5.captureTree(g, scales);
                    H5G.close(g);
                    for i = 1:numel(tree.stopped)
                        mestra.internal.Reader.note(d, ['/' name{1}], ...
                            'E41', tree.stopped{i});
                    end
                    d.unknownGroups{end + 1} = ...
                        struct('name', name{1}, 'tree', tree);
                elseif strcmp(kind, 'dataset')
                    d.unknownGroups{end + 1} = ...
                        struct('name', name{1}, 'tree', []);
                else
                    mestra.internal.Reader.note(d, ['/' name{1}], ...
                        mestra.internal.Reader.kindRule(kind), ...
                        [mestra.internal.Reader.describeKind(kind, ...
                            'group') ', not followed']);
                end
            end
        end

        function out = join(existing, found)
        %join  Add a list of records to a struct array, once.
        %   Appending one record at a time copies the whole array
        %   every time, so reading n slots that way costs n squared
        %   and the copying overtakes every HDF5 call in the loop: on
        %   a file with four thousand scalars it was most of the open.
        %   Collecting the records and joining them here is one copy.
            out = existing;
            if isempty(found), return, end
            out = [existing found{:}];
        end

        function id = strictIdentifier(skipped)
        %strictIdentifier  The rule a strict refusal is raised under.
        %   Every entry of `skipped` begins with the identifier of the
        %   rule it breaks.  A link that is not a hard link comes
        %   first, because it is the one thing a reader must never
        %   follow; otherwise the first entry decides.
            id = 'mestra:E41';
            first = '';
            for i = 1:numel(skipped)
                parts = strsplit(skipped{i}, ' ');
                if isempty(parts), continue, end
                if strcmp(parts{1}, 'E40')
                    id = 'mestra:E40';
                    return
                end
                if isempty(first), first = parts{1}; end
            end
            if ~isempty(first)
                id = ['mestra:' first];
            end
        end

        function strict = strictOption(args)
        %strictOption  The 'Strict' name-value pair, default true.
            strict = true;
            if isempty(args), return, end
            if mod(numel(args), 2) ~= 0
                error('mestra:reader', ...
                      'options come in name and value pairs');
            end
            for i = 1:2:numel(args)
                if ~ischar(args{i}) || ~strcmpi(args{i}, 'Strict')
                    error('mestra:reader', ...
                          'the only option is ''Strict''');
                end
                strict = logical(args{i + 1});
            end
        end

        function fid = openFile(path)
        %openFile  H5F.open, with any failure named as this reader's.
            try
                fid = H5F.open(path, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
            catch err
                % E01, as every other reader says: a file that is not
                % HDF5 has no `format`, so it is not a mestra file.
                error('mestra:E01', ...
                      ['"%s" would not open as an HDF5 file: %s'], path, ...
                      regexprep(strtrim(err.message), '\s+', ' '));
            end
        end

        function tf = hasGroup(loc, name)
        %hasGroup  True when a name is a hard-linked group.
            tf = mestra.internal.Reader.hasKind(loc, name, 'group');
        end

        function tf = hasKind(loc, name, wanted)
        %hasKind  True when a name is a hard link of the kind wanted.
            tf = false;
            try
                if ~H5L.exists(loc, name, 'H5P_DEFAULT')
                    return
                end
            catch
                return
            end
            tf = strcmp(mestra.internal.H5.childType(loc, name), wanted);
        end

        function tf = isKind(d, gid, name, wanted, path)
        %isKind  True when a member is the kind wanted, and otherwise
        %   records on the dataset why it was passed over.
            kind = mestra.internal.H5.childType(gid, name);
            tf = strcmp(kind, wanted);
            if ~tf
                mestra.internal.Reader.note(d, path, ...
                    mestra.internal.Reader.kindRule(kind), ...
                    [mestra.internal.Reader.describeKind(kind, wanted) ...
                     ', not followed']);
            end
        end

        function id = kindRule(kind)
        %kindRule  Which rule a member of the wrong sort breaks.
            switch kind
                case {'soft', 'external'}
                    id = 'E40';
                otherwise
                    id = 'E41';
            end
        end

        function text = describeKind(kind, wanted)
        %describeKind  What was found where something else was needed.
            switch kind
                case 'soft'
                    text = ['a soft link, which this format does not ' ...
                            'define'];
                case 'external'
                    text = ['an external link, which names another ' ...
                            'file this reader never opens'];
                case 'unreadable'
                    text = 'an object that would not open';
                otherwise
                    text = sprintf('a %s where a %s is required', ...
                                   kind, wanted);
            end
        end

        function tf = knownRootChild(root, name)
        %knownRootChild  True for a group or scale this version knows.
            if ismember(name, mestra.internal.Reader.ROOT_GROUPS)
                tf = true; return
            end
            if any(strcmp(name, {'row', 'row_support'}))
                tf = true; return
            end
            prefixes = {'component_', 'draw_', 'group_', 'category_'};
            for i = 1:numel(prefixes)
                p = prefixes{i};
                if numel(name) > numel(p) && strncmp(name, p, numel(p))
                    tf = true; return
                end
            end
            tf = false;
            try
                if strcmp(mestra.internal.H5.childType(root, name), 'dataset')
                    did = H5D.open(root, name);
                    info = mestra.internal.H5.dsetInfo(did);
                    H5D.close(did);
                    tf = info.isScale;  % any other scale is the container's
                end
            catch
            end
        end

        function major = majorVersion(format)
        %majorVersion  The n of "mestra/n", or NaN.
            major = NaN;
            if isempty(format), return, end
            parts = strsplit(format, '/');
            if numel(parts) ~= 2 || ~strcmp(parts{1}, 'mestra'), return, end
            v = str2double(parts{2});
            if ~isnan(v) && v == floor(v) && v >= 0
                major = v;
            end
        end

        function n = rowCount(fid)
        %rowCount  The length of the file-level `row` scale.
        %   Section 21 makes that the row count, so a reader learns it
        %   even from a file with no row-dimensioned dataset at all.
            n = 0;
            if ~mestra.internal.Reader.hasKind(fid, 'row', 'dataset')
                return
            end
            did = H5D.open(fid, 'row');
            info = mestra.internal.H5.dsetInfo(did);
            H5D.close(did);
            if ~isempty(info.dims)
                n = info.dims(1);
            end
        end

        function inspect(did, info, d, path, map, wanted)
        %inspect  Record what would stop this dataset being read.
        %   Section 23 makes a reader refuse a dataset with a filter
        %   that is not gzip or shuffle (E29), and section 21 requires
        %   exactly one dimension scale on every axis (E25), without
        %   which the axis has no name to permute by.  Both are noted
        %   here so that a strict read refuses the file and a lenient
        %   one says what it passed over.
        %
        %   A leading extent that disagrees with the `row` dimension it
        %   is attached to is E16, and is noted here for the same
        %   reason: a dataset that says it holds three rows in a file
        %   of two is not a dataset a caller can line up against the
        %   keys, and handing it back would be worse than refusing it
        %   (docs/api-conventions.md, section 2).
            if nargin < 5, map = []; end
            if nargin < 6, wanted = {}; end
            if ~isempty(info.dims)
                leading = mestra.internal.H5.scaleNames(did, 0, map);
                if ~isempty(leading) && strcmp(leading(1).name, 'row') && ...
                        leading(1).length >= 0 && ...
                        info.dims(1) ~= leading(1).length
                    mestra.internal.Reader.note(d, path, 'E16', sprintf( ...
                        ['the leading extent is %d where the row ' ...
                         'dimension it is attached to has %d'], ...
                        info.dims(1), leading(1).length));
                end
            end
            for i = 1:size(info.filters, 1)
                id = info.filters(i, 1);
                ok = (id == 1 && info.filters(i, 2) >= 1 && ...
                      info.filters(i, 2) <= 9) || id == 2;
                if ~ok
                    mestra.internal.Reader.note(d, path, 'E29', sprintf( ...
                        ['the filter %d is not gzip or shuffle, which ' ...
                         'are the only two section 23 allows'], id));
                end
            end
            for axis = 1:numel(info.dims)
                found = mestra.internal.H5.scaleNames(did, axis - 1, map);
                if isempty(found)
                    mestra.internal.Reader.note(d, path, 'E25', sprintf( ...
                        'axis %d carries no dimension scale', axis - 1));
                elseif numel(found) > 1
                    mestra.internal.Reader.note(d, path, 'E25', sprintf( ...
                        'axis %d carries %d dimension scales', ...
                        axis - 1, numel(found)));
                elseif isempty(found(1).name) || ~found(1).hasName
                    mestra.internal.Reader.note(d, path, 'E25', sprintf( ...
                        ['axis %d is attached to something this reader ' ...
                         'cannot name'], axis - 1));
                elseif numel(wanted) >= axis && ~isempty(wanted{axis}) && ...
                        ~strcmp(found(1).name, wanted{axis})
                    % Where the position decides the one name section 21
                    % allows -- a table, a support's own cell datasets --
                    % another name is the wrong dimension.
                    mestra.internal.Reader.note(d, path, 'E25', sprintf( ...
                        'axis %d carries "%s" where section 21 names "%s"', ...
                        axis - 1, found(1).name, wanted{axis}));
                end
            end
        end

        function [values, ok] = readValues(did, info, d, path)
        %readValues  Read a dataset, noting why if it will not read.
        %   Returns [] and OK false when the data could not be had,
        %   with the reason recorded under the rule it breaks.  A
        %   dataset with no elements reads as an empty array of its
        %   own shape and OK true: empty is a value, not a failure.
            try
                if strcmp(info.type, 'string')
                    out = mestra.internal.H5.decodeStrings(did, info);
                    switch out.verdict
                        case 'replaced'
                            mestra.internal.Reader.note(d, path, 'E26', ...
                                ['a stored byte is not valid UTF-8; ' ...
                                 'this binding returned a replacement ' ...
                                 'character for it']);
                        case 'decoded'
                            mestra.internal.Reader.note(d, path, 'E41', ...
                                ['this binding decoded the strings to ' ...
                                 'text and the stored bytes are gone']);
                        otherwise
                            mestra.internal.Reader.checkStrings(out.bytes, ...
                                                                d, path);
                    end
                end
                values = mestra.internal.H5.readData(did, info);
                ok = true;
            catch err
                ok = false;
                id = 'E41';
                if strcmp(err.identifier, 'mestra:E41'), id = 'E41'; end
                mestra.internal.Reader.note(d, path, id, ...
                    regexprep(strtrim(err.message), '\s+', ' '));
                values = [];
            end
        end

        function checkStrings(bytes, d, path)
        %checkStrings  E26 over the stored bytes of a string dataset.
        %   Section 25 allows a NUL only in the trailing padding, and
        %   requires the rest to be valid UTF-8.  A reader that cannot
        %   recover the bytes of a string must say so rather than
        %   return something else (section 25), and E26 is a
        %   structural rule, so a strict read refuses the file
        %   (docs/api-conventions.md, section 2).  One finding per
        %   rule per object: the first bad entry is the one reported.
            if isempty(d), return, end
            for i = 1:size(bytes, 2)
                [ok, why] = mestra.internal.Text.checkStringBytes( ...
                    bytes(:, i)');
                if ~ok
                    mestra.internal.Reader.note(d, path, 'E26', sprintf( ...
                        'entry %d has %s', i - 1, why));
                    return
                end
            end
        end

        function names = axisNames(did, ndims, map)
        %axisNames  The logical dimension name of each axis, FILE order.
            names = cell(1, ndims);
            for axis = 1:ndims
                found = mestra.internal.H5.scaleNames(did, axis - 1, map);
                if isempty(found) || isempty(found(1).name)
                    names{axis} = '';
                else
                    names{axis} = mestra.Dataset.logicalDim(found(1).name);
                end
            end
        end

        function rec = readKey(g, name, eager, d, map)
        %readKey  One key column.
            if nargin < 4, d = []; end
            if nargin < 5, map = []; end
            H5 = mestra.internal.H5;
            did = H5D.open(g, name);
            info = H5.dsetInfo(did);
            rec = mestra.Dataset.emptyKey();
            rec(1).name = name;
            path = ['/keys/' name];
            R = @mestra.internal.Reader;
            R().checkAttrs(d, did, path);
            rec(1).role = R().str(did, 'role');
            rec(1).units = R().str(did, 'units');
            rec(1).category = R().str(did, 'category');
            rec(1).trajectoryGroup = R().str(did, 'trajectory_group');
            rec(1).parent = R().str(did, 'parent');
            rec(1).lower = R().num(did, 'lower');
            rec(1).upper = R().num(did, 'upper');
            rec(1).dtype = info.type;
            rec(1).chunk = info.chunk;
            rec(1).filters = info.filters;
            rec(1).strSize = info.strSize;
            mestra.internal.Reader.inspect(did, info, d, path, map);
            if eager
                values = mestra.internal.Reader.readValues(did, info, d, path);
                rec(1).values = reshape(values, 1, []);
            else
                rec(1).values = [];
            end
            H5D.close(did);
        end

        function rec = readScalar(g, name, eager, d, map)
        %readScalar  One scalar slot, stored or served by a callable.
            if nargin < 4, d = []; end
            if nargin < 5, map = []; end
            H5 = mestra.internal.H5;
            rec = mestra.Dataset.emptyScalar();
            rec(1).name = name;
            rec(1).dims = {'row'};
            path = ['/scalars/' name];
            if strcmp(H5.childType(g, name), 'group')
                oid = H5G.open(g, name);
                R = @mestra.internal.Reader;
                R().noteLinks(d, oid, path);
                R().inspectMembers(d, oid, path, map);
                R().checkAttrs(d, oid, path);
                rec(1).units = R().str(oid, 'units');
                rec(1).source = R().str(oid, 'source');
                rec(1).output = R().str(oid, 'output');
                rec(1).statistic = R().str(oid, 'statistic');
                rec(1).of = R().str(oid, 'of');
                rec(1).quantile = R().num(oid, 'quantile');
                rec(1).level = R().num(oid, 'level');
                rec(1).method = R().str(oid, 'method');
                rec(1).values = [];
                rec(1).dtype = '';
                rec(1).dims = {};
                mestra.internal.Reader.checkSlotKind(rec(1).source, true, ...
                                                     d, path);
                H5G.close(oid);
                return
            end
            did = H5D.open(g, name);
            info = H5.dsetInfo(did);
            R = @mestra.internal.Reader;
            R().checkAttrs(d, did, path);
            rec(1).units = R().str(did, 'units');
            rec(1).source = R().str(did, 'source');
            rec(1).output = R().str(did, 'output');
            rec(1).statistic = R().str(did, 'statistic');
            rec(1).of = R().str(did, 'of');
            rec(1).quantile = R().num(did, 'quantile');
            rec(1).level = R().num(did, 'level');
            rec(1).method = R().str(did, 'method');
            rec(1).dtype = info.type;
            rec(1).chunk = info.chunk;
            rec(1).filters = info.filters;
            mestra.internal.Reader.checkSlotKind(rec(1).source, false, d, path);
            mestra.internal.Reader.inspect(did, info, d, path, map);
            if eager
                values = mestra.internal.Reader.readValues(did, info, d, path);
                rec(1).values = reshape(values, 1, []);
            else
                rec(1).values = [];
            end
            H5D.close(did);
        end

        function rec = readSupport(g, name, eager, map, d)
        %readSupport  One support, its cells and its arrays.
            if nargin < 4, map = []; end
            if nargin < 5, d = []; end
            H5 = mestra.internal.H5;
            if nargin < 4, map = []; end
            sid = H5.openGroup(g, name);
            rec = mestra.Dataset.emptySupport();
            rec(1).name = name;
            path = ['/supports/' name];
            mestra.internal.Reader.noteLinks(d, sid, path);
            mestra.internal.Reader.checkAttrs(d, sid, path);
            rec(1).kind = mestra.internal.Reader.str(sid, 'kind');
            rec(1).nNodes = mestra.internal.Reader.num(sid, 'n_nodes');
            rec(1).nCells = mestra.internal.Reader.num(sid, 'n_cells');
            rec(1).supportId = mestra.internal.Reader.str(sid, 'support_id');
            if isempty(rec(1).nNodes), rec(1).nNodes = 0; end
            if isempty(rec(1).nCells), rec(1).nCells = 0; end
            rec(1).cellTypes = [];
            rec(1).cellOffsets = [];
            rec(1).cellConnectivity = [];
            rec(1).coordinates = [];
            rec(1).nodeArrays = mestra.Dataset.emptySlot();
            rec(1).cellArrays = mestra.Dataset.emptySlot();
            rec(1).groupsPresent = {};
            for g2 = {'node_arrays', 'cell_arrays'}
                if H5.exists(sid, g2{1})
                    rec(1).groupsPresent{end + 1} = g2{1};
                end
            end

            plain = {'cell_types', 'cellTypes', 'cell'; ...
                     'cell_offsets', 'cellOffsets', 'cell_plus_one'; ...
                     'cell_connectivity', 'cellConnectivity', 'index'};
            for i = 1:size(plain, 1)
                if mestra.internal.Reader.hasKind(sid, plain{i, 1}, 'dataset')
                    did = H5D.open(sid, plain{i, 1});
                    info = H5.dsetInfo(did);
                    mestra.internal.Reader.inspect(did, info, d, ...
                        [path '/' plain{i, 1}], map, plain(i, 3));
                    rec(1).(plain{i, 2}) = ...
                        reshape(H5.readData(did, info), 1, []);
                    H5D.close(did);
                end
            end
            if any(strcmp(H5.childType(sid, 'coordinates'), ...
                          {'group', 'dataset'}))
                rec(1).coordinates = mestra.internal.Reader.readSlot( ...
                    sid, 'coordinates', name, 'node', eager, map, d, ...
                    [path '/coordinates']);
            end
            arrays = {'node_arrays', 'node'; 'cell_arrays', 'cell'};
            for a = 1:size(arrays, 1)
                if ~mestra.internal.Reader.hasGroup(sid, arrays{a, 1})
                    continue
                end
                ag = H5.openGroup(sid, arrays{a, 1});
                found = {};
                for nm = H5.children(ag)
                    if ~any(strcmp(H5.childType(ag, nm{1}), ...
                                   {'group', 'dataset'}))
                        continue
                    end
                    slot = mestra.internal.Reader.readSlot(ag, nm{1}, ...
                        name, arrays{a, 2}, eager, map, d, ...
                        [path '/' arrays{a, 1} '/' nm{1}]);
                    found{end + 1} = slot; %#ok<AGROW>
                end
                if strcmp(arrays{a, 2}, 'node')
                    rec(1).nodeArrays = mestra.internal.Reader.join( ...
                        rec(1).nodeArrays, found);
                else
                    rec(1).cellArrays = mestra.internal.Reader.join( ...
                        rec(1).cellArrays, found);
                end
                H5G.close(ag);
            end
            % A dataset this version does not know, inside a support
            % group, is a public object, so the byte-level rules are
            % checked on it as on any other (section 14); it is not
            % read, and a rewrite lists it as lossy.
            known = {'coordinates', 'node_arrays', 'cell_arrays', ...
                     'cell_types', 'cell_offsets', 'cell_connectivity', ...
                     'node', 'cell', 'cell_plus_one', 'index', 'row'};
            rec(1).extraGroups = struct('name', {}, 'tree', {});
            for nm = H5.children(sid)
                if ismember(nm{1}, known), continue, end
                if strcmp(H5.childType(sid, nm{1}), 'group')
                    % A group this version does not know: copied whole,
                    % so that a rewrite carries it (section 28).
                    eg = H5G.open(sid, nm{1});
                    tree = H5.captureTree(eg, map);
                    H5G.close(eg);
                    rec(1).extraGroups(end + 1) = ...
                        struct('name', nm{1}, 'tree', tree);
                    continue
                end
                if ~strcmp(H5.childType(sid, nm{1}), 'dataset')
                    continue
                end
                did = H5D.open(sid, nm{1});
                info = H5.dsetInfo(did);
                if ~info.isScale
                    mestra.internal.Reader.inspect(did, info, d, ...
                        [path '/' nm{1}], map);
                end
                H5D.close(did);
            end
            H5G.close(sid);
        end

        function rec = readSlot(g, name, supportName, location, eager, ...
                                map, d, path)
            if nargin < 6, map = []; end
            if nargin < 7, d = []; end
            if nargin < 8, path = ['/supports/' supportName '/' name]; end
        %readSlot  One array slot, stored or served by a callable.
            H5 = mestra.internal.H5;
            R = @mestra.internal.Reader;
            rec = mestra.Dataset.emptySlot();
            rec(1).name = name;
            rec(1).location = location;
            rec(1).support = supportName;
            isGroup = strcmp(H5.childType(g, name), 'group');
            if isGroup
                oid = H5G.open(g, name);
                R().noteLinks(d, oid, path);
                R().inspectMembers(d, oid, path, map);
            else
                oid = H5D.open(g, name);
            end
            R().checkAttrs(d, oid, path);
            rec(1).role = R().str(oid, 'role');
            rec(1).varies = R().str(oid, 'varies');
            rec(1).units = R().str(oid, 'units');
            rec(1).source = R().str(oid, 'source');
            rec(1).output = R().str(oid, 'output');
            rec(1).statistic = R().str(oid, 'statistic');
            rec(1).of = R().str(oid, 'of');
            rec(1).category = R().str(oid, 'category');
            rec(1).derivedFrom = R().str(oid, 'derived_from');
            rec(1).recipe = R().str(oid, 'recipe');
            rec(1).reference = R().str(oid, 'reference');
            rec(1).quantile = R().num(oid, 'quantile');
            rec(1).level = R().num(oid, 'level');
            rec(1).method = R().str(oid, 'method');
            rec(1).components = R().num(oid, 'components');
            rec(1).recomputed = R().num(oid, 'recomputed');
            if ~isempty(rec(1).recomputed)
                rec(1).recomputed = rec(1).recomputed ~= 0;
            end
            mestra.internal.Reader.checkSlotKind(rec(1).source, isGroup, ...
                                                 d, path);
            if isGroup
                rec(1).values = [];
                rec(1).dims = {};
                rec(1).shape = [];
                rec(1).dtype = '';
                rec(1).chunk = [];
                rec(1).filters = zeros(0, 2);
                H5G.close(oid);
                return
            end
            info = H5.dsetInfo(oid);
            names = mestra.internal.Reader.axisNames(oid, numel(info.dims), ...
                                                     map);
            rec(1).dims = fliplr(names);
            % The extents in the file's own order, so that a lazy read
            % can still say what shape a slot has (section 29).
            rec(1).shape = reshape(double(info.dims), 1, []);
            rec(1).dtype = info.type;
            rec(1).chunk = info.chunk;
            rec(1).filters = info.filters;
            R().inspect(oid, info, d, path, map);
            if eager
                [values, ok] = R().readValues(oid, info, d, path);
                if ~ok
                    rec(1).values = [];
                else
                    % A zero-length extent, which a zero-row file or a
                    % support no row is on gives a row-varying slot, is
                    % kept: the empty array still has its shape.
                    rec(1).values = reshape(values, [fliplr(info.dims) 1 1]);
                end
            else
                rec(1).values = [];
            end
            H5D.close(oid);
        end

        function [rec, limits] = readCallable(g, id, d, eager, map)
        %readCallable  One callable: its type, its dictionary and, when
        %   the type is registered, the object itself.  A reader that
        %   does not know the type keeps the dictionary and must not
        %   interpret it (section 25).
        %
        %   With `eager` false the dictionary is walked and not read.
        %   Section 7 of docs/api-conventions.md: an open never reads a
        %   dataset inside a callable's dictionary.  The walk still
        %   decides what the structure decides, so a dictionary nested
        %   past the cap is E41 from the open as it is from the read,
        %   which is what section 30 asks of the hostile subset; what
        %   the bytes decide waits for the read.  The dictionary
        %   itself is then dropped rather than handed back with its
        %   arrays missing, and with it the object built from it:
        %   `id`, `type` and `repr` are attributes, so a metadata open
        %   still names every callable and mestra.info still prints
        %   one.
            if nargin < 3, d = []; end
            if nargin < 4, eager = true; end
            gid = mestra.internal.H5.openGroup(g, id);
            mestra.internal.Reader.checkAttrs(d, gid, ['/callables/' id], ...
                                              {'type', 'repr'});
            rec = mestra.Dataset.emptyCallable();
            rec(1).id = id;
            rec(1).type = mestra.internal.Reader.str(gid, 'type');
            rec(1).repr = mestra.internal.Reader.str(gid, 'repr');
            rec(1).obj = [];
            if nargin < 5, map = []; end
            [dict, problems] = mestra.internal.Codec.read(gid, true, 0, ...
                                                          eager, map);
            limits = {};
            for i = 1:numel(problems)
                if numel(problems{i}) > 4 && strcmp(problems{i}(1:4), 'U03 ')
                    limits{end + 1} = {'E41', problems{i}(5:end)}; %#ok<AGROW>
                elseif numel(problems{i}) > 4 && ...
                        any(strcmp(problems{i}(1:4), {'E40 ', 'E25 '}))
                    limits{end + 1} = {problems{i}(1:3), ...
                                       problems{i}(5:end)}; %#ok<AGROW>
                end
            end
            if ~eager
                rec(1).dict = containers.Map('KeyType', 'char', ...
                                             'ValueType', 'any');
                H5G.close(gid);
                return
            end
            rec(1).dict = dict;
            if ~isempty(rec(1).type) && mestra.Registry.isKnown(rec(1).type)
                try
                    rec(1).obj = mestra.Registry.create(rec(1).type, ...
                                                        rec(1).dict);
                catch
                    rec(1).obj = [];
                end
            end
            H5G.close(gid);
        end

        function v = str(oid, name)
        %str  A scalar string attribute, or '' when there is not one.
        %   An attribute whose dataspace is not scalar is not a value
        %   this format defines (section 18), so it is passed over
        %   rather than handed on as an array.  checkAttrs is what
        %   records that it was passed over, under the rule it breaks.
            v = '';
            [value, ok] = mestra.internal.H5.scalarAttr(oid, name);
            if ok && ischar(value) && (isempty(value) || isrow(value))
                v = value;
            end
        end

        function v = num(oid, name)
        %num  A scalar numeric attribute, or [] when there is not one.
            v = [];
            [value, ok] = mestra.internal.H5.scalarAttr(oid, name);
            if ok && isnumeric(value) && isscalar(value)
                v = double(value);
            end
        end

        function checkAttrs(d, oid, path, only)
        %checkAttrs  E19 and E26 over one object's attributes.
        %   The rules are section 18's and live in
        %   mestra.internal.Attrs, so that the reader and the
        %   validator cannot disagree about them.  They are decided
        %   from attributes alone, which is why mestra.open applies
        %   them as mestra.read does without reading an array
        %   (section 29).  A W11 finding is an attribute this version
        %   does not know and cannot use: the validator reports it and
        %   the file is still accepted, so the reader passes over it
        %   without refusing the file.
            if isempty(d), return, end
            if nargin > 3
                found = mestra.internal.Attrs.findings(oid, only);
            else
                found = mestra.internal.Attrs.findings(oid);
            end
            for i = 1:numel(found)
                if strcmp(found(i).id, 'W11'), continue, end
                mestra.internal.Reader.note(d, path, found(i).id, ...
                                            found(i).message);
            end
        end

        function checkSlotKind(source, isGroup, d, path)
        %checkSlotKind  E30: a slot with `source = data` stored as a
        %   group, or a slot served by a callable stored as a dataset.
        %
        %   Section 19 makes the two kinds of slot distinguishable
        %   without reading any data, and a reader that takes one for
        %   the other reads a callable slot as an empty array or an
        %   array as a slot with no data.  Neither is something to
        %   hand back, so a strict read refuses the file.
            if isempty(source), return, end
            servedByCallable = numel(source) > 9 && ...
                               strncmp(source, 'callable:', 9);
            % A source that is neither word is E36 and says nothing
            % about which kind of slot was meant.
            if isGroup && strcmp(source, 'data')
                mestra.internal.Reader.note(d, path, 'E30', sprintf( ...
                    ['source is "%s" and the slot is a group; a slot ' ...
                     'holding data is a dataset'], source));
            elseif ~isGroup && servedByCallable
                mestra.internal.Reader.note(d, path, 'E30', ...
                    ['source names a callable and the slot is a ' ...
                     'dataset; a slot served by a callable is a group ' ...
                     'with no data']);
            end
        end

        function checkRowsPerSupport(d)
        %checkRowsPerSupport  E16 in an unaligned file: a row-varying
        %   slot on a support has one entry per row that references it
        %   (section 22), which only /row_support says.  A metadata
        %   open reads that column for this (conventions section 7),
        %   so the open and the read refuse the same file.
            if numel(d.supports) < 2 || isempty(d.rowSupport), return, end
            order = mestra.internal.H5.sortByBytes({d.supports.name});
            for i = 1:numel(d.supports)
                s = d.supports(i);
                want = sum(double(d.rowSupport) == ...
                           find(strcmp(order, s.name), 1) - 1);
                slots = [s.nodeArrays s.cellArrays];
                if ~isempty(s.coordinates), slots = [slots s.coordinates]; end
                for j = 1:numel(slots)
                    a = slots(j);
                    if ~strcmp(a.varies, 'row') || ~strcmp(a.source, 'data') || ...
                            isempty(a.shape) || a.shape(1) == want
                        continue
                    end
                    if strcmp(a.name, 'coordinates')
                        where = ['/supports/' s.name '/coordinates'];
                    else
                        where = ['/supports/' s.name '/' a.location ...
                                 '_arrays/' a.name];
                    end
                    mestra.internal.Reader.note(d, where, 'E16', sprintf( ...
                        ['%d entries where %d rows of this file are on ' ...
                         'the support'], a.shape(1), want));
                end
            end
        end

        function inspectMembers(d, gid, path, map)
        %inspectMembers  The byte-level rules on a dataset found inside
        %   a slot a callable serves, which holds none by right but is
        %   a public object if it does.
            if isempty(d), return, end
            for name = mestra.internal.H5.children(gid)
                if ~strcmp(mestra.internal.H5.childType(gid, name{1}), ...
                           'dataset')
                    continue
                end
                did = H5D.open(gid, name{1});
                info = mestra.internal.H5.dsetInfo(did);
                if ~info.isScale
                    mestra.internal.Reader.inspect(did, info, d, ...
                        [path '/' name{1}], map);
                end
                H5D.close(did);
            end
        end

        function noteLinks(d, gid, path)
        %noteLinks  E40 for each member of a public group that is a
        %   soft or an external link.  Used where the reader does not
        %   otherwise walk the members: a support group's own, a slot
        %   a callable serves, and /notes.  None is followed.
            if isempty(d), return, end
            for name = mestra.internal.H5.children(gid)
                kind = mestra.internal.H5.childType(gid, name{1});
                if any(strcmp(kind, {'soft', 'external'}))
                    mestra.internal.Reader.note(d, [path '/' name{1}], ...
                        'E40', sprintf('a %s link, not followed', kind));
                end
            end
        end

        function note(d, path, id, why)
        %note  One line for something the reader passed over.
        %   With no dataset to record on, as when a single support is
        %   read on its own, there is nothing to record and the caller
        %   is asking only for the values.
            if isempty(d), return, end
            d.skipped{end + 1} = sprintf('%s %s: %s', id, path, why);
        end
    end
end
