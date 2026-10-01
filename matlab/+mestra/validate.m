function out = validate(path)
%MESTRA.VALIDATE  Check a file against the rules of section 14.
%
%   OUT = MESTRA.VALIDATE(PATH) returns a struct with
%
%       errors     a sorted cell array of the error identifiers the
%                  file produces, with no duplicates
%       warnings   the same for the warnings
%       findings   a struct array with one entry per finding: the rule
%                  id, the HDF5 path it was found at, and one line of
%                  plain language
%       valid      true when `errors` is empty
%
%   The identifiers are exactly those of specification section 14.  A
%   retired identifier is never emitted: E07 and W09 were retired on
%   2026-09-20 and nothing here produces them.
%
%   A file whose `format` names another major version produces E01 and
%   nothing else: section 28 says a reader must refuse it outright and
%   must not try to read it partially.
%
%   W10 uses a small units parser (mestra.internal.Units) that checks
%   the UDUNITS grammar and not the names, because shipping a unit
%   database is not what the rule asks for.
%
%   A file is untrusted input, so the pass is per object: an object
%   that will not open or will not read stops that object and nothing
%   else, and the pass carries on to the end of the file.  Two rules
%   of section 14 cover what is found that way:
%
%       E40   a link in the public tree that is not a hard link: a
%             soft link, whether it resolves, dangles or loops, or an
%             external link.  It is reported and never followed
%       E41   an object the reader could not read, with its path: a
%             malformed header or attribute, nesting deeper than the
%             cap of mestra.limits, or a dataset above the stated
%             maximum element count
%
%   A file that is not HDF5 at all, or that is truncated, raises
%   mestra:reader rather than a library error.
%
%   Example
%
%       r = mestra.validate('case.mes');
%       if ~r.valid
%           for f = r.findings
%               fprintf('%s %s %s\n', f.id, f.path, f.message);
%           end
%       end
%
%   See also mestra.read, mestra.write, mestra.Dataset.

    rep = mestra.internal.Report();
    fid = mestra.internal.Reader.openFile(path);
    closeFile = onCleanup(@() H5F.close(fid)); %#ok<NASGU>
    % One pass over one file: every object's attribute names are
    % listed once and answered from that list afterwards.
    closePass = mestra.internal.H5.pass(); %#ok<NASGU>
    root = H5G.open(fid, '/');
    closeRoot = onCleanup(@() H5G.close(root)); %#ok<NASGU>

    ctx = struct();
    ctx.rep = rep;
    ctx.fid = fid;
    ctx.root = root;
    % Section 21: scale names come from a map built by a
    % bounded walk of our own and never from a path lookup.
    ctx.scales = mestra.internal.H5.scaleMap(fid);

    if ~checkFormat(ctx)
        out = rep.result();
        return
    end
    ctx = guard(ctx, '/', @() gather(ctx), ctx);
    % Each pass is separate, so that one that cannot finish leaves the
    % others to run.  That is what makes a finding late in a file
    % reachable when something early in it will not read.
    guard(ctx, '/', @() checkRoot(ctx));
    guard(ctx, '/', @() checkDimensionScales(ctx));
    guard(ctx, '/categories', @() checkCategories(ctx));
    guard(ctx, '/keys', @() checkKeys(ctx));
    guard(ctx, '/scalars', @() checkScalars(ctx));
    guard(ctx, '/row_support', @() checkRowSupport(ctx));
    guard(ctx, '/supports', @() checkSupports(ctx));
    guard(ctx, '/callables', @() checkCallables(ctx));
    guard(ctx, '/', @() checkUnknown(ctx));
    guard(ctx, '/private', @() checkPrivate(ctx));
    out = rep.result();
end

% ============================================== per-object recovery

function out = guard(ctx, path, fn, fallback)
%guard  Run one check; a failure is a finding and not the end.
%   Whatever went wrong, the object is E41 and carries its path, so a
%   caller knows which one was passed over and the pass continues.
    if nargin < 4, fallback = []; end
    try
        if nargout > 0
            out = fn();
        else
            fn();
            out = [];
        end
    catch err
        out = fallback;
        ctx.rep.add('E41', path, '%s', ...
                    regexprep(strtrim(err.message), '\s+', ' '));
    end
end

function tf = followable(ctx, gid, name, path)
%followable  True for a hard link; anything else is E40 or E41.
    kind = mestra.internal.H5.childType(gid, name);
    tf = any(strcmp(kind, {'group', 'dataset', 'other'}));
    if ~tf
        switch kind
            case 'soft'
                ctx.rep.add('E40', path, ...
                    ['a soft link, which this format does not define ' ...
                     'and this reader does not follow']);
            case 'external'
                ctx.rep.add('E40', path, ...
                    ['an external link, which names another file; ' ...
                     'this reader never opens one']);
            otherwise
                ctx.rep.add('E41', path, 'an object that would not open');
        end
    end
end

function linksIn(ctx, gid, path)
%linksIn  E40 for every member of a public group whose members are not
%   otherwise visited: /notes, which holds attributes, and a slot a
%   callable serves, which is a group holding nothing.  A link there
%   is in the public tree as much as one under /keys.
    for name = mestra.internal.H5.children(gid)
        followable(ctx, gid, name{1}, [path '/' name{1}]);
    end
end

% ===================================================== the vocabulary

function names = keyRoles()
    names = {'design', 'condition', 'time', 'categorical', 'group', ...
             'split', 'id', 'status'};
end

function names = arrayRoles()
    names = {'coordinates', 'field', 'label', 'weight', 'normal', 'derived'};
end

function t = cellTypeTable()
%cellTypeTable  The codes of section 20 and the nodes each takes.
    t = containers.Map({1, 3, 5, 7, 9, 10, 12, 13, 14, 21, 22, 23, 24, ...
                        25, 26, 27}, ...
                       {1, 2, 3, -1, 4, 4, 8, 6, 5, 3, 6, 8, 10, 20, 15, 13});
end

% ===================================================== opening checks

function ok = checkFormat(ctx)
    H5 = mestra.internal.H5;
    ok = false;
    if ~H5.hasAttr(ctx.root, 'format')
        ctx.rep.add('E01', '/', 'the root has no format attribute');
        ctx.rep.add('E17', '/', 'format is missing');
        return
    end
    [value, ok] = H5.scalarAttr(ctx.root, 'format');
    if ~ok || ~ischar(value), value = ''; end
    major = mestra.internal.Reader.majorVersion(value);
    if isnan(major) || major ~= 0
        ctx.rep.add('E01', '/', ...
            'format is "%s"; this reader accepts "mestra/0" only', value);
        return
    end
    ok = true;
end

function ctx = gather(ctx)
%gather  Read once what every check needs.
    H5 = mestra.internal.H5;
    ctx.aligned = [];
    v = mestra.internal.Reader.num(ctx.root, 'aligned');
    if ~isempty(v), ctx.aligned = v; end
    ctx.generalisationGroup = ...
        mestra.internal.Reader.str(ctx.root, 'generalisation_group');

    ctx.rowCount = 0;
    ctx.rowUnlimited = true;
    if mestra.internal.Reader.hasKind(ctx.fid, 'row', 'dataset')
        did = H5D.open(ctx.fid, 'row');
        info = H5.dsetInfo(did);
        H5D.close(did);
        if ~isempty(info.dims), ctx.rowCount = info.dims(1); end
        ctx.rowUnlimited = ~isempty(info.maxdims) && info.maxdims(1) < 0;
    end

    ctx.categories = containers.Map('KeyType', 'char', 'ValueType', 'any');
    ctx.categoryNames = {};
    if mestra.internal.Reader.hasGroup(ctx.fid, 'categories')
        g = H5.openGroup(ctx.fid, 'categories');
        for name = H5.children(g)
            ctx.categoryNames{end + 1} = name{1};
            ctx.categories(name{1}) = {};
            if ~strcmp(H5.childType(g, name{1}), 'dataset'), continue, end
            did = H5D.open(g, name{1});
            try
                info = H5.dsetInfo(did);
                if strcmp(info.type, 'string')
                    ctx.categories(name{1}) = H5.readData(did, info);
                end
            catch
                % The table cannot be read; checkCategories says so and
                % every value checked against it is simply not checked.
            end
            H5D.close(did);
        end
        H5G.close(g);
    end

    ctx.keys = struct('name', {}, 'role', {}, 'values', {}, 'type', {}, ...
                      'category', {}, 'trajectoryGroup', {});
    if mestra.internal.Reader.hasGroup(ctx.fid, 'keys')
        g = H5.openGroup(ctx.fid, 'keys');
        for name = H5.children(g)
            rec.name = name{1};
            rec.role = '';
            rec.values = [];
            rec.type = '';
            rec.category = '';
            rec.trajectoryGroup = '';
            if strcmp(H5.childType(g, name{1}), 'dataset')
                did = H5D.open(g, name{1});
                try
                    info = H5.dsetInfo(did);
                    rec.type = info.type;
                    try
                        rec.values = reshape(H5.readData(did, info), 1, []);
                    catch
                        % The values are unreadable; checkKeys reports
                        % it and every value-based rule is skipped for
                        % this key alone.
                    end
                    rec.role = strAttr(did, 'role');
                    rec.category = strAttr(did, 'category');
                    rec.trajectoryGroup = strAttr(did, 'trajectory_group');
                catch
                end
                H5D.close(did);
            end
            ctx.keys(end + 1) = rec;
        end
        H5G.close(g);
    end

    % A support is a group reached by a hard link; anything else under
    % /supports is reported where it is met (E40, E41) and is not a
    % support, so it neither counts for E37 nor takes a place in the
    % support order of section 22.
    ctx.supportNames = {};
    ctx.supportMembers = {};
    if mestra.internal.Reader.hasGroup(ctx.fid, 'supports')
        g = H5.openGroup(ctx.fid, 'supports');
        ctx.supportMembers = H5.children(g);
        isGroup = cellfun(@(n) strcmp(H5.childType(g, n), 'group'), ...
                          ctx.supportMembers);
        ctx.supportNames = ctx.supportMembers(isGroup);
        H5G.close(g);
    end

    ctx.rowSupport = [];
    ctx.hasRowSupport = H5.exists(ctx.fid, '/row_support');
    if mestra.internal.Reader.hasKind(ctx.fid, 'row_support', 'dataset')
        did = H5D.open(ctx.fid, 'row_support');
        info = H5.dsetInfo(did);
        try
            ctx.rowSupport = reshape(double(H5.readData(did, info)), 1, []);
        catch
        end
        H5D.close(did);
    end

    ctx.callableIds = {};
    if mestra.internal.Reader.hasGroup(ctx.fid, 'callables')
        g = H5.openGroup(ctx.fid, 'callables');
        ctx.callableIds = H5.children(g);
        H5G.close(g);
    end

    ctx.groupKeys = {};
    for i = 1:numel(ctx.keys)
        if strcmp(ctx.keys(i).role, 'group')
            ctx.groupKeys{end + 1} = ctx.keys(i).name;
        end
    end
end

% ========================================================= root rules

function checkRoot(ctx)
    H5 = mestra.internal.H5;
    rep = ctx.rep;
    for name = {'format', 'writer', 'created'}
        if ~H5.hasAttr(ctx.root, name{1})
            rep.add('E17', '/', '%s is missing', name{1});
        end
    end
    checkAttrEncodings(ctx, ctx.root, '/');
    if ~H5.hasAttr(ctx.root, 'aligned')
        rep.add('E39', '/', 'aligned is required on the root group');
    end
    if ~isempty(ctx.groupKeys) && ...
       ~H5.hasAttr(ctx.root, 'generalisation_group')
        rep.add('E39', '/', ...
            ['the file declares a group key, so generalisation_group ' ...
             'is required']);
    end
    if H5.hasAttr(ctx.root, 'created')
        v = mestra.internal.Reader.str(ctx.root, 'created');
        if ~isempty(v) && ~mestra.internal.Text.iso8601Utc(v)
            rep.add('W14', '/', ...
                'created "%s" is not an ISO 8601 UTC timestamp', v);
        end
    end

    n = numel(ctx.supportNames);
    if n > 1
        rep.add('W05', '/supports', ...
            ['%d supports: index-aligned operations are not ' ...
             'available'], n);
    end
    if ~isempty(ctx.aligned)
        if ctx.aligned ~= 0 && n > 1
            rep.add('E37', '/', ...
                'aligned is true and the file declares %d supports', n);
        elseif ctx.aligned == 0 && n <= 1
            rep.add('E37', '/', ...
                'aligned is false and the file declares %d support(s)', n);
        end
        if ctx.aligned ~= 0 && ctx.hasRowSupport
            rep.add('E28', '/row_support', ...
                '/row_support is present in an aligned file');
        elseif ctx.aligned == 0 && ~ctx.hasRowSupport
            rep.add('E28', '/', ...
                '/row_support is missing from an unaligned file');
        end
    end
    if ~ctx.rowUnlimited
        rep.add('E27', '/row', 'row is not an unlimited dimension');
    end
end

% =================================================== category tables

function checkCategories(ctx)
    H5 = mestra.internal.H5;
    if ~mestra.internal.Reader.hasGroup(ctx.fid, 'categories'), return, end
    g = H5.openGroup(ctx.fid, 'categories');
    closer = onCleanup(@() H5G.close(g)); %#ok<NASGU>
    for name = H5.children(g)
        path = ['/categories/' name{1}];
        checkName(ctx, name{1}, path);
        if ~followable(ctx, g, name{1}, path), continue, end
        if ~strcmp(H5.childType(g, name{1}), 'dataset')
            ctx.rep.add('E20', path, 'a category table must be a dataset');
            ctx.rep.add('E41', path, ...
                'a category table that is not a dataset cannot be read');
            continue
        end
        guard(ctx, path, @() checkOneCategory(ctx, g, name{1}, path));
    end
end

function checkOneCategory(ctx, g, name, path)
    H5 = mestra.internal.H5;
    did = H5D.open(g, name);
    closer = onCleanup(@() H5D.close(did)); %#ok<NASGU>
    info = H5.dsetInfo(did);
    if ~strcmp(info.type, 'string')
        ctx.rep.add('E20', path, ...
            'a category table must be a fixed-length UTF-8 string');
    else
        checkStringDataset(ctx, did, info, path);
    end
    checkAttrEncodings(ctx, did, path);
    checkScales(ctx, did, info, path, {['category_' name]});
end

function checkStringDataset(ctx, did, info, path)
%checkStringDataset  E26 and W13 over the stored bytes.
%   MATLAB's HDF5 interface decodes a fixed-length string to text
%   before this package sees it, so the stored bytes are not always
%   recoverable.  What can still be decided, and how:
%
%     the characters came back one per stored byte  everything is
%       decidable: the UTF-8 check and the NUL check run on the real
%       bytes and E26 means what it says;
%     the count is right and characters above 255 came back  MATLAB
%       substituted one replacement character per byte it could not
%       decode, so the bytes were not valid UTF-8: that is E26;
%     the count changed  MATLAB decoded valid multi-byte UTF-8 and the
%       bytes are gone. Nothing about them is decidable, so this is
%       E41 and not a guess at E26.
    H5 = mestra.internal.H5;
    try
        out = H5.decodeStrings(did, info);
    catch err
        ctx.rep.add('E41', path, ...
            'the strings could not be read: %s', ...
            regexprep(strtrim(err.message), '\s+', ' '));
        return
    end
    switch out.verdict
        case 'replaced'
            ctx.rep.add('E26', path, ...
                ['a stored byte is not valid UTF-8; this binding ' ...
                 'returned a replacement character for it']);
            return
        case 'decoded'
            ctx.rep.add('E41', path, ...
                ['this binding decoded the strings to text, so the ' ...
                 'stored bytes cannot be checked']);
            return
    end
    raw = out.bytes;
    longest = 0;
    for i = 1:size(raw, 2)
        [ok, why] = mestra.internal.Text.checkStringBytes(raw(:, i)');
        if ~ok
            ctx.rep.add('E26', path, 'entry %d has %s', i - 1, why);
        end
        last = find(raw(:, i) ~= 0, 1, 'last');
        if isempty(last), last = 0; end
        longest = max(longest, last);
    end
    if info.strSize > max(longest, 1)
        ctx.rep.add('W13', path, ...
            'the size is %d bytes where %d would do', ...
            info.strSize, max(longest, 1));
    end
end

% ============================================================== keys

function checkKeys(ctx)
    H5 = mestra.internal.H5;
    rep = ctx.rep;
    if ~mestra.internal.Reader.hasGroup(ctx.fid, 'keys'), return, end
    g = H5.openGroup(ctx.fid, 'keys');
    closer = onCleanup(@() H5G.close(g)); %#ok<NASGU>

    counts = containers.Map(keyRoles(), num2cell(zeros(1, numel(keyRoles()))));
    for name = H5.children(g)
        path = ['/keys/' name{1}];
        checkName(ctx, name{1}, path);
        if ~followable(ctx, g, name{1}, path), continue, end
        if ~strcmp(H5.childType(g, name{1}), 'dataset')
            rep.add('E41', path, ...
                'a key is a dataset, and this is not one, so there is no key column here to read');
            continue
        end
        role = guard(ctx, path, @() checkOneKey(ctx, g, name{1}, path), '');
        if ~isempty(role) && ismember(role, keyRoles())
            counts(role) = counts(role) + 1;
        end
    end

    for role = {'time', 'split', 'id', 'status'}
        if counts(role{1}) > 1
            rep.add('E03', '/keys', ...
                '%d keys have the role %s where at most one may', ...
                counts(role{1}), role{1});
        end
    end

    guard(ctx, '/keys', @() checkSplitLeak(ctx));
    guard(ctx, '/keys', @() checkTrajectories(ctx));
    guard(ctx, '/keys', @() checkStatus(ctx));
    guard(ctx, '/categories', @() checkUnusedCategories(ctx));
end

function role = checkOneKey(ctx, g, name, path)
    H5 = mestra.internal.H5;
    rep = ctx.rep;
    did = H5.openDataset(g, name);
    closer = onCleanup(@() H5D.close(did)); %#ok<NASGU>
    info = H5.dsetInfo(did);
    checkAttrEncodings(ctx, did, path);
    checkKnownAttrs(ctx, did, path, ...
        {'role', 'units', 'lower', 'upper', 'category', ...
         'trajectory_group', 'parent'});
    checkScales(ctx, did, info, path);
    checkChunking(ctx, did, info, path, ctx.rowCount);

    role = strAttr(did, 'role');
    if isempty(role) || ~ismember(role, keyRoles())
        rep.add('E02', path, 'the role is "%s"', role);
    end

    units = strAttr(did, 'units');
    if ismember(role, {'design', 'condition', 'time'})
        if ~H5.hasAttr(did, 'units')
            rep.add('E39', path, 'a %s key needs units', role);
        end
    end
    if ~isempty(units) && ~mestra.internal.Units.parses(units)
        rep.add('W10', path, 'the units "%s" do not parse', units);
    end
    if ismember(role, {'categorical', 'group', 'split', 'status'})
        if ~H5.hasAttr(did, 'category')
            rep.add('E39', path, 'a %s key needs a category table', role);
        end
    end
    if strcmp(role, 'time') && ~isempty(ctx.groupKeys) && ...
       ~H5.hasAttr(did, 'trajectory_group')
        rep.add('E39', path, ...
            'the time key needs trajectory_group when the file has groups');
    end

    checkKeyDtype(ctx, role, info.type, path);
    if numel(info.dims) ~= 1
        rep.add('E16', path, ...
            'a key has %d dimensions where it must have exactly one', ...
            numel(info.dims));
    end
    if ~isempty(info.dims) && info.dims(1) ~= ctx.rowCount
        rep.add('E16', path, ...
            'the key holds %d values where the file has %d rows', ...
            info.dims(1), ctx.rowCount);
    end

    values = [];
    try
        values = H5.readData(did, info);
    catch err
        rep.add('E41', path, 'the values would not read: %s', ...
                regexprep(strtrim(err.message), '\s+', ' '));
    end
    checkKeyValues(ctx, did, path, role, values);
    if strcmp(info.type, 'string')
        guard(ctx, path, @() checkStringDataset(ctx, did, info, path));
    end
end

function checkKeyDtype(ctx, role, type, path)
    switch role
        case {'design', 'condition', 'time'}
            allowed = {'float64'};
        case {'categorical', 'group', 'split', 'status'}
            allowed = {'int32', 'int64'};
        case 'id'
            allowed = {'int64', 'string'};
        otherwise
            return
    end
    if ~ismember(type, allowed)
        ctx.rep.add('E20', path, ...
            'a %s key is stored as %s where section 19 allows %s', ...
            role, type, strjoin(allowed, ' or '));
    end
end

function checkKeyValues(ctx, did, path, role, values)
    H5 = mestra.internal.H5;
    rep = ctx.rep;
    if isempty(values), return, end
    if ismember(role, {'categorical', 'group', 'split', 'status'})
        table = categoryEntries(ctx, strAttr(did, 'category'));
        if ~isempty(table) || H5.hasAttr(did, 'category')
            n = numel(table);
            bad = values(values < 0 | values >= n);
            if ~isempty(bad)
                rep.add('E10', path, ...
                    'the value %g is outside a table of %d entries', ...
                    double(bad(1)), n);
            end
        end
        return
    end
    if ~isnumeric(values), return, end
    lower = numAttr(did, 'lower');
    upper = numAttr(did, 'upper');
    v = double(reshape(values, 1, []));
    outside = false(1, numel(v));
    if ~isempty(lower), outside = outside | (v < lower); end
    if ~isempty(upper), outside = outside | (v > upper); end
    if any(outside)
        % Section 5 of docs/api-conventions.md: a rule that could fire
        % once per row fires once, with the count and the first three
        % rows, so that the report stays a report on a long file.
        idx = find(outside) - 1;
        rep.add('W04', path, ...
            ['%s are outside the declared bounds [%s, %s] (%s); widen ' ...
             'the bounds or leave those rows out'], ...
            mestra.internal.Report.someRows(idx, numel(v)), ...
            bound(lower), bound(upper), ...
            mestra.internal.Report.whichRows(idx));
    elseif ~isempty(lower) && ~isempty(upper) && ~isempty(values)
        finite = double(values(isfinite(double(values))));
        if isempty(finite), return, end
        observed = max(finite) - min(finite);
        declared = upper - lower;
        % The rule does not apply when the observed width
        % is zero, which covers a file with no rows, a key with one
        % distinct value, and a key with no finite value at all.
        if observed > 0 && declared > 4 * observed
            rep.add('W08', path, ...
                ['the declared width %g is more than four times the ' ...
                 'observed width %g'], declared, observed);
        end
    end
end

function checkSplitLeak(ctx)
    split = findKey(ctx, 'split', 'role');
    if isempty(split) || isempty(ctx.generalisationGroup), return, end
    unit = findKey(ctx, ctx.generalisationGroup, 'name');
    if isempty(unit) || isempty(unit.values) || isempty(split.values)
        return
    end
    if numel(unit.values) ~= numel(split.values), return, end
    units = unique(double(unit.values));
    table = categoryEntries(ctx, unit.category);
    for i = 1:numel(units)
        sides = unique(double(split.values(double(unit.values) == units(i))));
        if numel(sides) > 1
            % The message names the unit that leaked and then says why
            % it matters, which is what turns a warning into something
            % a user acts on.
            ctx.rep.add('W01', ['/keys/' split.name], ...
                ['the rows of %s %s are on both sides of the split, so ' ...
                 'this is not a generalisation test'], ...
                unit.name, categoryLabel(table, units(i)));
            return
        end
    end
end

function s = categoryLabel(table, id)
%categoryLabel  The entry a category id names, or the id itself when
%   the file carries no table for it.
    j = double(id) + 1;
    if ~isempty(table) && j >= 1 && j <= numel(table)
        s = table{j};
    else
        s = sprintf('%d', double(id));
    end
end

function checkTrajectories(ctx)
    t = findKey(ctx, 'time', 'role');
    if isempty(t) || isempty(t.values), return, end
    times = reshape(double(t.values), 1, []);
    groups = ones(1, numel(times));
    if ~isempty(t.trajectoryGroup)
        g = findKey(ctx, t.trajectoryGroup, 'name');
        if ~isempty(g) && numel(g.values) == numel(times)
            groups = reshape(double(g.values), 1, []);
        end
    end
    for v = unique(groups)
        seq = times(groups == v);
        if any(diff(seq) <= 0)
            ctx.rep.add('E09', ['/keys/' t.name], ...
                'time is not strictly increasing within a trajectory');
            return
        end
    end
end

function checkStatus(ctx)
    s = findKey(ctx, 'status', 'role');
    if isempty(s) || isempty(s.values), return, end
    table = categoryEntries(ctx, s.category);
    converged = find(strcmp(table, 'converged'), 1) - 1;
    missing = isempty(converged);
    if missing, converged = -1; end
    v = double(reshape(s.values, 1, []));
    bad = find(v ~= converged) - 1;
    if isempty(bad), return, end
    if missing
        tail = ['; this file''s status table has no entry called ' ...
                '"converged", which is the word section 3 excludes ' ...
                'rows against'];
    else
        tail = ['; a row that is not converged is excluded from ' ...
                'modelling unless it is asked for'];
    end
    ctx.rep.add('W02', ['/keys/' s.name], ...
        '%s have a status other than converged (%s)%s', ...
        mestra.internal.Report.someRows(bad, numel(v)), ...
        mestra.internal.Report.whichRows(bad), tail);
end

function checkUnusedCategories(ctx)
    for i = 1:numel(ctx.groupKeys)
        k = findKey(ctx, ctx.groupKeys{i}, 'name');
        if isempty(k) || isempty(k.category), continue, end
        table = categoryEntries(ctx, k.category);
        used = unique(double(k.values));
        for c = 0:numel(table) - 1
            if ~any(used == c)
                % W07 is about the group key, so it is reported at the
                % key, as the other three implementations report it.
                ctx.rep.add('W07', ['/keys/' k.name], ...
                    'entry %d ("%s") is used by no row', c, table{c + 1});
            end
        end
    end
end

% =========================================================== scalars

function checkScalars(ctx)
    H5 = mestra.internal.H5;
    if ~mestra.internal.Reader.hasGroup(ctx.fid, 'scalars'), return, end
    g = H5.openGroup(ctx.fid, 'scalars');
    closer = onCleanup(@() H5G.close(g)); %#ok<NASGU>
    for name = H5.children(g)
        path = ['/scalars/' name{1}];
        checkName(ctx, name{1}, path);
        if ~followable(ctx, g, name{1}, path), continue, end
        guard(ctx, path, @() checkOneScalar(ctx, g, name{1}, path));
    end
end

function checkOneScalar(ctx, g, name, path)
    H5 = mestra.internal.H5;
    kind = H5.childType(g, name);
    if strcmp(kind, 'group')
        isGroup = true;
        oid = H5.openGroup(g, name);
    elseif strcmp(kind, 'dataset')
        isGroup = false;
        oid = H5.openDataset(g, name);
    else
        ctx.rep.add('E41', path, ...
            'a scalar is a dataset or a group, and this is neither, so there is nothing here to read');
        return
    end
    if isGroup
        linksIn(ctx, oid, path);
    end
    checkAttrEncodings(ctx, oid, path);
    checkKnownAttrs(ctx, oid, path, ...
        {'units', 'source', 'output', 'statistic', 'of', 'quantile', ...
         'level', 'method'});
    if ~H5.hasAttr(oid, 'units')
        ctx.rep.add('E11', path, 'a scalar needs units');
    else
        units = strAttr(oid, 'units');
        if ~isempty(units) && ~mestra.internal.Units.parses(units)
            ctx.rep.add('W10', path, ...
                'the units "%s" do not parse', units);
        end
    end
    checkSource(ctx, oid, path, isGroup);
    checkStatistic(ctx, oid, path);
    if isGroup
        H5G.close(oid);
        return
    end
    info = H5.dsetInfo(oid);
    if ~strcmp(info.type, 'float64')
        ctx.rep.add('E20', path, ...
            'a scalar is stored as %s where float64 is required', ...
            info.type);
    end
    if numel(info.dims) ~= 1
        ctx.rep.add('E16', path, ...
            'a scalar has %d dimensions where it must have exactly one', ...
            numel(info.dims));
    end
    if ~isempty(info.dims) && info.dims(1) ~= ctx.rowCount
        ctx.rep.add('E16', path, ...
            'the scalar holds %d values where the file has %d rows', ...
            info.dims(1), ctx.rowCount);
    end
    checkScales(ctx, oid, info, path);
    checkChunking(ctx, oid, info, path, ctx.rowCount);
    try
        values = H5.readData(oid, info);
        if isnumeric(values)
            v = double(reshape(values, 1, []));
            bad = find(~isfinite(v)) - 1;
            if ~isempty(bad)
                ctx.rep.add('W03', path, ...
                    '%s hold a non-finite value (%s)', ...
                    mestra.internal.Report.someRows(bad, numel(v)), ...
                    mestra.internal.Report.whichRows(bad));
            end
            if isBand(oid)
                neg = find(v < 0) - 1;
                if ~isempty(neg)
                    ctx.rep.add('W16', path, ...
                        ['%s hold a value below zero in a band, which ' ...
                         'is a half-width and is never negative (%s)'], ...
                        mestra.internal.Report.someRows(neg, numel(v)), ...
                        mestra.internal.Report.whichRows(neg));
                end
            end
        end
    catch err
        ctx.rep.add('E41', path, 'the values would not read: %s', ...
                    regexprep(strtrim(err.message), '\s+', ' '));
    end
    H5D.close(oid);
end

% ======================================================= row_support

function checkRowSupport(ctx)
    if ~ctx.hasRowSupport, return, end
    H5 = mestra.internal.H5;
    did = H5D.open(ctx.fid, '/row_support');
    closer = onCleanup(@() H5D.close(did)); %#ok<NASGU>
    info = H5.dsetInfo(did);
    if ~strcmp(info.type, 'int32')
        ctx.rep.add('E20', '/row_support', ...
            '/row_support is stored as %s where int32 is required', ...
            info.type);
    end
    checkScales(ctx, did, info, '/row_support');
    % E16 covers /row_support as it covers a key: one value per row.
    if numel(info.dims) ~= 1
        ctx.rep.add('E16', '/row_support', ...
            '/row_support has one dimension, row, and this has %d', ...
            numel(info.dims));
    elseif info.dims(1) ~= ctx.rowCount
        ctx.rep.add('E16', '/row_support', ...
            '%d values in a file of %d rows', info.dims(1), ctx.rowCount);
    end
    checkChunking(ctx, did, info, '/row_support', ctx.rowCount);
    n = numel(ctx.supportNames);
    bad = ctx.rowSupport(ctx.rowSupport < 0 | ctx.rowSupport >= n);
    if ~isempty(bad)
        ctx.rep.add('E06', '/row_support', ...
            'a row references support %d where the file declares %d', ...
            bad(1), n);
    end
    for i = 1:n
        if ~any(ctx.rowSupport == i - 1)
            ctx.rep.add('W15', ['/supports/' ctx.supportNames{i}], ...
                'no row references this support');
        end
    end
end

% ========================================================== supports

function checkSupports(ctx)
    H5 = mestra.internal.H5;
    if ~mestra.internal.Reader.hasGroup(ctx.fid, 'supports'), return, end
    g = H5.openGroup(ctx.fid, 'supports');
    closer = onCleanup(@() H5G.close(g)); %#ok<NASGU>
    for i = 1:numel(ctx.supportMembers)
        name = ctx.supportMembers{i};
        path = ['/supports/' name];
        checkName(ctx, name, path);
        if ~followable(ctx, g, name, path), continue, end
        if ~strcmp(H5.childType(g, name), 'group')
            ctx.rep.add('E41', path, ...
                'a support is a group, and this is not one, so there is no support here to read');
            % Still a public dataset, so the byte-level rules hold.
            guard(ctx, path, @() checkUnknownDataset(ctx, g, name, path));
            continue
        end
        index = find(strcmp(ctx.supportNames, name), 1) - 1;
        guard(ctx, path, @() checkOneSupport(ctx, g, name, index));
    end
end

function checkOneSupport(ctx, parent, name, index)
    H5 = mestra.internal.H5;
    rep = ctx.rep;
    path = ['/supports/' name];
    sid = H5.openGroup(parent, name);
    closer = onCleanup(@() H5G.close(sid)); %#ok<NASGU>
    checkAttrEncodings(ctx, sid, path);
    checkKnownAttrs(ctx, sid, path, ...
        {'kind', 'n_nodes', 'n_cells', 'support_id'});
    for a = {'kind', 'n_nodes', 'n_cells', 'support_id'}
        if ~H5.hasAttr(sid, a{1})
            rep.add('E39', path, '%s is required on a support', a{1});
        end
    end
    kind = strAttr(sid, 'kind');
    nNodes = numAttr(sid, 'n_nodes');
    nCells = numAttr(sid, 'n_cells');
    if isempty(nNodes), nNodes = 0; end
    if isempty(nCells), nCells = 0; end

    has = @(n) H5.exists(sid, n);
    cellNames = {'cell_types', 'cell_offsets', 'cell_connectivity'};
    present = cellfun(has, cellNames);
    if strcmp(kind, 'mesh')
        if ~all(present)
            rep.add('E38', path, ...
                'a mesh support is missing %s', ...
                strjoin(cellNames(~present), ', '));
        end
    elseif any(present) || has('cell')
        rep.add('E38', path, ...
            'a support of kind %s carries cell data', kind);
    end

    types = []; offsets = []; conn = [];
    if has('cell_types')
        types = plainValues(ctx, sid, 'cell_types', path, 'uint8');
    end
    if has('cell_offsets')
        offsets = plainValues(ctx, sid, 'cell_offsets', path, 'int64');
    end
    if has('cell_connectivity')
        conn = plainValues(ctx, sid, 'cell_connectivity', path, 'int64');
    end
    if strcmp(kind, 'mesh')
        checkCells(ctx, path, types, offsets, conn, nNodes, nCells);
    else
        % A cell dataset on another kind is E38 alone, and the count
        % it declares is not a cell count.
        checkCells(ctx, path, types, offsets, conn, nNodes, []);
    end

    % ------------------------------------------------- support_id
    record.kind = kind;
    record.nNodes = nNodes;
    record.cellTypes = types;
    record.cellOffsets = offsets;
    record.cellConnectivity = conn;
    record.coordinates = [];
    % Coordinates a callable serves are a group with no values; a mesh
    % support's id is computed from its cells alone (section 24).
    if has('coordinates') && strcmp( ...
            mestra.internal.H5.childType(sid, 'coordinates'), 'dataset')
        did = H5D.open(sid, 'coordinates');
        info = H5.dsetInfo(did);
        try
            record.coordinates.values = H5.readData(did, info);
            record.coordinates.dims = cell(1, numel(info.dims));
        catch
            record.coordinates = [];
        end
        H5D.close(did);
    end
    stored = strAttr(sid, 'support_id');
    if ~isempty(stored)
        try
            computed = mestra.supportId(record);
            if ~strcmp(computed, stored)
                rep.add('E08', path, ...
                    'the support_id does not match the stored arrays');
            end
        catch err
            rep.add('E41', path, ...
                'the support_id could not be computed: %s', ...
                regexprep(strtrim(err.message), '\s+', ' '));
        end
    end

    % ----------------------------------------------------- arrays
    if ismember(kind, {'mesh', 'axis'}) && ~has('coordinates')
        rep.add('E03', path, 'a %s support has no coordinates array', kind);
    end
    if has('coordinates')
        guard(ctx, [path '/coordinates'], @() checkSlot(ctx, sid, ...
            'coordinates', [path '/coordinates'], 'node', nNodes, ...
            nCells, kind, index));
    end
    pairs = {'node_arrays', 'node'; 'cell_arrays', 'cell'};
    for p = 1:size(pairs, 1)
        if ~has(pairs{p, 1}), continue, end
        groupPath = [path '/' pairs{p, 1}];
        if ~strcmp(H5.childType(sid, pairs{p, 1}), 'group')
            rep.add('E41', groupPath, ...
                '%s is a group of slots, and this is not a group, so there are no slots here to read', ...
                pairs{p, 1});
            continue
        end
        ag = H5.openGroup(sid, pairs{p, 1});
        for nm = H5.children(ag)
            slotPath = [groupPath '/' nm{1}];
            checkName(ctx, nm{1}, slotPath);
            if ~followable(ctx, ag, nm{1}, slotPath), continue, end
            location = pairs{p, 2};
            guard(ctx, slotPath, @() checkSlot(ctx, ag, nm{1}, slotPath, ...
                location, nNodes, nCells, kind, index));
        end
        H5G.close(ag);
    end

    known = [cellNames {'coordinates', 'node_arrays', 'cell_arrays', ...
                        'node', 'cell', 'cell_plus_one', 'index', 'row'}];
    checked = [cellNames {'coordinates', 'node_arrays', 'cell_arrays'}];
    for nm = H5.children(sid)
        if ~followable(ctx, sid, nm{1}, [path '/' nm{1}])
            continue
        end
        % W11 is an attribute or a group (sections 14 and 28); a
        % dataset this version does not know is checked below instead.
        if ~ismember(nm{1}, known) && strcmp(H5.childType(sid, nm{1}), 'group')
            rep.add('W11', [path '/' nm{1}], ...
                'this reader does not know this object; it is ignored');
        end
        if ~ismember(nm{1}, checked)
            checkUnknownDataset(ctx, sid, nm{1}, [path '/' nm{1}]);
        end
    end
end

function checkUnknownDataset(ctx, gid, name, path)
%checkUnknownDataset  The byte-level rules on a dataset this version
%   does not describe, inside a group it does.
%
%   Section 14 says of the rules of sections 18 to 25: "They are
%   checked on the public objects only. /private is not checked, and
%   neither is any group this version of the format does not know".
%   A dataset this version does not know, inside a group it does, is
%   neither of the two exceptions, so it is a public object and the
%   rules are checked on it.  That is the reading taken here; the
%   other reading, that only the objects this version names are
%   checked, would leave a file free to put an unchunked
%   row-dimensioned dataset anywhere it liked and still pass.
%
%   Only the rules that need nothing this version does not know are
%   decidable: section 23's, which are about the row dimension the
%   dataset is attached to and not about what the dataset means.  A
%   dimension scale is not subject to them (section 23), and a
%   dataset this version does know is checked as the slot it is.
    H5 = mestra.internal.H5;
    if ~mestra.internal.Reader.hasKind(gid, name, 'dataset'), return, end
    did = H5D.open(gid, name);
    closer = onCleanup(@() H5D.close(did)); %#ok<NASGU>
    info = H5.dsetInfo(did);
    if info.isScale || isempty(info.dims), return, end
    checkScales(ctx, did, info, path);
    leading = H5.scaleNames(did, 0, ctx.scales);
    if isempty(leading) || ~strcmp(leading(1).name, 'row'), return, end
    checkChunking(ctx, did, info, path, info.dims(1));
end

function [values, info] = plainValues(ctx, sid, name, path, wanted)
    H5 = mestra.internal.H5;
    did = H5D.open(sid, name);
    info = H5.dsetInfo(did);
    values = [];
    if ~strcmp(info.type, wanted)
        ctx.rep.add('E20', [path '/' name], ...
            '%s is stored as %s where %s is required', name, info.type, ...
            wanted);
    end
    checkAttrEncodings(ctx, did, [path '/' name]);
    checkScales(ctx, did, info, [path '/' name]);
    try
        values = reshape(double(H5.readData(did, info)), 1, []);
    catch
    end
    H5D.close(did);
end

function checkCells(ctx, path, types, offsets, conn, nNodes, nCells)
    rep = ctx.rep;
    if isempty(types) && isempty(offsets) && isempty(conn), return, end
    if ~isempty(nCells) && numel(types) ~= nCells
        % The cell count a mesh declares is the length of its
        % cell_types.  A cell array is checked against the declaration;
        % the declaration is checked here against the one dataset that
        % defines it, or a mesh with no cell arrays could declare any
        % count at all.  nCells is [] on any other kind.
        rep.add('E05', [path '/cell_types'], ...
            'the support declares %d cells and cell_types holds %d', ...
            nCells, numel(types));
    end
    table = cellTypeTable();
    for i = 1:numel(types)
        if ~table.isKey(types(i))
            rep.add('E21', [path '/cell_types'], ...
                'cell type %d is not in the table of section 20', types(i));
        end
    end
    if ~isempty(offsets)
        if offsets(1) ~= 0
            rep.add('E23', [path '/cell_offsets'], ...
                'the first offset is %d and not 0', offsets(1));
        elseif any(diff(offsets) < 0)
            rep.add('E23', [path '/cell_offsets'], ...
                'the offsets are not non-decreasing');
        elseif offsets(end) ~= numel(conn)
            rep.add('E23', [path '/cell_offsets'], ...
                'the last offset is %d and the connectivity holds %d', ...
                offsets(end), numel(conn));
        end
    end
    if numel(offsets) == numel(types) + 1
        for j = 1:numel(types)
            span = offsets(j + 1) - offsets(j);
            if ~table.isKey(types(j)), continue, end
            want = table(types(j));
            if want < 0
                if span < 3
                    rep.add('E22', [path '/cell_connectivity'], ...
                        ['polygon %d has %d nodes where 3 or more are ' ...
                         'needed'], j - 1, span);
                end
            elseif span ~= want
                rep.add('E22', [path '/cell_connectivity'], ...
                    'cell %d has %d nodes where type %d takes %d', ...
                    j - 1, span, types(j), want);
            end
        end
    end
    if ~isempty(conn) && any(conn < 0 | conn >= nNodes)
        rep.add('E24', [path '/cell_connectivity'], ...
            'a connectivity value is outside [0, %d)', nNodes);
    end
end

% ============================================================= slots

function checkSlot(ctx, parent, name, path, location, nNodes, nCells, ...
                   kind, supportIndex)
    H5 = mestra.internal.H5;
    rep = ctx.rep;
    kind_ = H5.childType(parent, name);
    isGroup = strcmp(kind_, 'group');
    if isGroup
        oid = H5G.open(parent, name);
    elseif strcmp(kind_, 'dataset')
        oid = H5D.open(parent, name);
    else
        rep.add('E41', path, ...
            'a slot is a dataset or a group, and this is neither, so there is nothing here to read');
        return
    end
    if isGroup
        linksIn(ctx, oid, path);
    end
    checkAttrEncodings(ctx, oid, path);
    checkKnownAttrs(ctx, oid, path, ...
        {'role', 'varies', 'units', 'components', 'source', 'output', ...
         'statistic', 'of', 'quantile', 'level', 'method', 'category', ...
         'recomputed', ...
         'derived_from', 'recipe', 'reference'});

    role = strAttr(oid, 'role');
    varies = strAttr(oid, 'varies');
    units = strAttr(oid, 'units');
    if isempty(role) || ~ismember(role, arrayRoles())
        rep.add('E02', path, 'the role is "%s"', role);
    end
    if ~H5.hasAttr(oid, 'varies')
        rep.add('E39', path, 'varies is required on an array slot');
    end
    if ~H5.hasAttr(oid, 'components')
        rep.add('E31', path, 'components is required on an array slot');
    end
    if strcmp(role, 'field') && ~H5.hasAttr(oid, 'units')
        rep.add('E11', path, 'a field needs units');
    elseif any(strcmp(role, {'derived', 'coordinates'})) && ...
           ~H5.hasAttr(oid, 'units')
        rep.add('E39', path, 'a %s array needs units', role);
    end
    if ~isempty(units) && ~mestra.internal.Units.parses(units)
        rep.add('W10', path, 'the units "%s" do not parse', units);
    end
    if strcmp(role, 'derived') && ...
       (~H5.hasAttr(oid, 'derived_from') || ~H5.hasAttr(oid, 'recipe'))
        rep.add('E13', path, ...
            'a derived array needs derived_from and recipe');
    end
    % Present and false is no more a mark than absent (W06).
    if any(strcmp(role, {'weight', 'normal'})) && ...
       ~isequal(double(numAttr(oid, 'recomputed')), 1)
        rep.add('W06', path, ...
            'a %s array does not say it was recomputed', role);
    end
    if strcmp(kind, 'axis') && strcmp(role, 'coordinates') && ...
       ~strcmp(varies, 'none')
        rep.add('E35', path, ...
            'the coordinates of an axis support must have varies = none');
    end
    checkSource(ctx, oid, path, isGroup);
    checkStatistic(ctx, oid, path);

    % ---- E04: section 5 names three values of varies, for a slot a
    % callable serves as much as for one that holds data
    if ~isempty(varies) && ~any(strcmp(varies, {'none', 'row'})) && ...
            ~strncmp(varies, 'group:', 6)
        rep.add('E04', path, 'varies is "%s", not none, row or group:<k>', ...
                varies);
    end

    if isGroup
        H5G.close(oid);
        return
    end
    info = H5.dsetInfo(oid);
    checkArrayDtype(ctx, role, info.type, path);
    checkScales(ctx, oid, info, path);
    names = mestra.internal.Reader.axisNames(oid, numel(info.dims), ...
                                             ctx.scales);

    % ---- E04: a varies naming a group key the file does not declare
    if numel(varies) > 6 && strncmp(varies, 'group:', 6)
        if ~ismember(varies(7:end), ctx.groupKeys)
            rep.add('E04', path, ...
                ['varies names the group key "%s", which the file ' ...
                 'does not declare'], varies(7:end));
        end
    end

    % ---- E04: the leading dimension against `varies`
    if ~any(cellfun(@isempty, names))
        leading = '';
        if ~isempty(names), leading = names{1}; end
        if strcmp(varies, 'none')
            if any(strcmp(leading, {'row'})) || ...
               (numel(leading) > 6 && strncmp(leading, 'group:', 6))
                rep.add('E04', path, ...
                    'varies is none and the leading dimension is %s', ...
                    leading);
            end
        elseif ~isempty(varies) && ~strcmp(leading, varies)
            rep.add('E04', path, ...
                'varies is %s and the leading dimension is %s', ...
                varies, leading);
        end
    end

    % ---- E05: the node or cell extent against the support
    axis = find(strcmp(names, location), 1);
    if ~isempty(axis)
        want = nNodes;
        if strcmp(location, 'cell'), want = nCells; end
        if info.dims(axis) ~= want
            rep.add('E05', path, ...
                'the %s extent is %d where the support has %d', ...
                location, info.dims(axis), want);
        end
    end

    % ---- E31: components against the component extent
    declared = numAttr(oid, 'components');
    cAxis = find(strcmp(names, 'component'), 1);
    if ~isempty(declared) && ~isempty(cAxis) && info.dims(cAxis) ~= declared
        rep.add('E31', path, ...
            'components is %d and the component dimension is %d', ...
            declared, info.dims(cAxis));
    end

    % ---- E34 and E16: the leading extent
    if ~isempty(varies) && numel(varies) > 6 && strncmp(varies, 'group:', 6)
        table = groupCategoryCount(ctx, varies(7:end));
        if ~isempty(table) && ~isempty(info.dims) && info.dims(1) ~= table
            rep.add('E34', path, ...
                'the leading extent is %d and the group has %d categories', ...
                info.dims(1), table);
        end
    elseif strcmp(varies, 'row') && ~isempty(info.dims)
        want = ctx.rowCount;
        if ~isempty(ctx.aligned) && ctx.aligned == 0 && ~isempty(ctx.rowSupport)
            want = sum(ctx.rowSupport == supportIndex);
        end
        if info.dims(1) ~= want
            rep.add('E16', path, ...
                ['the leading extent is %d where %d rows are on this ' ...
                 'support'], info.dims(1), want);
        end
        checkChunking(ctx, oid, info, path, info.dims(1));
    end

    if strcmp(role, 'label')
        checkLabelValues(ctx, oid, info, path);
    elseif any(strcmp(role, {'field', 'derived'}))
        try
            values = mestra.internal.H5.readData(oid, info);
            if isnumeric(values)
                bad = find(~isfinite(double(values(:))));
                if ~isempty(bad)
                    rep.add('W03', path, '%s', ...
                            nonFinitePhrase(bad, varies, info.dims));
                end
                if isBand(oid)
                    neg = find(double(values(:)) < 0);
                    if ~isempty(neg)
                        rep.add('W16', path, '%s', ...
                                negativeBandPhrase(neg, varies, info.dims));
                    end
                end
            end
        catch err
            rep.add('E41', path, 'the values would not read: %s', ...
                    regexprep(strtrim(err.message), '\s+', ' '));
        end
    end
    H5D.close(oid);
end

function checkLabelValues(ctx, oid, info, path)
    name = strAttr(oid, 'category');
    if isempty(name), return, end
    table = categoryEntries(ctx, name);
    try
        values = double(mestra.internal.H5.readData(oid, info));
    catch
        return
    end
    bad = values(values < 0 | values >= numel(table));
    if ~isempty(bad)
        ctx.rep.add('E10', path, ...
            'the label value %g is outside a table of %d entries', ...
            bad(1), numel(table));
    end
end

function checkArrayDtype(ctx, role, type, path)
    switch role
        case {'coordinates', 'field', 'derived', 'weight', 'normal'}
            allowed = {'float64'};
        case 'label'
            allowed = {'int32', 'int64'};
        otherwise
            return
    end
    if ~ismember(type, allowed)
        ctx.rep.add('E20', path, ...
            'a %s array is stored as %s where section 19 allows %s', ...
            role, type, strjoin(allowed, ' or '));
    end
end

% ========================================================= callables

function checkCallables(ctx)
    H5 = mestra.internal.H5;
    if ~mestra.internal.Reader.hasGroup(ctx.fid, 'callables'), return, end
    g = H5.openGroup(ctx.fid, 'callables');
    closer = onCleanup(@() H5G.close(g)); %#ok<NASGU>
    for name = H5.children(g)
        path = ['/callables/' name{1}];
        checkName(ctx, name{1}, path);
        if ~followable(ctx, g, name{1}, path), continue, end
        if ~strcmp(H5.childType(g, name{1}), 'group')
            ctx.rep.add('E41', path, ...
                'a callable is a group, and this is not one, so there is no dictionary here to read');
            continue
        end
        guard(ctx, path, @() checkOneCallable(ctx, g, name{1}, path));
    end
end

function checkOneCallable(ctx, g, name, path)
    H5 = mestra.internal.H5;
    cid = H5.openGroup(g, name);
    closer = onCleanup(@() H5G.close(cid)); %#ok<NASGU>
    if ~H5.hasAttr(cid, 'type')
        ctx.rep.add('E15', path, 'a callable group has no type');
    end
    checkDictionaryScales(ctx, cid, path, 0);
    [~, problems] = mestra.internal.Codec.read(cid, true);
    for i = 1:numel(problems)
        if numel(problems{i}) > 4 && strcmp(problems{i}(1:4), 'U03 ')
            ctx.rep.add('E41', path, '%s', problems{i}(5:end));
        elseif numel(problems{i}) > 4 && strcmp(problems{i}(1:4), 'E40 ')
            ctx.rep.add('E40', path, '%s', problems{i}(5:end));
        elseif numel(problems{i}) > 4 && strcmp(problems{i}(1:4), 'E25 ')
            continue    % checkDictionaryScales reports it at the dataset
        else
            ctx.rep.add('E32', path, '%s', problems{i});
        end
    end
end

function checkDictionaryScales(ctx, gid, path, depth)
%checkDictionaryScales  E25 and E33 over a callable's dictionary:
%   section 21 names the scale on axis i of a dictionary dataset
%   mestra_<dataset>_d<i>, so a scale of any other name is the wrong
%   one even when it is a legal name somewhere else, and section 17
%   holds a dictionary key to the netCDF-4 name rule (E33).
    if depth > mestra.internal.Limits.get('maxDepth'), return, end
    H5 = mestra.internal.H5;
    for name = H5.publicAttrNames(gid)
        if mestra.internal.Text.reserved(name{1}), continue, end
        if depth == 0 && any(strcmp(name{1}, {'type', 'repr'})), continue, end
        % An attribute's name is reported at the object carrying it.
        checkName(ctx, name{1}, path);
    end
    for name = H5.children(gid)
        if mestra.internal.Text.reserved(name{1}), continue, end
        kind = H5.childType(gid, name{1});
        sub = [path '/' name{1}];
        checkName(ctx, name{1}, sub);
        if strcmp(kind, 'group')
            g = H5G.open(gid, name{1});
            checkDictionaryScales(ctx, g, sub, depth + 1);
            H5G.close(g);
        elseif strcmp(kind, 'dataset')
            did = H5D.open(gid, name{1});
            info = H5.dsetInfo(did);
            if ~info.isScale && ~isempty(info.dims)
                wanted = arrayfun(@(i) sprintf('mestra_%s_d%d', ...
                    name{1}, i - 1), 1:numel(info.dims), ...
                    'UniformOutput', false);
                guard(ctx, sub, @() checkScales(ctx, did, info, sub, wanted));
            end
            H5D.close(did);
        end
    end
end

% =========================================================== general

function checkSource(ctx, oid, path, isGroup)
    H5 = mestra.internal.H5;
    rep = ctx.rep;
    if ~H5.hasAttr(oid, 'source')
        rep.add('E39', path, 'source is required on a slot');
        return
    end
    source = strAttr(oid, 'source');
    if strcmp(source, 'data')
        if isGroup
            rep.add('E30', path, ...
                'source is data and the slot is stored as a group');
        end
        return
    end
    if numel(source) > 9 && strncmp(source, 'callable:', 9)
        if ~isGroup
            rep.add('E30', path, ...
                'source names a callable and the slot is a dataset');
        end
        id = source(10:end);
        if ~ismember(id, ctx.callableIds)
            rep.add('E14', path, ...
                ['source names the callable "%s", which the file does ' ...
                 'not hold'], id);
        end
        if ~H5.hasAttr(oid, 'output')
            rep.add('E39', path, ...
                'output is required when source names a callable');
        end
        return
    end
    rep.add('E36', path, ...
        'source is "%s" and must be data or callable:<id>', source);
end

function checkStatistic(ctx, oid, path)
%checkStatistic  E12: a statistic with what it needs (section 9), and
%   on a slot a callable serves, one a callable produces (section 10).
    H5 = mestra.internal.H5;
    if ~H5.hasAttr(oid, 'statistic'), return, end
    statistic = strAttr(oid, 'statistic');
    served = false;
    if H5.hasAttr(oid, 'source')
        source = strAttr(oid, 'source');
        served = numel(source) > 9 && strncmp(source, 'callable:', 9);
    end
    if strcmp(statistic, 'quantile') && ~H5.hasAttr(oid, 'quantile')
        ctx.rep.add('E12', path, ...
            'the statistic is quantile and no quantile is given');
    end
    if ~any(strcmp(statistic, {'value', 'draw'})) && ~H5.hasAttr(oid, 'of')
        ctx.rep.add('E12', path, ...
            'the statistic is %s and does not say what it is of', statistic);
    end
    if served && any(strcmp(statistic, {'std', 'quantile', 'draw'}))
        ctx.rep.add('E12', path, ...
            ['a callable returns a mean and at most a band (section ' ...
             '10), so a slot it serves is value, mean or band, not %s'], ...
            statistic);
    end
    if strcmp(statistic, 'band') && ~served
        if ~H5.hasAttr(oid, 'level')
            ctx.rep.add('E12', path, ...
                'a band states the coverage it claims with level');
        else
            level = numAttr(oid, 'level');
            if isnumeric(level) && isscalar(level) && ...
                    ~(level > 0 && level < 1)
                ctx.rep.add('E12', path, ...
                    ['level is a coverage fraction in (0, 1), and this ' ...
                     'is %g'], level);
            end
        end
        if ~H5.hasAttr(oid, 'method')
            ctx.rep.add('E12', path, ...
                'a band says how it was made with method');
        end
    end
end

function tf = isBand(oid)
%isBand  True when the slot's statistic is band (section 9).
    H5 = mestra.internal.H5;
    tf = H5.hasAttr(oid, 'statistic') && ...
         strcmp(strAttr(oid, 'statistic'), 'band');
end

function checkAttrEncodings(ctx, oid, path)
%checkAttrEncodings  E19, E26 and the variable-length string ban.
%   The rules themselves are in mestra.internal.Attrs, because the
%   reader applies the same ones to refuse a file rather than open it
%   half understood (docs/api-conventions.md, section 2).
    found = mestra.internal.Attrs.findings(oid);
    for i = 1:numel(found)
        ctx.rep.add(found(i).id, path, '%s', found(i).message);
    end
end

function checkKnownAttrs(ctx, oid, path, known)
%checkKnownAttrs  W11 for an attribute this version does not know.
    for name = mestra.internal.H5.publicAttrNames(oid)
        if ~ismember(name{1}, known)
            ctx.rep.add('W11', path, ...
                ['the attribute %s is not one this version knows; it ' ...
                 'is ignored'], name{1});
        end
    end
end

function checkDimensionScales(ctx)
%checkDimensionScales  E42 and E43, over the scale map built once.
%   Both rules are about the dimension scales themselves rather than
%   about anything attached to them, and the map of section 21 has
%   already seen every one of them, so this pass reads nothing more.
%
%   Section 14 keeps it to the public objects: the byte-level rules
%   of sections 18 to 25 are not checked inside /private, which
%   section 29 forbids a reader to interpret at all.
    want = mestra.internal.H5.crtOrderTrackedIndexed();
    addresses = ctx.scales.keys();
    for i = 1:numel(addresses)
        s = ctx.scales(addresses{i});
        if isPrivate(s.path), continue, end
        if bitand(s.order, want) ~= want
            ctx.rep.add('E42', s.path, ...
                ['this dimension scale was created without attribute ' ...
                 'creation order tracked and indexed, so it can carry ' ...
                 'no more than 4085 attachments (section 21)']);
        end
        if s.unlimited && ~strcmp(s.name, 'row') && ...
           ~dictionaryZeroAxis(s.path, s.length)
            ctx.rep.add('E43', s.path, ...
                ['the dimension "%s" is unlimited; only `row`, ' ...
                 'file-level or support-local, may be'], s.name);
        end
    end
end

function tf = isPrivate(path)
%isPrivate  True for /private and anything under it.
    tf = strcmp(path, '/private') || strncmp(path, '/private/', 9);
end

function tf = dictionaryZeroAxis(path, len)
%dictionaryZeroAxis  The one unlimited dimension that is not `row`.
%   Section 25 requires the zero-length axis of a dataset inside a
%   callable's dictionary to be unlimited, because HDF5 has no other
%   legal way to write a zero-length axis, and section 19 names that
%   exception where it forbids the rest.
    tf = len == 0 && strncmp(path, '/callables/', 11);
end

function checkScales(ctx, did, info, path, wanted)
%checkScales  E25 for every axis of a dataset.
%   A dimension scale is not itself subject to this rule and carries
%   no scale on its own axis (section 14), so this is called only on
%   the datasets the format defines.  WANTED, when given, is the one
%   name section 21 allows on each axis, where the position decides
%   it: category_<table> on a table, mestra_<dataset>_d<i> in a
%   dictionary.
    if nargin < 5, wanted = {}; end
    H5 = mestra.internal.H5;
    for axis = 1:numel(info.dims)
        found = H5.scaleNames(did, axis - 1, ctx.scales);
        if isempty(found)
            ctx.rep.add('E25', path, ...
                'axis %d carries no dimension scale', axis - 1);
            continue
        end
        if numel(found) > 1
            ctx.rep.add('E25', path, ...
                'axis %d carries %d dimension scales', axis - 1, ...
                numel(found));
            continue
        end
        if isempty(found(1).name)
            ctx.rep.add('E25', path, ...
                ['axis %d is attached to something this reader could ' ...
                 'not resolve to a dimension'], axis - 1);
            continue
        end
        if ~found(1).hasName
            % CLASS without NAME is half a scale: netCDF-C writes both
            % and a reader that needs the sentence would find nothing.
            ctx.rep.add('E25', path, ...
                'axis %d names the scale "%s", which has no NAME attribute', ...
                axis - 1, found(1).name);
            continue
        end
        why = scaleNameProblem(found(1).name, info.dims(axis));
        if isempty(why) && numel(wanted) >= axis && ...
                ~isempty(wanted{axis}) && ~strcmp(found(1).name, wanted{axis})
            why = sprintf('the scale is "%s" where section 21 names "%s"', ...
                          found(1).name, wanted{axis});
        end
        if ~isempty(why)
            ctx.rep.add('E25', path, 'axis %d: %s', axis - 1, why);
        end
        % Section 19 puts the axis under the same rule as the scale:
        % every dimension but `row` is created with a maximum extent
        % equal to its length, so an axis that can grow makes its own
        % dimension's name false.
        if numel(info.maxdims) >= axis && info.maxdims(axis) < 0 && ...
           ~strcmp(found(1).name, 'row') && ...
           ~dictionaryZeroAxis(path, info.dims(axis))
            ctx.rep.add('E43', path, ...
                ['axis %d is unlimited and carries the dimension ' ...
                 '"%s", which is not `row`'], axis - 1, found(1).name);
        end
    end
end

function why = scaleNameProblem(name, len)
%scaleNameProblem  Whether a scale's link name is one section 21 allows.
    why = '';
    fixed = {'row', 'node', 'cell', 'cell_plus_one', 'index'};
    if ismember(name, fixed), return, end
    parameterised = {'component_', 'draw_'};
    for i = 1:numel(parameterised)
        p = parameterised{i};
        if numel(name) > numel(p) && strncmp(name, p, numel(p))
            n = str2double(name(numel(p) + 1:end));
            if isnan(n) || n ~= len
                why = sprintf(['the scale is "%s" and the axis has ' ...
                               'length %d'], name, len);
            end
            return
        end
    end
    for p = {'group_', 'category_', 'mestra_'}
        if numel(name) > numel(p{1}) && strncmp(name, p{1}, numel(p{1}))
            return
        end
    end
    why = sprintf('"%s" is not a dimension name section 21 defines', name);
end

function s = nonFinitePhrase(bad, varies, dims)
%nonFinitePhrase  W03 for an array, once, with the count and the
%   first three rows when the array has rows to name.
%
%   MATLAB reads a C-order dataset with its axes reversed, so the
%   file's leading `row` axis is the array's trailing one and a
%   linear index divided by the size of one row gives the row.
    n = numel(bad);
    if strcmp(varies, 'row') && numel(dims) >= 1 && dims(1) > 0
        block = max(1, prod(double(dims(2:end))));
        rows = unique(floor((double(bad(:)') - 1) / block));
        s = sprintf(['%s hold a non-finite value (%s), %d value(s) in ' ...
                     'all'], ...
                    mestra.internal.Report.someRows(rows, dims(1)), ...
                    mestra.internal.Report.whichRows(rows), n);
    else
        s = sprintf(['%d value(s) are non-finite; this array does not ' ...
                     'vary along rows, so there is no row to name'], n);
    end
end

function s = negativeBandPhrase(neg, varies, dims)
%negativeBandPhrase  W16 for an array, once, with the count and the
%   first three rows when the array has rows to name.
    n = numel(neg);
    if strcmp(varies, 'row') && numel(dims) >= 1 && dims(1) > 0
        block = max(1, prod(double(dims(2:end))));
        rows = unique(floor((double(neg(:)') - 1) / block));
        s = sprintf(['%s hold a value below zero in a band, which is a ' ...
                     'half-width and is never negative (%s), %d ' ...
                     'value(s) in all'], ...
                    mestra.internal.Report.someRows(rows, dims(1)), ...
                    mestra.internal.Report.whichRows(rows), n);
    else
        s = sprintf(['%d value(s) are below zero in a band, which is a ' ...
                     'half-width; this array does not vary along rows, ' ...
                     'so there is no row to name'], n);
    end
end

function s = bound(v)
%bound  A bound, or a dash when the file declares none.
    if isempty(v)
        s = '-';
    else
        s = strtrim(sprintf('%g', double(v)));
    end
end

function checkChunking(ctx, did, info, path, nRows)
%checkChunking  E27, E29 and W12 for a row-dimensioned dataset.
    rep = ctx.rep;
    if isempty(info.chunk)
        rep.add('E27', path, 'a row-dimensioned dataset is not chunked');
    end
    for i = 1:size(info.filters, 1)
        id = info.filters(i, 1);
        if id == 1
            level = info.filters(i, 2);
            if level < 1 || level > 9
                rep.add('E29', path, 'gzip at level %d', level);
            end
        elseif id ~= 2
            rep.add('E29', path, ...
                'filter %d is not gzip or shuffle', id);
        end
    end
    if isempty(info.chunk), return, end
    rest = info.dims(2:end);
    itemsize = mestra.internal.Writer.itemSize(info.type);
    if strcmp(info.type, 'string'), itemsize = info.strSize; end
    % Section 23: the row count in the default is the length of the
    % row dimension the leading axis is attached to, and not the
    % dataset's own leading extent.
    attached = mestra.internal.H5.scaleNames(did, 0, ctx.scales);
    if ~isempty(attached) && attached(1).length >= 0
        nRows = attached(1).length;
    end
    c = mestra.internal.Writer.rowChunk(itemsize, rest, nRows);
    want = [c rest];
    if numel(want) ~= numel(info.chunk) || any(want ~= info.chunk)
        rep.add('W12', path, ...
            'the chunk is %s where the default of section 23 is %s', ...
            mat2str(info.chunk), mat2str(want));
    end
end

function checkName(ctx, name, path)
%checkName  E33 for a producer-chosen name.
    if ~mestra.internal.Text.legalName(name)
        ctx.rep.add('E33', path, ...
            '"%s" is not a legal netCDF-4 name', name);
    elseif mestra.internal.Text.reserved(name)
        ctx.rep.add('E33', path, ...
            '"%s" begins with the reserved prefix mestra_', name);
    end
end

function checkUnknown(ctx)
%checkUnknown  W11 for a root attribute or group this version ignores.
    H5 = mestra.internal.H5;
    if strcmp(H5.childType(ctx.root, 'notes'), 'group')
        notes = H5G.open(ctx.root, 'notes');
        linksIn(ctx, notes, '/notes');
        % /notes is public: a dataset there is held to the byte-level
        % rules as any public dataset is.
        for name = H5.children(notes)
            guard(ctx, ['/notes/' name{1}], @() checkUnknownDataset( ...
                ctx, notes, name{1}, ['/notes/' name{1}]));
        end
        H5G.close(notes);
    end
    known = mestra.internal.Reader.ROOT_ATTRS;
    for name = H5.publicAttrNames(ctx.root)
        if ~ismember(name{1}, known)
            ctx.rep.add('W11', '/', ...
                'the root attribute %s is not one this version knows', ...
                name{1});
        end
    end
    for name = H5.children(ctx.root)
        if ~followable(ctx, ctx.root, name{1}, ['/' name{1}])
            continue
        end
        if ~mestra.internal.Reader.knownRootChild(ctx.root, name{1})
            ctx.rep.add('W11', ['/' name{1}], ...
                'this reader does not know this object; it is ignored');
        end
    end
end

function checkPrivate(ctx)
%checkPrivate  E18, decided from what a reader can see.
%   A required public attribute or object absent, by any of E02, E11,
%   E13, E15, E17, E31 or E39, in a file that also carries a
%   `/private` group.  It is reported beside that rule and never by
%   interpreting `/private`, which section 29 forbids.
    if ~mestra.internal.H5.exists(ctx.fid, '/private'), return, end
    triggers = {'E02', 'E11', 'E13', 'E15', 'E17', 'E31', 'E39'};
    for i = 1:numel(triggers)
        if ctx.rep.has(triggers{i})
            ctx.rep.add('E18', '/private', ...
                ['%s found a required public thing missing in a file ' ...
                 'that also carries a private group'], triggers{i});
            return
        end
    end
end

% ========================================================== fetching

function v = strAttr(oid, name)
    v = mestra.internal.Reader.str(oid, name);
end

function v = numAttr(oid, name)
    v = mestra.internal.Reader.num(oid, name);
end

function entries = categoryEntries(ctx, name)
    entries = {};
    if ~isempty(name) && ctx.categories.isKey(name)
        entries = ctx.categories(name);
    end
end

function n = groupCategoryCount(ctx, keyName)
    n = [];
    k = findKey(ctx, keyName, 'name');
    if isempty(k) || isempty(k.category), return, end
    n = numel(categoryEntries(ctx, k.category));
end

function k = findKey(ctx, value, field)
    k = [];
    for i = 1:numel(ctx.keys)
        if strcmp(ctx.keys(i).(field), value)
            k = ctx.keys(i);
            return
        end
    end
end
