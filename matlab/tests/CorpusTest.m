classdef CorpusTest < matlab.unittest.TestCase
%CorpusTest  The conformance corpus, run case by case.
%
%   Every golden file under vectors/cases ships with what every
%   implementation must find (specification sections 15 and 30).  This
%   runs all of it: the validator's outcome by rule identifier, every
%   support id, every probe bit for bit, every codec round trip, every
%   worked evaluation, and, for every file that validates cleanly, a
%   read then a write then the structural comparison of section 30.
%
%   See also run_tests, mestra.validate, mestra.read, mestra.write.

    properties (TestParameter)
        caseName
    end

    methods (Static)
        function names = allCases()
            cases = corpusRoot();
            listing = dir(cases);
            names = {};
            for i = 1:numel(listing)
                n = listing(i).name;
                if n(1) == '.' || ~listing(i).isdir, continue, end
                names{end + 1} = n; %#ok<AGROW>
            end
            names = sort(names);
        end

        function p = caseFile(name)
            p = fullfile(corpusRoot(), name, 'case.mes');
        end

        function requireCase(testCase, name)
        %requireCase  Skip a case that is generated on demand and is
        %   not there, naming the command that writes it, rather than
        %   failing on a file nobody promised to commit.  `wide_keys`
        %   is sixteen megabytes; its expected.json is committed, so
        %   that an implementation knows it exists and knows to
        %   generate it first (specification section 30).
            testCase.assumeEqual( ...
                exist(CorpusTest.caseFile(name), 'file'), 2, ...
                sprintf(['%s is generated on demand and is not ' ...
                         'there; run python vectors/generate.py ' ...
                         '--on-demand'], name));
        end

        function e = expected(name)
            text = fileread(fullfile(corpusRoot(), name, 'expected.json'));
            e = jsondecode(text);
        end

        function counts = coverage()
        %coverage  What the corpus asks for, counted.
            counts = struct('cases', 0, 'valid', 0, 'probes', 0, ...
                            'instanceProbes', 0, 'cellProbes', 0, ...
                            'drawProbes', 0, 'supportIds', 0, ...
                            'dictionaries', 0, 'evaluationProbes', 0);
            names = CorpusTest.allCases();
            counts.cases = numel(names);
            for i = 1:numel(names)
                e = CorpusTest.expected(names{i});
                if isempty(CorpusTest.asCellstr(e.validator.errors))
                    counts.valid = counts.valid + 1;
                end
                list = e.probes;
                if isstruct(list), list = num2cell(list); end
                counts.probes = counts.probes + numel(list);
                for j = 1:numel(list)
                    p = list{j};
                    if isfield(p, 'instance')
                        counts.instanceProbes = counts.instanceProbes + 1;
                    end
                    if isfield(p, 'draw')
                        counts.drawProbes = counts.drawProbes + 1;
                    end
                    if ~isempty(strfind(p.slot, 'cell_')) %#ok<STREMP>
                        counts.cellProbes = counts.cellProbes + 1;
                    end
                end
                if isstruct(e.support_ids)
                    counts.supportIds = counts.supportIds + ...
                        numel(fieldnames(e.support_ids));
                end
                if isstruct(e.codec)
                    counts.dictionaries = counts.dictionaries + ...
                        numel(fieldnames(e.codec));
                end
                evals = e.evaluation;
                if isstruct(evals), evals = num2cell(evals); end
                for j = 1:numel(evals)
                    p = evals{j}.probes;
                    if isstruct(p), p = num2cell(p); end
                    counts.evaluationProbes = counts.evaluationProbes + ...
                        numel(p);
                end
            end
        end

        function out = asCellstr(v)
        %asCellstr  jsondecode gives a cell, a char or an empty; make
        %   it a row cell array of char every time.
            if isempty(v)
                out = {};
            elseif ischar(v)
                out = {v};
            elseif iscell(v)
                out = reshape(cellstr(string(v)), 1, []);
            else
                out = reshape(cellstr(string(v)), 1, []);
            end
        end
    end

    methods (TestClassSetup)
        function defineCases(testCase) %#ok<MANU>
        end
    end

    methods (TestParameterDefinition, Static)
        function caseName = initialiseCases()
            caseName = CorpusTest.allCases();
        end
    end

    methods (Test)

        function validatorOutcome(testCase, caseName)
        %validatorOutcome  The errors and warnings, exactly.
            CorpusTest.requireCase(testCase, caseName);
            e = CorpusTest.expected(caseName);
            r = mestra.validate(CorpusTest.caseFile(caseName));
            wantE = sort(CorpusTest.asCellstr(e.validator.errors));
            wantW = sort(CorpusTest.asCellstr(e.validator.warnings));
            gotE = sort(r.errors);
            gotW = sort(r.warnings);
            if isempty(gotE), gotE = {}; end
            if isempty(gotW), gotW = {}; end
            testCase.verifyEqual(gotE, wantE, ...
                sprintf('%s: errors are [%s] and should be [%s]', ...
                        caseName, strjoin(gotE, ','), strjoin(wantE, ',')));
            testCase.verifyEqual(gotW, wantW, ...
                sprintf('%s: warnings are [%s] and should be [%s]', ...
                        caseName, strjoin(gotW, ','), strjoin(wantW, ',')));
            % A retired identifier must never be emitted.
            testCase.verifyFalse(any(ismember({'E07', 'W09'}, ...
                                              [gotE gotW])), ...
                'a retired rule identifier was emitted');
            % One finding per rule per object (docs/api-conventions.md,
            % section 5).  It is checked here, on every case, because
            % this is where the corpus is validated anyway.
            if ~isempty(r.findings)
                seen = arrayfun(@(f) [f.id ' ' f.path], r.findings, ...
                                'UniformOutput', false);
                testCase.verifyEqual(numel(unique(seen)), numel(seen), ...
                    sprintf('%s: a rule fired twice at one path', caseName));
            end
        end

        function supportIds(testCase, caseName)
        %supportIds  Every digest, computed from the stored arrays.
            CorpusTest.requireCase(testCase, caseName);
            e = CorpusTest.expected(caseName);
            if ~isfield(e, 'support_ids') || isempty(e.support_ids)
                return
            end
            names = fieldnames(e.support_ids);
            if isempty(names), return, end
            file = CorpusTest.caseFile(caseName);
            fid = H5F.open(file, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
            closer = onCleanup(@() H5F.close(fid)); %#ok<NASGU>
            map = mestra.internal.H5.scaleMap(fid);
            g = H5G.open(fid, '/supports');
            for i = 1:numel(names)
                name = names{i};
                record = mestra.internal.Reader.readSupport(g, name, true, ...
                                                            map);
                got = mestra.supportId(record);
                testCase.verifyEqual(got, e.support_ids.(name), ...
                    sprintf('%s: the support id of %s', caseName, name));
            end
            H5G.close(g);
        end

        function probes(testCase, caseName)
        %probes  Every stored value, found by dimension name.
            CorpusTest.requireCase(testCase, caseName);
            e = CorpusTest.expected(caseName);
            if isempty(e.probes), return, end
            d = mestra.read(CorpusTest.caseFile(caseName));
            list = e.probes;
            if isstruct(list), list = num2cell(list); end
            for i = 1:numel(list)
                p = list{i};
                got = CorpusTest.probeValue(d, p);
                want = mestra.internal.Text.fromDecimal(p.value);
                testCase.verifyTrue(CorpusTest.bitEqual(got, want), ...
                    sprintf('%s: probe %d on %s is %s and should be %s', ...
                            caseName, i, p.slot, ...
                            mestra.internal.Text.decimal(got), p.value));
            end
        end

        function codecRoundTrip(testCase, caseName)
        %codecRoundTrip  Every callable's dictionary, in the tagged form.
            CorpusTest.requireCase(testCase, caseName);
            e = CorpusTest.expected(caseName);
            if ~isfield(e, 'codec') || isempty(fieldnames(e.codec))
                return
            end
            d = mestra.read(CorpusTest.caseFile(caseName));
            ids = fieldnames(e.codec);
            for i = 1:numel(ids)
                record = d.callable(ids{i});
                got = mestra.internal.Codec.toTagged(record.dict);
                CorpusTest.compareTagged(testCase, got, e.codec.(ids{i}), ...
                    sprintf('%s/%s', caseName, ids{i}));
            end
        end

        function evaluations(testCase, caseName)
        %evaluations  Every worked evaluation, bit for bit.
            CorpusTest.requireCase(testCase, caseName);
            e = CorpusTest.expected(caseName);
            if isempty(e.evaluation), return, end
            list = e.evaluation;
            if isstruct(list), list = num2cell(list); end
            d = mestra.read(CorpusTest.caseFile(caseName));
            for i = 1:numel(list)
                entry = list{i};
                t = CorpusTest.keysTable(entry.keys);
                materialised = mestra.evaluate(d, t);
                probeList = entry.probes;
                if isstruct(probeList), probeList = num2cell(probeList); end
                for j = 1:numel(probeList)
                    p = probeList{j};
                    got = CorpusTest.probeValue(materialised, p);
                    want = mestra.internal.Text.fromDecimal(p.value);
                    testCase.verifyTrue(CorpusTest.bitEqual(got, want), ...
                        sprintf(['%s: evaluation of %s, probe %d on %s ' ...
                                 'is %s and should be %s'], caseName, ...
                                entry.callable, j, p.slot, ...
                                mestra.internal.Text.decimal(got), p.value));
                end
            end
        end

        function readWriteCompare(testCase, caseName)
        %readWriteCompare  Read, write, compare by section 30's rule.
            CorpusTest.requireCase(testCase, caseName);
            e = CorpusTest.expected(caseName);
            if ~isempty(CorpusTest.asCellstr(e.validator.errors))
                return      % an invalid file is not required to round trip
            end
            d = mestra.read(CorpusTest.caseFile(caseName));
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() CorpusTest.removeIfPresent(out));
            mestra.write(d, out);
            differences = mestra.internal.Compare.structural( ...
                CorpusTest.caseFile(caseName), out);
            testCase.verifyEmpty(differences, ...
                sprintf('%s: %s', caseName, strjoin(differences, '; ')));
        end

        function lazyRowRange(testCase, caseName)
        %lazyRowRange  One slot, one row range, nothing else read.
            CorpusTest.requireCase(testCase, caseName);
            e = CorpusTest.expected(caseName);
            if ~isempty(CorpusTest.asCellstr(e.validator.errors)), return, end
            d = mestra.open(CorpusTest.caseFile(caseName));
            found = d.slots();
            for i = 1:numel(found)
                slot = found(i).slot;
                if ~strcmp(slot.source, 'data'), continue, end
                if isempty(slot.dims) || ~strcmp(slot.dims{end}, 'row')
                    continue
                end
                full = mestra.read(CorpusTest.caseFile(caseName));
                whole = CorpusTest.slotByPath(full, found(i).path);
                % A key or a scalar is a row vector here, so its row
                % count is its length and not its extent along dims.
                if isscalar(whole.dims)
                    n = numel(whole.values);
                else
                    n = size(whole.values, numel(whole.dims));
                end
                if n == 0, continue, end
                part = d.readRows(found(i).path, [1 1]);
                subs = repmat({':'}, 1, numel(whole.dims));
                subs{numel(whole.dims)} = 1;
                wanted = whole.values(subs{:});
                testCase.verifyEqual(part.values(:), wanted(:), ...
                    sprintf('%s: lazy read of %s', caseName, found(i).path));
                testCase.verifyEqual(part.dims, whole.dims, ...
                    sprintf('%s: lazy dims of %s', caseName, found(i).path));
                return
            end
        end
    end

    % ------------------------------------------------------ helpers

    methods (Static)

        function removeIfPresent(path)
            if exist(path, 'file') == 2
                delete(path);
            end
        end

        function tf = bitEqual(a, b)
        %bitEqual  Float64 equality by bits, so that NaN equals NaN.
            if isnan(a) && isnan(b), tf = true; return, end
            tf = isequal(typecast(double(a), 'uint64'), ...
                         typecast(double(b), 'uint64'));
        end

        function t = keysTable(keys)
        %keysTable  An expected.json keys object as a MATLAB table.
            names = fieldnames(keys);
            columns = cell(1, numel(names));
            for i = 1:numel(names)
                raw = keys.(names{i});
                if ischar(raw), raw = {raw}; end
                values = zeros(numel(raw), 1);
                for j = 1:numel(raw)
                    values(j) = mestra.internal.Text.fromDecimal(raw{j});
                end
                columns{i} = values;
            end
            t = table(columns{:}, 'VariableNames', names');
        end

        function slot = slotByPath(d, path)
            found = d.slots();
            i = find(strcmp({found.path}, path), 1);
            slot = found(i).slot;
        end

        function value = probeValue(d, p)
        %probeValue  One probe, resolved by dimension name.
            [values, dims] = CorpusTest.arrayFor(d, p.slot);
            if isempty(dims)
                % A dataset inside a callable's dictionary has no
                % dimension name the format defines, so the probe's
                % index fields apply to its axes in order.
                idx = CorpusTest.positionalIndex(p, ndims(values));
            else
                idx = ones(1, numel(dims));
                for axis = 1:numel(dims)
                    idx(axis) = CorpusTest.indexFor(p, dims{axis});
                end
            end
            subs = num2cell(idx);
            value = double(values(subs{:}));
        end

        function idx = positionalIndex(p, n)
            order = {'row', 'instance', 'draw', 'node', 'component', 'index'};
            found = [];
            for i = 1:numel(order)
                if isfield(p, order{i})
                    found(end + 1) = double(p.(order{i})) + 1; %#ok<AGROW>
                end
            end
            % A mestra.Array holds its data with the file's own
            % subscripts, so the indices go on in file order.
            idx = ones(1, max(n, numel(found)));
            idx(1:numel(found)) = found;
        end

        function i = indexFor(p, dim)
            switch dim
                case 'row'
                    i = double(p.row) + 1;
                case 'draw'
                    i = double(p.draw) + 1;
                case {'node', 'cell'}
                    i = double(p.node) + 1;
                case 'component'
                    i = double(p.component) + 1;
                case 'index'
                    i = double(p.index) + 1;
                case 'cell_plus_one'
                    i = double(p.cell_plus_one) + 1;
                otherwise
                    if numel(dim) > 6 && strcmp(dim(1:6), 'group:')
                        i = double(p.instance) + 1;
                    else
                        error('mestra:probe', ...
                              'no index for the dimension "%s"', dim);
                    end
            end
        end

        function [values, dims] = arrayFor(d, path)
        %arrayFor  The array a probe names, with its dimension names.
            parts = strsplit(path, '/');
            parts = parts(~cellfun(@isempty, parts));
            switch parts{1}
                case 'keys'
                    values = d.key(parts{2}).values;
                    dims = {'row'};
                case 'scalars'
                    values = d.scalar(parts{2}).values;
                    dims = {'row'};
                case 'row_support'
                    values = d.rowSupport;
                    dims = {'row'};
                case 'supports'
                    s = d.support(parts{2});
                    switch parts{3}
                        case 'cell_types'
                            values = s.cellTypes; dims = {'cell'};
                        case 'cell_offsets'
                            values = s.cellOffsets; dims = {'cell_plus_one'};
                        case 'cell_connectivity'
                            values = s.cellConnectivity; dims = {'index'};
                        case 'coordinates'
                            values = s.coordinates.values;
                            dims = s.coordinates.dims;
                        case 'node_arrays'
                            a = d.nodeArray(parts{2}, parts{4});
                            values = a.values; dims = a.dims;
                        case 'cell_arrays'
                            a = d.cellArray(parts{2}, parts{4});
                            values = a.values; dims = a.dims;
                        otherwise
                            error('mestra:probe', ...
                                  'no slot at "%s"', path);
                    end
                case 'callables'
                    record = d.callable(parts{2});
                    node = record.dict;
                    for i = 3:numel(parts)
                        node = node(parts{i});
                    end
                    values = node.data;
                    dims = {};
                otherwise
                    error('mestra:probe', 'no slot at "%s"', path);
            end
        end

        function compareTagged(testCase, got, want, where)
        %compareTagged  Two values in the tagged form of section 30.
            testCase.verifyEqual(got.t, want.t, ...
                sprintf('%s: the tag is %s and should be %s', where, ...
                        got.t, want.t));
            switch want.t
                case 'dict'
                    gotNames = sort(fieldnames(got.v));
                    wantNames = sort(fieldnames(want.v));
                    testCase.verifyEqual(gotNames, wantNames, ...
                        sprintf('%s: the keys differ', where));
                    for i = 1:numel(wantNames)
                        if ~ismember(wantNames{i}, gotNames), continue, end
                        CorpusTest.compareTagged(testCase, ...
                            got.v.(wantNames{i}), want.v.(wantNames{i}), ...
                            [where '/' wantNames{i}]);
                    end
                case 'array'
                    testCase.verifyEqual(got.dtype, want.dtype, ...
                        sprintf('%s: the dtype differs', where));
                    testCase.verifyEqual(double(got.shape(:))', ...
                        double(want.shape(:))', ...
                        sprintf('%s: the shape differs', where));
                    gotData = got.data;
                    wantData = want.data;
                    if ischar(wantData), wantData = {wantData}; end
                    testCase.verifyEqual(numel(gotData), numel(wantData), ...
                        sprintf('%s: the element count differs', where));
                    for i = 1:min(numel(gotData), numel(wantData))
                        if strcmp(want.dtype, 'float64')
                            a = mestra.internal.Text.fromDecimal(gotData{i});
                            b = mestra.internal.Text.fromDecimal(wantData{i});
                            testCase.verifyTrue(CorpusTest.bitEqual(a, b), ...
                                sprintf('%s: element %d differs', where, i));
                        else
                            testCase.verifyEqual(double(gotData(i)), ...
                                double(wantData(i)), ...
                                sprintf('%s: element %d differs', where, i));
                        end
                    end
                case 'strings'
                    testCase.verifyEqual(double(got.shape(:))', ...
                        double(want.shape(:))', ...
                        sprintf('%s: the shape differs', where));
                    wantData = want.data;
                    if ischar(wantData), wantData = {wantData}; end
                    testCase.verifyEqual(reshape(got.data, 1, []), ...
                        reshape(cellstr(string(wantData)), 1, []), ...
                        sprintf('%s: the strings differ', where));
                case 'f64'
                    a = mestra.internal.Text.fromDecimal(got.v);
                    b = mestra.internal.Text.fromDecimal(want.v);
                    testCase.verifyTrue(CorpusTest.bitEqual(a, b), ...
                        sprintf('%s: the value differs', where));
                case 'null'
                    % nothing more to compare
                otherwise
                    testCase.verifyEqual(got.v, want.v, ...
                        sprintf('%s: the value differs', where));
            end
        end
    end
end
