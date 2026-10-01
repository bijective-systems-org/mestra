classdef PackageTest < matlab.unittest.TestCase
%PackageTest  The pieces the corpus does not reach on its own.
%
%   The worked digests of section 24, the units parser behind W10, the
%   permute-by-name rule of sections 4 and 29, the affine callable's
%   summation order, the error identifiers a mistake produces, and the
%   exactness of the two ways of parsing a "%.17e" string.
%
%   See also CorpusTest, run_tests.

    methods (Test)

        function workedDigests(testCase)
        %workedDigests  The three digests section 24 writes out.
            Sha = mestra.internal.Sha;
            mesh = [Sha.int64le(6), uint8([9 9]), Sha.int64le([0 4 8]), ...
                    Sha.int64le([0 1 4 3 1 2 5 4])];
            testCase.verifyEqual(numel(mesh), 98, ...
                'the byte string of section 24 is 98 bytes long');
            meshDigest = ['96df395d80ef548444562292de441525' ...
                          'ba0b5c8ad00a8dadff19a19c943936c7'];
            testCase.verifyEqual(Sha.hex256(mesh), meshDigest);
            axis = [Sha.int64le(4), Sha.float64le([0.0 0.5 1.0 1.5])];
            axisDigest = ['57467fe7370808bdb0ad01b95d963f59' ...
                          'e8bc6762f90f96049453ae33bb05a54c'];
            testCase.verifyEqual(Sha.hex256(axis), axisDigest);
            noneDigest = ['af5570f5a1810b7af78caf4bc70a660f' ...
                          '0df51e42baf91d4de5b2328de0e83dfc'];
            testCase.verifyEqual(Sha.hex256(Sha.int64le(0)), noneDigest);
            emptyDigest = ['e3b0c44298fc1c149afbf4c8996fb924' ...
                           '27ae41e4649b934ca495991b7852b855'];
            testCase.verifyEqual(Sha.hex256(uint8([])), emptyDigest, ...
                'the digest of no bytes at all');
        end

        function unitsParser(testCase)
        %unitsParser  What W10 accepts and what it does not.
            good = {'1', 'm', 'Pa', 's', 'm2', 'W m-2', 'W', 'K', ...
                    'm2 s-1', 'degree', 'kg/(m s)', 'm s-2', 'J kg-1 K-1'};
            for i = 1:numel(good)
                testCase.verifyTrue(mestra.internal.Units.parses(good{i}), ...
                    sprintf('"%s" should parse', good{i}));
            end
            bad = {'kg/(m s', '', 'm/', 'm^', '(', ')m'};
            for i = 1:numel(bad)
                testCase.verifyFalse(mestra.internal.Units.parses(bad{i}), ...
                    sprintf('"%s" should not parse', bad{i}));
            end
        end

        function recursiveRoutinesAreCapped(testCase)
        %recursiveRoutinesAreCapped  Nothing recurses on a file's say-so.
        %   The units parser is the one recursive routine that runs on
        %   text rather than on structure, so it is capped the same way
        %   the walks are: a string nested past the cap does not parse,
        %   which is W10, and W10 is what an unparseable string is.
            deep = [repmat('(', 1, 2000) 'm' repmat(')', 1, 2000)];
            testCase.verifyFalse(mestra.internal.Units.parses(deep), ...
                'deeply nested parentheses do not parse');
            testCase.verifyTrue(mestra.internal.Units.parses('((m))'), ...
                'but a reasonable nesting still does');
            testCase.verifyFalse( ...
                mestra.internal.Units.parses(repmat('m', 1, 100000)), ...
                'and an absurdly long string is refused outright');
        end

        function limitsAreSettable(testCase)
        %limitsAreSettable  A real file may need more than the default.
            old = mestra.limits();
            restore = onCleanup(@() mestra.limits(old));
            testCase.verifyEqual(old.maxDepth, 64);
            previous = mestra.limits('maxDepth', 8);
            testCase.verifyEqual(previous.maxDepth, 64);
            testCase.verifyEqual(mestra.internal.Limits.get('maxDepth'), 8);
            mestra.limits(previous);
            testCase.verifyEqual(mestra.internal.Limits.get('maxDepth'), 64);
            testCase.verifyError(@() mestra.limits('nonesuch', 1), ...
                                 'mestra:limits');
            testCase.verifyError(@() mestra.limits('maxDepth', -1), ...
                                 'mestra:limits');
        end

        function affineOrderIsLoopsAndNotAProduct(testCase)
        %affineOrderIsLoopsAndNotAProduct  How the order is guaranteed.
        %   Section 27 fixes the summation: accumulate over the keys in
        %   the declared order, add b last, no fused multiply-add.  The
        %   guarantee is structural, not documentary: mestra.Affine.call
        %   writes the sum out as three nested loops over rows, outputs
        %   and keys, and the only arithmetic on the path is a scalar
        %   multiply and a scalar add.  A matrix product would be free
        %   to reassociate and is therefore never used; this checks the
        %   source says so, and then checks a case where the orders
        %   disagree in the last bit.
            source = fileread(which('mestra.Affine'));
            body = extractAfter(source, 'function out = call(obj');
            body = extractBefore(body, 'function d = toDict');
            testCase.verifyEmpty(regexp(body, 'A\s*\*', 'once'), ...
                'call must not multiply a matrix by anything');
            testCase.verifyEmpty(regexp(body, "\*\s*x\b(?!\()", 'once'), ...
                'nor a vector');
            testCase.verifySubstring(body, 'acc = acc + A(j, k) * x(r, k)');
            testCase.verifySubstring(body, 'y(r, j) = acc + b(j)');

            % Adding b first instead of last changes the last bit here.
            a = mestra.Affine({'mach', 'alpha'}, struct('cl', ...
                struct('A', [2.0 0.1], 'b', 0.05, 'shape', [])));
            t = table(0.5, 4.0, 'VariableNames', {'mach', 'alpha'});
            out = a.call(t);
            bLast = out('cl').mean.data(1);
            bFirst = 0.05 + 2.0 * 0.5 + 0.1 * 4.0;
            testCase.verifyEqual(typecast(bLast, 'uint64'), ...
                                 typecast(1.45, 'uint64'), ...
                                 'b added last gives exactly 1.45');
            testCase.verifyNotEqual(typecast(bFirst, 'uint64'), ...
                                    typecast(1.45, 'uint64'), ...
                                    'and the other order does not');
        end

        function permuteByName(testCase)
        %permuteByName  The idiom sections 4 and 29 require.
            a = reshape(1:24, [2 3 4]);       % component node row
            names = {'component', 'node', 'row'};
            b = mestra.permute(a, names, {'row', 'node', 'component'});
            testCase.verifyEqual(size(b), [4 3 2]);
            testCase.verifyEqual(b(3, 2, 1), a(1, 2, 3));
            c = mestra.permute(a, names, {'row', 'node', 'component', 'draw'});
            testCase.verifyEqual(size(c), [4 3 2]);
            oneComponent = reshape(1:12, [1 3 4]);
            d = mestra.permute(oneComponent, names, {'row', 'node'});
            testCase.verifyEqual(size(d), [4 3]);
            % 'instance' stands for the one group axis, as in Dims
            grouped = {'component', 'node', 'group:member'};
            e = mestra.permute(oneComponent, grouped, {'instance', 'node'});
            testCase.verifyEqual(size(e), [4 3]);
            testCase.verifyEqual(e(3, 2), oneComponent(1, 2, 3));
            testCase.verifyError(@() mestra.permute(a, names, ...
                                                   {'instance', 'node'}), ...
                                 'mestra:dims');
            testCase.verifyError(@() mestra.permute(a, names, {'row'}), ...
                'mestra:dims');
        end

        function affineSummationOrder(testCase)
        %affineSummationOrder  The worked example of section 27.
            outputs = struct( ...
                'cl', struct('A', [2.0 0.1], 'b', 0.05, 'shape', []), ...
                'pressure', struct('A', [1 0; 2 0; 3 0.5; 4 0.5; 5 1; 6 1], ...
                                   'b', [0; 0.1; 0.2; 0.3; 0.4; 0.5], ...
                                   'shape', [6 1]));
            a = mestra.Affine({'mach', 'alpha'}, outputs);
            t = table(0.5, 4.0, 'VariableNames', {'mach', 'alpha'});
            out = a.call(t);
            cl = out('cl').mean;
            testCase.verifyEqual(cl.data(1), 1.45, ...
                'b is added last, which makes this exactly 1.45');
            testCase.verifyEmpty(out('cl').uncertainty, ...
                'no band on this output');
            p = out('pressure').mean;
            testCase.verifyEqual(p.shape, [1 6 1]);
            wanted = [0.5 1.1 3.7 4.3 6.9 7.5];
            for i = 1:6
                testCase.verifyEqual(p.data(1, i, 1), wanted(i), ...
                    sprintf('pressure node %d', i - 1));
            end
        end

        function affineRoundTrip(testCase)
        %affineRoundTrip  toDict, then fromDict, then the same answers.
            outputs = struct('cl', struct('A', [2.0 0.1], 'b', 0.05, ...
                                          'shape', []));
            a = mestra.Affine({'mach', 'alpha'}, outputs);
            b = mestra.Affine.fromDict(a.toDict());
            testCase.verifyEqual(b.keys, a.keys);
            t = table(0.5, 4.0, 'VariableNames', {'mach', 'alpha'});
            fromB = b.call(t);
            fromA = a.call(t);
            testCase.verifyEqual(fromB('cl').mean.data, fromA('cl').mean.data);
            testCase.verifyTrue(mestra.Registry.isKnown('affine'));
            testCase.verifyEqual(a.type(), 'affine');
        end

        function affineCarriesAConstantBand(testCase)
        %affineCarriesAConstantBand  Section 27: uncertainty, level
        %   and method, all three or none, the same in every row.
            outputs = struct( ...
                'cl', struct('A', [2.0 0.1], 'b', 0.05, 'shape', [], ...
                             'uncertainty', 0.02, 'level', 0.95, ...
                             'method', 'constant band'), ...
                'pressure', struct('A', [1 0; 2 0; 3 0.5; 4 0.5; 5 1; 6 1], ...
                                   'b', [0; 0.1; 0.2; 0.3; 0.4; 0.5], ...
                                   'shape', [6 1], ...
                                   'uncertainty', ...
                                   [0.05; 0.1; 0.15; 0.2; 0.25; 0.3], ...
                                   'level', 0.68, 'method', 'constant band'));
            a = mestra.Affine({'mach', 'alpha'}, outputs);
            t = table([0.5; 0.6], [4.0; 5.0], ...
                      'VariableNames', {'mach', 'alpha'});
            out = a.call(t);
            cl = out('cl');
            testCase.verifyEqual(cl.uncertainty.data, [0.02; 0.02]);
            testCase.verifyEqual(cl.level, 0.95);
            testCase.verifyEqual(cl.method, 'constant band');
            p = out('pressure');
            testCase.verifyEqual(p.uncertainty.shape, [2 6 1]);
            testCase.verifyEqual(p.uncertainty.data(2, 6, 1), 0.3);
            testCase.verifyEqual(p.level, 0.68);
            b = mestra.Affine.fromDict(a.toDict());
            back = b.call(t);
            testCase.verifyEqual(back('cl').uncertainty.data, ...
                                 cl.uncertainty.data);
            testCase.verifyEqual(back('pressure').method, 'constant band');
            bad = struct('cl', struct('A', [2.0 0.1], 'b', 0.05, ...
                                      'shape', [], 'level', 0.95));
            testCase.verifyError(@() mestra.Affine({'mach', 'alpha'}, bad), ...
                                 'mestra:affine');
        end

        function aPredictionIsAMeanAndAtMostABand(testCase)
        %aPredictionIsAMeanAndAtMostABand  Section 10: the record a
        %   callable returns, and what it refuses.
            m = mestra.Array([1; 2]);
            plain = mestra.Callable.prediction(m);
            testCase.verifyEmpty(plain.uncertainty);
            testCase.verifyEmpty(plain.level);
            banded = mestra.Callable.prediction(m, mestra.Array([0.1; 0.2]), ...
                                                0.95, 'm');
            testCase.verifyEqual(banded.level, 0.95);
            u = mestra.Array([0.1; 0.2]);
            testCase.verifyError(@() mestra.Callable.prediction(m, u, 1.96, 'm'), ...
                                 'mestra:prediction');
            testCase.verifyError(@() mestra.Callable.prediction(m, u, 0.95, ''), ...
                                 'mestra:prediction');
            testCase.verifyError(@() mestra.Callable.prediction( ...
                m, mestra.Array(0.1), 0.95, 'm'), 'mestra:prediction');
            testCase.verifyError(@() mestra.Callable.prediction( ...
                m, mestra.Array([-0.1; 0.2]), 0.95, 'm'), 'mestra:prediction');
            testCase.verifyError(@() mestra.Callable.prediction(m, [], 0.95), ...
                                 'mestra:prediction');
        end

        function affineRefusesExtraKeys(testCase)
        %affineRefusesExtraKeys  Section 27 allows nothing else.
            d = containers.Map('KeyType', 'char', 'ValueType', 'any');
            d('keys') = mestra.Array({'mach'});
            d('outputs') = containers.Map('KeyType', 'char', ...
                                          'ValueType', 'any');
            d('extra') = 1;
            testCase.verifyError(@() mestra.Affine.fromDict(d), ...
                                 'mestra:affine');
        end

        function decimalParsing(testCase)
        %decimalParsing  str2double against sscanf on every corpus float.
        %   The corpus writes floats as "%.17e" and compares by bits.
        %   sscanf with %lf is the C library's own strtod; str2double is
        %   MATLAB's own.  If they ever disagreed, only the first would
        %   be safe to use.  On this corpus they do not.
            values = PackageTest.corpusDecimals();
            testCase.verifyGreaterThan(numel(values), 100, ...
                'the corpus should hand over a few hundred floats');
            disagreements = 0;
            for i = 1:numel(values)
                text = values{i};
                a = mestra.internal.Text.fromDecimal(text);
                b = str2double(text);
                if ~(isnan(a) && isnan(b)) && ...
                   ~isequal(typecast(a, 'uint64'), typecast(b, 'uint64'))
                    disagreements = disagreements + 1;
                end
            end
            testCase.verifyEqual(disagreements, 0, ...
                'str2double and sscanf(%lf) agree on every corpus float');
        end

        function roundTripOfText(testCase)
        %roundTripOfText  "%.17e" survives a trip through a float64.
            values = [1.45 0.1 -0.0 1e300 1e-300 0.8];
            for v = values
                text = mestra.internal.Text.decimal(v);
                back = mestra.internal.Text.fromDecimal(text);
                testCase.verifyEqual(typecast(back, 'uint64'), ...
                                     typecast(v, 'uint64'), ...
                                     sprintf('%s does not round trip', text));
            end
            testCase.verifyEqual(mestra.internal.Text.decimal(NaN), 'nan');
            testCase.verifyEqual(mestra.internal.Text.decimal(Inf), 'inf');
            testCase.verifyEqual(mestra.internal.Text.decimal(-Inf), '-inf');
        end

        function buildFromArrays(testCase)
        %buildFromArrays  A dataset in a handful of calls, then read back.
            d = mestra.Dataset();
            d.writer = 'mestra matlab tests';
            d.created = '2026-09-19T00:00:00Z';
            d.addCategoryTable('member', {'wing_a', 'wing_b'});
            d.addKey('mach', [0.4 0.8], 'condition', '1', ...
                     'Lower', 0.1, 'Upper', 0.9);
            d.addKey('member', int32([0 1]), 'group', 'Category', 'member');
            d.setGeneralisationGroup('member');
            coords = cat(3, [0 1 2 0 1 2; 0 0 0 1 1 1], ...
                            [0 1.5 3 0 1.5 3; 0 0 0 1 1 1]);
            d.addMeshSupport('s0', coords, uint8([9 9]), int64([0 4 8]), ...
                             int64([0 1 4 3 1 2 5 4]), 'm', ...
                             'Dims', {'component', 'node', 'group:member'});
            p = [101 102 103 104 105 106; 201 202 203 204 205 206];
            d.addNodeArray('s0', 'pressure', p, 'field', 'Pa', ...
                           'Dims', {'row', 'node'});
            d.addScalar('cl', [0.25 0.55], '1');

            meshDigest = ['96df395d80ef548444562292de441525' ...
                          'ba0b5c8ad00a8dadff19a19c943936c7'];
            testCase.verifyEqual(d.support('s0').supportId, meshDigest, ...
                'the support id is filled in by the builder');

            out = [tempname() '.mes'];
            cleanup = onCleanup(@() PackageTest.removeIfPresent(out));
            mestra.write(d, out);
            r = mestra.validate(out);
            testCase.verifyEmpty(r.errors, strjoin(r.errors, ','));
            testCase.verifyEmpty(r.warnings, strjoin(r.warnings, ','));

            back = mestra.read(out);
            a = back.nodeArray('s0', 'pressure');
            q = mestra.permute(a.values, a.dims, {'row', 'node', 'component'});
            testCase.verifyEqual(q(2, 4, 1), 204);
            c = back.support('s0').coordinates;
            cc = mestra.permute(c.values, c.dims, ...
                                {'group:member', 'node', 'component'});
            testCase.verifyEqual(cc(2, 3, 1), 3);
        end

        function errorIdentifiersNameTheRule(testCase)
        %errorIdentifiersNameTheRule  A mistake says which rule it broke.
            d = mestra.Dataset();
            testCase.verifyError( ...
                @() d.addNodeArray('nothing', 'p', 1, 'field', '1'), ...
                'mestra:noSupport');
            dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
            dict('bad') = {1, 'two'};
            testCase.verifyError(@() PackageTest.writeDict(dict), ...
                                 'mestra:E32');
            reserved = containers.Map('KeyType', 'char', 'ValueType', 'any');
            reserved('mestra_x') = 1;
            testCase.verifyError(@() PackageTest.writeDict(reserved), ...
                                 'mestra:E33');
        end

        function refusesAnotherMajorVersion(testCase)
        %refusesAnotherMajorVersion  E01, and never a partial read.
            d = mestra.Dataset();
            d.format = 'mestra/1';
            d.created = '2026-09-19T00:00:00Z';
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() PackageTest.removeIfPresent(out));
            % A file of another major version is E01, so writing it
            % takes the escape hatch section 2 of the conventions
            % gives for deliberately invalid files.
            mestra.write(d, out, 'Check', false);
            testCase.verifyError(@() mestra.read(out), 'mestra:E01');
            r = mestra.validate(out);
            testCase.verifyEqual(r.errors, {'E01'});
        end

        function nonAsciiIsRefused(testCase)
        %nonAsciiIsRefused  The one thing MATLAB cannot do, said plainly.
        %   Section 25 puts strings in UTF-8 and counts the declared
        %   size in bytes.  MATLAB's HDF5 interface writes a
        %   fixed-length string only from ASCII characters and decodes
        %   one to text before this package sees it, so a non-ASCII
        %   string can neither be written nor read back faithfully.
        %   The package says so rather than writing or returning
        %   something else.
            d = mestra.Dataset();
            d.created = '2026-09-19T00:00:00Z';
            d.addCategoryTable('member', {'aile', ['fl' char(252) 'gel']});
            d.addKey('member', int32([0 1]), 'group', ...
                     'Category', 'member');
            d.setGeneralisationGroup('member');
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() PackageTest.removeIfPresent(out));
            testCase.verifyError(@() mestra.write(d, out), ...
                                 'mestra:matlabAscii');
        end

        function opaqueSubtreesSurvive(testCase)
        %opaqueSubtreesSurvive  /private is copied and not interpreted.
        %   Section 29 forbids a reader to interpret /private, and a
        %   writer must not lose it.  err_e18 is the corpus case that
        %   carries one; it is invalid, so the corpus round trip skips
        %   it and this test does it instead.
            source = fullfile(corpusRoot(), 'err_e18', 'case.mes');
            d = mestra.read(source);
            testCase.verifyNotEmpty(d.privateTree, ...
                'the private group was captured');
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() PackageTest.removeIfPresent(out));
            % err_e18 is an invalid case on purpose, so this is the
            % one place the round trip takes the escape hatch.
            mestra.write(d, out, 'Check', false);
            differences = mestra.internal.Compare.structural(source, out);
            testCase.verifyEmpty(differences, strjoin(differences, '; '));
        end

        function openReadsNoArray(testCase)
        %openReadsNoArray  Section 29: opening reports without reading.
            cases = corpusRoot();
            d = mestra.open(fullfile(cases, 'mesh_two_rows', 'case.mes'));
            testCase.verifyEqual(d.nRows, 2);
            testCase.verifyEmpty(d.key('mach').values);
            testCase.verifyEmpty(d.nodeArray('s0', 'pressure').values);
            testCase.verifyEqual(d.nodeArray('s0', 'pressure').dims, ...
                                 {'component', 'node', 'row'});
            testCase.verifyEqual(d.support('s0').supportId(1:8), '96df395d');
        end

        function anUnknownDatasetInAKnownGroupIsChecked(testCase)
        %anUnknownDatasetInAKnownGroupIsChecked  Section 14 says the
        %   byte-level rules "are checked on the public objects only.
        %   /private is not checked, and neither is any group this
        %   version of the format does not know".  A dataset this
        %   version does not know, inside a group it does, is neither
        %   exception, so it is a public object and the rules are
        %   checked on it.  The rules that need nothing this version
        %   does not know are section 23's: a dataset with a row
        %   dimension must be chunked (E27).
        %
        %   Such a file can be read as E27 or as nothing at all, which
        %   is why it is pinned here.  Both names below are the same
        %   rule, and the one this
        %   version happens to know as a support's own dimension scale
        %   draws no W11 while the other does.
            for probe = {{'row', false}, {'extra', true}}
                name = probe{1}{1};
                unknownName = probe{1}{2};
                path = [tempname() '.mes'];
                cleanup = onCleanup( ...
                    @() PackageTest.removeIfPresent(path)); %#ok<NASGU>
                PackageTest.putContiguousRowDataset( ...
                    fullfile(corpusRoot(), 'mesh_two_rows', 'case.mes'), ...
                    path, name);
                r = mestra.validate(path);
                testCase.verifyTrue(ismember('E27', r.errors), sprintf( ...
                    ['a contiguous row-dimensioned dataset named %s ' ...
                     'in a support group: [%s]'], name, ...
                    strjoin(r.errors, ' ')));
                found = r.findings(strcmp({r.findings.id}, 'E27'));
                testCase.verifyEqual(found(1).path, ...
                                     ['/supports/s0/' name]);
                testCase.verifyEqual(ismember('W11', r.warnings), ...
                                     unknownName, strjoin(r.warnings, ' '));
            end
        end

        function byteOrderIsPartOfTheEncoding(testCase)
        %byteOrderIsPartOfTheEncoding  Section 18 names little-endian
        %   types for every number and section 19 stores every numeric
        %   dataset little-endian.  A check of class and size alone,
        %   which H5T gives without the byte order, passed both.
            path = [tempname() '.mes'];
            cleanup = onCleanup( ...
                @() PackageTest.removeIfPresent(path)); %#ok<NASGU>
            copyfile(fullfile(corpusRoot(), 'mesh_two_rows', 'case.mes'), ...
                     path);
            fileattrib(path, '+w');
            fid = H5F.open(path, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
            gid = H5G.open(fid, '/supports/s0');
            H5A.delete(gid, 'n_cells');
            tid = H5T.copy('H5T_STD_I64BE');
            sid = H5S.create('H5S_SCALAR');
            aid = H5A.create(gid, 'n_cells', tid, sid, 'H5P_DEFAULT');
            H5A.write(aid, 'H5T_NATIVE_LLONG', int64(2));
            H5A.close(aid); H5S.close(sid); H5T.close(tid);
            H5G.close(gid); H5F.close(fid);
            r = mestra.validate(path);
            testCase.verifyEqual(r.errors, {'E19'});
            testCase.verifyError(@() mestra.read(path), 'mestra:E19');

            % A big-endian dataset is E20 alone, a semantic rule, so
            % the file reads, and the values are the ones it holds.
            big = fullfile(corpusRoot(), 'err_e20_big_endian', 'case.mes');
            r = mestra.validate(big);
            testCase.verifyEqual(r.errors, {'E20'});
            d = mestra.read(big);
            testCase.verifyEqual(d.scalar('cl').values(:)', [0.25 0.55]);
        end
    end

    methods (Static)

        function putContiguousRowDataset(src, dst, name)
        %putContiguousRowDataset  A copy of SRC with one contiguous
        %   dataset of NAME in /supports/s0, attached to the file's
        %   `row` scale.  Built here rather than committed: it is one
        %   dataset added to a corpus file.
            copyfile(src, dst);
            fileattrib(dst, '+w');
            fid = H5F.open(dst, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
            closer = onCleanup(@() H5F.close(fid)); %#ok<NASGU>
            gid = H5G.open(fid, '/supports/s0');
            did = mestra.internal.H5.createDataset(gid, name, 'float64', ...
                                                   2, 2, []);
            mestra.internal.H5.writeData(did, 'float64', [1 2]);
            scale = H5D.open(fid, 'row');
            H5DS.attach_scale(did, scale, 0);
            H5D.close(scale);
            H5D.close(did);
            H5G.close(gid);
        end

        function removeIfPresent(path)
            if exist(path, 'file') == 2
                delete(path);
            end
        end

        function writeDict(dict)
            out = [tempname() '.mes'];
            cleanup = onCleanup( ...
                @() PackageTest.removeIfPresent(out)); %#ok<NASGU>
            d = mestra.Dataset();
            d.created = '2026-09-19T00:00:00Z';
            d.addCallable('m1', dict, 'Type', 'example');
            mestra.write(d, out);
        end

        function values = corpusDecimals()
        %corpusDecimals  Every "%.17e" string the corpus carries.
            values = {};
            cases = corpusRoot();
            listing = dir(cases);
            for i = 1:numel(listing)
                n = listing(i).name;
                if n(1) == '.' || ~listing(i).isdir, continue, end
                e = jsondecode(fileread(fullfile(cases, n, 'expected.json')));
                values = [values PackageTest.harvest(e)]; %#ok<AGROW>
            end
        end

        function out = harvest(node)
        %harvest  Every string that looks like a "%.17e" float.
            out = {};
            if ischar(node)
                if ~isempty(regexp(node, '^-?\d\.\d+e[+-]\d+$', 'once'))
                    out = {node};
                end
            elseif iscell(node)
                for i = 1:numel(node)
                    out = [out PackageTest.harvest(node{i})]; %#ok<AGROW>
                end
            elseif isstruct(node)
                for k = 1:numel(node)
                    names = fieldnames(node(k));
                    for i = 1:numel(names)
                        found = PackageTest.harvest(node(k).(names{i}));
                        out = [out found]; %#ok<AGROW>
                    end
                end
            end
        end
    end
end
