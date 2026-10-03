classdef ConventionsTest < matlab.unittest.TestCase
%ConventionsTest  The cross-language API conventions, in MATLAB.
%
%   docs/api-conventions.md fixes the argument order, the defaults,
%   the refusals, the validator's granularity and the shape of the
%   messages, so that a user moving between the four implementations
%   meets the same API in four idioms.  Every rule of it that this
%   language can be held to is checked here.
%
%   See also PostTest, PackageTest, CorpusTest.

    methods (Test)

        % ------------------------------------- 1. building

        function argumentOrderIsTheAgreedOne(testCase)
        %argumentOrderIsTheAgreedOne  name, values, role, units.
        %   Section 1: `add_key(name, values, role, units)` in that
        %   order in every language.  In MATLAB the role and the
        %   units may also be given by name, and the two spellings
        %   build the same thing.
            a = ConventionsTest.twoRows();
            a.addKey('alpha', [1 2], 'condition', 'degree');
            a.addScalar('cd', [0.1 0.2], '1');
            a.addNodeArray('s0', 'temperature', [300 301 302 303 304 305], ...
                           'field', 'K', 'Dims', {'component', 'node'});

            b = ConventionsTest.twoRows();
            b.addKey('alpha', [1 2], 'Role', 'condition', 'Units', 'degree');
            b.addScalar('cd', [0.1 0.2], 'Units', '1');
            b.addNodeArray('s0', 'temperature', [300 301 302 303 304 305], ...
                           'Role', 'field', 'Units', 'K', ...
                           'Dims', {'component', 'node'});

            testCase.verifyEqual(b.key('alpha'), a.key('alpha'));
            testCase.verifyEqual(b.scalar('cd'), a.scalar('cd'));
            testCase.verifyEqual(b.nodeArray('s0', 'temperature'), ...
                                 a.nodeArray('s0', 'temperature'));
        end

        function anArrayRoleDefaultsToField(testCase)
        %anArrayRoleDefaultsToField  Section 1 writes the array
        %   builders as (support, name, values, units, dims), with no
        %   role, because an array is usually a field.  Both readings
        %   of the fourth argument work, and a role is never mistaken
        %   for a unit: the two vocabularies do not overlap.
            d = ConventionsTest.twoRows();
            d.addNodeArray('s0', 'a', ones(1, 6), 'Pa', ...
                           'Dims', {'component', 'node'});
            d.addNodeArray('s0', 'b', ones(1, 6), 'field', 'Pa', ...
                           'Dims', {'component', 'node'});
            testCase.verifyEqual(d.nodeArray('s0', 'a').role, 'field');
            testCase.verifyEqual(d.nodeArray('s0', 'a').units, 'Pa');
            testCase.verifyEqual(d.nodeArray('s0', 'b').role, 'field');
            testCase.verifyEqual(d.nodeArray('s0', 'b').units, 'Pa');
        end

        function theOldOrderIsRefusedAndNamed(testCase)
        %theOldOrderIsRefusedAndNamed  The order this package used
        %   before the conventions is caught and the new one stated,
        %   rather than quietly taking a role for a column of values.
            d = mestra.Dataset();
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addKey('mach', 'condition', [0.4 0.8]), ...
                'mestra:arguments');
            testCase.verifySubstring(err.message, ...
                'addKey(name, values, role, units)');
        end

        function dimsSettlesWhatAnArrayVariesAlong(testCase)
        %dimsSettlesWhatAnArrayVariesAlong  Section 1: a `row` axis
        %   means row, a `group:<k>` axis means that group, neither
        %   means none.  A default of `row` instead would write a file
        %   the validator rejects.
            d = ConventionsTest.twoRows();
            d.addNodeArray('s0', 'rowwise', [1 2 3 4 5 6; 7 8 9 10 11 12], ...
                           'field', '1', 'Dims', {'row', 'node'});
            d.addNodeArray('s0', 'fixed', 1:6, 'field', '1', ...
                           'Dims', {'component', 'node'});
            d.addNodeArray('s0', 'perMember', [1 2 3 4 5 6; 7 8 9 10 11 12], ...
                           'field', '1', 'Dims', {'group:member', 'node'});
            d.addNodeArray('s0', 'byInstance', ...
                           [1 2 3 4 5 6; 7 8 9 10 11 12], 'field', '1', ...
                           'Dims', {'instance', 'node'});
            testCase.verifyEqual(d.nodeArray('s0', 'rowwise').varies, 'row');
            testCase.verifyEqual(d.nodeArray('s0', 'fixed').varies, 'none');
            testCase.verifyEqual(d.nodeArray('s0', 'perMember').varies, ...
                                 'group:member');
            testCase.verifyEqual(d.nodeArray('s0', 'byInstance').varies, ...
                                 'group:member', ...
                'the corpus calls a group axis "instance"');
        end

        function aVariesDisagreeingWithDimsIsRefused(testCase)
        %aVariesDisagreeingWithDimsIsRefused  With the rule id, at
        %   build time.  A disagreement about the row axis is the
        %   row count, which is E16; any other is a leading dimension
        %   that disagrees with `varies`, which is E04.
            d = ConventionsTest.twoRows();
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addNodeArray('s0', 'bad', 1:6, 'field', '1', ...
                                   'Dims', {'component', 'node'}, ...
                                   'Varies', 'row'), 'mestra:E16');
            testCase.verifyTrue(startsWith(err.message, 'E16: '), ...
                'the message names the rule first');
            testCase.verifySubstring(err.message, ...
                '/supports/s0/node_arrays/bad');
            testCase.verifySubstring(err.message, 'Varies');

            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addNodeArray('s0', 'bad2', 1:6, 'field', '1', ...
                                   'Dims', {'component', 'node'}, ...
                                   'Varies', 'group:member'), 'mestra:E04');
            testCase.verifyTrue(startsWith(err.message, 'E04: '));
        end

        function anArrayTheWrongWayRoundIsRefusedAtBuildTime(testCase)
        %anArrayTheWrongWayRoundIsRefusedAtBuildTime  Section 1: every
        %   builder refuses at build time, with the rule id, what the
        %   validator would refuse.  An array whose node count
        %   disagrees with its support is E05, and the commonest way
        %   to arrive at one is to hand over an array the wrong way
        %   round; the message says so.
            d = ConventionsTest.twoRows();
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addNodeArray('s0', 'p', ones(2, 5), 'field', 'Pa', ...
                                   'Dims', {'row', 'node'}), 'mestra:E05');
            testCase.verifyTrue(startsWith(err.message, 'E05: '));
            testCase.verifySubstring(err.message, 'a support of 6');
            testCase.verifySubstring(err.message, 'Dims');

            % An array named the other way round is caught the same
            % way, because its node axis then has the length of the
            % row axis.
            ConventionsTest.errorFrom(testCase, ...
                @() d.addCellArray('s0', 'q', ones(2, 6), 'field', '1', ...
                                   'Dims', {'row', 'cell'}), 'mestra:E05');
        end

        function aCategoryValueOutsideItsTableIsRefused(testCase)
        %aCategoryValueOutsideItsTableIsRefused  E10, at build time,
        %   for a key and for a label.
            d = mestra.Dataset();
            d.addCategoryTable('member', {'wing_a', 'wing_b'});
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addKey('member', int32([0 2]), 'group', ...
                             'Category', 'member'), 'mestra:E10');
            testCase.verifyTrue(startsWith(err.message, 'E10: '));
            testCase.verifySubstring(err.message, 'ids 0 to 1');

            e = ConventionsTest.twoRows();
            e.addCategoryTable('faces', {'upper', 'lower'});
            ConventionsTest.errorFrom(testCase, ...
                @() e.addNodeArray('s0', 'f', int32([0 1 2 0 1 0]), ...
                                   'label', 'Category', 'faces', ...
                                   'Dims', {'component', 'node'}), ...
                'mestra:E10');
        end

        function boundsDefaultToTheObservedRange(testCase)
        %boundsDefaultToTheObservedRange  Section 1: the same arrays
        %   written by any implementation must give the same file,
        %   so that W04 and W08 are decidable on every file.
            d = mestra.Dataset();
            d.addKey('mach', [0.4 0.9 0.6], 'condition', '1');
            testCase.verifyEqual(d.key('mach').lower, 0.4);
            testCase.verifyEqual(d.key('mach').upper, 0.9);

            d.addKey('alpha', [1 NaN 3], 'design', 'degree');
            testCase.verifyEqual(d.key('alpha').lower, 1, ...
                'a non-finite value is not a bound');
            testCase.verifyEqual(d.key('alpha').upper, 3);

            d.addKey('re', [1 2 3], 'condition', '1', ...
                     'Lower', 0.5, 'Upper', 10);
            testCase.verifyEqual(d.key('re').lower, 0.5, ...
                'a caller who wants a wider domain says so');
            testCase.verifyEqual(d.key('re').upper, 10);

            d.addCategoryTable('member', {'a', 'b'});
            d.addKey('member', int32([0 1]), 'group', 'Category', 'member');
            testCase.verifyEmpty(d.key('member').lower, ...
                ['a category id has no domain of validity, so a group ' ...
                 'key takes no bounds']);
        end

        function theCategoryTableIsTheOneWay(testCase)
        %theCategoryTableIsTheOneWay  Section 1: addCategoryTable and
        %   then 'Category'.  No inline alternative, and the old name
        %   says where it went (X5).
            d = mestra.Dataset();
            d.addCategoryTable('member', {'wing_a', 'wing_b'});
            testCase.verifyEqual(d.category('member').entries, ...
                                 {'wing_a', 'wing_b'});
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addCategory('x', {'a'}), 'mestra:renamed');
            testCase.verifySubstring(err.message, 'addCategoryTable');
        end

        function theGeneralisationGroupHasASetter(testCase)
        %theGeneralisationGroupHasASetter  Section 1: a dataset-level
        %   setter in every language (X6), which refuses a key that is
        %   not a group.
            d = mestra.Dataset();
            d.addCategoryTable('member', {'a', 'b'});
            d.addKey('member', int32([0 1]), 'group', 'Category', 'member');
            d.addKey('mach', [0.4 0.8], 'condition', '1');
            d.setGeneralisationGroup('member');
            testCase.verifyEqual(d.generalisationGroup, 'member');
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.setGeneralisationGroup('mach'), 'mestra:E03');
            testCase.verifyTrue(startsWith(err.message, 'E03: '));
        end

        function aCallableSlotTakesTheArrayBuilderOrder(testCase)
        %aCallableSlotTakesTheArrayBuilderOrder  Section 1: the array
        %   builders' order plus the callable and the output.
            d = ConventionsTest.twoRows();
            A = mestra.Affine({'mach'}, struct('p', struct( ...
                    'A', [1; 1; 1; 1; 1; 1], 'b', zeros(6, 1), ...
                    'shape', [6 1])));
            d.addCallable('m1', A);
            d.addCallableSlot('s0', 'p', 'field', 'Pa', 'm1', 'p', ...
                              'Components', 1);
            slot = d.nodeArray('s0', 'p');
            testCase.verifyEqual(slot.source, 'callable:m1');
            testCase.verifyEqual(slot.output, 'p');
            testCase.verifyEqual(slot.units, 'Pa');
            testCase.verifyEqual(slot.components, 1);
        end

        function aCallableMayServeAMeshSupportsCoordinates(testCase)
        %aCallableMayServeAMeshSupportsCoordinates  Section 10: a
        %   model of the geometry itself.  The support is added with
        %   its node count and no coordinates, and setCallableCoordinates
        %   is setCoordinates with the values dropped and the callable
        %   and its output added.
            d = mestra.Dataset();
            d.writer = 'mestra matlab tests';
            d.created = '2026-09-19T00:00:00Z';
            d.addKey('mach', [], 'condition', '1', ...
                     'Lower', 0.1, 'Upper', 0.9);
            d.addSupport('s0', 'mesh', 6, 'CellTypes', uint8([9 9]), ...
                         'CellOffsets', int64([0 4 8]), ...
                         'CellConnectivity', int64([0 1 4 3 1 2 5 4]));
            % x = x0 (1 + mach): the mesh stretches along x with mach;
            % A runs over the flattened (node, component) order.
            base = [0 0; 1 0; 2 0; 0 1; 1 1; 2 1];
            stretch = zeros(12, 1);
            stretch(1:2:11) = base(:, 1);
            b = reshape(base', [], 1);
            A = mestra.Affine({'mach'}, struct('coordinates', struct( ...
                    'A', stretch, 'b', b, 'shape', [6 2])));
            % Which callable fills the slot is required; that it exists
            % is what write checks, as for every callable slot here.
            testCase.verifyError( ...
                @() d.setCallableCoordinates('s0', 'm', '', ...
                                             'coordinates', 'Components', 2), ...
                'mestra:E14');
            testCase.verifyError( ...
                @() d.setCallableCoordinates('s0', 'm', 'm1', 'coordinates'), ...
                'mestra:E31');
            d.addCallable('m1', A);
            d.setCallableCoordinates('s0', 'm', 'm1', 'coordinates', ...
                                     'Components', 2);
            slot = d.support('s0').coordinates;
            testCase.verifyEqual(slot.source, 'callable:m1');
            testCase.verifyEqual(slot.output, 'coordinates');
            testCase.verifyEqual(slot.role, 'coordinates');
            testCase.verifyEqual(slot.varies, 'row');
            testCase.verifyError( ...
                @() d.setCallableCoordinates('s0', 'm', 'm1', ...
                                             'coordinates', 'Components', 2), ...
                'mestra:E03');

            out = [tempname() '.mes'];
            cleanup = onCleanup(@() ConventionsTest.removeIfPresent(out));
            mestra.write(d, out);
            r = mestra.validate(out);
            testCase.verifyEmpty(r.errors);
            back = mestra.read(out);
            served = back.support('s0').coordinates;
            testCase.verifyEqual(served.source, 'callable:m1');
            testCase.verifyEmpty(served.values);
            testCase.verifyError( ...
                @() mestra.computeWeights(back, 's0', 'cell'), ...
                'mestra:computeWeights');
            e = mestra.evaluate(back, table(0.5, 'VariableNames', {'mach'}));
            c = e.support('s0').coordinates;
            testCase.verifyEqual(c.source, 'data');
            got = mestra.permute(c.values, c.dims, {'row', 'node', 'component'});
            testCase.verifyEqual(got(1, 3, 1), 3.0);
            testCase.verifyEqual(got(1, 6, 2), 1.0);

            % An axis support's coordinates are its identity and are
            % stored.
            d.addAxisSupport('t', [0 1 2], 's');
            testCase.verifyError( ...
                @() d.setCallableCoordinates('t', 's', 'm1', ...
                                             'coordinates', 'Components', 1), ...
                'mestra:E03');
        end

        function aCallableFileCanBeBuiltFromArrays(testCase)
        %aCallableFileCanBeBuiltFromArrays  End to end, on the worked
        %   example of docs/example.md: build a file with no rows and
        %   two callable slots, write it, and evaluate it to 1.45.
        %
        %   It is the one place that shows how to write a callable
        %   file in MATLAB, built with addCallable and
        %   addCallableSlot.
            A = mestra.Affine({'mach', 'alpha'}, struct( ...
                'cl', struct('A', [2.0 0.1], 'b', 0.05, 'shape', []), ...
                'pressure', struct( ...
                    'A', [1 0; 2 0; 3 0.5; 4 0.5; 5 1; 6 1], ...
                    'b', [0; 0.1; 0.2; 0.3; 0.4; 0.5], 'shape', [6 1])));

            d = mestra.Dataset();
            d.writer = 'mestra matlab tests';
            d.created = '2026-09-19T00:00:00Z';
            d.addKey('mach', [], 'condition', '1', ...
                     'Lower', 0.1, 'Upper', 0.9);
            d.addKey('alpha', [], 'condition', 'degree', ...
                     'Lower', -2, 'Upper', 10);
            d.addMeshSupport('s0', [0 1 2 0 1 2; 0 0 0 1 1 1], ...
                             uint8([9 9]), int64([0 4 8]), ...
                             int64([0 1 4 3 1 2 5 4]), 'm', ...
                             'Dims', {'component', 'node'});
            d.addCallable('m1', A);
            d.addScalar('cl', [], '1', 'Callable', 'm1', 'Output', 'cl');
            d.addCallableSlot('s0', 'pressure', 'field', 'Pa', ...
                              'm1', 'pressure', 'Components', 1);

            out = [tempname() '.mes'];
            cleanup = onCleanup(@() ConventionsTest.removeIfPresent(out));
            mestra.write(d, out);
            r = mestra.validate(out);
            testCase.verifyEmpty(r.errors, strjoin(r.errors, ','));

            back = mestra.read(out);
            t = table(0.5, 4.0, 'VariableNames', {'mach', 'alpha'});
            e = mestra.evaluate(back, t);
            testCase.verifyEqual(e.scalar('cl').values, 1.45, ...
                                 'AbsTol', 1e-12);
            p = e.nodeArray('s0', 'pressure');
            q = mestra.permute(p.values, p.dims, {'row', 'node'});
            testCase.verifyEqual(q(1, 3), 3.7, 'AbsTol', 1e-12);
        end

        function aBandSlotTakesTheUncertaintyAndItsLevel(testCase)
        %aBandSlotTakesTheUncertaintyAndItsLevel  Specification
        %   section 10: which part of a record fills a slot is its
        %   statistic, and evaluation writes level and method on the
        %   band.
            file = fullfile(corpusRoot(), 'affine_band', 'case.mes');
            d = mestra.read(file);
            t = table([0.1; 0.9], [-2.0; 10.0], ...
                      'VariableNames', {'mach', 'alpha'});
            e = mestra.evaluate(d, t);
            band = e.scalar('cl_band');
            testCase.verifyEqual(band.source, 'data');
            testCase.verifyEqual(band.statistic, 'band');
            testCase.verifyEqual(band.level, 0.95);
            testCase.verifyEqual(band.method, 'constant band');
            testCase.verifyEqual(band.values, [0.02 0.02]);
            pb = e.nodeArray('s0', 'pressure_band');
            testCase.verifyEqual(pb.level, 0.68);
            q = mestra.permute(pb.values, pb.dims, {'row', 'node'});
            testCase.verifyEqual(q(2, 6), 0.3);
            testCase.verifyEmpty(e.scalar('cl').level);
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() ...
                ConventionsTest.removeIfPresent(out)); %#ok<NASGU>
            mestra.write(e, out);
            r = mestra.validate(out);
            testCase.verifyEmpty(r.errors, strjoin(r.errors, ','));
            testCase.verifyEmpty(r.warnings, strjoin(r.warnings, ','));
            back = mestra.read(out);
            testCase.verifyEqual(back.scalar('cl_band').level, 0.95);
            testCase.verifyEqual(back.scalar('cl_band').method, ...
                                 'constant band');
        end

        function aBandSlotNeedsARecordWithABand(testCase)
        %aBandSlotNeedsARecordWithABand  A band slot served by an
        %   output with no uncertainty cannot be filled.
            A = mestra.Affine({'mach'}, struct('cl', ...
                struct('A', 2.0, 'b', 0.05, 'shape', [])));
            d = mestra.Dataset();
            d.writer = 'mestra matlab tests';
            d.created = '2026-09-19T00:00:00Z';
            d.addKey('mach', [], 'condition', '1', ...
                     'Lower', 0.1, 'Upper', 0.9);
            d.addCallable('m1', A);
            d.addScalar('cl', [], '1', 'Callable', 'm1', 'Output', 'cl');
            d.addScalar('cl_band', [], '1', 'Callable', 'm1', ...
                        'Output', 'cl', 'Statistic', 'band', 'Of', 'cl');
            t = table(0.5, 'VariableNames', {'mach'});
            testCase.verifyError(@() mestra.evaluate(d, t), 'mestra:E12');
        end

        % ------------------------------------- 2. writing

        function writeRefusesAFileItsOwnValidatorRejects(testCase)
        %writeRefusesAFileItsOwnValidatorRejects  Section 2.  A writer
        %   that produces invalid files silently is the one failure
        %   mode an open format cannot afford.
            d = ConventionsTest.twoRows();
            % Built past the builder's own check, which is the only
            % way to make an invalid dataset now.
            d.supports(1).nodeArrays = ...
                ConventionsTest.brokenSlot(d.support('s0'));
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() ConventionsTest.removeIfPresent(out));
            err = ConventionsTest.errorFrom(testCase, ...
                @() mestra.write(d, out), 'mestra:E16');
            testCase.verifySubstring(err.message, 'E16');
            testCase.verifySubstring(err.message, '''Check'', false');
            testCase.verifyEqual(exist(out, 'file'), 0, ...
                'a refused write leaves no file behind');
        end

        function checkFalseWritesItAnyway(testCase)
        %checkFalseWritesItAnyway  Section 2's escape hatch, which is
        %   how a deliberately invalid file is made.
            d = ConventionsTest.twoRows();
            d.supports(1).nodeArrays = ...
                ConventionsTest.brokenSlot(d.support('s0'));
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() ConventionsTest.removeIfPresent(out));
            mestra.write(d, out, 'Check', false);
            testCase.verifyEqual(exist(out, 'file'), 2);
            r = mestra.validate(out);
            testCase.verifyEqual(r.errors, {'E16'});
        end

        function aRefusedWriteLeavesTheOldFileAlone(testCase)
        %aRefusedWriteLeavesTheOldFileAlone  The file outlives the
        %   session that made it, so a refusal must not destroy one.
            good = ConventionsTest.twoRows();
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() ConventionsTest.removeIfPresent(out));
            mestra.write(good, out);
            before = dir(out);

            bad = ConventionsTest.twoRows();
            bad.supports(1).nodeArrays = ...
                ConventionsTest.brokenSlot(bad.support('s0'));
            testCase.verifyError(@() mestra.write(bad, out), 'mestra:E16');
            after = dir(out);
            testCase.verifyEqual(after.bytes, before.bytes);
            testCase.verifyTrue(mestra.validate(out).valid);
        end

        function failedPublicationPreservesExistingBytes(testCase)
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() ConventionsTest.removeIfPresent(out));
            mestra.write(ConventionsTest.twoRows(), out);
            fid = fopen(out, 'rb'); before = fread(fid, Inf, '*uint8'); fclose(fid);
            testCase.verifyError(@() mestra.internal.publishFile( ...
                [out '.missing'], out), 'mestra:write');
            fid = fopen(out, 'rb'); after = fread(fid, Inf, '*uint8'); fclose(fid);
            testCase.verifyEqual(after, before);
            testCase.verifyTrue(mestra.validate(out).valid);
            % Successful replacement of an existing file uses the same path.
            mestra.write(ConventionsTest.twoRows(), out);
            testCase.verifyTrue(mestra.validate(out).valid);
        end

        function failedWriteDoesNotMoveIntoADirectory(testCase)
            folder = tempname(); mkdir(folder);
            cleanup = onCleanup(@() rmdir(folder, 's'));
            out = fullfile(folder, 'target.mes'); mkdir(out);
            testCase.verifyError(@() mestra.write( ...
                ConventionsTest.twoRows(), out), 'mestra:write');
            entries = dir(folder);
            testCase.verifyEqual({entries(~ismember({entries.name}, {'.', '..'})).name}, ...
                                 {'target.mes'});
            testCase.verifyTrue(isfolder(out));
        end

        function aDestinationThatCannotTakeAFileIsRefusedFirst(testCase)
        %aDestinationThatCannotTakeAFileIsRefusedFirst  The staging and
        %   the publish both happen in the destination's directory, so
        %   a destination that cannot take a file is refused before
        %   anything is validated or created, as mestra:write naming
        %   the path, with nothing left behind.
            folder = tempname(); mkdir(folder);
            cleanup = onCleanup(@() rmdir(folder, 's'));
            taken = fullfile(folder, 'taken.mes'); mkdir(taken);
            probes = {taken, fullfile(folder, 'gone', 'x.mes'), ''};
            for i = 1:numel(probes)
                err = ConventionsTest.errorFrom(testCase, ...
                    @() mestra.write(ConventionsTest.twoRows(), probes{i}), ...
                    'mestra:write');
                testCase.verifyFalse(contains(err.message, 'Java'), ...
                    'the refusal is this package''s own and not the JVM''s');
            end
            entries = dir(folder);
            testCase.verifyEqual({entries(~ismember({entries.name}, {'.', '..'})).name}, ...
                                 {'taken.mes'});
            inside = dir(taken);
            testCase.verifyEmpty(inside(~ismember({inside.name}, {'.', '..'})));
        end

        function theBuilderRefusesWhatSection18Forbids(testCase)
        %theBuilderRefusesWhatSection18Forbids  A bound that is not
        %   finite, units that are not text, a name netCDF-4 would not
        %   take: each refused at build time with its rule, so that
        %   the writer never emits what its own validator would refuse.
            d = ConventionsTest.twoRows();
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addKey('alpha', [1 2], 'condition', 'degree', ...
                             'Lower', NaN, 'Upper', 3), 'mestra:E19');
            testCase.verifySubstring(err.message, 'Lower');
            testCase.verifySubstring(err.message, '/keys/alpha');
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addKey('alpha', [1 2], 'condition', 3), 'mestra:E19');
            testCase.verifySubstring(err.message, 'units');
            ConventionsTest.errorFrom(testCase, ...
                @() d.addScalar('cd', [1 2], 3), 'mestra:E19');
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addKey('al/pha', [1 2], 'condition', '1'), 'mestra:E33');
            testCase.verifyTrue(startsWith(err.message, 'E33: '));
            ConventionsTest.errorFrom(testCase, ...
                @() d.addScalar('mestra_cd', [1 2], '1'), 'mestra:E33');
            ConventionsTest.errorFrom(testCase, ...
                @() d.addCategoryTable('a b', {'x'}), 'mestra:E33');
            % Section 18: a name begins with a letter, a digit or an
            % underscore.  "." was legal, and HDF5 reads it as the
            % group itself.
            for name = {'.', '.x', '-x', '+x'}
                ConventionsTest.errorFrom(testCase, ...
                    @() d.addKey(name{1}, [1 2], 'condition', '1'), ...
                    'mestra:E33');
            end
        end

        function everythingTheBuilderMakesValidatesClean(testCase)
        %everythingTheBuilderMakesValidatesClean  Including the chunk
        %   shape, which is M4: the writer picks the default of
        %   section 23 and its own validator does not warn about it.
            d = ConventionsTest.twoRows();
            d.addNodeArray('s0', 'pressure', ...
                           [101 102 103 104 105 106
                            201 202 203 204 205 206], ...
                           'field', 'Pa', 'Dims', {'row', 'node'});
            d.addNodeArray('s0', 'cad_edge_t', linspace(0, 1, 6), ...
                           'field', '1', 'Dims', {'component', 'node'});
            d.addScalar('cl', [0.25 0.55], '1');
            out = [tempname() '.mes'];
            cleanup = onCleanup(@() ConventionsTest.removeIfPresent(out));
            mestra.write(d, out);
            r = mestra.validate(out);
            testCase.verifyEmpty(r.errors, strjoin(r.errors, ','));
            testCase.verifyEmpty(r.warnings, strjoin(r.warnings, ','));
        end

        function anEvaluatedFileValidatesClean(testCase)
        %anEvaluatedFileValidatesClean  Section 2: a file this package
        %   writes validates clean, and an evaluated file is one of
        %   them.  The trap is a chunk carried over from a file of no
        %   rows onto a file of two, which every validator then
        %   warns about (W12).  Section 23
        %   measures the default against the row dimension the leading
        %   axis is attached to, and after evaluation that is the
        %   table's row count and not the source file's.
            for name = {'affine_zero_rows', 'callable_two_slots'}
                file = fullfile(corpusRoot(), name{1}, 'case.mes');
                e = CorpusTest.expected(name{1});
                entry = e.evaluation;
                if iscell(entry), entry = entry{1}; else, entry = entry(1); end
                d = mestra.read(file);
                out = [tempname() '.mes'];
                cleanup = onCleanup(@() ...
                    ConventionsTest.removeIfPresent(out)); %#ok<NASGU>
                mestra.write(mestra.evaluate(d, ...
                    CorpusTest.keysTable(entry.keys)), out);
                r = mestra.validate(out);
                testCase.verifyEmpty(r.errors, ...
                    sprintf('%s: %s', name{1}, strjoin(r.errors, ',')));
                testCase.verifyEmpty(r.warnings, ...
                    sprintf('%s: %s', name{1}, strjoin(r.warnings, ',')));
                testCase.verifyFalse( ...
                    ConventionsTest.hasRootGroup(out, 'callables'), ...
                    sprintf(['%s: an evaluated file has no /callables ' ...
                             'group at all'], name{1}));
            end
        end

        function anEvaluatedFileHasNoCallablesGroup(testCase)
        %anEvaluatedFileHasNoCallablesGroup  Section 7: evaluating
        %   turns every callable slot into a stored slot, so the
        %   result has no callable to keep and `/callables` is absent
        %   from the file, not present and empty.  Three different
        %   things could be left there; this is the settled one.
            file = fullfile(corpusRoot(), 'callable_two_slots', 'case.mes');
            e = CorpusTest.expected('callable_two_slots');
            entry = e.evaluation;
            if iscell(entry), entry = entry{1}; else, entry = entry(1); end
            d = mestra.read(file);
            testCase.verifyNotEmpty(d.callables, ...
                'the file it came from has a callable');
            evaluated = mestra.evaluate(d, CorpusTest.keysTable(entry.keys));
            testCase.verifyEmpty(evaluated.callables);
            testCase.verifyFalse(ismember('callables', ...
                                          evaluated.groupsPresent), ...
                'the group the source carried does not come with it');
            for slot = evaluated.slots()
                testCase.verifyEqual(slot.slot.source, 'data', ...
                    'every callable slot now holds data');
            end
            out = [tempname() '.mes'];
            cleanup = onCleanup( ...
                @() ConventionsTest.removeIfPresent(out)); %#ok<NASGU>
            mestra.write(evaluated, out);
            testCase.verifyFalse( ...
                ConventionsTest.hasRootGroup(out, 'callables'));
            % And the file it came from still has its callable: the
            % dataset evaluate was given is not changed.
            testCase.verifyNotEmpty(d.callables);
        end

        function aStrictReadRefusesAStructuralFault(testCase)
        %aStrictReadRefusesAStructuralFault  Section 2: a read is
        %   strict by default and refuses a file that breaks a
        %   structural rule, because half of such a file is worse
        %   than none of it.
            for probe = {{'err_e01', 'E01'}, {'err_e16', 'E16'}, ...
                         {'err_e19', 'E19'}, {'err_e25', 'E25'}, ...
                         {'err_e26', 'E26'}, {'err_e29', 'E29'}, ...
                         {'err_e30', 'E30'}}
                name = probe{1}{1};
                rule = probe{1}{2};
                file = fullfile(corpusRoot(), name, 'case.mes');
                err = [];
                try
                    mestra.read(file);
                catch err %#ok<CTCH>
                end
                testCase.verifyNotEmpty(err, ...
                    sprintf('%s was read without a word', name));
                testCase.verifyEqual(err.identifier, ['mestra:' rule], ...
                                     err.identifier);
                if ~strcmp(rule, 'E01')
                    d = mestra.read(file, 'Strict', false);
                    testCase.verifyTrue( ...
                        any(startsWith(d.skipped, [rule ' '])), ...
                        sprintf('%s: %s', name, strjoin(d.skipped, '; ')));
                end
            end
        end

        function theMetadataOpenNamesWhatTheReadNames(testCase)
        %theMetadataOpenNamesWhatTheReadNames  Section 30: opening a
        %   file for its metadata alone must refuse with the same ids
        %   as an operation that reads a slot, rather than return
        %   something.  Two entry points naming different rules is the
        %   fault this guards against; the nine structural rules of
        %   section 2 of docs/api-conventions.md are decided
        %   from attributes and dataspaces, so the open reaches the
        %   same verdict without reading an array.
        %
        %   Section 7 names the one place the two part company: an
        %   open never reads a dataset inside a callable's dictionary,
        %   so a dictionary holding a dataset above the element cap is
        %   E41 from the read and nothing from the open.  No file in
        %   any corpus is one, so the last file below is built here,
        %   and the exemption is checked rather than assumed.  The
        %   open still walks the dictionary, which is why a dictionary
        %   nested past the cap is still E41 from both.
            files = {};
            for name = CorpusTest.allCases()
                files{end + 1} = CorpusTest.caseFile(name{1}); %#ok<AGROW>
            end
            for name = HostileSubsetTest.allCases()
                files{end + 1} = HostileSubsetTest.caseFile(name{1}); %#ok<AGROW>
            end
            for name = HostileTest.allCases()
                files{end + 1} = HostileTest.caseFile(name{1}); %#ok<AGROW>
            end
            oversized = [tempname() '.mes'];
            cleanup = onCleanup( ...
                @() ConventionsTest.removeIfPresent(oversized)); %#ok<NASGU>
            ConventionsTest.putOversizedDictionaryDataset( ...
                fullfile(corpusRoot(), 'affine_zero_rows', 'case.mes'), ...
                oversized);
            files{end + 1} = oversized;
            exempted = 0;
            for i = 1:numel(files)
                file = files{i};
                opened = ConventionsTest.identifierOf(@() mestra.open(file));
                read = ConventionsTest.identifierOf(@() mestra.read(file));
                if isempty(opened) && strcmp(read, 'mestra:E41')
                    % Section 29 puts the element cap on an eager read
                    % alone, so an open that returns here is right.
                    exempted = exempted + 1;
                    continue
                end
                testCase.verifyEqual(opened, read, sprintf( ...
                    '%s: the open says "%s" and the read says "%s"', ...
                    file, opened, read));
            end
            testCase.verifyEqual(exempted, 1, ...
                ['the built file is the only one the element cap parts, ' ...
                 'and it does']);

            % The open reads no dictionary and says so by leaving it
            % empty, rather than handing back one with its arrays
            % missing.  The read has it.
            file = fullfile(corpusRoot(), 'callable_two_slots', 'case.mes');
            testCase.verifyEqual( ...
                mestra.open(file).callable('m2').dict.Count, uint64(0));
            testCase.verifyGreaterThan( ...
                mestra.read(file).callable('m2').dict.Count, 0);
            testCase.verifyEqual(mestra.open(file).callable('m2').type, ...
                mestra.read(file).callable('m2').type, ...
                'and names it from its attributes either way');
        end

        function aSemanticFaultNeverStopsARead(testCase)
        %aSemanticFaultNeverStopsARead  Section 2: a missing unit or a
        %   bad split is a finding about a file that still opens, so
        %   that mestra.info works on the files a user most needs to
        %   inspect.
            for name = {'err_e02', 'err_e11', 'err_e39', 'warn_w01'}
                file = fullfile(corpusRoot(), name{1}, 'case.mes');
                d = mestra.read(file);
                testCase.verifyClass(d, 'mestra.Dataset');
                testCase.verifyNotEmpty(mestra.info(file, 'String', true));
            end
        end

        % ------------------------------------- 5. validator output

        function perRowRulesReportOnceWithACountAndRows(testCase)
        %perRowRulesReportOnceWithACountAndRows  Section 5: W02, W03
        %   and W04 could each fire once per row and fire once, with
        %   the count and the first three row indices.
            for probe = {{'warn_w02', 'W02'}, {'warn_w03', 'W03'}, ...
                         {'warn_w04', 'W04'}}
                name = probe{1}{1};
                id = probe{1}{2};
                r = mestra.validate(fullfile(corpusRoot(), name, 'case.mes'));
                hits = r.findings(strcmp({r.findings.id}, id));
                testCase.verifyNumElements(hits, 1, ...
                    sprintf('%s fires once on %s', id, name));
                testCase.verifyTrue( ...
                    ~isempty(regexp(hits(1).message, 'row', 'once')), ...
                    sprintf('%s names a row: %s', id, hits(1).message));
                testCase.verifyTrue( ...
                    ~isempty(regexp(hits(1).message, '\d', 'once')), ...
                    sprintf('%s gives a count: %s', id, hits(1).message));
            end
        end

        function oneFindingPerRulePerObject(testCase)
        %oneFindingPerRulePerObject  Section 5: a rule that has
        %   already fired at a path does not fire again, so a report
        %   has one line per thing that is wrong and not one line per
        %   way of noticing it.
        %
        %   CorpusTest checks this on every case of the corpus, where
        %   the file is validated anyway.  These are the cases where
        %   one object breaks a rule in more than one way, which is
        %   where a second finding would come from.
            for name = {'err_e16', 'err_e19', 'err_e25', 'err_e31', ...
                        'err_e38', 'warn_w03'}
                r = mestra.validate(fullfile(corpusRoot(), name{1}, ...
                                             'case.mes'));
                testCase.verifyNotEmpty(r.findings, name{1});
                keys = arrayfun(@(f) [f.id ' ' f.path], r.findings, ...
                                'UniformOutput', false);
                testCase.verifyEqual(numel(unique(keys)), numel(keys), ...
                    sprintf('%s repeats a rule at one path', name{1}));
            end
        end

        function reportPrintsTheAgreedLines(testCase)
        %reportPrintsTheAgreedLines  Section 5: `<id> <path>:
        %   <message>` and then `<n> error(s), <m> warning(s)`.
            r = mestra.validate(fullfile(corpusRoot(), 'warn_w02', ...
                                         'case.mes'));
            text = mestra.report(r, 'String', true);
            lines = strsplit(strtrim(text), newline);
            testCase.verifyTrue( ...
                ~isempty(regexp(lines{1}, '^W02 /keys/\w+: \S', 'once')), ...
                lines{1});
            testCase.verifyTrue( ...
                ~isempty(regexp(lines{end}, ...
                                '^\d+ error\(s\), \d+ warning\(s\)$', ...
                                'once')), lines{end});
            testCase.verifyEqual(mestra.report(r, 'File', ...
                                 ConventionsTest.devNull()), 0, ...
                'the return value is the error count');
        end

        function infoPrintsTheFieldsOfSectionFive(testCase)
        %infoPrintsTheFieldsOfSectionFive  Name, role, units, bounds,
        %   category, trajectory group and parent for every key; kind,
        %   counts and id for every support; a named shape, units and
        %   source for every slot.
            text = mestra.info(fullfile(corpusRoot(), ...
                'transient_fixed_mesh', 'case.mes'), 'String', true);
            for want = {'t', 'time', 'units s', 'trajectory group run', ...
                        'kind mesh', '6 node(s)', '2 cell(s)', 'id ', ...
                        'row=5 node=6 component=1', 'units K', ...
                        'source data', 'category run'}
                testCase.verifySubstring(text, want{1});
            end
        end

        function infoNamesACallableAndItsOutput(testCase)
        %infoNamesACallableAndItsOutput  Section 5 asks for the
        %   callable id and the output of a callable slot.
            text = mestra.info(fullfile(corpusRoot(), 'affine_zero_rows', ...
                                        'case.mes'), 'String', true);
            testCase.verifySubstring(text, 'source callable ');
            testCase.verifySubstring(text, 'output ');
            testCase.verifySubstring(text, 'callables');
        end

        function infoPrintsAShapeItNeverRead(testCase)
        %infoPrintsAShapeItNeverRead  Section 29: opening a file must
        %   not read any array, and must still report every slot with
        %   its attributes.  mestra.info goes through mestra.open, so
        %   the shapes it prints come from the dataspaces alone.
            file = fullfile(corpusRoot(), 'mesh_two_rows', 'case.mes');
            d = mestra.open(file);
            slots = d.slots();
            for i = 1:numel(slots)
                testCase.verifyEmpty(slots(i).slot.values, ...
                    sprintf('%s was read', slots(i).path));
            end
            text = mestra.info(file, 'String', true);
            testCase.verifySubstring(text, 'row=2 node=6 component=1');
            testCase.verifySubstring(text, 'kind mesh');
        end

        % ------------------------------------- 6. messages

        function everyBuilderRefusalNamesItsRuleFirst(testCase)
        %everyBuilderRefusalNamesItsRuleFirst  Section 6: the rule id
        %   first, then the object path, then what to do.  The model
        %   is Python's E04 message.
            probes = { ...
                @() ConventionsTest.twoRows().addKey('x', [1 2], 'wrong'), ...
                @() ConventionsTest.twoRows().addKey('x', [1 2], 'design'), ...
                @() ConventionsTest.twoRows().addScalar('x', [1 2]), ...
                @() ConventionsTest.twoRows().addNodeArray('s0', 'x', ...
                        1:6, 'field'), ...
                @() ConventionsTest.twoRows().addNodeArray('s0', 'x', ...
                        1:6, 'nonsense', '1'), ...
                @() ConventionsTest.twoRows().addCallableSlot('s0', 'x', ...
                        'field', 'Pa', 'm1', 'x')};
            for i = 1:numel(probes)
                err = [];
                try
                    probes{i}();
                catch err %#ok<CTCH>
                end
                testCase.verifyNotEmpty(err, sprintf('probe %d refused', i));
                id = err.identifier;
                testCase.verifyTrue(startsWith(id, 'mestra:'), id);
                rule = id(8:end);
                if ~isempty(regexp(rule, '^[EW]\d\d$', 'once'))
                    testCase.verifyTrue( ...
                        startsWith(err.message, [rule ': ']), ...
                        sprintf('%s: %s', rule, err.message));
                    testCase.verifySubstring(err.message, '/');
                end
            end
        end

        function theVectorMessageSaysWhatMatlabVectorsAre(testCase)
        %theVectorMessageSaysWhatMatlabVectorsAre  M3: MATLAB has no
        %   one-dimensional array, so naming one axis of a plain
        %   vector can never be right, and the message says so and
        %   says what to write instead.
            d = ConventionsTest.twoRows();
            err = ConventionsTest.errorFrom(testCase, ...
                @() d.addNodeArray('s0', 'x', linspace(0, 1, 6), ...
                                   'field', '1', 'Dims', {'node'}), ...
                'mestra:dims');
            testCase.verifySubstring(err.message, '1-by-N');
            testCase.verifySubstring(err.message, 'component');
        end

        function aNameThatIsNotInTheFileCarriesNoRuleId(testCase)
        %aNameThatIsNotInTheFileCarriesNoRuleId  The identifiers of
        %   section 14 say something about
        %   a file, and a caller who catches one to detect a malformed
        %   file must not catch a typo as well.
            d = mestra.read(fullfile(corpusRoot(), 'mesh_two_rows', ...
                                     'case.mes'));
            err = ConventionsTest.errorFrom(testCase, ...
                @() mestra.integrate(d, 'CL'), 'mestra:integrate');
            testCase.verifySubstring(err.message, 'CL');
            testCase.verifySubstring(err.message, 'pressure');
        end

        function theLeakedUnitIsNamedInW01(testCase)
        %theLeakedUnitIsNamedInW01  The model message of section 6
        %   names the unit that leaked and then says why it matters.
            r = mestra.validate(fullfile(corpusRoot(), 'warn_w01', ...
                                         'case.mes'));
            hits = r.findings(strcmp({r.findings.id}, 'W01'));
            testCase.verifyNumElements(hits, 1);
            testCase.verifySubstring(hits(1).message, 'member');
            testCase.verifySubstring(hits(1).message, ...
                                     'not a generalisation test');
        end
    end

    methods (Static)

        function tf = hasRootGroup(path, name)
        %hasRootGroup  True when the file carries that root group.
            listing = h5info(path, '/');
            tf = false;
            for i = 1:numel(listing.Groups)
                if strcmp(listing.Groups(i).Name, ['/' name])
                    tf = true;
                    return
                end
            end
        end

        function putOversizedDictionaryDataset(src, dst)
        %putOversizedDictionaryDataset  A copy of SRC whose callable's
        %   dictionary holds one dataset above the element cap,
        %   declared and never written, so the file stays small.
            copyfile(src, dst);
            fileattrib(dst, '+w');
            fid = H5F.open(dst, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
            closer = onCleanup(@() H5F.close(fid)); %#ok<NASGU>
            listing = h5info(dst, '/callables');
            gid = H5G.open(fid, listing.Groups(1).Name);
            n = mestra.limits().maxElements + 1000;
            did = mestra.internal.H5.createDataset(gid, 'oversized', ...
                                                   'float64', n, n, 1024);
            % Its own dimension scale, as section 21 names it, so that
            % the element cap is the only thing wrong with the file.
            % A scale chunked over its whole length would be one chunk
            % past the library's 4 GiB limit, so it is chunked as the
            % dataset is; nothing is ever written to it either way.
            sid = mestra.internal.H5.createDataset(gid, ...
                'mestra_oversized_d0', 'float32', n, n, 1024, [], 0, ...
                mestra.internal.H5.crtOrderTrackedIndexed());
            H5DS.set_scale(sid, sprintf('%s%10d', ...
                mestra.internal.H5.SENTENCE, n));
            H5DS.attach_scale(did, sid, 0);
            H5D.close(sid);
            H5D.close(did);
            H5G.close(gid);
        end

        function id = identifierOf(thunk)
        %identifierOf  The identifier a call refused with, or ''.
            id = '';
            try
                thunk();
            catch err
                id = err.identifier;
            end
        end


        function d = twoRows()
        %twoRows  A two-row, two-member, six-node dataset, built the
        %   way the conventions say to build one.
            d = mestra.Dataset();
            d.writer = 'mestra matlab tests';
            d.created = '2026-09-19T00:00:00Z';
            d.addCategoryTable('member', {'wing_a', 'wing_b'});
            d.addKey('mach', [0.4 0.8], 'condition', '1');
            d.addKey('member', int32([0 1]), 'group', 'Category', 'member');
            d.setGeneralisationGroup('member');
            coords = cat(3, [0 1 2 0 1 2; 0 0 0 1 1 1], ...
                            [0 1.5 3 0 1.5 3; 0 0 0 1 1 1]);
            d.addMeshSupport('s0', coords, uint8([9 9]), int64([0 4 8]), ...
                             int64([0 1 4 3 1 2 5 4]), 'm', ...
                             'Dims', {'component', 'node', 'group:member'});
        end

        function slots = brokenSlot(support)
        %brokenSlot  A node array whose leading extent is one where
        %   the file has two rows: E16, reached past the builder.
            slots = support.nodeArrays;
            rec = mestra.Dataset.emptySlot();
            rec(1).name = 'e16';
            rec(1).role = 'field';
            rec(1).units = '1';
            rec(1).varies = 'row';
            rec(1).source = 'data';
            rec(1).values = ones(1, 6, 1);
            rec(1).dims = {'component', 'node', 'row'};
            rec(1).shape = [1 6 1];
            rec(1).dtype = 'float64';
            rec(1).location = 'node';
            rec(1).support = support.name;
            rec(1).components = 1;
            slots = mestra.Dataset.append(slots, rec, 'e16');
        end

        function err = errorFrom(testCase, fcn, identifier)
            err = [];
            try
                fcn();
            catch err %#ok<CTCH>
            end
            testCase.verifyNotEmpty(err, 'the call was expected to refuse');
            testCase.verifyEqual(err.identifier, identifier, err.message);
        end

        function fid = devNull()
            persistent handle
            if isempty(handle) || handle < 0
                handle = fopen('/dev/null', 'w');
            end
            fid = handle;
        end

        function removeIfPresent(path)
            if exist(path, 'file') == 2
                delete(path);
            end
        end
    end
end
