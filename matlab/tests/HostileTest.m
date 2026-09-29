classdef HostileTest < matlab.unittest.TestCase
%HostileTest  Files built to break a reader that trusts them.
%
%   The conformance corpus says what a conforming file means.  These
%   files are the other half: none of them conforms, and none of them
%   is the sort of mistake a writer makes by accident.  They exist to
%   fix what happens when a reader is handed one anyway.
%
%   The contract every one of them is held to:
%
%     * mestra.read, mestra.open, mestra.validate and the dataset's
%       readRows method either return, or raise an error whose
%       identifier is mestra: followed by the rule of section 14 that
%       the file breaks, or mestra:reader when the file will not open
%       at all.  No library stack trace reaches the caller, nothing
%       recurses until the stack gives out, and nothing tries to
%       allocate more memory than mestra.limits allows;
%     * mestra.validate always returns.  An object it cannot read is
%       E41 with that object's path, a link it will not follow is
%       E40, and the pass carries on to the end of the file;
%     * no file is ever opened that the caller did not name.
%
%   A per-file time budget stands in for a timeout.  MATLAB cannot
%   interrupt a call in its own process without a toolbox, so what is
%   asserted is that each file finished well inside the budget; a
%   genuine hang would stop the suite, which the wall clock of
%   whatever runs it catches.  Nothing here can hang by construction:
%   every walk is bounded by mestra.limits and no loop depends on a
%   number the file chose.
%
%   The files are built by tests/hostile/make_hostile.py with h5py,
%   which can say things MATLAB's HDF5 interface cannot say at all.
%
%   See also CorpusTest, PackageTest, mestra.limits, run_tests.

    properties (Constant)
        % Seconds one hostile file may take through all four entry
        % points.  Generous: the slowest is a few hundred milliseconds.
        BUDGET = 30

        % The identifiers a caller may ever see from these files: a
        % rule of section 14 (E01 for a file that will not open at
        % all), or mestra:reader for a failure below the rules.
        ALLOWED = {'mestra:E01', 'mestra:reader', 'mestra:E19', ...
                   'mestra:E25', 'mestra:E26', 'mestra:E29', ...
                   'mestra:E40', 'mestra:E41'}
    end

    properties (TestParameter)
        hostileCase
    end

    methods (TestParameterDefinition, Static)
        function hostileCase = initialiseCases()
            hostileCase = HostileTest.allCases();
        end
    end

    methods (Static)

        function dir_ = casesDir()
            here = fileparts(mfilename('fullpath'));
            dir_ = fullfile(here, 'hostile', 'cases');
        end

        function names = allCases()
            listing = dir(fullfile(HostileTest.casesDir(), '*.mes'));
            names = {};
            for i = 1:numel(listing)
                [~, stem] = fileparts(listing(i).name);
                % external_target is the file an external link points
                % at; it is a case only in that it must never be read.
                if strcmp(stem, 'external_target'), continue, end
                names{end + 1} = stem; %#ok<AGROW>
            end
            names = sort(names);
        end

        function p = caseFile(name)
            p = fullfile(HostileTest.casesDir(), [name '.mes']);
        end
    end

    methods (Test)

        function survivesEveryEntryPoint(testCase, hostileCase)
        %survivesEveryEntryPoint  The contract above, for one file.
            path = HostileTest.caseFile(hostileCase);
            started = tic;

            % ---- validate always returns -------------------------
            outcome = [];
            try
                outcome = mestra.validate(path);
            catch err
                testCase.verifyTrue(ismember(err.identifier, ...
                    HostileTest.ALLOWED), ...
                    sprintf('%s: validate raised %s', hostileCase, ...
                            err.identifier));
            end
            if ~isempty(outcome)
                testCase.verifyTrue(iscell(outcome.errors));
                testCase.verifyTrue(iscell(outcome.warnings));
                testCase.verifyTrue(iscell(outcome.errors));
                for i = 1:numel(outcome.findings)
                    f = outcome.findings(i);
                    testCase.verifyNotEmpty(f.path, ...
                        sprintf('%s: the finding %s has no path', ...
                                hostileCase, f.id));
                    testCase.verifyFalse(any(strcmp(f.id, {'E07', 'W09'})), ...
                        'a retired rule identifier was emitted');
                end
            end

            % ---- open and read ------------------------------------
            for fn = {@mestra.open, @mestra.read}
                try
                    d = fn{1}(path, 'Strict', false);
                    testCase.verifyClass(d, 'mestra.Dataset');
                    testCase.verifyTrue(iscell(d.skipped));
                catch err
                    testCase.verifyTrue(ismember(err.identifier, ...
                        HostileTest.ALLOWED), ...
                        sprintf('%s: a reader raised %s: %s', ...
                                hostileCase, err.identifier, err.message));
                end
            end

            % ---- readRows over every slot the file offers ---------
            try
                d = mestra.open(path, 'Strict', false);
                slots = d.slots();
                for i = 1:numel(slots)
                    try
                        d.readRows(slots(i).path, [1 1]);
                    catch err
                        testCase.verifyTrue(ismember(err.identifier, ...
                            [HostileTest.ALLOWED {'mestra:rowRange'}]), ...
                            sprintf('%s: readRows on %s raised %s', ...
                                    hostileCase, slots(i).path, ...
                                    err.identifier));
                    end
                end
            catch err
                testCase.verifyTrue(ismember(err.identifier, ...
                    HostileTest.ALLOWED), ...
                    sprintf('%s: open raised %s', hostileCase, ...
                            err.identifier));
            end

            elapsed = toc(started);
            testCase.verifyLessThan(elapsed, HostileTest.BUDGET, ...
                sprintf('%s took %.1f s, past the %d s budget', ...
                        hostileCase, elapsed, HostileTest.BUDGET));
        end

        function externalLinkIsNeverFollowed(testCase)
        %externalLinkIsNeverFollowed  The one that reads another file.
        %   The target carries a support and a scalar of its own.  A
        %   reader that followed the link would report them as this
        %   file's; this one reports the link and reads neither.
            path = HostileTest.caseFile('link_external');
            testCase.verifyError(@() mestra.read(path), 'mestra:E40', ...
                'a strict read refuses a file with an external link');
            d = mestra.read(path, 'Strict', false);
            testCase.verifyEqual(numel(d.supports), 1, ...
                'only the support this file holds itself');
            testCase.verifyEqual(numel(d.callables), 0, ...
                'the target''s callables group is not this file''s');
            testCase.verifyGreaterThanOrEqual(numel(d.skipped), 4, ...
                'every external link is reported');
            for i = 1:numel(d.skipped)
                testCase.verifySubstring(d.skipped{i}, 'not followed');
            end
            r = mestra.validate(path);
            testCase.verifyTrue(ismember('E40', r.errors), ...
                'the links are reported as E40');
            for i = 1:numel(r.findings)
                testCase.verifyEmpty( ...
                    strfind(r.findings(i).path, 'external_target'), ...
                    'no finding may come from the other file'); %#ok<STRCL1>
            end
        end

        function softLinksAreReportedNotFollowed(testCase)
        %softLinksAreReportedNotFollowed  Dangling and cyclic alike.
            for name = {'link_soft_dangling', 'link_soft_cycle'}
                r = mestra.validate(HostileTest.caseFile(name{1}));
                testCase.verifyTrue(ismember('E40', r.errors), ...
                    sprintf('%s: the links are reported', name{1}));
                d = mestra.read(HostileTest.caseFile(name{1}), 'Strict', false);
                testCase.verifyGreaterThanOrEqual(numel(d.skipped), 4, ...
                    sprintf('%s: four links, four notes', name{1}));
            end
        end

        function enormousShapeIsRefusedNotAttempted(testCase)
        %enormousShapeIsRefusedNotAttempted  Ten to the twelve elements.
        %   Opening must cost nothing, one row range must read one row,
        %   and the whole dataset must be refused by the stated limit
        %   rather than by whatever the machine happens to have.
        %
        %   A strict read never gets as far as the limit, and should
        %   not: a dataset of a million rows in a file of two breaks
        %   E16, which is a structural rule, and a strict read refuses
        %   a file that breaks one (docs/api-conventions.md, section
        %   2).  The element limit is what refuses the dataset once
        %   the caller has asked for it anyway, which is a non-strict
        %   read, a row range wider than the limit, or the validator;
        %   all three are checked below.
            path = HostileTest.caseFile('huge_shape');
            started = tic;
            d = mestra.open(path, 'Strict', false);
            testCase.verifyLessThan(toc(started), 5, ...
                'opening must not touch the data');

            err = [];
            try
                mestra.read(path);
            catch err %#ok<CTCH>
            end
            testCase.verifyNotEmpty(err, 'a strict read must refuse it');
            testCase.verifyEqual(err.identifier, 'mestra:E16', ...
                'the first structural rule it breaks is the row count');

            started = tic;
            lenient = mestra.read(path, 'Strict', false);
            testCase.verifyLessThan(toc(started), 10, ...
                'and it still never attempts the allocation');
            testCase.verifyTrue( ...
                any(startsWith(lenient.skipped, 'E41 ')), ...
                sprintf(['the element limit is what refuses the data: ' ...
                         '%s'], strjoin(lenient.skipped, '; ')));
            testCase.verifyTrue( ...
                any(contains(lenient.skipped, 'enormous')), ...
                'and it names the dataset it would not read');

            one = d.readRows('/supports/s0/node_arrays/enormous', [1 1]);
            testCase.verifyEqual(numel(one.values), 1e6, ...
                'one row of it, and not the other million');

            testCase.verifyError( ...
                @() d.readRows('/supports/s0/node_arrays/wide_row', [1 1]), ...
                'mestra:E41', ...
                'a single row larger than the limit is refused too');

            r = mestra.validate(path);
            testCase.verifyTrue(ismember('E41', r.errors), ...
                'the validator records what it could not read');
        end

        function aFailedReadDoesNotEndThePass(testCase)
        %aFailedReadDoesNotEndThePass  The rule the C++ review found.
        %   /keys/aaa_broken has a checksum that does not match, and
        %   sorts before /keys/zzz_late, which is missing its units.
        %   A validator that stopped at the first failure would never
        %   report E39.
            r = mestra.validate(HostileTest.caseFile('read_fails_midway'));
            testCase.verifyTrue(ismember('E39', r.errors), ...
                'the finding after the unreadable object is still found');
            testCase.verifyTrue(ismember('E41', r.errors), ...
                'and the unreadable object is recorded');
            found = false;
            for i = 1:numel(r.findings)
                hit = strfind(r.findings(i).path, 'aaa_broken');
                if strcmp(r.findings(i).id, 'E41') && ~isempty(hit)
                    found = true;
                end
            end
            testCase.verifyTrue(found, ...
                'the unclassified finding names the object');
        end

        function deepNestingIsBounded(testCase)
        %deepNestingIsBounded  Committed depth, and thirty thousand.
        %   The committed files are a thousand levels, which is enough
        %   to prove the bound.  The thirty thousand case is built
        %   here instead of committed, because it is four megabytes
        %   of object headers and nothing else.
            for name = {'deep_callables', 'deep_root'}
                d = mestra.read(HostileTest.caseFile(name{1}), 'Strict', false);
                testCase.verifyNotEmpty(d.skipped, ...
                    sprintf('%s: the bound is reported', name{1}));
            end

            path = [tempname() '.mes'];
            cleanup = onCleanup(@() HostileTest.removeIfPresent(path));
            HostileTest.buildDeepFile(path, 30000);
            started = tic;
            d = mestra.read(path, 'Strict', false);
            r = mestra.validate(path);
            testCase.verifyLessThan(toc(started), HostileTest.BUDGET, ...
                'thirty thousand levels finished inside the budget');
            testCase.verifyNotEmpty(d.skipped, ...
                'the reader says it stopped');
            testCase.verifyTrue(ismember('E41', r.errors) || ...
                                ismember('W11', r.warnings), ...
                'and the validator reports it');
        end

        function hardLinkCycleTerminates(testCase)
        %hardLinkCycleTerminates  Two groups, one link, infinite depth.
            path = HostileTest.caseFile('hard_link_cycle');
            started = tic;
            d = mestra.read(path, 'Strict', false);
            testCase.verifyLessThan(toc(started), HostileTest.BUDGET);
            testCase.verifyNotEmpty(d.skipped, ...
                'the walk says where it stopped');
        end

        function arrayAttributesAreRefusedNotUsed(testCase)
        %arrayAttributesAreRefusedNotUsed  A scalar is one value.
            for name = {'attr_array_root', 'attr_array_key', ...
                        'attr_array_slot'}
                path = HostileTest.caseFile(name{1});
                r = mestra.validate(path);
                testCase.verifyTrue(ismember('E19', r.errors), ...
                    sprintf('%s: a named attribute that is not a scalar', ...
                            name{1}));
                d = mestra.read(path, 'Strict', false);
                testCase.verifyTrue(ischar(d.format));
                testCase.verifyTrue(isscalar(d.aligned) || ...
                                    islogical(d.aligned));
                for i = 1:numel(d.keys)
                    testCase.verifyTrue(ischar(d.keys(i).units));
                    testCase.verifyTrue(isempty(d.keys(i).lower) || ...
                                        isscalar(d.keys(i).lower));
                    testCase.verifyTrue(isempty(d.keys(i).upper) || ...
                                        isscalar(d.keys(i).upper));
                end
                slots = d.slots();
                for i = 1:numel(slots)
                    slot = slots(i).slot;
                    if ~isfield(slot, 'components'), continue, end
                    testCase.verifyTrue(isempty(slot.components) || ...
                                        isscalar(slot.components), ...
                        'components is one number or none');
                    testCase.verifyTrue(isempty(slot.quantile) || ...
                                        isscalar(slot.quantile), ...
                        'quantile is one number or none');
                end
            end
        end

        function wrongKindIsPassedOverNotOpened(testCase)
        %wrongKindIsPassedOverNotOpened  A key that is a group, and so on.
            path = HostileTest.caseFile('kind_swap');
            d = mestra.read(path, 'Strict', false);
            testCase.verifyGreaterThanOrEqual(numel(d.skipped), 3);
            names = {d.keys.name};
            testCase.verifyFalse(ismember('regime', names), ...
                'a key that is a group is not a key');
            supports = {d.supports.name};
            testCase.verifyFalse(ismember('s1', supports), ...
                'a support that is a dataset is not a support');
            testCase.verifyTrue(ismember('s0', supports));
            testCase.verifyTrue(ismember('s2', supports), ...
                'the other real support is still read');
            testCase.verifyEmpty(d.support('s2').nodeArrays, ...
                'node_arrays that is a dataset holds no arrays');
            testCase.verifyEmpty({d.categories.name}, ...
                'a category table that is a group is not a table');
            % An object of the wrong kind is one the reader cannot read
            % as what its place says it is: E41 in every language, and
            % not a missing attribute (E39) or a missing type (E15).
            r = mestra.validate(path);
            testCase.verifyTrue(ismember('E41', r.errors));
            testCase.verifyFalse(ismember('E39', r.errors));
            testCase.verifyFalse(ismember('E15', r.errors));
        end

        function filtersDoNotUpsetTheDescription(testCase)
        %filtersDoNotUpsetTheDescription  Twenty client values, and an
        %   identifier no build has.
            for name = {'filter_many_cd', 'filter_unknown_id'}
                r = mestra.validate(HostileTest.caseFile(name{1}));
                testCase.verifyTrue(ismember('E29', r.errors), ...
                    sprintf('%s: the filter is reported', name{1}));
            end
        end

        function externalTargetIsNotAmongTheCases(testCase)
        %externalTargetIsNotAmongTheCases  Guard the guard.
            testCase.verifyFalse(ismember('external_target', ...
                HostileTest.allCases()), ...
                'the link target is not run as a case of its own');
            testCase.verifyEqual(exist(fullfile(HostileTest.casesDir(), ...
                'external_target.mes'), 'file'), 2, ...
                'but it is there, so the link really points somewhere');
        end
    end

    methods (Static)

        function removeIfPresent(path)
            if exist(path, 'file') == 2
                delete(path);
            end
        end

        function buildDeepFile(path, depth)
        %buildDeepFile  A file nested `depth` groups deep, from MATLAB.
        %   Built rather than committed: thirty thousand groups is four
        %   megabytes of object headers, which does not belong in a
        %   repository whose conformance files are thirty kilobytes.
            fcpl = mestra.internal.H5.plist('H5P_FILE_CREATE');
            fid = H5F.create(path, 'H5F_ACC_TRUNC', fcpl, 'H5P_DEFAULT');
            H5P.close(fcpl);
            root = H5G.open(fid, '/');
            mestra.internal.H5.writeStrAttr(root, 'created', ...
                                            '2026-09-20T00:00:00Z');
            mestra.internal.H5.writeStrAttr(root, 'format', 'mestra/0');
            mestra.internal.H5.writeStrAttr(root, 'writer', 'hostile test');
            mestra.internal.H5.writeNumAttr(root, 'aligned', int8(1), 'int8');
            scale = mestra.internal.H5.makeScale(root, 'row', 0, true);
            H5D.close(scale);
            gcpl = mestra.internal.H5.plist('H5P_GROUP_CREATE');
            here = H5G.create(root, 'extras', 'H5P_DEFAULT', gcpl, ...
                              'H5P_DEFAULT');
            for i = 1:depth
                next = H5G.create(here, 'g', 'H5P_DEFAULT', gcpl, ...
                                  'H5P_DEFAULT');
                H5G.close(here);
                here = next;
            end
            H5G.close(here);
            H5P.close(gcpl);
            H5G.close(root);
            H5F.close(fid);
        end
    end
end
