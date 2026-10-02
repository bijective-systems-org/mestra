classdef Attrs
%Attrs  The encodings section 18 requires of an attribute, in one place.
%
%   Two callers need the same answer.  The validator reports E19 and
%   E26 against a path; the reader refuses a file that breaks one of
%   them, because both are structural rules and section 2 of
%   docs/api-conventions.md has a strict read refuse a file that breaks
%   a structural rule.  They ask this class rather than each carrying
%   its own reading of section 18, so that the two cannot drift apart.
%
%   Everything here is decided from attributes alone, which is what
%   lets mestra.open apply the same rules as mestra.read without
%   reading an array (section 29).
%
%   See also mestra.validate, mestra.internal.Reader.

    methods (Static)

        function m = kinds()
        %kinds  The encoding section 18 requires of each named attribute.
            m = containers.Map( ...
                {'format', 'writer', 'created', 'generalisation_group', ...
                 'role', 'units', 'category', 'trajectory_group', ...
                 'parent', 'kind', 'support_id', 'varies', 'source', ...
                 'output', 'statistic', 'of', 'derived_from', 'recipe', ...
                 'reference', 'type', 'repr', 'aligned', 'recomputed', ...
                 'n_nodes', 'n_cells', 'components', 'lower', 'upper', ...
                 'quantile', 'level', 'method'}, ...
                {'string', 'string', 'string', 'string', ...
                 'string', 'string', 'string', 'string', ...
                 'string', 'string', 'string', 'string', 'string', ...
                 'string', 'string', 'string', 'string', 'string', ...
                 'string', 'string', 'string', 'int8', 'int8', ...
                 'int64', 'int64', 'int64', 'float64', 'float64', ...
                 'float64', 'float64', 'string'});
        end

        function found = findings(oid, only)
        %findings  Every section 18 fault on one object's attributes.
        %   With ONLY, a cell array of names, just those attributes are
        %   looked at: on a callable's group every other attribute is
        %   an entry of its dictionary, which section 25 encodes and the
        %   codec checks, and the null sentinel there is legal.
        %   A 1-by-n struct array with fields `id`, `attr` and
        %   `message`, in the order the attributes come back in.  W11
        %   is included, because an attribute this version does not
        %   know and cannot use is a warning and not an error; a
        %   caller that only refuses errors passes over it.
            H5 = mestra.internal.H5;
            found = mestra.internal.Attrs.none();
            kinds = mestra.internal.Attrs.kinds();
            names = H5.publicAttrNames(oid);
            if nargin > 1
                names = names(ismember(names, only));
            end
            for name = names
                try
                    % One open of the attribute answers both what it
                    % is and what it holds.  Every finding below needs
                    % one or the other, and asking separately opened
                    % each attribute of each object two or three times.
                    info = H5.attrDetail(oid, name{1});
                catch err
                    found = mestra.internal.Attrs.add(found, 'E41', ...
                        name{1}, sprintf( ...
                        'the attribute %s would not be described: %s', ...
                        name{1}, regexprep(strtrim(err.message), '\s+', ' ')));
                    continue
                end
                if strcmp(info.type, 'vlstring')
                    found = mestra.internal.Attrs.add(found, 'E19', ...
                        name{1}, sprintf( ...
                        'the attribute %s is a variable-length string', ...
                        name{1}));
                    continue
                end
                if ~info.scalar
                    % Section 18 gives every attribute this format
                    % names a scalar dataspace, so an array there is
                    % the wrong encoding whatever its type.
                    if kinds.isKey(name{1})
                        found = mestra.internal.Attrs.add(found, 'E19', ...
                            name{1}, sprintf( ...
                            ['the attribute %s is an array where a ' ...
                             'scalar is required'], name{1}));
                    else
                        found = mestra.internal.Attrs.add(found, 'W11', ...
                            name{1}, sprintf( ...
                            ['the attribute %s is not a scalar and is ' ...
                             'ignored'], name{1}));
                    end
                    continue
                end
                if kinds.isKey(name{1})
                    want = kinds(name{1});
                    if ~strcmp(info.type, want)
                        found = mestra.internal.Attrs.add(found, 'E19', ...
                            name{1}, sprintf( ...
                            'the attribute %s is %s where %s is required', ...
                            name{1}, info.type, want));
                        continue
                    end
                    if strcmp(want, 'string') && ...
                            (info.cset ~= H5ML.get_constant_value( ...
                                 'H5T_CSET_UTF8') || ...
                             info.strpad ~= H5ML.get_constant_value( ...
                                 'H5T_STR_NULLPAD'))
                        % Section 18: a string is fixed-length UTF-8
                        % with NUL padding, and the character set and
                        % the padding are part of that encoding.
                        found = mestra.internal.Attrs.add(found, 'E19', ...
                            name{1}, sprintf( ...
                            ['the attribute %s is not a UTF-8 string ' ...
                             'with NUL padding'], name{1}));
                        continue
                    end
                    if strcmp(want, 'int8')
                        v = info.raw;
                        if ~any(double(v) == [0 1])
                            found = mestra.internal.Attrs.add(found, 'E19', ...
                                name{1}, sprintf( ...
                                'the boolean %s has the value %g', ...
                                name{1}, double(v)));
                        end
                    elseif strcmp(want, 'float64') && ...
                            ~isempty(info.raw) && ~isfinite(double(info.raw))
                        % Section 18: an attribute that declares a
                        % bound, a level or a quantile must be finite,
                        % and the four float64 attributes are those.
                        found = mestra.internal.Attrs.add(found, 'E19', ...
                            name{1}, sprintf( ...
                            ['the attribute %s declares a bound, a level ' ...
                             'or a quantile and must be finite'], name{1}));
                    end
                end
                if strcmp(info.type, 'string')
                    try
                        bytes = H5.stringBytes(info.raw, name{1});
                        [ok, why] = ...
                            mestra.internal.Text.checkStringBytes(bytes);
                        if ~ok
                            found = mestra.internal.Attrs.add(found, 'E26', ...
                                name{1}, sprintf( ...
                                'the attribute %s has %s', name{1}, why));
                        end
                    catch err
                        found = mestra.internal.Attrs.add(found, 'E41', ...
                            name{1}, sprintf( ...
                            ['the attribute %s could not be read as ' ...
                             'bytes: %s'], name{1}, ...
                            regexprep(strtrim(err.message), '\s+', ' ')));
                    end
                end
            end
        end
    end

    methods (Static)

        function found = freeFindings(oid)
        %freeFindings  Section 18 over attributes the format does not
        %   name, as those under /notes are (section 11): a string
        %   among them is stored as every string is, fixed-length UTF-8
        %   padded with NUL (E19), and holds legal text (E26).  A number
        %   is the producer's choice, and so is the shape.
            H5 = mestra.internal.H5;
            found = mestra.internal.Attrs.none();
            for name = H5.publicAttrNames(oid)
                try
                    info = H5.attrDetail(oid, name{1});
                catch err
                    found = mestra.internal.Attrs.add(found, 'E41', ...
                        name{1}, sprintf( ...
                        'the attribute %s would not be described: %s', ...
                        name{1}, regexprep(strtrim(err.message), '\s+', ' ')));
                    continue
                end
                if strcmp(info.type, 'vlstring')
                    found = mestra.internal.Attrs.add(found, 'E19', ...
                        name{1}, sprintf( ...
                        'the attribute %s is a variable-length string', ...
                        name{1}));
                elseif strcmp(info.type, 'string') && ...
                        (info.cset ~= H5ML.get_constant_value('H5T_CSET_UTF8') || ...
                         info.strpad ~= H5ML.get_constant_value('H5T_STR_NULLPAD'))
                    found = mestra.internal.Attrs.add(found, 'E19', ...
                        name{1}, sprintf( ...
                        ['the attribute %s is not a UTF-8 string with ' ...
                         'NUL padding'], name{1}));
                elseif strcmp(info.type, 'string') && info.scalar
                    bytes = H5.stringBytes(info.raw, name{1});
                    [ok, why] = mestra.internal.Text.checkStringBytes(bytes);
                    if ~ok
                        found = mestra.internal.Attrs.add(found, 'E26', ...
                            name{1}, sprintf('the attribute %s has %s', ...
                                             name{1}, why));
                    end
                end
            end
        end
    end

    methods (Static, Access = private)

        function found = none()
        %none  An empty finding list of the right shape.
            found = struct('id', {}, 'attr', {}, 'message', {});
        end

        function found = add(found, id, attr, message)
            found(end + 1) = struct('id', id, 'attr', attr, ...
                                    'message', message); %#ok<AGROW>
        end
    end
end
