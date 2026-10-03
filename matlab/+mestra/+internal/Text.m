classdef Text
%Text  Name, string and timestamp checks the format needs.
%
%   See also mestra.validate.

    methods (Static)

        function tf = legalName(name)
        %legalName  True when a name is a legal netCDF-4 name.
        %   Section 18: not empty, no "/" and no NUL, not beginning or
        %   ending with a space, built from letters, digits,
        %   underscore, hyphen, "." and "+", and beginning with a
        %   letter, a digit or an underscore.
            tf = false;
            if isempty(name), return, end
            if name(1) == ' ' || name(end) == ' ', return, end
            if any(name(1) == '-.+'), return, end
            for i = 1:numel(name)
                c = name(i);
                ok = isletter(c) || any(c == '0123456789_-.+') || ...
                     double(c) > 127;
                if ~ok, return, end
            end
            tf = true;
        end

        function tf = reserved(name)
        %reserved  True when a producer-chosen name uses the reserved
        %   prefix "mestra_" (section 18, E33).
            tf = numel(name) >= 7 && strcmp(name(1:7), 'mestra_');
        end

        function [ok, reason] = checkStringBytes(bytes)
        %checkStringBytes  Validate one stored fixed-length string.
        %   Returns false with a reason when the bytes are not valid
        %   UTF-8, or hold a NUL anywhere but in the trailing padding
        %   (E26).
            bytes = uint8(bytes(:)');
            last = find(bytes ~= 0, 1, 'last');
            if isempty(last)
                ok = true; reason = ''; return
            end
            if any(bytes(1:last) == 0)
                ok = false;
                reason = 'a NUL byte before the trailing padding';
                return
            end
            [ok, reason] = mestra.internal.Text.validUtf8(bytes(1:last));
        end

        function [ok, reason] = validUtf8(bytes)
        %validUtf8  True when a byte vector is well-formed UTF-8.
            ok = false;
            reason = 'not valid UTF-8';
            i = 1;
            n = numel(bytes);
            while i <= n
                b = double(bytes(i));
                if b < 128
                    extra = 0;
                elseif b >= 194 && b <= 223
                    extra = 1;
                elseif b >= 224 && b <= 239
                    extra = 2;
                elseif b >= 240 && b <= 244
                    extra = 3;
                else
                    return
                end
                if i + extra > n, return, end
                for j = 1:extra
                    c = double(bytes(i + j));
                    if c < 128 || c > 191, return, end
                end
                i = i + extra + 1;
            end
            ok = true;
            reason = '';
        end

        function tf = iso8601Utc(text)
        %iso8601Utc  True when a timestamp is ISO 8601 in UTC.
        %   Accepts "YYYY-MM-DDThh:mm:ssZ", the same with a fractional
        %   second, and the "+00:00" spelling of the zone.
            tf = false;
            if isempty(text), return, end
            pattern = ['^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' ...
                       '(\.\d+)?(Z|\+00:00)$'];
            tf = ~isempty(regexp(text, pattern, 'once'));
        end

        function s = decimal(x)
        %decimal  A float64 in the C format "%.17e", or the three
        %   non-finite spellings the corpus uses.
            if isnan(x)
                s = 'nan';
            elseif isinf(x)
                if x > 0, s = 'inf'; else, s = '-inf'; end
            else
                s = sprintf('%.17e', x);
            end
        end

        function x = fromDecimal(text)
        %fromDecimal  Parse a "%.17e" string back to float64, exactly.
        %   sscanf with %lf is used rather than str2double because it
        %   is the C library's own strtod and is exact by construction;
        %   str2double agrees on every value in the corpus, and the
        %   test suite checks that it does.
            text = strtrim(char(text));
            switch lower(text)
                case 'nan', x = NaN; return
                case 'inf', x = Inf; return
                case '-inf', x = -Inf; return
            end
            x = sscanf(text, '%lf', 1);
            if isempty(x)
                error('mestra:internal', 'cannot parse "%s"', text);
            end
        end
    end
end
