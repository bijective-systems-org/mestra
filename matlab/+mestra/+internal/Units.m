classdef Units
%Units  The units grammar W10 is decided by (specification section 32).
%
%   A units string parses when it matches the grammar of section 32,
%   whatever names it uses: "furlong fortnight-1" parses and "m//s"
%   does not.  Deciding that needs no unit database, so this class has
%   none; converting between units is a tool's business.
%
%   The grammar is written out in section 32 together with the five
%   readings that give one string one verdict.  Each rule below is one
%   production of it.  A rule takes the string and a position and
%   returns whether it matched and the position after what it matched;
%   on a mismatch the position it returns means nothing, and the
%   caller carries on from the position it already holds.
%
%       mestra.internal.Units.parses('W m-2')                  % true
%       mestra.internal.Units.parses('days since 2000-01-01')  % true
%       mestra.internal.Units.parses('m per')                  % false
%
%   See also mestra.validate.

    properties (Constant, Access = private)
        % How deep groups and log references together may nest.  The
        % string comes out of a file, and the bound keeps a recursive
        % parser safe on it.
        MaxDepth = 32
        % The longest string, in bytes of UTF-8, looked at at all.
        MaxBytes = 4096
        % Words that are operators after a power and names elsewhere.
        ShiftWords = {'since', 'from', 'after', 'ref'}
        LogNames = {'log', 'lg', 'ln', 'lb'}
    end

    methods (Static)

        function tf = parses(text)
        %parses  True when a units string matches the grammar.
            tf = false;
            if isstring(text) && isscalar(text)
                text = char(text);
            end
            if ~ischar(text) || (~isempty(text) && ~isrow(text))
                return
            end
            % Characters before bytes: a string of more characters
            % than the bound has more bytes too, and need not be
            % encoded to be refused.
            if numel(text) > mestra.internal.Units.MaxBytes || ...
               numel(unicode2native(text, 'UTF-8')) > ...
               mestra.internal.Units.MaxBytes
                return
            end
            [ok, at] = mestra.internal.Units.units(text, 1, 0);
            tf = ok && at == numel(text) + 1;
        end
    end

    methods (Static, Access = private)

        % units = { sp } product [ shift ] { sp }
        function [ok, at] = units(s, at, depth)
            at = mestra.internal.Units.spaces(s, at);
            [ok, at] = mestra.internal.Units.product(s, at, depth);
            if ~ok, return, end
            [shifted, after] = mestra.internal.Units.shift(s, at);
            if shifted
                at = after;
            end
            at = mestra.internal.Units.spaces(s, at);
        end

        % product = power { [ operator ] power }
        function [ok, at] = product(s, at, depth)
            [ok, at] = mestra.internal.Units.power(s, at, depth);
            if ~ok, return, end
            while true
                save = at;
                at = mestra.internal.Units.spaces(s, at);
                c = mestra.internal.Units.ch(s, at);
                if isempty(c) || c == ')' || c == '@' || ...
                   mestra.internal.Units.shiftWordAt(s, at)
                    at = save;
                    return
                end
                if any(c == '*./-')
                    at = at + 1;
                elseif mestra.internal.Units.word(s, at, 'per')
                    at = at + 3;
                else
                    % Juxtaposition: a space, or nothing at all,
                    % between two powers is a product.
                    [next, after] = mestra.internal.Units.power(s, at, depth);
                    if next
                        at = after;
                        continue
                    end
                    at = save;
                    return
                end
                at = mestra.internal.Units.spaces(s, at);
                [ok, at] = mestra.internal.Units.power(s, at, depth);
                if ~ok, return, end
            end
        end

        % power = basic [ integer | ( "^" | "**" ) number ]
        function [ok, at] = power(s, at, depth)
            [ok, at] = mestra.internal.Units.basic(s, at, depth);
            if ~ok, return, end
            c = mestra.internal.Units.ch(s, at);
            if mestra.internal.Units.isDigit(c) || ...
               (~isempty(c) && any(c == '+-') && ...
                mestra.internal.Units.isDigit( ...
                    mestra.internal.Units.ch(s, at + 1)))
                % An integer power follows its basic with nothing
                % between: "m-1" is m to the power -1.
                [~, at] = mestra.internal.Units.digits(s, at + 1, Inf);
            elseif isequal(c, '^')
                [ok, at] = mestra.internal.Units.number(s, at + 1);
            elseif at + 1 <= numel(s) && strcmp(s(at:at + 1), '**')
                [ok, at] = mestra.internal.Units.number(s, at + 2);
            end
        end

        % basic = name | number | "(" units ")" | logname logref
        function [ok, at] = basic(s, at, depth)
            c = mestra.internal.Units.ch(s, at);
            if isequal(c, '(')
                [ok, at] = mestra.internal.Units.group(s, at, depth);
                return
            end
            if mestra.internal.Units.isLetter(c)
                start = at;
                [ok, at] = mestra.internal.Units.name(s, at);
                if ismember(s(start:at - 1), ...
                            mestra.internal.Units.LogNames) && ...
                   isequal(mestra.internal.Units.ch(s, at), '(')
                    [opened, after] = ...
                        mestra.internal.Units.reference(s, at);
                    if opened
                        [ok, at] = mestra.internal.Units.logref( ...
                            s, after, depth);
                    end
                end
                return
            end
            [ok, at] = mestra.internal.Units.number(s, at);
        end

        % name = letter { letter | digit }, less its trailing digits,
        % which are its power: "m2" is m squared, "m2s" is one name.
        function [ok, at] = name(s, at)
            ok = mestra.internal.Units.isLetter(mestra.internal.Units.ch(s, at));
            if ~ok, return, end
            while at <= numel(s) && ...
                  (mestra.internal.Units.isLetter(s(at)) || ...
                   mestra.internal.Units.isDigit(s(at)))
                at = at + 1;
            end
            while mestra.internal.Units.isDigit(s(at - 1))
                at = at - 1;
            end
        end

        % "(" units ")"
        function [ok, at] = group(s, at, depth)
            ok = false;
            if depth >= mestra.internal.Units.MaxDepth, return, end
            [ok, at] = mestra.internal.Units.units(s, at + 1, depth + 1);
            if ~ok || ~isequal(mestra.internal.Units.ch(s, at), ')')
                ok = false;
                return
            end
            at = at + 1;
        end

        % The opening of a log reference, "(re " or "(re:", after the
        % name of a logarithm.  When it is not there the "(" opens a
        % group instead.
        function [ok, at] = reference(s, at)
            ok = false;
            at = mestra.internal.Units.spaces(s, at + 1);
            if at + 1 > numel(s) || ~strcmp(s(at:at + 1), 're')
                return
            end
            at = at + 2;
            if isequal(mestra.internal.Units.ch(s, at), ':')
                at = mestra.internal.Units.spaces(s, at + 1);
                ok = true;
                return
            end
            after = mestra.internal.Units.spaces(s, at);
            ok = after > at;
            at = after;
        end

        % The rest of "lg(re 1 mW)": a product and its ")".
        function [ok, at] = logref(s, at, depth)
            ok = false;
            if depth >= mestra.internal.Units.MaxDepth, return, end
            [ok, at] = mestra.internal.Units.product(s, at, depth + 1);
            if ~ok, return, end
            at = mestra.internal.Units.spaces(s, at);
            ok = isequal(mestra.internal.Units.ch(s, at), ')');
            at = at + 1;
        end

        % shift = { sp } ( "@" | "since" | ... ) { sp }
        %         ( timestamp | number )
        function [ok, at] = shift(s, at)
            ok = false;
            at = mestra.internal.Units.spaces(s, at);
            if isequal(mestra.internal.Units.ch(s, at), '@')
                at = at + 1;
            else
                words = mestra.internal.Units.ShiftWords;
                found = false;
                for i = 1:numel(words)
                    if mestra.internal.Units.word(s, at, words{i})
                        at = at + numel(words{i});
                        found = true;
                        break
                    end
                end
                if ~found, return, end
            end
            at = mestra.internal.Units.spaces(s, at);
            [ok, after] = mestra.internal.Units.timestamp(s, at);
            if ok
                at = after;
                return
            end
            [ok, at] = mestra.internal.Units.number(s, at);
        end

        % number = [ sign ] ( digits [ "." digits ] | "." digits )
        %          [ ( "e" | "E" ) [ sign ] digits ]
        function [ok, at] = number(s, at)
            ok = false;
            at = mestra.internal.Units.sign(s, at);
            [whole, at] = mestra.internal.Units.digits(s, at, Inf);
            if isequal(mestra.internal.Units.ch(s, at), '.')
                [fraction, at] = mestra.internal.Units.digits(s, at + 1, Inf);
                if whole == 0 && fraction == 0, return, end
            elseif whole == 0
                return
            end
            c = mestra.internal.Units.ch(s, at);
            if ~isempty(c) && any(c == 'eE')
                after = mestra.internal.Units.sign(s, at + 1);
                [n, after] = mestra.internal.Units.digits(s, after, Inf);
                if n > 0
                    at = after;
                end
            end
            ok = true;
        end

        % timestamp = date [ ( "T" | sp { sp } ) clock [ { sp } zone ] ]
        % A clock or a zone that does not match is not part of it.
        function [ok, at] = timestamp(s, at)
            [ok, at] = mestra.internal.Units.date(s, at);
            if ~ok, return, end
            save = at;
            if isequal(mestra.internal.Units.ch(s, at), 'T')
                at = at + 1;
            else
                at = mestra.internal.Units.spaces(s, at);
                if at == save, return, end
            end
            [timed, after] = mestra.internal.Units.clock(s, at);
            if ~timed
                at = save;
                return
            end
            at = after;
            [zoned, after] = mestra.internal.Units.zone( ...
                s, mestra.internal.Units.spaces(s, at));
            if zoned
                at = after;
            end
        end

        % date = [ sign ] digits "-" d [ d ] [ "-" d [ d ] ]
        function [ok, at] = date(s, at)
            ok = false;
            at = mestra.internal.Units.sign(s, at);
            [n, at] = mestra.internal.Units.digits(s, at, Inf);
            if n == 0 || ~isequal(mestra.internal.Units.ch(s, at), '-')
                return
            end
            [n, at] = mestra.internal.Units.digits(s, at + 1, 2);
            if n == 0, return, end
            if isequal(mestra.internal.Units.ch(s, at), '-') && ...
               mestra.internal.Units.isDigit(mestra.internal.Units.ch(s, at + 1))
                [~, at] = mestra.internal.Units.digits(s, at + 1, 2);
            end
            ok = true;
        end

        % clock = d [ d ] ":" d d [ ":" d d [ "." { d } ] ]
        function [ok, at] = clock(s, at)
            ok = false;
            [n, at] = mestra.internal.Units.digits(s, at, 2);
            if n == 0 || ~isequal(mestra.internal.Units.ch(s, at), ':')
                return
            end
            [n, at] = mestra.internal.Units.digits(s, at + 1, 2);
            if n ~= 2, return, end
            if isequal(mestra.internal.Units.ch(s, at), ':') && ...
               mestra.internal.Units.isDigit(mestra.internal.Units.ch(s, at + 1))
                [n, at] = mestra.internal.Units.digits(s, at + 1, 2);
                if n ~= 2, return, end
                if isequal(mestra.internal.Units.ch(s, at), '.')
                    [~, at] = mestra.internal.Units.digits(s, at + 1, Inf);
                end
            end
            ok = true;
        end

        % zone = name | sign d [ d ] [ [ ":" ] d d ]
        function [ok, at] = zone(s, at)
            c = mestra.internal.Units.ch(s, at);
            if ~isempty(c) && any(c == '+-')
                [n, at] = mestra.internal.Units.digits(s, at + 1, 2);
                ok = n > 0;
                if ~ok, return, end
                after = at;
                if isequal(mestra.internal.Units.ch(s, after), ':')
                    after = after + 1;
                end
                [n, after] = mestra.internal.Units.digits(s, after, 2);
                if n == 2
                    at = after;
                end
                return
            end
            [ok, at] = mestra.internal.Units.name(s, at);
        end

        % ------------------------------------------------- characters

        function c = ch(s, at)
        %ch  The character at a position, or '' past the end.
            if at <= numel(s)
                c = s(at);
            else
                c = '';
            end
        end

        function at = spaces(s, at)
        %spaces  Past any run of U+0020; a tab is not a space here.
            while at <= numel(s) && s(at) == ' '
                at = at + 1;
            end
        end

        function at = sign(s, at)
            if at <= numel(s) && any(s(at) == '+-')
                at = at + 1;
            end
        end

        function [n, at] = digits(s, at, most)
        %digits  Past at most MOST decimal digits; N is how many.
            n = 0;
            while n < most && at <= numel(s) && ...
                  mestra.internal.Units.isDigit(s(at))
                at = at + 1;
                n = n + 1;
            end
        end

        function tf = word(s, at, w)
        %word  W at AT as a whole word, not run on into a name.
            last = at + numel(w) - 1;
            tf = last <= numel(s) && strcmp(s(at:last), w);
            if tf && last < numel(s)
                after = s(last + 1);
                tf = ~(mestra.internal.Units.isLetter(after) || ...
                       mestra.internal.Units.isDigit(after));
            end
        end

        function tf = shiftWordAt(s, at)
            words = mestra.internal.Units.ShiftWords;
            tf = false;
            for i = 1:numel(words)
                if mestra.internal.Units.word(s, at, words{i})
                    tf = true;
                    return
                end
            end
        end

        function tf = isLetter(c)
        %isLetter  A character a name is made of besides the digits:
        %   an ASCII letter, one of _ % ' ", or anything outside ASCII.
            tf = ~isempty(c) && ...
                 ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || ...
                  any(c == '_%''"') || double(c) > 127);
        end

        function tf = isDigit(c)
            tf = ~isempty(c) && c >= '0' && c <= '9';
        end
    end
end
