function out = report(findings, varargin)
%MESTRA.REPORT  Print what a validation run found.
%
%   MESTRA.REPORT(R) prints one line per finding and then the
%   summary, in the form every implementation prints
%   (docs/api-conventions.md, section 5):
%
%       <id> <path>: <message>
%       <n> error(s), <m> warning(s)
%
%   R is what MESTRA.VALIDATE returns, or its `findings` field alone.
%
%       r = mestra.validate('case.mes');
%       mestra.report(r);
%       W02 /keys/status: 2 of 6 rows have a status other than
%           converged (rows 1, 4); a row that is not converged is
%           excluded from modelling unless it is asked for
%       0 error(s), 1 warning(s)
%
%   N = MESTRA.REPORT(R) also returns the number of errors, so a
%   script can end on it the way a command-line tool exits on it:
%
%       if mestra.report(mestra.validate(path)) > 0, exit(1); end
%
%   S = MESTRA.REPORT(R, 'String', true) returns the same text
%   instead of printing it.
%
%   MESTRA.REPORT(R, 'File', fid) prints to an open file or to
%   2 for standard error.
%
%   Row indices in a finding are the file's, counted from 0, so that
%   two implementations report the same row of the same file by the
%   same number.
%
%   See also mestra.validate, mestra.info, mestra.read.

    p = inputParser();
    p.addParameter('String', false);
    p.addParameter('File', 1);
    p.parse(varargin{:});

    if isstruct(findings) && isscalar(findings) && ...
            isfield(findings, 'findings')
        f = findings.findings;
    else
        f = findings;
    end
    if isempty(f)
        f = struct('id', {}, 'path', {}, 'message', {});
    end

    lines = cell(1, numel(f) + 1);
    nErrors = 0;
    nWarnings = 0;
    for i = 1:numel(f)
        id = f(i).id;
        if ~isempty(id)
            switch id(1)
                case 'E', nErrors = nErrors + 1;
                case 'W', nWarnings = nWarnings + 1;
                case 'U', nErrors = nErrors + 1;
            end
        end
        path = oneLine(f(i).path);
        if isempty(path), path = '/'; end
        if isempty(f(i).message)
            lines{i} = sprintf('%s %s', id, path);
        else
            lines{i} = sprintf('%s %s: %s', id, path, oneLine(f(i).message));
        end
    end
    lines{end} = sprintf('%d error(s), %d warning(s)', nErrors, nWarnings);
    text = [strjoin(lines, newline) newline];

    if p.Results.String
        out = text;
        return
    end
    fprintf(p.Results.File, '%s', text);
    if nargout > 0
        out = nErrors;
    end
end

function s = oneLine(s)
%oneLine  A control character, which an illegal name can hold, written
%   as \xHH, so that a finding stays on its one line.
    bad = double(s) < 32 | double(s) == 127;
    if ~any(bad), return, end
    parts = cell(1, numel(s));
    for i = 1:numel(s)
        if bad(i)
            parts{i} = sprintf('\\x%02x', double(s(i)));
        else
            parts{i} = s(i);
        end
    end
    s = [parts{:}];
end
