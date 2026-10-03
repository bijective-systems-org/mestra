function write(dataset, path, varargin)
%MESTRA.WRITE  Write a mestra.Dataset to a conforming file.
%
%   MESTRA.WRITE(DATASET, PATH) validates what it is about to write
%   and refuses to leave an invalid file behind.  The file is built,
%   checked with MESTRA.VALIDATE, and only then put at PATH; on any
%   error it is deleted and the error names the rule, with every
%   finding in its message.  A writer that produces invalid files
%   silently is the one failure mode an open format cannot afford,
%   because the file outlives the session that made it
%   (docs/api-conventions.md, section 2).
%
%   Warnings do not stop a write.  They are findings about a file
%   that is nonetheless conforming, and MESTRA.VALIDATE reports them
%   whenever the caller asks.
%
%   MESTRA.WRITE(DATASET, PATH, 'Check', false) writes without
%   validating, which is how a deliberately invalid file is made.
%
%       mestra.write(d, 'out.mes');
%       mestra.write(broken, 'e16.mes', 'Check', false);
%
%   It writes the file specification sections 13 to 25 define:
%
%     * every string, in an attribute or a dataset, fixed length,
%       UTF-8, padded on the right with NUL bytes, and never variable
%       length (section 18);
%     * a dimension scale on every axis of every dataset, named as
%       section 21 requires and written as netCDF-C writes a dimension
%       with no coordinate variable;
%     * `row` unlimited in every file, every row-dimensioned dataset
%       chunked with its non-row extents full, and the default chunk
%       length of section 23 unless the dataset carries another one;
%     * gzip and shuffle as the only filters (section 23);
%     * no HDF5 fill value anywhere (section 19);
%     * object time tracking off on every object, so that two runs
%       given the same dataset differ only where the data differs.
%
%   A slot whose `source` is a callable is written as an empty group
%   with the slot's attributes and no data; a slot holding data is a
%   dataset (section 19).  Callables go through the dictionary codec
%   of sections 17 and 25, with the dictionary's keys visited in
%   ascending order of their UTF-8 bytes so that two writers given the
%   same dictionary produce the same file.
%
%   /notes, /private and any group this version does not know are
%   written back exactly as they were read, so a read-then-write loses
%   nothing.
%
%   The file is created fresh; an existing file at PATH is replaced.
%   Checked publication uses Java's atomic move. It requires MATLAB's
%   JVM and a filesystem supporting atomic replacement; otherwise it
%   fails without deleting the previous file. There is no copy/delete
%   fallback and no power-loss durability guarantee.
%
%   STRINGS ARE ASCII HERE.  The format puts every string in a
%   fixed-length UTF-8 field and counts the size in bytes, and
%   MATLAB's HDF5 interface cannot carry that faithfully: H5D.write
%   and H5A.write refuse a character above 127 outright, H5D.read and
%   H5A.read decode before this package sees the bytes, and HDF5 will
%   not convert between the ASCII and UTF-8 character sets.  So this
%   package refuses rather than corrupting: a string with a byte
%   above 127, written or read, raises mestra:matlabAscii and says
%   what happened.  Keep key names, category entries, callable ids
%   and string ids to ASCII and nothing here applies; the whole
%   conformance corpus is ASCII and passes in full.
%
%   Example
%
%       d = mestra.read('in.mes');
%       mestra.write(d, 'out.mes');       % structurally equal
%
%   See also mestra.read, mestra.validate, mestra.Dataset.

    if ~isa(dataset, 'mestra.Dataset')
        error('mestra:write', ...
              'the first argument must be a mestra.Dataset');
    end
    p = inputParser();
    p.addParameter('Check', true);
    p.parse(varargin{:});
    refuseADestinationThatCannotTakeAFile(path);
    % One text form from here on: the staging name below is built by
    % concatenation, which joins a string scalar element by element
    % into a string array rather than into one name.
    path = mestra.internal.Args.text(path);
    if ~p.Results.Check
        mestra.internal.Writer.save(dataset, path);
        return
    end

    % Build it beside the destination and only move it there once its
    % own validator accepts it, so that a refused write never replaces
    % what was there and never leaves a half file behind.
    [folder, base, ext] = fileparts(path);
    if isempty(folder), folder = '.'; end
    [~, tag] = fileparts(tempname());
    tmp = fullfile(folder, ['.' base ext '.mestra-' tag]);
    cleanup = onCleanup(@() removeIfPresent(tmp));
    mestra.internal.Writer.save(dataset, tmp);
    r = mestra.validate(tmp);
    bad = [r.errors r.unclassified];
    if ~isempty(bad)
        error(['mestra:' bad{1}], ...
              ['%s: mestra.write refused to write %s because its own ' ...
               'validator rejects it:\n%s\nFix the dataset, or pass ' ...
               '''Check'', false to write it anyway'], bad{1}, path, ...
              mestra.report(r, 'String', true));
    end
    mestra.internal.publishFile(tmp, path);
end

function refuseADestinationThatCannotTakeAFile(path)
%refuseADestinationThatCannotTakeAFile  The staging and the publish
%   both happen in the destination's directory, so a destination that
%   cannot take a file is refused here, naming the argument, before
%   anything is validated or created.  A failure after this point is
%   the operating system's error and leaves an existing destination
%   as it was.
    if ~mestra.internal.Args.isText(path) || isempty(char(path))
        error('mestra:write', ...
              'path is empty; give the name of the file to write');
    end
    path = char(path);
    if isfolder(path)
        error('mestra:write', ...
              ['%s names a directory, which a file cannot replace; ' ...
               'give a file name'], path);
    end
    folder = fileparts(path);
    if isempty(folder), folder = '.'; end
    if ~isfolder(folder)
        error('mestra:write', ...
              ['the directory %s does not exist; the file is staged ' ...
               'and published there, so create it first'], folder);
    end
    [~, attributes] = fileattrib(folder);
    if isstruct(attributes) && ~attributes.UserWrite
        error('mestra:write', ...
              ['the directory %s is not writable; the file is staged ' ...
               'and published there'], folder);
    end
end

function removeIfPresent(path)
    if exist(path, 'file') == 2
        delete(path);
    end
end
