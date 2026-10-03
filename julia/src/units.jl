# The units grammar W10 is decided by (section 32).
#
# A units string parses when it matches the grammar, whatever names it
# uses: "furlong fortnight-1" parses and "m//s" does not.  This is the
# grammar and not a units database; it says whether a string matches
# and nothing else.  Each function below is one production of section
# 32: it returns true and moves past what it matched, or returns false
# and leaves the position for its caller to restore.
#
# The string is read byte by byte.  Every byte of a UTF-8 sequence for
# a character outside ASCII is 0x80 or above, and every such character
# is a letter of a name, so reading bytes gives the verdict reading
# characters would.

mutable struct UnitsCursor
    s::Vector{UInt8}
    i::Int
end

"""How deep a units string may nest its parentheses.  A units string
comes out of a file, so its depth is the file's choice, and the parser
below is recursive."""
const MAX_UNITS_DEPTH = 32

"""The longest units string, in bytes, this parser will look at."""
const MAX_UNITS_BYTES = 4096

const SHIFT_WORDS = ("since", "from", "after", "ref")
const LOG_NAMES = ("log", "lg", "ln", "lb")

ubyte(c::UnitsCursor, offset::Int = 0) =
    c.i + offset <= length(c.s) ? c.s[c.i + offset] : 0x00

isudigit(b::UInt8) = UInt8('0') <= b <= UInt8('9')

# A character a name is made of, besides the digits: an ASCII letter,
# one of _ % ' ", or any character outside ASCII.
isuletter(b::UInt8) =
    UInt8('A') <= b <= UInt8('Z') || UInt8('a') <= b <= UInt8('z') ||
    b == UInt8('_') || b == UInt8('%') || b == UInt8('\'') ||
    b == UInt8('"') || b > 0x7f

function uspaces!(c::UnitsCursor)
    start = c.i
    while ubyte(c) == UInt8(' ')
        c.i += 1
    end
    return c.i - start
end

function udigits!(c::UnitsCursor, most::Int = typemax(Int))
    start = c.i
    while isudigit(ubyte(c)) && c.i - start < most
        c.i += 1
    end
    return c.i - start
end

function uhas(c::UnitsCursor, w::AbstractString)
    n = ncodeunits(w)
    c.i + n - 1 <= length(c.s) || return false
    return view(c.s, c.i:c.i + n - 1) == codeunits(w)
end

# `w` here as a whole word: not run on into a name.
function uword(c::UnitsCursor, w::AbstractString)
    uhas(c, w) || return false
    after = ubyte(c, ncodeunits(w))
    return !(isuletter(after) || isudigit(after))
end

function ushiftword(c::UnitsCursor)
    for w in SHIFT_WORDS
        uword(c, w) && return ncodeunits(w)
    end
    return 0
end

"""
    parse_units(s) -> Bool

True when the string matches the units grammar of section 32.
"""
function parse_units(s::AbstractString)
    ncodeunits(s) > MAX_UNITS_BYTES && return false
    c = UnitsCursor(Vector{UInt8}(codeunits(String(s))), 1)
    return uunits!(c, 0) && c.i > length(c.s)
end

function uunits!(c::UnitsCursor, depth::Int)
    uspaces!(c)
    uproduct!(c, depth) || return false
    save = c.i
    ushift!(c) || (c.i = save)
    uspaces!(c)
    return true
end

function uproduct!(c::UnitsCursor, depth::Int)
    upower!(c, depth) || return false
    while true
        save = c.i
        uspaces!(c)
        b = ubyte(c)
        if b == 0x00 || b == UInt8(')') || b == UInt8('@') ||
           ushiftword(c) > 0
            c.i = save
            return true
        end
        if b in (UInt8('*'), UInt8('.'), UInt8('/'), UInt8('-'))
            c.i += 1
        elseif uword(c, "per")
            c.i += 3
        else
            # Juxtaposition: a space, or nothing at all, between two
            # powers is a product.
            upower!(c, depth) && continue
            c.i = save
            return true
        end
        uspaces!(c)
        upower!(c, depth) || return false
    end
end

function upower!(c::UnitsCursor, depth::Int)
    ubasic!(c, depth) || return false
    b = ubyte(c)
    if isudigit(b) || ((b == UInt8('+') || b == UInt8('-')) &&
                       isudigit(ubyte(c, 1)))
        c.i += 1
        udigits!(c)
    elseif b == UInt8('^') || (b == UInt8('*') && ubyte(c, 1) == UInt8('*'))
        c.i += b == UInt8('^') ? 1 : 2
        return unumber!(c)
    end
    return true
end

function ubasic!(c::UnitsCursor, depth::Int)
    b = ubyte(c)
    b == UInt8('(') && return ugroup!(c, depth)
    if isuletter(b)
        start = c.i
        uname!(c)
        if String(c.s[start:c.i - 1]) in LOG_NAMES && ubyte(c) == UInt8('(')
            save = c.i
            ureference!(c) && return ulogref!(c, depth)
            c.i = save
        end
        return true
    end
    return unumber!(c)
end

function uname!(c::UnitsCursor)
    isuletter(ubyte(c)) || return false
    while isuletter(ubyte(c)) || isudigit(ubyte(c))
        c.i += 1
    end
    # The trailing digits of a name are its power: "m2" is m squared,
    # and "m2s" is one name.
    while isudigit(c.s[c.i - 1])
        c.i -= 1
    end
    return true
end

function ugroup!(c::UnitsCursor, depth::Int)
    depth >= MAX_UNITS_DEPTH && return false
    c.i += 1
    (uunits!(c, depth + 1) && ubyte(c) == UInt8(')')) || return false
    c.i += 1
    return true
end

# The opening of a logarithmic reference, "(re " or "(re:", after a
# name of a logarithm.  When it is there the "(" belongs to the
# reference; when it is not, the "(" opens a group.
function ureference!(c::UnitsCursor)
    c.i += 1
    uspaces!(c)
    uhas(c, "re") || return false
    c.i += 2
    if ubyte(c) == UInt8(':')
        c.i += 1
        uspaces!(c)
        return true
    end
    return uspaces!(c) > 0
end

# The rest of "lg(re 1 mW)": a product and its ")".
function ulogref!(c::UnitsCursor, depth::Int)
    depth >= MAX_UNITS_DEPTH && return false
    uproduct!(c, depth + 1) || return false
    uspaces!(c)
    ubyte(c) == UInt8(')') || return false
    c.i += 1
    return true
end

function ushift!(c::UnitsCursor)
    uspaces!(c)
    if ubyte(c) == UInt8('@')
        c.i += 1
    else
        n = ushiftword(c)
        n == 0 && return false
        c.i += n
    end
    uspaces!(c)
    save = c.i
    utimestamp!(c) && return true
    c.i = save
    return unumber!(c)
end

function usign!(c::UnitsCursor)
    (ubyte(c) == UInt8('+') || ubyte(c) == UInt8('-')) && (c.i += 1)
    return c
end

function unumber!(c::UnitsCursor)
    usign!(c)
    whole = udigits!(c)
    if ubyte(c) == UInt8('.')
        c.i += 1
        fraction = udigits!(c)
        whole == 0 && fraction == 0 && return false
    elseif whole == 0
        return false
    end
    if ubyte(c) == UInt8('e') || ubyte(c) == UInt8('E')
        save = c.i
        c.i += 1
        usign!(c)
        udigits!(c) == 0 && (c.i = save)
    end
    return true
end

function utimestamp!(c::UnitsCursor)
    udate!(c) || return false
    save = c.i
    if ubyte(c) == UInt8('T')
        c.i += 1
    elseif uspaces!(c) == 0
        return true
    end
    if !uclock!(c)
        c.i = save
        return true
    end
    save = c.i
    uspaces!(c)
    uzone!(c) || (c.i = save)
    return true
end

function udate!(c::UnitsCursor)
    usign!(c)
    (udigits!(c) > 0 && ubyte(c) == UInt8('-')) || return false
    c.i += 1
    udigits!(c, 2) > 0 || return false
    if ubyte(c) == UInt8('-') && isudigit(ubyte(c, 1))
        c.i += 1
        udigits!(c, 2)
    end
    return true
end

function uclock!(c::UnitsCursor)
    (udigits!(c, 2) > 0 && ubyte(c) == UInt8(':')) || return false
    c.i += 1
    udigits!(c, 2) == 2 || return false
    if ubyte(c) == UInt8(':') && isudigit(ubyte(c, 1))
        c.i += 1
        udigits!(c, 2) == 2 || return false
        if ubyte(c) == UInt8('.')
            c.i += 1
            udigits!(c)
        end
    end
    return true
end

function uzone!(c::UnitsCursor)
    if ubyte(c) == UInt8('+') || ubyte(c) == UInt8('-')
        c.i += 1
        udigits!(c, 2) > 0 || return false
        save = c.i
        ubyte(c) == UInt8(':') && (c.i += 1)
        udigits!(c, 2) == 2 || (c.i = save)
        return true
    end
    return uname!(c)
end
