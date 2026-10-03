"""Units strings: the grammar W10 is decided by, and their dimensions.

SPEC.md section 32 gives the grammar. A units string parses when it
matches it, whatever names it uses: "furlong fortnight-1" parses and
"m//s" does not. That verdict is `is_parseable`, and it is all the
validator asks (W10). It needs no unit database.

`parse` goes further for a tool that wants to compare two quantities.
It reduces a string to the dimensions of the seven SI base units,
using a table of the names CF files use with the SI prefixes, and
refuses a string whose names are not in that table or that carries a
shift, so that a tool never combines two quantities it could not
check (section 3).

    >>> is_parseable("days since 2000-01-01")
    True
    >>> parse("W m-2").dimensions
    {'kg': 1, 's': -3}
    >>> parse("kg/(m s") is None
    True
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import limits

__all__ = ["Unit", "parse", "is_parseable", "same_dimensions"]

#: The base dimensions every other unit is reduced to.
BASE = ("m", "kg", "s", "A", "K", "mol", "cd")

#: Dimensionless names, kept apart so that "1", "rad" and "degree"
#: all parse and all come out with no dimensions.
_DIMENSIONLESS = {
    "1", "one", "unitless", "dimensionless", "none", "count", "percent",
    "rad", "radian", "radians", "sr", "steradian", "degree", "degrees",
    "deg", "arcdeg", "arcminute", "arcsecond", "dB", "decibel", "PLdB",
}

#: name -> the exponents of the base dimensions.
_UNITS: dict[str, dict[str, int]] = {
    "m": {"m": 1}, "metre": {"m": 1}, "meter": {"m": 1},
    "g": {"kg": 1}, "gram": {"kg": 1},
    "s": {"s": 1}, "second": {"s": 1}, "sec": {"s": 1},
    "A": {"A": 1}, "ampere": {"A": 1},
    "K": {"K": 1}, "kelvin": {"K": 1},
    "mol": {"mol": 1}, "mole": {"mol": 1},
    "cd": {"cd": 1}, "candela": {"cd": 1},
    "Hz": {"s": -1}, "hertz": {"s": -1},
    "N": {"kg": 1, "m": 1, "s": -2}, "newton": {"kg": 1, "m": 1, "s": -2},
    "Pa": {"kg": 1, "m": -1, "s": -2},
    "pascal": {"kg": 1, "m": -1, "s": -2},
    "bar": {"kg": 1, "m": -1, "s": -2},
    "atm": {"kg": 1, "m": -1, "s": -2},
    "psi": {"kg": 1, "m": -1, "s": -2},
    "J": {"kg": 1, "m": 2, "s": -2}, "joule": {"kg": 1, "m": 2, "s": -2},
    "W": {"kg": 1, "m": 2, "s": -3}, "watt": {"kg": 1, "m": 2, "s": -3},
    "C": {"A": 1, "s": 1}, "coulomb": {"A": 1, "s": 1},
    "V": {"kg": 1, "m": 2, "s": -3, "A": -1},
    "volt": {"kg": 1, "m": 2, "s": -3, "A": -1},
    "F": {"kg": -1, "m": -2, "s": 4, "A": 2},
    "ohm": {"kg": 1, "m": 2, "s": -3, "A": -2},
    "S": {"kg": -1, "m": -2, "s": 3, "A": 2},
    "Wb": {"kg": 1, "m": 2, "s": -2, "A": -1},
    "T": {"kg": 1, "s": -2, "A": -1},
    "H": {"kg": 1, "m": 2, "s": -2, "A": -2},
    "lm": {"cd": 1}, "lx": {"cd": 1, "m": -2},
    "Bq": {"s": -1}, "Gy": {"m": 2, "s": -2}, "Sv": {"m": 2, "s": -2},
    "kat": {"mol": 1, "s": -1},
    "L": {"m": 3}, "l": {"m": 3}, "litre": {"m": 3}, "liter": {"m": 3},
    "t": {"kg": 1}, "tonne": {"kg": 1},
    "min": {"s": 1}, "minute": {"s": 1},
    "h": {"s": 1}, "hour": {"s": 1}, "hr": {"s": 1},
    "d": {"s": 1}, "day": {"s": 1}, "week": {"s": 1},
    "year": {"s": 1}, "yr": {"s": 1},
    "degree_C": {"K": 1}, "degC": {"K": 1}, "celsius": {"K": 1},
    "degree_F": {"K": 1}, "degF": {"K": 1},
    "ft": {"m": 1}, "foot": {"m": 1}, "feet": {"m": 1},
    "in": {"m": 1}, "inch": {"m": 1}, "yd": {"m": 1}, "yard": {"m": 1},
    "mi": {"m": 1}, "mile": {"m": 1}, "nmi": {"m": 1},
    "lb": {"kg": 1}, "lbm": {"kg": 1},
    "lbf": {"kg": 1, "m": 1, "s": -2},
    "slug": {"kg": 1},
    "knot": {"m": 1, "s": -1}, "kt": {"m": 1, "s": -1},
}

#: The SI prefixes, and the ones UDUNITS spells out.
_PREFIXES = {
    "y", "z", "a", "f", "p", "n", "u", "µ", "μ", "m", "c",
    "d", "da", "h", "k", "M", "G", "T", "P", "E", "Z", "Y",
    "yocto", "zepto", "atto", "femto", "pico", "nano", "micro",
    "milli", "centi", "deci", "deca", "hecto", "kilo", "mega", "giga",
    "tera", "peta", "exa", "zetta", "yotta",
}

_TOKEN = re.compile(r"""
    (?P<space>\s+)
  | (?P<number>[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?
              |\.[0-9]+(?:[eE][-+]?[0-9]+)?
              |[0-9]+[eE][-+]?[0-9]+)
  | (?P<divide>/|per(?![A-Za-z_]))
  | (?P<name>[A-Za-z_µμ][A-Za-z_µμ]*)
  | (?P<integer>[0-9]+)
  | (?P<power>\^|\*\*)
  | (?P<times>\*|·)
  | (?P<percent>%)
  | (?P<open>\()
  | (?P<close>\))
  | (?P<sign>[-+])
""", re.VERBOSE)


@dataclass(frozen=True)
class Unit:
    """A parsed unit: the string it came from and its dimensions."""

    text: str
    dimensions: dict[str, int] = field(default_factory=dict)

    @property
    def dimensionless(self) -> bool:
        """True when nothing is left after reduction, as for "1"."""
        return not self.dimensions

    def __str__(self) -> str:
        return self.text


class _Token:
    __slots__ = ("kind", "text")

    def __init__(self, kind: str, text: str) -> None:
        self.kind = kind
        self.text = text


class _ParseError(Exception):
    pass


def _scan(text: str) -> Iterator[_Token]:
    at = 0
    while at < len(text):
        match = _TOKEN.match(text, at)
        if match is None:
            raise _ParseError("unexpected character %r" % text[at])
        at = match.end()
        kind = match.lastgroup or ""
        if kind == "space":
            yield _Token("space", " ")
            continue
        yield _Token(kind, match.group())


def _lookup(name: str) -> dict[str, int]:
    """The dimensions of one name, trying the SI prefixes."""
    if name in _DIMENSIONLESS:
        return {}
    if name in _UNITS:
        return dict(_UNITS[name])
    for cut in (1, 2, 5):
        head, tail = name[:cut], name[cut:]
        if head in _PREFIXES and tail:
            if tail in _DIMENSIONLESS:
                return {}
            if tail in _UNITS:
                return dict(_UNITS[tail])
    raise _ParseError("unknown unit %r" % name)


def _combine(left: dict[str, int], right: dict[str, int],
             sign: int) -> dict[str, int]:
    out = dict(left)
    for base, power in right.items():
        value = out.get(base, 0) + sign * power
        if value:
            out[base] = value
        elif base in out:
            del out[base]
    return out


def _power(dims: dict[str, int], exponent: int) -> dict[str, int]:
    return {base: power * exponent
            for base, power in dims.items() if power * exponent}


class _Parser:
    """Recursive descent over the token stream."""

    def __init__(self, text: str) -> None:
        self.tokens = list(_scan(text))
        self.at = 0
        #: How many parentheses deep the parser is. A units string
        #: is one or two levels; thousands is an attack on a
        #: recursive descent parser and not a unit (limits).
        self.depth = 0

    def peek(self) -> _Token | None:
        while (self.at < len(self.tokens)
               and self.tokens[self.at].kind == "space"):
            self.at += 1
        if self.at >= len(self.tokens):
            return None
        return self.tokens[self.at]

    def peek_raw(self) -> _Token | None:
        if self.at >= len(self.tokens):
            return None
        return self.tokens[self.at]

    def take(self) -> _Token:
        token = self.peek()
        if token is None:
            raise _ParseError("the string ends too early")
        self.at += 1
        return token

    def parse(self) -> dict[str, int]:
        dims = self.expression()
        left = self.peek()
        if left is not None:
            raise _ParseError("trailing %r" % left.text)
        return dims

    def expression(self) -> dict[str, int]:
        dims = self.term()
        while True:
            token = self.peek()
            if token is None or token.kind == "close":
                return dims
            if token.kind == "divide":
                self.take()
                dims = _combine(dims, self.term(), -1)
            elif token.kind == "times":
                self.take()
                dims = _combine(dims, self.term(), 1)
            elif token.kind in ("name", "number", "open", "integer",
                                "percent"):
                dims = _combine(dims, self.term(), 1)
            else:
                raise _ParseError("unexpected %r" % token.text)

    def term(self) -> dict[str, int]:
        dims = self.factor()
        token = self.peek_raw()
        if token is not None and token.kind in ("integer", "sign"):
            # An exponent written against the name, "m2" or "s-1".
            return _power(dims, self.exponent())
        after = self.peek()
        if after is not None and after.kind == "power":
            self.take()
            return _power(dims, self.exponent())
        return dims

    def exponent(self) -> int:
        token = self.take()
        sign = 1
        if token.kind == "sign":
            sign = -1 if token.text == "-" else 1
            token = self.take()
        if token.kind != "integer":
            raise _ParseError("an exponent must be an integer")
        return sign * int(token.text)

    def factor(self) -> dict[str, int]:
        token = self.take()
        if token.kind == "open":
            if self.depth >= limits.MAX_UNITS_DEPTH:
                raise _ParseError("more than %d parentheses deep"
                                  % limits.MAX_UNITS_DEPTH)
            self.depth += 1
            dims = self.expression()
            self.depth -= 1
            close = self.peek()
            if close is None or close.kind != "close":
                raise _ParseError("a group is not closed")
            self.take()
            return dims
        if token.kind in ("number", "integer", "percent"):
            return {}
        if token.kind == "name":
            return _lookup(token.text)
        raise _ParseError("unexpected %r" % token.text)


# ------------------------------------------------------------ the grammar

#: The words that are operators after a power and names anywhere else.
_SHIFT_WORDS = ("since", "from", "after", "ref")
_LOG_NAMES = ("log", "lg", "ln", "lb")


def _letter(c: str) -> bool:
    """A character a name is made of, besides the digits: an ASCII
    letter, one of _ % ' ", or any character outside ASCII."""
    return len(c) == 1 and (("A" <= c <= "Z") or ("a" <= c <= "z")
                            or c in "_%'\"" or ord(c) > 127)


def _digit(c: str) -> bool:
    return len(c) == 1 and "0" <= c <= "9"


class _Grammar:
    """Section 32 read literally: one recursive descent, which says
    whether a string matches and nothing else. Every method returns
    True and moves past what it matched, or returns False and leaves
    the position to its caller to restore."""

    def __init__(self, text: str) -> None:
        self.s = text
        self.at = 0

    def char(self, offset: int = 0) -> str:
        at = self.at + offset
        return self.s[at] if at < len(self.s) else ""

    def spaces(self) -> int:
        start = self.at
        while self.char() == " ":
            self.at += 1
        return self.at - start

    def digits(self, most: int | None = None) -> int:
        start = self.at
        while _digit(self.char()) and (most is None
                                       or self.at - start < most):
            self.at += 1
        return self.at - start

    def word(self, word: str) -> bool:
        """`word` here as a whole word: not run on into a name."""
        end = self.at + len(word)
        if self.s[self.at:end] != word:
            return False
        after = self.s[end] if end < len(self.s) else ""
        return not (after and (_letter(after) or _digit(after)))

    def units(self, depth: int) -> bool:
        self.spaces()
        if not self.product(depth):
            return False
        save = self.at
        if not self.shift():
            self.at = save
        self.spaces()
        return True

    def product(self, depth: int) -> bool:
        if not self.power(depth):
            return False
        while True:
            save = self.at
            self.spaces()
            c = self.char()
            if c == "" or c == ")" or c == "@" or any(
                    self.word(w) for w in _SHIFT_WORDS):
                self.at = save
                return True
            if c in "*./-":
                self.at += 1
            elif self.word("per"):
                self.at += 3
            else:
                # Juxtaposition: a space, or nothing at all, between
                # two powers is a product.
                if self.power(depth):
                    continue
                self.at = save
                return True
            self.spaces()
            if not self.power(depth):
                return False

    def power(self, depth: int) -> bool:
        if not self.basic(depth):
            return False
        c = self.char()
        if _digit(c) or (c in ("+", "-") and _digit(self.char(1))):
            self.at += 1
            self.digits()
        elif c == "^" or self.s.startswith("**", self.at):
            self.at += 1 if c == "^" else 2
            return self.number()
        return True

    def basic(self, depth: int) -> bool:
        c = self.char()
        if c == "(":
            return self.group(depth)
        if _letter(c):
            start = self.at
            self.name()
            if self.s[start:self.at] in _LOG_NAMES and self.char() == "(":
                save = self.at
                if self.reference():
                    return self.logref(depth)
                self.at = save
            return True
        return self.number()

    def name(self) -> bool:
        if not _letter(self.char()):
            return False
        while _letter(self.char()) or _digit(self.char()):
            self.at += 1
        # The trailing digits of a name are its power: "m2" is m
        # squared, and "m2s" is one name.
        while _digit(self.s[self.at - 1]):
            self.at -= 1
        return True

    def group(self, depth: int) -> bool:
        if depth >= limits.MAX_UNITS_DEPTH:
            return False
        self.at += 1
        if not self.units(depth + 1) or self.char() != ")":
            return False
        self.at += 1
        return True

    def reference(self) -> bool:
        """The opening of a logarithmic reference, "(re " or "(re:",
        after a name of a logarithm. When it is there the "(" belongs
        to the reference; when it is not, the "(" opens a group."""
        self.at += 1
        self.spaces()
        if self.s[self.at:self.at + 2] != "re":
            return False
        self.at += 2
        if self.char() == ":":
            self.at += 1
            self.spaces()
            return True
        return self.spaces() > 0

    def logref(self, depth: int) -> bool:
        """The rest of "lg(re 1 mW)": a product and its ")"."""
        if depth >= limits.MAX_UNITS_DEPTH:
            return False
        if not self.product(depth + 1):
            return False
        self.spaces()
        if self.char() != ")":
            return False
        self.at += 1
        return True

    def shift(self) -> bool:
        self.spaces()
        if self.char() == "@":
            self.at += 1
        else:
            for w in _SHIFT_WORDS:
                if self.word(w):
                    self.at += len(w)
                    break
            else:
                return False
        self.spaces()
        save = self.at
        if self.timestamp():
            return True
        self.at = save
        return self.number()

    def sign(self) -> None:
        if self.char() in ("+", "-"):
            self.at += 1

    def number(self) -> bool:
        self.sign()
        whole = self.digits()
        fraction = 0
        if self.char() == ".":
            self.at += 1
            fraction = self.digits()
            if not whole and not fraction:
                return False
        elif not whole:
            return False
        if self.char() in ("e", "E"):
            save = self.at
            self.at += 1
            self.sign()
            if not self.digits():
                self.at = save
        return True

    def timestamp(self) -> bool:
        if not self.date():
            return False
        save = self.at
        if self.char() == "T":
            self.at += 1
        elif not self.spaces():
            return True
        if not self.clock():
            self.at = save
            return True
        save = self.at
        self.spaces()
        if not self.zone():
            self.at = save
        return True

    def date(self) -> bool:
        self.sign()
        if not self.digits() or self.char() != "-":
            return False
        self.at += 1
        if not self.digits(2):
            return False
        if self.char() == "-" and _digit(self.char(1)):
            self.at += 1
            self.digits(2)
        return True

    def clock(self) -> bool:
        if not self.digits(2) or self.char() != ":":
            return False
        self.at += 1
        if self.digits(2) != 2:
            return False
        if self.char() == ":" and _digit(self.char(1)):
            self.at += 1
            if self.digits(2) != 2:
                return False
            if self.char() == ".":
                self.at += 1
                self.digits()
        return True

    def zone(self) -> bool:
        if self.char() in ("+", "-"):
            self.at += 1
            if not self.digits(2):
                return False
            save = self.at
            if self.char() == ":":
                self.at += 1
            if self.digits(2) != 2:
                self.at = save
            return True
        return self.name()


def is_parseable(text: str) -> bool:
    """True when `text` matches the units grammar of section 32.

    This is the W10 verdict. It looks at the shape of the string and
    never at its names, so a unit this module has never heard of
    parses as long as it is written the way a unit is written.
    """
    if not isinstance(text, str):
        return False
    try:
        if len(text.encode("utf-8")) > limits.MAX_UNITS_LENGTH:
            return False
    except UnicodeEncodeError:
        return False
    grammar = _Grammar(text)
    return grammar.units(0) and grammar.at == len(text)


def parse(text: str) -> Unit | None:
    """The dimensions of a units string, or None when there are none
    this module can give.

    None when the string is outside the grammar (W10), and also when
    it is grammatical but names a unit outside this module's table or
    carries a shift: such a string is valid, and a tool still must
    not combine it with anything.
    """
    if not is_parseable(text):
        return None
    try:
        return Unit(text, _Parser(text.strip()).parse())
    except _ParseError:
        return None
    except RecursionError:                          # pragma: no cover
        return None


def same_dimensions(left: str, right: str) -> bool:
    """True when both parse and their dimensions agree.

    Tools must refuse to combine unparseable units, so a string
    `parse` gives no dimensions for is never the same as anything.
    """
    a, b = parse(left), parse(right)
    if a is None or b is None:
        return False
    return a.dimensions == b.dimensions
