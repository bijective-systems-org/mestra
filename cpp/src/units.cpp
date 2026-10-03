// The units grammar W10 is decided by (SPEC.md section 32).
//
// A units string parses when it matches the grammar, whatever names it
// uses: "furlong fortnight-1" parses and "m//s" does not.  This is the
// grammar and not a units database, and it says whether a string
// matches and nothing else.  Each method below is one production of
// section 32; it returns true and moves past what it matched, or
// returns false and leaves the position for its caller to restore.
//
// The string is read byte by byte.  Every byte of a UTF-8 sequence for
// a character outside ASCII is 0x80 or above, and every such character
// is a letter of a name, so reading bytes gives the verdict reading
// characters would.

#include "mestra/validate.hpp"

#include <cstddef>
#include <string>

namespace mestra {
namespace {

// The bounds of section 32: the string comes out of a file, and the
// recursion through parentheses is bounded rather than trusted.
const std::size_t kMaxBytes = 4096;
const int kMaxDepth = 32;

bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool is_letter(char c) {
  const unsigned char u = static_cast<unsigned char>(c);
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' ||
         c == '%' || c == '\'' || c == '"' || u > 127;
}

class Grammar {
 public:
  explicit Grammar(const std::string& s) : s_(s) {}

  bool whole() { return units(0) && at_ == s_.size(); }

 private:
  char at(std::size_t offset = 0) const {
    return at_ + offset < s_.size() ? s_[at_ + offset] : '\0';
  }

  std::size_t spaces() {
    const std::size_t start = at_;
    while (at() == ' ') ++at_;
    return at_ - start;
  }

  std::size_t digits(std::size_t most = std::string::npos) {
    const std::size_t start = at_;
    while (is_digit(at()) && at_ - start < most) ++at_;
    return at_ - start;
  }

  // `w` here as a whole word: not run on into a name.
  bool word(const char* w) const {
    const std::string text(w);
    if (s_.compare(at_, text.size(), text) != 0) return false;
    const char after = at(text.size());
    return !(is_letter(after) || is_digit(after));
  }

  bool shift_word(std::size_t* length) const {
    for (const char* w : {"since", "from", "after", "ref"}) {
      if (word(w)) {
        *length = std::string(w).size();
        return true;
      }
    }
    return false;
  }

  bool units(int depth) {
    spaces();
    if (!product(depth)) return false;
    const std::size_t save = at_;
    if (!shift()) at_ = save;
    spaces();
    return true;
  }

  bool product(int depth) {
    if (!power(depth)) return false;
    for (;;) {
      const std::size_t save = at_;
      spaces();
      const char c = at();
      std::size_t length = 0;
      if (c == '\0' || c == ')' || c == '@' || shift_word(&length)) {
        at_ = save;
        return true;
      }
      if (c == '*' || c == '.' || c == '/' || c == '-') {
        ++at_;
      } else if (word("per")) {
        at_ += 3;
      } else {
        // Juxtaposition: a space, or nothing at all, between two
        // powers is a product.
        if (power(depth)) continue;
        at_ = save;
        return true;
      }
      spaces();
      if (!power(depth)) return false;
    }
  }

  bool power(int depth) {
    if (!basic(depth)) return false;
    const char c = at();
    if (is_digit(c) || ((c == '+' || c == '-') && is_digit(at(1)))) {
      ++at_;
      digits();
    } else if (c == '^' || (c == '*' && at(1) == '*')) {
      at_ += c == '^' ? 1 : 2;
      return number();
    }
    return true;
  }

  bool basic(int depth) {
    const char c = at();
    if (c == '(') return group(depth);
    if (is_letter(c)) {
      const std::size_t start = at_;
      name();
      const std::string n = s_.substr(start, at_ - start);
      if ((n == "log" || n == "lg" || n == "ln" || n == "lb") &&
          at() == '(') {
        const std::size_t save = at_;
        if (reference()) return logref(depth);
        at_ = save;
      }
      return true;
    }
    return number();
  }

  bool name() {
    if (!is_letter(at())) return false;
    while (is_letter(at()) || is_digit(at())) ++at_;
    // The trailing digits of a name are its power: "m2" is m squared,
    // and "m2s" is one name.
    while (is_digit(s_[at_ - 1])) --at_;
    return true;
  }

  bool group(int depth) {
    if (depth >= kMaxDepth) return false;
    ++at_;
    if (!units(depth + 1) || at() != ')') return false;
    ++at_;
    return true;
  }

  // The opening of a logarithmic reference, "(re " or "(re:", after a
  // name of a logarithm.  When it is there the "(" belongs to the
  // reference; when it is not, the "(" opens a group.
  bool reference() {
    ++at_;
    spaces();
    if (s_.compare(at_, 2, "re") != 0) return false;
    at_ += 2;
    if (at() == ':') {
      ++at_;
      spaces();
      return true;
    }
    return spaces() > 0;
  }

  // The rest of "lg(re 1 mW)": a product and its ")".
  bool logref(int depth) {
    if (depth >= kMaxDepth) return false;
    if (!product(depth + 1)) return false;
    spaces();
    if (at() != ')') return false;
    ++at_;
    return true;
  }

  bool shift() {
    spaces();
    std::size_t length = 0;
    if (at() == '@') {
      ++at_;
    } else if (shift_word(&length)) {
      at_ += length;
    } else {
      return false;
    }
    spaces();
    const std::size_t save = at_;
    if (timestamp()) return true;
    at_ = save;
    return number();
  }

  void sign() {
    if (at() == '+' || at() == '-') ++at_;
  }

  bool number() {
    sign();
    const std::size_t whole = digits();
    if (at() == '.') {
      ++at_;
      const std::size_t fraction = digits();
      if (whole == 0 && fraction == 0) return false;
    } else if (whole == 0) {
      return false;
    }
    if (at() == 'e' || at() == 'E') {
      const std::size_t save = at_;
      ++at_;
      sign();
      if (digits() == 0) at_ = save;
    }
    return true;
  }

  bool timestamp() {
    if (!date()) return false;
    std::size_t save = at_;
    if (at() == 'T') {
      ++at_;
    } else if (spaces() == 0) {
      return true;
    }
    if (!clock()) {
      at_ = save;
      return true;
    }
    save = at_;
    spaces();
    if (!zone()) at_ = save;
    return true;
  }

  bool date() {
    sign();
    if (digits() == 0 || at() != '-') return false;
    ++at_;
    if (digits(2) == 0) return false;
    if (at() == '-' && is_digit(at(1))) {
      ++at_;
      digits(2);
    }
    return true;
  }

  bool clock() {
    if (digits(2) == 0 || at() != ':') return false;
    ++at_;
    if (digits(2) != 2) return false;
    if (at() == ':' && is_digit(at(1))) {
      ++at_;
      if (digits(2) != 2) return false;
      if (at() == '.') {
        ++at_;
        digits();
      }
    }
    return true;
  }

  bool zone() {
    if (at() == '+' || at() == '-') {
      ++at_;
      if (digits(2) == 0) return false;
      const std::size_t save = at_;
      if (at() == ':') ++at_;
      if (digits(2) != 2) at_ = save;
      return true;
    }
    return name();
  }

  const std::string& s_;
  std::size_t at_ = 0;
};

}  // namespace

bool units_parse(const std::string& units) {
  if (units.size() > kMaxBytes) return false;
  return Grammar(units).whole();
}

}  // namespace mestra
