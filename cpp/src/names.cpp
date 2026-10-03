#include "names.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace mestra {
namespace internal {

bool legal_netcdf_name(const std::string& name) {
  if (name.empty()) return false;
  if (name.front() == ' ' || name.back() == ' ') return false;
  // Section 18: the first character is a letter, a digit or an
  // underscore, as netCDF-C requires.
  const unsigned char first = static_cast<unsigned char>(name.front());
  if (!std::isalnum(first) && first != '_') return false;
  for (const char raw : name) {
    const unsigned char c = static_cast<unsigned char>(raw);
    if (c == '\0' || c == '/') return false;
    if (std::isalnum(c) || c == '_' || c == '-' || c == '.' || c == '+') {
      continue;
    }
    // Section 18 lists the ASCII set a name is built from; anything
    // outside it, including a space in the middle, is not a legal
    // netCDF-4 name for this format's purposes.
    return false;
  }
  return true;
}

bool reserved_name(const std::string& name) {
  return name.compare(0, 7, "mestra_") == 0;
}

bool valid_utf8(const std::string& s) {
  std::size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    std::size_t extra = 0;
    unsigned int code = 0;
    if (c == 0x00) return false;
    if (c < 0x80) {
      ++i;
      continue;
    } else if ((c & 0xE0u) == 0xC0u) {
      extra = 1;
      code = c & 0x1Fu;
    } else if ((c & 0xF0u) == 0xE0u) {
      extra = 2;
      code = c & 0x0Fu;
    } else if ((c & 0xF8u) == 0xF0u) {
      extra = 3;
      code = c & 0x07u;
    } else {
      return false;
    }
    if (i + extra >= s.size()) return false;
    for (std::size_t k = 1; k <= extra; ++k) {
      const unsigned char n = static_cast<unsigned char>(s[i + k]);
      if ((n & 0xC0u) != 0x80u) return false;
      code = (code << 6) | (n & 0x3Fu);
    }
    if (extra == 1 && code < 0x80u) return false;
    if (extra == 2 && code < 0x800u) return false;
    if (extra == 3 && code < 0x10000u) return false;
    if (code > 0x10FFFFu) return false;
    if (code >= 0xD800u && code <= 0xDFFFu) return false;
    i += extra + 1;
  }
  return true;
}

std::string strip_nul(const std::string& s) {
  std::size_t n = s.size();
  while (n > 0 && s[n - 1] == '\0') --n;
  return s.substr(0, n);
}

bool embedded_nul(const std::string& s) {
  const std::string stripped = strip_nul(s);
  return stripped.find('\0') != std::string::npos;
}

bool iso8601_utc(const std::string& s) {
  // YYYY-MM-DDThh:mm:ss, then an optional fractional part, then "Z" or
  // "+00:00".  Anything else is not a UTC timestamp in this format.
  if (s.size() < 20) return false;
  auto digits = [&](std::size_t at, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
      if (at + k >= s.size()) return false;
      if (!std::isdigit(static_cast<unsigned char>(s[at + k]))) return false;
    }
    return true;
  };
  if (!digits(0, 4) || s[4] != '-' || !digits(5, 2) || s[7] != '-' ||
      !digits(8, 2)) {
    return false;
  }
  if (s[10] != 'T') return false;
  if (!digits(11, 2) || s[13] != ':' || !digits(14, 2) || s[16] != ':' ||
      !digits(17, 2)) {
    return false;
  }
  std::size_t at = 19;
  if (at < s.size() && s[at] == '.') {
    ++at;
    std::size_t n = 0;
    while (at < s.size() && std::isdigit(static_cast<unsigned char>(s[at]))) {
      ++at;
      ++n;
    }
    if (n == 0) return false;
  }
  const std::string zone = s.substr(at);
  return zone == "Z" || zone == "+00:00" || zone == "-00:00";
}

std::string format_f64(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.17e", v);
  return std::string(buffer);
}

std::string format_i64(std::int64_t v) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%lld",
                static_cast<long long>(v));
  return std::string(buffer);
}

std::string basename(const std::string& path) {
  const std::size_t at = path.find_last_of('/');
  return at == std::string::npos ? path : path.substr(at + 1);
}

std::string group_of_varies(const std::string& varies) {
  if (varies.compare(0, 6, "group:") == 0) return varies.substr(6);
  return std::string();
}

std::string callable_of_source(const std::string& source) {
  // "callable:" alone names no callable: it is E36, like any other
  // source that is neither word (section 19).
  if (source.size() > 9 && source.compare(0, 9, "callable:") == 0) {
    return source.substr(9);
  }
  return std::string();
}

namespace {

// A simple identifier takes an exponent straight after it: "m" to the
// third is "m3".  Anything else -- a product, a quotient, or a name
// that already carries an exponent -- is bracketed first, because
// "m2" cubed is not "m23".  Both forms are in the grammar of the W10
// parser.
bool simple_identifier(const std::string& s) {
  if (s.empty()) return false;
  if (!std::isalpha(static_cast<unsigned char>(s[0]))) return false;
  for (const char c : s) {
    if (!std::isalpha(static_cast<unsigned char>(c)) && c != '_') {
      return false;
    }
  }
  return true;
}

}  // namespace

std::string units_power(const std::string& units, int power) {
  if (units.empty() || units == "1" || power == 1) return units;
  if (power == 0) return "1";
  const std::string exponent = format_i64(power);
  if (simple_identifier(units)) return units + exponent;
  return "(" + units + ")" + exponent;
}

std::string units_product(const std::string& a, const std::string& b) {
  if (a.empty() || a == "1") return b;
  if (b.empty() || b == "1") return a;
  return a + " " + b;
}

std::vector<std::string> split_spaces(const std::string& s) {
  std::vector<std::string> out;
  std::size_t at = 0;
  while (at < s.size()) {
    while (at < s.size() && s[at] == ' ') ++at;
    const std::size_t start = at;
    while (at < s.size() && s[at] != ' ') ++at;
    if (at > start) out.push_back(s.substr(start, at - start));
  }
  return out;
}

}  // namespace internal
}  // namespace mestra
