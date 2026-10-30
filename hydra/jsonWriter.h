#pragma once
#include <charconv>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace lava_json {
// Numeric scene payloads only: callers supply JSON punctuation and escaped
// keys. Avoid locale processing, stream state and temporary per-vector
// allocations.
class Writer {
  std::string text;

public:
  explicit Writer(size_t capacity = 0) { text.reserve(capacity); }
  Writer &operator<<(std::string_view value) {
    text.append(value);
    return *this;
  }
  Writer &operator<<(const std::string &value) {
    text.append(value);
    return *this;
  }
  Writer &operator<<(const char *value) {
    text.append(value);
    return *this;
  }
  Writer &operator<<(char value) {
    text.push_back(value);
    return *this;
  }
  Writer &operator<<(bool value) {
    text.append(value ? "true" : "false");
    return *this;
  }
  template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
  Writer &operator<<(T value) {
    char buffer[64];
    auto result = [&] {
      if constexpr (std::is_floating_point_v<T>)
        return std::to_chars(buffer, buffer + sizeof(buffer), value,
                             std::chars_format::general, 9);
      else
        return std::to_chars(buffer, buffer + sizeof(buffer), value);
    }();
    if (result.ec != std::errc())
      throw std::runtime_error("formatting Lava scene number");
    text.append(buffer, result.ptr);
    return *this;
  }
  // Hex of little-endian IEEE float32 bytes, avoiding decimal conversion on
  // both sides of the existing JSON ABI. No GPU image data passes through this
  // path.
  template <size_t N, class Getter> Writer &vectors(size_t count, Getter get) {
    text.append("{\"f32le\":\"");
    constexpr char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < count; ++i) {
      auto value = get(i);
      char encoded[N * 8];
      for (size_t j = 0; j < N; ++j) {
        float number = static_cast<float>(value[j]);
        uint32_t bits;
        std::memcpy(&bits, &number, sizeof(bits));
        for (size_t b = 0; b < 4; ++b) {
          auto byte = (bits >> (b * 8)) & 255;
          encoded[j * 8 + b * 2] = digits[byte >> 4];
          encoded[j * 8 + b * 2 + 1] = digits[byte & 15];
        }
      }
      text.append(encoded, sizeof(encoded));
    }
    text.append("\"}");
    return *this;
  }
  std::string str() { return std::move(text); }
};
} // namespace lava_json
