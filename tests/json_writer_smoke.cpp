#include "../hydra/jsonWriter.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
int main() {
  lava_json::Writer payload;
  payload << "{\"points\":[[" << 0.125 << ',' << -2.5f << ',' << 3
          << "]],\"indices\":[" << size_t(0) << ',' << 1 << ',' << 2
          << "],\"visible\":" << true << '}';
  assert(payload.str() ==
         "{\"points\":[[0.125,-2.5,3]],\"indices\":[0,1,2],\"visible\":true}");
  // Preserve the existing nine-significant-digit transport precision, including
  // very small normals, large coordinates, negative zero and double extremes.
  for (double value : {0., -0., 1. / 3., -1234567.89, 1e-40, -1e30,
                       std::numeric_limits<double>::min(),
                       std::numeric_limits<double>::max()}) {
    lava_json::Writer fast;
    fast << value;
    std::ostringstream old;
    old << std::setprecision(9) << value;
    double a = std::strtod(fast.str().c_str(), nullptr);
    double b = std::strtod(old.str().c_str(), nullptr);
    assert(a == b && std::signbit(a) == std::signbit(b));
  }
  lava_json::Writer packed;
  packed.vectors<3>(
      1, [](size_t) { return std::array<float, 3>{0.f, -0.f, 1.f}; });
  assert(packed.str() == "{\"f32le\":\"00000000000000800000803f\"}");
  lava_json::Writer integers;
  integers << std::numeric_limits<unsigned long long>::max() << ','
           << std::numeric_limits<long long>::min();
  assert(integers.str() == "18446744073709551615,-9223372036854775808");
  std::cout
      << "PASS: JSON writer structure, numeric precision and integer bounds\n";
}
