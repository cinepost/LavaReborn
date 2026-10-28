#pragma once
#include <UT/UT_String.h>
#include <pxr/usd/sdf/layerUtils.h>
#include <pxr/usd/sdf/propertySpec.h>
#include <pxr/usd/usd/attribute.h>

PXR_NAMESPACE_OPEN_SCOPE
namespace lava {
inline std::string ExpandTexturePath(std::string path) {
  UT_String expanded(path.c_str());
  expanded.expandVariables();
  path = expanded.c_str();
  const std::string token = "%(UDIM)d";
  size_t pos = 0;
  while ((pos = path.find(token, pos)) != std::string::npos) {
    path.replace(pos, token.size(), "<UDIM>");
    pos += 6;
  }
  return path;
}
inline bool IsUdimPath(std::string const &path) {
  return path.find("<UDIM>") != std::string::npos ||
         path.find("%(UDIM)d") != std::string::npos;
}

inline std::string AnchorUdim(UsdAttribute const &attribute, std::string path,
                              UsdTimeCode time = UsdTimeCode::Default()) {
  path = ExpandTexturePath(path);
  if (path.find("<UDIM>") == std::string::npos || !attribute)
    return path;
  auto stack = attribute.GetPropertyStack(time);
  if (stack.empty())
    return path;
  // Force file-relative anchoring for a pattern that cannot itself resolve as
  // an existing file. Tile discovery/conversion remains in the async subsystem.
  if (!path.empty() && path[0] != '/' && path.find(':') == std::string::npos &&
      path.rfind("./", 0) != 0)
    path = "./" + path;
  return SdfComputeAssetPathRelativeToLayer(stack.front()->GetLayer(), path);
}
} // namespace lava
PXR_NAMESPACE_CLOSE_SCOPE
