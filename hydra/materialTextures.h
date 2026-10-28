#pragma once
#include "texturePaths.h"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <map>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/stage.h>
#include <set>
#include <sstream>

PXR_NAMESPACE_OPEN_SCOPE
namespace lava {
inline std::string JsonString(std::string const &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << unsigned(c) << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
inline std::string TextureInputs(HdMaterialNetwork const &network,
                                 HdMaterialNode const &terminal, bool preview,
                                 UsdPrim const &material = UsdPrim(),
                                 SdfPath const &cacheId = SdfPath()) {
  std::ostringstream out;
  bool first = true;
  for (auto const &connection : network.relationships) {
    if (connection.outputId != terminal.path)
      continue;
    std::string field = connection.outputName.GetString();
    if (preview) {
      static std::map<std::string, std::string> const names = {
          {"diffuseColor", "base_color"},
          {"metallic", "base_metalness"},
          {"roughness", "specular_roughness"},
          {"transmission", "transmission_weight"},
          {"transmissionColor", "transmission_color"},
          {"ior", "specular_ior"},
          {"opacity", "geometry_opacity"},
          {"normal", "normal"},
          {"bump", "bump"},
          {"emissiveColor", "emission_color"}};
      auto it = names.find(field);
      if (it == names.end()) {
        std::printf("Lava [WARNING]: unsupported connected PreviewSurface "
                    "input %s; using constant.\n",
                    field.c_str());
        continue;
      }
      field = it->second;
    }
    static std::set<std::string> const supported = {
#define LAVA_TEXTURE(name, index, color) #name,
#include "material_texture_fields.inc"
#undef LAVA_TEXTURE
    };
    if (!supported.count(field)) {
      std::printf("Lava [WARNING]: texture input %s is not implemented; using "
                  "constant.\n",
                  field.c_str());
      continue;
    }
    HdMaterialNode const *image = nullptr;
    for (auto const &node : network.nodes)
      if (node.path == connection.inputId)
        image = &node;
    if (!image || image->identifier != TfToken("UsdUVTexture")) {
      std::printf("Lava [WARNING]: only direct UsdUVTexture material "
                  "connections are supported for %s; using constant.\n",
                  field.c_str());
      continue;
    }
    auto parameter = [&](char const *name) -> VtValue {
      auto it = image->parameters.find(TfToken(name));
      return it == image->parameters.end() ? VtValue() : it->second;
    };
    auto asset = parameter("file");
    std::string path;
    if (asset.IsHolding<SdfAssetPath>()) {
      auto const &a = asset.UncheckedGet<SdfAssetPath>();
      path = IsUdimPath(a.GetAssetPath()) || a.GetResolvedPath().empty()
                 ? a.GetAssetPath()
                 : a.GetResolvedPath();
    } else if (asset.IsHolding<std::string>())
      path = asset.UncheckedGet<std::string>();
    path = ExpandTexturePath(path);
    if (path.empty()) {
      std::printf(
          "Lava [WARNING]: empty texture asset for %s; using constant.\n",
          field.c_str());
      continue;
    }
    if (path.find("<UDIM>") != std::string::npos && material) {
      auto sourcePath = image->path;
      if (sourcePath.HasPrefix(cacheId))
        sourcePath = sourcePath.ReplacePrefix(cacheId, material.GetPath());
      auto shader = material.GetStage()->GetPrimAtPath(sourcePath);
      if (shader)
        path = AnchorUdim(shader.GetAttribute(TfToken("inputs:file")), path);
    }
    auto token = [&](char const *name, std::string fallback) -> std::string {
      auto value = parameter(name);
      if (value.IsHolding<TfToken>())
        return value.UncheckedGet<TfToken>().GetString();
      return value.IsHolding<std::string>() ? value.UncheckedGet<std::string>()
                                            : fallback;
    };
    std::string space = token("sourceColorSpace", "auto");
    if (space == "sRGB")
      space = "srgb";
    if (space != "srgb" && space != "raw")
      space = "auto";
    std::string quality = token("lava:compressionQuality", "balanced");
    if (quality != "fast" && quality != "high")
      quality = "balanced";
    auto wrap = [&](char const *name) {
      auto w = token(name, "repeat");
      return w == "clamp" || w == "mirror" || w == "black"
                 ? w
                 : std::string("repeat");
    };
    auto channel = connection.inputName.GetString();
    if (channel != "r" && channel != "g" && channel != "b" && channel != "a" &&
        channel != "luminance")
      channel = "rgb";
    if (!first)
      out << ',';
    first = false;
    out << JsonString(field) << ":{\"file\":" << JsonString(path)
        << ",\"color_space\":" << JsonString(space)
        << ",\"channel\":" << JsonString(channel)
        << ",\"compression\":{\"quality\":" << JsonString(quality) << "}"
        << ",\"wrap_s\":" << JsonString(wrap("wrapS"))
        << ",\"wrap_t\":" << JsonString(wrap("wrapT"));
    for (auto const *name : {"scale", "bias"}) {
      auto value = parameter(name);
      if (value.IsHolding<GfVec4f>()) {
        auto v = value.UncheckedGet<GfVec4f>();
        out << ",\"" << name << "\":[" << v[0] << ',' << v[1] << ',' << v[2]
            << ',' << v[3] << ']';
      }
    }
    if (field == "normal")
      out << ",\"signed\":true";
    auto strength = parameter("lava:strength");
    if (strength.IsHolding<float>())
      out << ",\"strength\":" << strength.UncheckedGet<float>();
    out << '}';
  }
  return out.str();
}
inline bool HasInputConnection(HdMaterialNetwork const &network,
                               SdfPath const &path, TfToken const &input) {
  return std::any_of(network.relationships.begin(), network.relationships.end(),
                     [&](auto const &r) {
                       return r.outputId == path && r.outputName == input;
                     });
}
} // namespace lava
PXR_NAMESPACE_CLOSE_SCOPE
