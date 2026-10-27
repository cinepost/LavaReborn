#pragma once
#include "meshNormals.h"
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>

PXR_NAMESPACE_OPEN_SCOPE
namespace lava {
// Recover authored Principled inputs that Houdini's auto-preview loses (notably
// monoChannel and bump parameters). Read only; never modify the source stage.
inline UsdPrim PrincipledSource(HdSceneDelegate *delegate, SdfPath const &id,
                                UsdTimeCode *time = nullptr) {
  auto *imaging = dynamic_cast<UsdImagingDelegate *>(delegate);
  if (!imaging) {
    for (auto const &rprim : delegate->GetRenderIndex().GetRprimIds()) {
      auto *candidate = dynamic_cast<UsdImagingDelegate *>(
          delegate->GetRenderIndex().GetSceneDelegateForRprim(rprim));
      if (candidate && id.HasPrefix(candidate->GetDelegateID())) {
        imaging = candidate;
        break;
      }
    }
  }
  if (!imaging)
    return {};
  if (time)
    *time = imaging->GetTime();
  MeshSourceReader reader;
  auto material = reader.Source(imaging, id);
  if (!material)
    return {};
  SdfPathVector paths;
  auto output = material.GetAttribute(TfToken("outputs:vex:surface"));
  if (output)
    output.GetConnections(&paths);
  if (paths.empty())
    return {};
  auto shader = material.GetStage()->GetPrimAtPath(paths[0].GetPrimPath());
  SdfAssetPath asset;
  if (!shader ||
      !shader.GetAttribute(TfToken("info:sourceAsset")).Get(&asset) ||
      asset.GetAssetPath().find("principledshader") == std::string::npos)
    return {};
  return shader;
}
inline VtValue PrincipledValue(UsdPrim const &prim, char const *name,
                               UsdTimeCode time = UsdTimeCode::Default()) {
  auto attr = prim.GetAttribute(TfToken(std::string("inputs:") + name));
  VtValue value;
  if (attr) {
    attr.Get(&value, time);
    if (value.IsEmpty())
      value =
          attr.GetCustomDataByKey(TfToken("HoudiniPreviewTags:default_value"));
  }
  return value;
}
inline HdMaterialNetwork
PrincipledNetwork(HdMaterialNetwork network, HdMaterialNode const &terminal,
                  UsdPrim const &prim,
                  UsdTimeCode time = UsdTimeCode::Default()) {
  if (!prim)
    return network;
  auto scalar = [&](char const *name, float fallback) {
    auto v = PrincipledValue(prim, name, time);
    if (v.IsHolding<float>())
      return v.UncheckedGet<float>();
    if (v.IsHolding<int>())
      return float(v.UncheckedGet<int>());
    return fallback;
  };
  auto text = [&](char const *name, std::string fallback) {
    auto v = PrincipledValue(prim, name, time);
    if (v.IsHolding<std::string>())
      return v.UncheckedGet<std::string>();
    if (v.IsHolding<TfToken>())
      return v.UncheckedGet<TfToken>().GetString();
    return fallback;
  };
  auto remove = [&](std::string const &name) {
    auto &r = network.relationships;
    r.erase(std::remove_if(r.begin(), r.end(),
                           [&](auto const &c) {
                             return c.outputId == terminal.path &&
                                    c.outputName == TfToken(name);
                           }),
            r.end());
  };
  auto add = [&](char const *field, char const *file, char const *wrap,
                 char const *space, GfVec4f scale, GfVec4f bias,
                 std::string channel, float strength) {
    remove(field);
    auto asset = PrincipledValue(prim, file, time);
    if (!asset.IsHolding<SdfAssetPath>() ||
        asset.UncheckedGet<SdfAssetPath>().GetAssetPath().empty())
      return;
    HdMaterialNode image;
    image.path =
        terminal.path.AppendChild(TfToken(std::string("lava_") + field));
    image.identifier = TfToken("UsdUVTexture");
    image.parameters[TfToken("file")] = asset;
    auto colorspace =
        text(space, std::string(field) == "diffuseColor" ? "auto" : "raw");
    if (colorspace == "linear")
      colorspace = "raw";
    image.parameters[TfToken("sourceColorSpace")] =
        VtValue(TfToken(colorspace));
    image.parameters[TfToken("wrapS")] = image.parameters[TfToken("wrapT")] =
        VtValue(TfToken(text(wrap, "repeat")));
    image.parameters[TfToken("scale")] = VtValue(scale);
    image.parameters[TfToken("bias")] = VtValue(bias);
    image.parameters[TfToken("lava:strength")] = VtValue(strength);
    network.nodes.push_back(image);
    network.relationships.push_back(
        {image.path, TfToken(channel), terminal.path, TfToken(field)});
  };
  if (scalar("basecolor_useTexture", 0) != 0) {
    auto value = PrincipledValue(prim, "basecolor", time);
    GfVec3f color(.2f);
    if (value.IsHolding<GfVec3f>())
      color = value.UncheckedGet<GfVec3f>();
    color *= scalar("basecolor_textureIntensity", 1);
    add("diffuseColor", "basecolor_texture", "basecolor_textureWrap",
        "basecolor_textureColorSpace", GfVec4f(color[0], color[1], color[2], 1),
        GfVec4f(0), "rgb", 1);
  }
  for (auto name : {std::string("rough"), std::string("metallic")}) {
    if (scalar((name + "_useTexture").c_str(), 0) == 0)
      continue;
    int channel = int(scalar((name + "_monoChannel").c_str(), 0));
    const char *channels[] = {"luminance", "r", "g", "b"};
    float multiplier = scalar(name.c_str(), name == "rough" ? .3f : 0.f);
    add(name == "rough" ? "roughness" : "metallic", (name + "_texture").c_str(),
        (name + "_textureWrap").c_str(), (name + "_textureColorSpace").c_str(),
        GfVec4f(multiplier), GfVec4f(0), channels[std::clamp(channel, 0, 3)],
        1);
  }
  if (scalar("baseBumpAndNormal_enable", 0) != 0) {
    remove("normal");
    if (text("baseBumpAndNormal_type", "normal") == "bump")
      add("bump", "baseBump_bumpTexture", "baseBump_wrap",
          "baseBump_colorSpace", GfVec4f(1), GfVec4f(0), "luminance",
          scalar("baseBump_bumpScale", .05f));
    else {
      auto space = text("baseNormal_vectorSpace", "uvtangent");
      if (space != "uvtangent") {
        std::printf("Lava [WARNING]: Principled normal space %s is "
                    "unsupported; use tangent space.\n",
                    space.c_str());
      } else {
        float x = scalar("baseNormal_flipX", 0) != 0 ? -1.f : 1.f;
        float y = scalar("baseNormal_flipY", 0) != 0 ? -1.f : 1.f;
        add("normal", "baseNormal_texture", "baseNormal_wrap",
            "baseNormal_colorspace", GfVec4f(2 * x, 2 * y, 2, 1),
            GfVec4f(-x, -y, -1, 0), "rgb", scalar("baseNormal_scale", 1));
      }
    }
  }
  return network;
}
} // namespace lava
PXR_NAMESPACE_CLOSE_SCOPE
