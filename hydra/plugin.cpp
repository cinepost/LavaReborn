// Hydra 1 render delegate adapter. Vulkan execution lives exclusively in Rust.
#include "interop.h"
#include "lava.h"
#include "materialTextures.h"
#include "meshNormals.h"
#include "principledTextures.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <mutex>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/registryManager.h>
#include <pxr/base/vt/array.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/mesh.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/shaderDefUtils.h>
#include <sstream>

PXR_NAMESPACE_OPEN_SCOPE
namespace {
std::string Vec(GfVec3d const &p) {
  std::ostringstream o;
  o << std::setprecision(9) << '[' << p[0] << ',' << p[1] << ',' << p[2] << ']';
  return o.str();
}
// Renderer primvars are transported by UsdImaging for both lights and meshes.
bool BoolPrimvar(HdSceneDelegate *d, SdfPath const &id, char const *name,
                 bool fallback) {
  auto value = d->Get(id, TfToken(name));
  if (value.IsHolding<bool>())
    return value.UncheckedGet<bool>();
  if (value.IsHolding<VtBoolArray>() &&
      !value.UncheckedGet<VtBoolArray>().empty())
    return value.UncheckedGet<VtBoolArray>()[0];
  return fallback;
}
bool BoolLightParameter(HdSceneDelegate *d, SdfPath const &id, char const *name,
                        bool fallback) {
  auto value =
      d->GetLightParamValue(id, TfToken(std::string("primvars:") + name));
  return value.IsHolding<bool>() ? value.UncheckedGet<bool>() : fallback;
}
struct State : HdRenderParam {
  std::mutex mutex;
  uint64_t generation = 0;
  uint64_t geometryGeneration = 0;
  std::map<SdfPath, std::string> meshes, lights, materials;
  std::map<SdfPath, SdfPath> meshMaterials;
};
class Mesh final : public HdMesh {
public:
  Mesh(SdfPath const &id) : HdMesh(id) {}
  HdDirtyBits GetInitialDirtyBitsMask() const override {
    return HdChangeTracker::AllDirty;
  }
  TfTokenVector const &GetBuiltinPrimvarNames() const override {
    static TfTokenVector n = {HdTokens->points,
                              HdTokens->displayColor,
                              HdTokens->normals,
                              TfToken("st"),
                              TfToken("lava:geometryLight"),
                              TfToken("lava:renderLightGeometry")};
    return n;
  }
  void Sync(HdSceneDelegate *d, HdRenderParam *param, HdDirtyBits *bits,
            TfToken const &) override {
    auto *state = static_cast<State *>(param);
    if (!d->GetVisible(GetId())) {
      std::lock_guard<std::mutex> lock(state->mutex);
      if (state->meshes.erase(GetId())) {
        ++state->generation;
        ++state->geometryGeneration;
      }
      *bits = HdChangeTracker::Clean;
      return;
    }
    auto value = d->Get(GetId(), HdTokens->points);
    if (!value.IsHolding<VtVec3fArray>()) {
      *bits = HdChangeTracker::Clean;
      return;
    }
    auto const &points = value.UncheckedGet<VtVec3fArray>();
    auto topology = d->GetMeshTopology(GetId());
    auto transform = d->GetTransform(GetId());
    GfVec3f color(.5f);
    auto c = d->Get(GetId(), HdTokens->displayColor);
    if (c.IsHolding<VtVec3fArray>() && !c.UncheckedGet<VtVec3fArray>().empty())
      color = c.UncheckedGet<VtVec3fArray>()[0];
    std::ostringstream o;
    o << std::setprecision(9) << "{\"points\":[";
    for (size_t i = 0; i < points.size(); i++) {
      if (i)
        o << ',';
      o << Vec(transform.Transform(GfVec3d(points[i])));
    }
    o << "],\"face_counts\":[";
    auto counts = topology.GetFaceVertexCounts();
    for (size_t i = 0; i < counts.size(); i++) {
      if (i)
        o << ',';
      o << counts[i];
    }
    o << "],\"indices\":[";
    auto indices = topology.GetFaceVertexIndices();
    for (size_t i = 0; i < indices.size(); i++) {
      if (i)
        o << ',';
      o << indices[i];
    }
    o << "]";
    // Baking points to world space requires the inverse-transpose normal
    // matrix.
    o << ",\"reverse_orientation\":"
      << (((topology.GetOrientation() == TfToken("leftHanded")) !=
           (transform.GetDeterminant3() < 0.))
              ? "true"
              : "false");
    auto authoredNormals = lava::ReadMeshNormals(d, GetId(), topology);
    auto const &normalIndices = authoredNormals.indices;
    auto const &normalValue = authoredNormals.value;
    if (normalValue.IsHolding<VtVec3fArray>()) {
      auto const &normals = normalValue.UncheckedGet<VtVec3fArray>();
      auto interpolation = authoredNormals.interpolation;
      std::vector<int> corners;
      bool valid = interpolation != HdInterpolationCount &&
                   std::abs(transform.GetDeterminant3()) > 1e-20;
      size_t corner = 0;
      for (size_t face = 0; valid && face < counts.size(); ++face) {
        for (int j = 0; j < counts[face]; ++j, ++corner) {
          size_t index = interpolation == HdInterpolationConstant  ? 0
                         : interpolation == HdInterpolationUniform ? face
                         : interpolation == HdInterpolationFaceVarying
                             ? corner
                             : static_cast<size_t>(indices[corner]);
          if (!normalIndices.empty()) {
            if (index >= normalIndices.size() || normalIndices[index] < 0) {
              valid = false;
              break;
            }
            index = static_cast<size_t>(normalIndices[index]);
          }
          if (index >= normals.size()) {
            valid = false;
            break;
          }
          corners.push_back(static_cast<int>(index));
        }
      }
      std::vector<GfVec3d> worldNormals;
      auto normalMatrix = transform.GetInverse().GetTranspose();
      for (auto const &n : normals) {
        auto world = normalMatrix.TransformDir(GfVec3d(n));
        if (!std::isfinite(world[0]) || !std::isfinite(world[1]) ||
            !std::isfinite(world[2])) {
          valid = false;
          break;
        }
        if (world.GetLength() > 1e-20)
          world.Normalize();
        else
          world = GfVec3d(0);
        worldNormals.push_back(world);
      }
      if (valid && !normals.empty()) {
        o << ",\"normals\":[";
        for (size_t i = 0; i < worldNormals.size(); ++i) {
          if (i)
            o << ',';
          o << Vec(worldNormals[i]);
        }
        o << "],\"normal_indices\":[";
        for (size_t i = 0; i < corners.size(); ++i) {
          if (i)
            o << ',';
          o << corners[i];
        }
        o << ']';
      } else if (!normals.empty()) {
        std::printf("Lava [WARNING]: unusable authored normals on %s; using "
                    "geometric normals.\n",
                    GetId().GetText());
      }
    }
    VtIntArray uvIndices;
    auto uvValue = d->GetIndexedPrimvar(GetId(), TfToken("st"), &uvIndices);
    if (uvValue.IsHolding<VtVec2fArray>()) {
      auto const &uv = uvValue.UncheckedGet<VtVec2fArray>();
      bool faceVarying = false;
      for (auto const &descriptor :
           d->GetPrimvarDescriptors(GetId(), HdInterpolationFaceVarying))
        if (descriptor.name == TfToken("st"))
          faceVarying = true;
      bool valid = true;
      std::vector<int> corners;
      for (size_t i = 0; i < indices.size(); ++i) {
        size_t index = faceVarying ? i : static_cast<size_t>(indices[i]);
        if (!uvIndices.empty()) {
          if (index >= uvIndices.size()) {
            valid = false;
            break;
          }
          index = static_cast<size_t>(uvIndices[index]);
        }
        if (index >= uv.size()) {
          valid = false;
          break;
        }
        corners.push_back(static_cast<int>(index));
      }
      if (valid && !uv.empty()) {
        o << ",\"texcoords\":[";
        for (size_t i = 0; i < uv.size(); ++i) {
          if (i)
            o << ',';
          o << '[' << uv[i][0] << ',' << uv[i][1] << ']';
        }
        o << "],\"texcoord_indices\":[";
        for (size_t i = 0; i < corners.size(); ++i) {
          if (i)
            o << ',';
          o << corners[i];
        }
        o << ']';
      }
    }
    o << ",\"geometry_light\":"
      << (BoolPrimvar(d, GetId(), "lava:geometryLight", false) ? "true"
                                                               : "false")
      << ",\"render_light_geometry\":"
      << (BoolPrimvar(d, GetId(), "lava:renderLightGeometry", false) ? "true"
                                                                     : "false")
      << ",\"color\":" << Vec(GfVec3d(color)) << '}';
    std::lock_guard<std::mutex> lock(state->mutex);
    auto serialized = o.str();
    auto materialId = d->GetMaterialId(GetId());
    auto &stored = state->meshes[GetId()];
    if (stored != serialized || state->meshMaterials[GetId()] != materialId) {
      state->meshMaterials[GetId()] = materialId;
      stored = std::move(serialized);
      ++state->generation;
      ++state->geometryGeneration;
    }
    *bits = HdChangeTracker::Clean;
  }

protected:
  HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override {
    return bits;
  }
  void _InitRepr(TfToken const &, HdDirtyBits *) override {}
};
class Light final : public HdLight {
  TfToken type;

public:
  Light(SdfPath const &id, TfToken const &t) : HdLight(id), type(t) {}
  HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
  void Sync(HdSceneDelegate *d, HdRenderParam *param,
            HdDirtyBits *bits) override {
    auto get = [&](TfToken const &key, float fallback) {
      auto v = d->GetLightParamValue(GetId(), key);
      return v.IsHolding<float>() ? v.UncheckedGet<float>() : fallback;
    };
    auto matrix = d->GetTransform(GetId());
    GfVec3d position = matrix.Transform(GfVec3d(0));
    GfVec3d direction = matrix.TransformDir(GfVec3d(0, 0, -1)).GetNormalized();
    GfVec3f color(1);
    auto c = d->GetLightParamValue(GetId(), HdLightTokens->color);
    if (c.IsHolding<GfVec3f>())
      color = c.UncheckedGet<GfVec3f>();
    float intensity = get(HdLightTokens->intensity, 1) *
                      std::exp2(get(HdLightTokens->exposure, 0));
    std::ostringstream o;
    o << std::setprecision(9) << "{\"color\":" << Vec(GfVec3d(color))
      << ",\"intensity\":" << intensity;
    float cone = get(HdLightTokens->shapingConeAngle, 180);
    if (type == HdPrimTypeTokens->distantLight)
      o << ",\"type\":\"directional\",\"direction\":" << Vec(direction);
    else if (type == HdPrimTypeTokens->rectLight) {
      o << ",\"type\":\"area\",\"position\":" << Vec(position) << ",\"u\":"
        << Vec(matrix.TransformDir(GfVec3d(get(HdLightTokens->width, 1), 0, 0)))
        << ",\"v\":"
        << Vec(matrix.TransformDir(
               GfVec3d(0, -get(HdLightTokens->height, 1), 0)))
        << ",\"single_sided\":"
        << (BoolLightParameter(d, GetId(), "lava:singleSided", true) ? "true"
                                                                     : "false")
        << ",\"render_light_geometry\":"
        << (BoolLightParameter(d, GetId(), "lava:renderLightGeometry", false)
                ? "true"
                : "false");
    } else if (cone > 0 && cone < 90)
      o << ",\"type\":\"spot\",\"position\":" << Vec(position)
        << ",\"direction\":" << Vec(direction) << ",\"cone_degrees\":" << cone
        << ",\"softness\":" << get(HdLightTokens->shapingConeSoftness, 0);
    else
      o << ",\"type\":\"point\",\"position\":" << Vec(position);
    o << '}';
    auto *state = static_cast<State *>(param);
    std::lock_guard<std::mutex> lock(state->mutex);
    state->lights[GetId()] = o.str();
    ++state->generation;
    *bits = Clean;
  }
};
// Constant native OpenPBR/MaterialX OpenPBR terminals and PreviewSurface
// lowering.
class Material final : public HdMaterial {
public:
  Material(SdfPath const &id) : HdMaterial(id) {}
  HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
  void Sync(HdSceneDelegate *d, HdRenderParam *param,
            HdDirtyBits *bits) override {
    auto value = d->GetMaterialResource(GetId());
    if (std::getenv("LAVA_TRACE"))
      std::fprintf(stderr, "Lava material %s: %s\n", GetId().GetText(),
                   value.GetTypeName().c_str());
    float metallic = 0, transmission = 0, ior = 1.5f;
    GfVec3f color(.5f);
    bool hasColor = false;
    std::string openpbr;
    std::string textures;
    if (value.IsHolding<HdMaterialNetworkMap>()) {
      for (auto const &network : value.UncheckedGet<HdMaterialNetworkMap>().map)
        for (auto const &node : network.second.nodes) {
          if (std::getenv("LAVA_TRACE"))
            std::fprintf(stderr, "Lava material node: %s (%zu inputs)\n",
                         node.identifier.GetText(), node.parameters.size());
          if (node.identifier == TfToken("UsdPreviewSurface")) {
            UsdTimeCode materialTime;
            auto source = lava::PrincipledSource(d, GetId(), &materialTime);
            auto textureNetwork = lava::PrincipledNetwork(network.second, node,
                                                          source, materialTime);
            textures = lava::TextureInputs(textureNetwork, node, true);
            if (std::getenv("LAVA_TRACE"))
              std::fprintf(stderr, "Lava textures (%s): {%s}\n",
                           source ? "Principled" : "USD", textures.c_str());
            auto scalar = [&](const char *name, float fallback) {
              if (source) {
                const char *authored =
                    std::string(name) == "roughness" ? "rough" : name;
                auto value =
                    lava::PrincipledValue(source, authored, materialTime);
                if (value.IsHolding<float>())
                  return value.UncheckedGet<float>();
              }
              auto p = node.parameters.find(TfToken(name));
              return p != node.parameters.end() && p->second.IsHolding<float>()
                         ? p->second.UncheckedGet<float>()
                         : fallback;
            };
            metallic = scalar("metallic", 0);
            transmission = scalar("transmission", 1 - scalar("opacity", 1));
            ior = scalar("ior", 1.5f);
            std::ostringstream preview;
            preview << std::setprecision(9)
                    << "{\"base_metalness\":" << std::clamp(metallic, 0.f, 1.f)
                    << ",\"transmission_weight\":"
                    << std::clamp(scalar("transmission", 0), 0.f, 1.f)
                    << ",\"geometry_opacity\":"
                    << std::clamp(scalar("opacity", 1), 0.f, 1.f)
                    << ",\"specular_roughness\":"
                    << std::clamp(scalar("roughness", .5f), 0.f, 1.f)
                    << ",\"specular_ior\":" << std::clamp(ior, 1.f, 4.f);
            auto base = node.parameters.find(TfToken("diffuseColor"));
            GfVec3f previewBase(.18f);
            if (base != node.parameters.end() &&
                base->second.IsHolding<GfVec3f>())
              previewBase = base->second.UncheckedGet<GfVec3f>();
            if (source) {
              auto authored =
                  lava::PrincipledValue(source, "basecolor", materialTime);
              if (authored.IsHolding<GfVec3f>())
                previewBase = authored.UncheckedGet<GfVec3f>();
            }
            preview << ",\"base_color\":" << Vec(GfVec3d(previewBase));
            auto emissive = node.parameters.find(TfToken("emissiveColor"));
            if (emissive != node.parameters.end() &&
                emissive->second.IsHolding<GfVec3f>())
              preview << ",\"emission_luminance\":1,\"emission_color\":"
                      << Vec(GfVec3d(emissive->second.UncheckedGet<GfVec3f>()));
            if ((emissive == node.parameters.end() ||
                 !emissive->second.IsHolding<GfVec3f>()) &&
                lava::HasInputConnection(network.second, node.path,
                                         TfToken("emissiveColor")))
              preview << ",\"emission_luminance\":1";
            preview << "}";
            openpbr = preview.str();
            auto p = node.parameters.find(TfToken("diffuseColor"));
            if (p != node.parameters.end() && p->second.IsHolding<GfVec3f>()) {
              color = p->second.UncheckedGet<GfVec3f>();
              hasColor = true;
            }
          }
          if (node.identifier == TfToken("ND_open_pbr_surface_surfaceshader") ||
              node.identifier == TfToken("open_pbr_surface") ||
              node.identifier == TfToken("OpenPBRSurface")) {
            textures = lava::TextureInputs(network.second, node, false);
            std::ostringstream material;
            material << std::setprecision(9) << "{";
            bool first = true;
            auto input = [&](const char *name) -> VtValue {
              auto it = node.parameters.find(TfToken(name));
              return it == node.parameters.end() ? VtValue() : it->second;
            };
            auto key = [&](const char *name) {
              if (!first)
                material << ',';
              first = false;
              material << '\"' << name << "\":";
            };
#define LAVA_FLOAT(name)                                                       \
  {                                                                            \
    auto v = input(#name);                                                     \
    if (v.IsHolding<float>()) {                                                \
      key(#name);                                                              \
      material << v.UncheckedGet<float>();                                     \
    }                                                                          \
  }
#define LAVA_COLOR(name)                                                       \
  {                                                                            \
    auto v = input(#name);                                                     \
    if (v.IsHolding<GfVec3f>()) {                                              \
      key(#name);                                                              \
      material << Vec(GfVec3d(v.UncheckedGet<GfVec3f>()));                     \
    }                                                                          \
  }
#define LAVA_BOOL(name)                                                        \
  {                                                                            \
    auto v = input(#name);                                                     \
    if (v.IsHolding<bool>()) {                                                 \
      key(#name);                                                              \
      material << (v.UncheckedGet<bool>() ? "true" : "false");                 \
    }                                                                          \
  }
#include "material_fields.inc"
#undef LAVA_FLOAT
#undef LAVA_COLOR
#undef LAVA_BOOL
            material << "}";
            openpbr = material.str();
          }
        }
    }
    std::ostringstream out;
    out << std::setprecision(9)
        << ",\"material\":{\"metallic\":" << std::clamp(metallic, 0.f, 1.f)
        << ",\"transmission\":" << std::clamp(transmission, 0.f, 1.f)
        << ",\"ior\":" << std::clamp(ior, 1.f, 4.f);
    if (!openpbr.empty())
      out << ",\"openpbr\":" << openpbr;
    if (!textures.empty())
      out << ",\"textures\":{" << textures << "}";
    out << "}";
    if (hasColor)
      out << ",\"color\":" << Vec(GfVec3d(color));
    auto *state = static_cast<State *>(param);
    std::lock_guard<std::mutex> lock(state->mutex);
    auto json = out.str();
    if (state->materials[GetId()] != json) {
      state->materials[GetId()] = json;
      ++state->generation;
      ++state->geometryGeneration;
    }
    *bits = Clean;
  }
};
struct RenderSession {
  LavaSession *raw = lava_create();
  mutable std::mutex mutex;
  uint64_t revision = 0;
  uint64_t reportedError = 0;
  unsigned target = 0;
  bool glFailed = false;
  std::unique_ptr<LavaGlInterop> gl;
  RenderSession() { lava_enable_interop(raw, true); }
  ~RenderSession() {
    LavaGlInterop::Retire(gl.release());
    lava_destroy(raw);
  }
  int64_t status() {
    auto result = lava_status(raw, revision);
    if (result < 0 && reportedError != revision) {
      reportedError = revision;
      TF_WARN("Lava: %s", lava_error(raw));
    }
    return result;
  }
};
class Buffer final : public HdRenderBuffer {
  GfVec3i dims{0};
  HdFormat format = HdFormatInvalid;
  std::vector<float> pixels;
  mutable std::mutex pixelsMutex;
  std::atomic<unsigned> mapped{0};
  std::shared_ptr<RenderSession> session;
  uint64_t revision = 0;
  bool depth = false;
  mutable std::unique_ptr<LavaGlInterop> placeholder;
  void readback() {
    if (!session || IsMapped())
      return;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->revision != revision)
      return;
    std::vector<float> incoming(pixels.size());
    auto done = depth ? lava_poll_depth(session->raw, revision, incoming.data(),
                                        incoming.size())
                      : lava_poll(session->raw, revision, incoming.data(),
                                  incoming.size());
    auto channels = depth ? 1u : 4u;
    if (done > 0)
      for (unsigned y = 0; y < GetHeight(); ++y)
        std::copy_n(incoming.data() + size_t(y) * GetWidth() * channels,
                    size_t(GetWidth()) * channels,
                    pixels.data() +
                        size_t(GetHeight() - 1 - y) * GetWidth() * channels);
  }

public:
  explicit Buffer(SdfPath const &id) : HdRenderBuffer(id) {}
  ~Buffer() override { LavaGlInterop::Retire(placeholder.release()); }
  bool Allocate(GfVec3i const &d, HdFormat f, bool) override {
    std::lock_guard<std::mutex> lock(pixelsMutex);
    if (d[0] <= 0 || d[1] <= 0 || d[2] != 1 ||
        (f != HdFormatFloat32Vec4 && f != HdFormatFloat32) || IsMapped())
      return false;
    dims = d;
    format = f;
    depth = f == HdFormatFloat32;
    pixels.assign(size_t(d[0]) * d[1] * (depth ? 1 : 4), depth ? 1e30f : 0.f);
    session.reset();
    return true;
  }
  unsigned GetWidth() const override { return dims[0]; }
  unsigned GetHeight() const override { return dims[1]; }
  unsigned GetDepth() const override { return dims[2]; }
  HdFormat GetFormat() const override { return format; }
  bool IsMultiSampled() const override { return false; }
  void *Map() override {
    std::lock_guard<std::mutex> lock(pixelsMutex);
    readback();
    ++mapped;
    return pixels.data();
  }
  void Unmap() override {
    std::lock_guard<std::mutex> lock(pixelsMutex);
    --mapped;
  }
  bool IsMapped() const override { return mapped.load() != 0; }
  // Resolve is deliberately metadata-only: Solaris next asks for GetResource;
  // reading pixels here would force a GPU -> CPU -> GPU roundtrip.
  void Resolve() override {}
  bool IsConverged() const override {
    std::lock_guard<std::mutex> lock(pixelsMutex);
    if (!session)
      return false;
    std::lock_guard<std::mutex> l(session->mutex);
    return session->revision == revision &&
           session->status() >= int64_t(session->target);
  }
  void Bind(std::shared_ptr<RenderSession> const &s, uint64_t r) {
    std::lock_guard<std::mutex> l(pixelsMutex);
    session = s;
    revision = r;
  }
  VtValue GetResource(bool) const override {
    std::lock_guard<std::mutex> lock(pixelsMutex);
    if (!LavaGlInterop::HasContext() || !GetWidth() || !GetHeight())
      return VtValue();
    if (!session) {
      try {
        if (!placeholder)
          placeholder = std::make_unique<LavaGlInterop>();
        return VtValue(
            placeholder->Get(nullptr, 0, depth, GetWidth(), GetHeight()));
      } catch (std::exception const &) {
        return VtValue();
      }
    }
    std::lock_guard<std::mutex> l(session->mutex);
    if (session->revision != revision || session->glFailed ||
        lava_frame_transport(session->raw, revision) == 1)
      return VtValue();
    try {
      if (!session->gl)
        session->gl = std::make_unique<LavaGlInterop>();
      auto texture = session->gl->Get(session->raw, revision, depth, GetWidth(),
                                      GetHeight());
      if (texture)
        return VtValue(texture);
    } catch (std::exception const &e) {
      session->glFailed = true;
      std::printf("Lava [WARNING]: Vulkan/OpenGL presentation failed; CPU AOV "
                  "fallback requires image transfer through system RAM: %s\n",
                  e.what());
      std::fflush(stdout);
      TF_WARN("Lava GPU interop unavailable; CPU AOV fallback: %s", e.what());
    }
    return VtValue();
  }

protected:
  void _Deallocate() override {
    std::lock_guard<std::mutex> l(pixelsMutex);
    pixels.clear();
    LavaGlInterop::Retire(placeholder.release());
    dims = GfVec3i(0);
    session.reset();
  }
};
class Delegate;
class Pass final : public HdRenderPass {
  Delegate *delegate;
  std::shared_ptr<RenderSession> session = std::make_shared<RenderSession>();
  std::string previous;
  uint64_t sceneGeneration = ~uint64_t(0);
  std::string sceneMode;
  uint64_t revision = 0;
  bool traced = false;
  std::atomic<bool> converged{false};

public:
  Pass(HdRenderIndex *index, HdRprimCollection const &collection, Delegate *d)
      : HdRenderPass(index, collection), delegate(d) {}
  ~Pass() override = default;
  bool IsConverged() const override {
    std::lock_guard<std::mutex> l(session->mutex);
    return session->revision && session->status() >= int64_t(session->target);
  }

protected:
  void _Execute(HdRenderPassStateSharedPtr const &,
                TfTokenVector const &) override;
};
class Delegate final : public HdRenderDelegate {
  State state;
  HdResourceRegistrySharedPtr registry = std::make_shared<HdResourceRegistry>();

public:
  explicit Delegate(HdRenderSettingsMap const &settings = {})
      : HdRenderDelegate(settings) {
    static std::once_flag materialDefinitions;
    std::call_once(materialDefinitions, [] {
      auto plugin = PlugRegistry::GetInstance().GetPluginWithName("hdLava");
      if (!plugin) {
        TF_WARN("Lava: unable to locate native OpenPBR shader definitions");
        return;
      }
      auto path = plugin->GetResourcePath() + "/openpbr_defs.usda";
      auto stage = UsdStage::Open(path);
      if (!stage) {
        TF_WARN("Lava: unable to open %s", path.c_str());
        return;
      }
      for (auto const &prim : stage->GetPseudoRoot().GetChildren()) {
        auto shader = UsdShadeShader(prim);
        if (shader)
          for (auto const &result :
               UsdShadeShaderDefUtils::GetNodeDiscoveryResults(shader, path))
            SdrRegistry::GetInstance().AddDiscoveryResult(result);
      }
    });
  }
  TfTokenVector GetShaderSourceTypes() const override {
    return {TfToken("glslfx"), TfToken("mtlx")};
  }
  TfTokenVector GetMaterialRenderContexts() const override {
    return {TfToken("mtlx"), TfToken()};
  }
  State &GetState() { return state; }
  HdRenderParam *GetRenderParam() const override {
    return const_cast<State *>(&state);
  }
  HdResourceRegistrySharedPtr GetResourceRegistry() const override {
    return registry;
  }
  TfTokenVector const &GetSupportedRprimTypes() const override {
    static TfTokenVector v = {HdPrimTypeTokens->mesh};
    return v;
  }
  TfTokenVector const &GetSupportedSprimTypes() const override {
    static TfTokenVector v = {
        HdPrimTypeTokens->camera, HdPrimTypeTokens->sphereLight,
        HdPrimTypeTokens->rectLight, HdPrimTypeTokens->distantLight,
        HdPrimTypeTokens->material};
    return v;
  }
  TfTokenVector const &GetSupportedBprimTypes() const override {
    static TfTokenVector v = {HdPrimTypeTokens->renderBuffer};
    return v;
  }
  HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex *i,
                                         HdRprimCollection const &c) override {
    return std::make_shared<Pass>(i, c, this);
  }
  HdInstancer *CreateInstancer(HdSceneDelegate *, SdfPath const &) override {
    TF_WARN("Lava prototype does not support instancers");
    return nullptr;
  }
  void DestroyInstancer(HdInstancer *i) override { delete i; }
  HdRprim *CreateRprim(TfToken const &t, SdfPath const &id) override {
    return t == HdPrimTypeTokens->mesh ? new Mesh(id) : nullptr;
  }
  void DestroyRprim(HdRprim *p) override {
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.meshes.erase(p->GetId());
      ++state.generation;
      ++state.geometryGeneration;
    }
    delete p;
  }
  HdSprim *CreateSprim(TfToken const &t, SdfPath const &id) override {
    if (t == HdPrimTypeTokens->camera)
      return new HdCamera(id);
    if (t == HdPrimTypeTokens->material)
      return new Material(id);
    auto const &types = GetSupportedSprimTypes();
    return std::find(types.begin(), types.end(), t) != types.end()
               ? new Light(id, t)
               : nullptr;
  }
  HdSprim *CreateFallbackSprim(TfToken const &t) override {
    return CreateSprim(t, SdfPath());
  }
  void DestroySprim(HdSprim *p) override {
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (state.materials.erase(p->GetId()))
        ++state.geometryGeneration;
      state.lights.erase(p->GetId());
      ++state.generation;
    }
    delete p;
  }
  HdBprim *CreateBprim(TfToken const &t, SdfPath const &id) override {
    return t == HdPrimTypeTokens->renderBuffer ? new Buffer(id) : nullptr;
  }
  HdBprim *CreateFallbackBprim(TfToken const &t) override {
    return CreateBprim(t, SdfPath());
  }
  void DestroyBprim(HdBprim *p) override { delete p; }
  void CommitResources(HdChangeTracker *) override {}
  HdAovDescriptor GetDefaultAovDescriptor(TfToken const &name) const override {
    if (name == HdAovTokens->color)
      return HdAovDescriptor(HdFormatFloat32Vec4, false, VtValue(GfVec4f(0)));
    if (name == HdAovTokens->depth)
      return HdAovDescriptor(HdFormatFloat32, false, VtValue(1e30f));
    return HdAovDescriptor();
  }
  HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override {
    return {{"Visibility generator (ray_query or raster)",
             TfToken("lava:visibility"), VtValue(std::string("ray_query"))},
            {"Rendering path", TfToken("lava:mode"),
             VtValue(std::string("interactive"))},
            {"Samples", TfToken("lava:samples"), VtValue(64)},
            {"Exposure", TfToken("lava:exposure"), VtValue(0.f)},
            {"Texture mode", TfToken("lava:textures"),
             VtValue(std::string("original"))},
            {"Maximum diffuse depth", TfToken("lava:diffuseDepth"), VtValue(3)},
            {"Maximum reflection depth", TfToken("lava:reflectionDepth"),
             VtValue(4)},
            {"Maximum refraction depth", TfToken("lava:refractionDepth"),
             VtValue(8)},
            {"Russian roulette start depth", TfToken("lava:rouletteStart"),
             VtValue(3)},
            {"Automatic viewport headlight", TfToken("lava:headlight"),
             VtValue(true)}};
  }
};
void Pass::_Execute(HdRenderPassStateSharedPtr const &passState,
                    TfTokenVector const &) {
  bool trace = std::getenv("LAVA_TRACE") && !traced;
  auto camera = passState->GetCamera();
  if (trace)
    std::fprintf(stderr, "Lava Execute: camera=%p, AOVs=%zu\n",
                 static_cast<void const *>(camera),
                 passState->GetAovBindings().size());
  if (!camera) {
    traced = true;
    return;
  }
  Buffer *output = nullptr;
  Buffer *depthOutput = nullptr;
  for (auto const &a : passState->GetAovBindings())
    if (a.aovName == HdAovTokens->color || a.aovName == HdAovTokens->depth) {
      auto *buffer = a.renderBuffer;
      if (!buffer && !a.renderBufferId.IsEmpty())
        buffer = dynamic_cast<HdRenderBuffer *>(GetRenderIndex()->GetBprim(
            HdPrimTypeTokens->renderBuffer, a.renderBufferId));
      if (a.aovName == HdAovTokens->color)
        output = dynamic_cast<Buffer *>(buffer);
      else
        depthOutput = dynamic_cast<Buffer *>(buffer);
    }
  if (trace)
    std::fprintf(stderr, "Lava color buffer=%p, size=%ux%u\n",
                 static_cast<void *>(output), output ? output->GetWidth() : 0,
                 output ? output->GetHeight() : 0);
  traced = true;
  if (!output || !output->GetWidth() || !output->GetHeight())
    return;
  auto &state = delegate->GetState();
  std::ostringstream o;
  auto depth = [&](const char *name, int fallback, int maximum) {
    return std::clamp(delegate->GetRenderSetting(TfToken(name), fallback), 0,
                      maximum);
  };
  o << std::setprecision(9) << "{\"settings\":{\"visibility\":\""
    << delegate->GetRenderSetting(TfToken("lava:visibility"),
                                  std::string("ray_query"))
    << "\",\"textures\":\""
    << delegate->GetRenderSetting(TfToken("lava:textures"),
                                  std::string("original"))
    << "\",\"diffuse_depth\":" << depth("lava:diffuseDepth", 3, 32)
    << ",\"reflection_depth\":" << depth("lava:reflectionDepth", 4, 32)
    << ",\"refraction_depth\":" << depth("lava:refractionDepth", 8, 32)
    << ",\"roulette_start\":" << depth("lava:rouletteStart", 3, 96)
    << "},\"camera\":{\"transform\":[";
  // Gf uses row vectors; copying row-major entries yields the transposed matrix
  // expected by Rust/GLSL column-vector convention.
  auto matrix = camera->GetTransform();
  for (int row = 0; row < 4; row++)
    for (int col = 0; col < 4; col++) {
      if (row || col)
        o << ',';
      o << matrix[row][col];
    }
  o << "],\"focal_length\":" << camera->GetFocalLength()
    << ",\"horizontal_aperture\":" << camera->GetHorizontalAperture()
    << ",\"vertical_aperture\":" << camera->GetVerticalAperture()
    << ",\"aperture_offset\":[" << camera->GetHorizontalApertureOffset() << ','
    << camera->GetVerticalApertureOffset()
    << "],\"near\":" << camera->GetClippingRange().GetMin()
    << ",\"far\":" << camera->GetClippingRange().GetMax()
    << ",\"f_stop\":" << camera->GetFStop()
    << ",\"focus_distance\":" << std::max(0.001f, camera->GetFocusDistance())
    << ",\"orthographic\":"
    << (camera->GetProjection() == HdCamera::Orthographic ? "true" : "false")
    << "},\"meshes\":[";
  uint64_t generation, geometryGeneration;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    generation = state.generation;
    geometryGeneration = state.geometryGeneration;
  }
  auto cameraKey = o.str();
  auto mode = delegate->GetRenderSetting(TfToken("lava:mode"),
                                         std::string("interactive"));
  int samples =
      std::max(1, delegate->GetRenderSetting(TfToken("lava:samples"), 64));
  float exposure = delegate->GetRenderSetting(TfToken("lava:exposure"), 0.f);
  bool headlight = delegate->GetRenderSetting(TfToken("lava:headlight"), true);
  auto key = std::to_string(headlight) + cameraKey +
             std::to_string(generation) + mode + std::to_string(samples) + ":" +
             std::to_string(exposure) + ":" +
             std::to_string(output->GetWidth()) + ":" +
             std::to_string(output->GetHeight());
  if (key != previous) {
    const bool viewOnly = revision && sceneGeneration == geometryGeneration &&
                          mode == "interactive" && sceneMode == mode;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      bool first = true;
      if (!viewOnly)
        for (auto const &p : state.meshes) {
          if (!first)
            o << ',';
          first = false;
          auto material = state.materials.find(state.meshMaterials[p.first]);
          if (material == state.materials.end())
            o << p.second;
          else {
            // Replace the trailing color field rather than emit duplicate JSON
            // keys.
            auto color = p.second.rfind(",\"color\":");
            auto overrideColor = material->second.find(",\"color\":");
            if (overrideColor == std::string::npos)
              o << p.second.substr(0, p.second.size() - 1) << material->second
                << '}';
            else
              o << p.second.substr(0, color) << material->second << '}';
          }
        }
      o << "],\"lights\":[";
      first = true;
      if (state.lights.empty() && mode == "interactive" && headlight) {
        auto direction = matrix.TransformDir(GfVec3d(0, 0, -1)).GetNormalized();
        o << "{\"type\":\"directional\",\"direction\":" << Vec(direction)
          << ",\"color\":[1,1,1],\"intensity\":3.14159265}";
      }
      for (auto const &p : state.lights) {
        if (!first)
          o << ',';
        first = false;
        o << p.second;
      }
    }
    o << "]}";
    auto json = o.str();
    if (std::getenv("LAVA_TRACE"))
      std::fprintf(
          stderr,
          "Lava submit: generation=%llu, bytes=%zu, samples=%d, view=%d\n",
          static_cast<unsigned long long>(generation), json.size(), samples,
          viewOnly);
    {
      std::lock_guard<std::mutex> l(session->mutex);
      revision =
          viewOnly
              ? lava_update_view(session->raw, json.c_str(), output->GetWidth(),
                                 output->GetHeight(), samples, exposure)
              : lava_submit(session->raw, json.c_str(), output->GetWidth(),
                            output->GetHeight(), mode == "final", samples,
                            exposure);
      session->revision = revision;
      session->target = samples;
      if (!revision)
        TF_WARN("Lava: %s", lava_error(session->raw));
      else {
        sceneGeneration = geometryGeneration;
        sceneMode = mode;
      }
    }
    previous = key;
  }
  output->Bind(session, revision);
  if (depthOutput)
    depthOutput->Bind(session, revision);
  {
    std::lock_guard<std::mutex> l(session->mutex);
    session->status();
  }
}
} // namespace
class LavaRendererPlugin final : public HdRendererPlugin {
public:
  HdRenderDelegate *CreateRenderDelegate() override { return new Delegate; }
  HdRenderDelegate *
  CreateRenderDelegate(HdRenderSettingsMap const &s) override {
    return new Delegate(s);
  }
  void DeleteRenderDelegate(HdRenderDelegate *d) override { delete d; }
  bool IsSupported(bool gpuEnabled = true) const override {
    return gpuEnabled && lava_is_supported();
  }
};
TF_REGISTRY_FUNCTION(TfType) {
  HdRendererPluginRegistry::Define<LavaRendererPlugin>();
}
PXR_NAMESPACE_CLOSE_SCOPE
