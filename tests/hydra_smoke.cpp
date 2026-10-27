#include "lava.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/vt/array.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/mesh.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/pxr.h>
#include <thread>
PXR_NAMESPACE_USING_DIRECTIVE
class Scene final : public HdSceneDelegate {
public:
  explicit Scene(HdRenderIndex *i) : HdSceneDelegate(i, SdfPath("/test")) {}
  VtValue Get(SdfPath const &, TfToken const &key) override {
    if (key == HdTokens->points)
      return VtValue(VtVec3fArray{GfVec3f(-1, -1, 0), GfVec3f(1, -1, 0),
                                  GfVec3f(0, 1, 0)});
    if (key == HdTokens->displayColor)
      return VtValue(VtVec3fArray{GfVec3f(.8, .2, .1)});
    return VtValue();
  }
  bool GetVisible(SdfPath const &) override { return true; }
  GfMatrix4d GetTransform(SdfPath const &id) override {
    GfMatrix4d m(1);
    if (id == SdfPath("/camera"))
      m.SetTranslate(GfVec3d(0, 0, 5));
    if (id == SdfPath("/light"))
      m.SetTranslate(GfVec3d(0, 2, 3));
    return m;
  }
  HdMeshTopology GetMeshTopology(SdfPath const &) override {
    return HdMeshTopology(TfToken("none"), HdTokens->rightHanded, VtIntArray{3},
                          VtIntArray{0, 1, 2});
  }
  VtValue GetCameraParamValue(SdfPath const &, TfToken const &key) override {
    if (key == HdCameraTokens->projection)
      return VtValue(HdCamera::Perspective);
    if (key == HdCameraTokens->horizontalAperture)
      return VtValue(3.6f);
    if (key == HdCameraTokens->verticalAperture)
      return VtValue(2.7f);
    if (key == HdCameraTokens->focalLength)
      return VtValue(5.f);
    if (key == HdCameraTokens->clippingRange)
      return VtValue(GfRange1f(.01, 1000));
    return VtValue();
  }
  VtValue GetLightParamValue(SdfPath const &, TfToken const &key) override {
    if (key == HdLightTokens->intensity)
      return VtValue(30.f);
    if (key == HdLightTokens->color)
      return VtValue(GfVec3f(1));
    return VtValue();
  }
};
int main() {
  auto &registry = HdRendererPluginRegistry::GetInstance();
  auto *plugin = registry.GetRendererPlugin(TfToken("LavaRendererPlugin"));
  if (!plugin) {
    std::cerr << "plugin missing\n";
    return 1;
  }
  auto *delegate = plugin->CreateRenderDelegate();
  delegate->SetRenderSetting(TfToken("lava:samples"), VtValue(3));
  auto *index = HdRenderIndex::New(delegate, {});
  int result = 0;
  {
    Scene scene(index);
    auto *param = delegate->GetRenderParam();
    auto *mesh =
        delegate->CreateRprim(HdPrimTypeTokens->mesh, SdfPath("/mesh"));
    auto *camera =
        delegate->CreateSprim(HdPrimTypeTokens->camera, SdfPath("/camera"));
    auto *light =
        delegate->CreateSprim(HdPrimTypeTokens->sphereLight, SdfPath("/light"));
    auto bits = mesh->GetInitialDirtyBitsMask();
    mesh->Sync(&scene, param, &bits, HdReprTokens->hull);
    bits = camera->GetInitialDirtyBitsMask();
    camera->Sync(&scene, param, &bits);
    bits = light->GetInitialDirtyBitsMask();
    light->Sync(&scene, param, &bits);
    SdfPath bufferPath("/test/color");
    index->InsertBprim(HdPrimTypeTokens->renderBuffer, &scene, bufferPath);
    auto *buffer = static_cast<HdRenderBuffer *>(
        index->GetBprim(HdPrimTypeTokens->renderBuffer, bufferPath));
    if (!buffer->Allocate(GfVec3i(64, 48, 1), HdFormatFloat32Vec4, false))
      return 2;
    auto state = delegate->CreateRenderPassState();
    state->SetCamera(static_cast<HdCamera *>(camera));
    HdRenderPassAovBinding binding;
    binding.aovName = HdAovTokens->color;
    binding.renderBufferId = bufferPath;
    state->SetAovBindings({binding});
    auto pass = delegate->CreateRenderPass(
        index, HdRprimCollection(HdTokens->geometry,
                                 HdReprSelector(HdReprTokens->hull)));
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    do {
      pass->Execute(state, {});
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (!pass->IsConverged() &&
             std::chrono::steady_clock::now() < deadline);
    auto *pixels = static_cast<float *>(buffer->Map());
    float minimum = 1e30f, maximum = -1e30f;
    for (size_t i = 0; i < 64 * 48 * 4; i += 4) {
      minimum = std::min(minimum, pixels[i]);
      maximum = std::max(maximum, pixels[i]);
    }
    buffer->Unmap();
    if (!buffer->IsConverged() || maximum - minimum < .01f ||
        !std::isfinite(maximum)) {
      std::cerr << "Hydra image invalid: " << minimum << ".." << maximum
                << "\n";
      result = 3;
    } else
      std::cout << "PASS: Hydra plugin mesh/camera/light sync -> Rust worker "
                   "-> linear color AOV\n";
    pass.reset();
    index->RemoveBprim(HdPrimTypeTokens->renderBuffer, bufferPath);
    delegate->DestroySprim(light);
    delegate->DestroySprim(camera);
    delegate->DestroyRprim(mesh);
  }
  delete index;
  plugin->DeleteRenderDelegate(delegate);
  registry.ReleasePlugin(plugin);
  return result;
}
