// Exercise normal USD population and Hdx task-driven sync, rather than manual
// prim Sync.
#include "../hydra/meshNormals.h"
#include "gl_context.h"
#include "lava.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <pxr/base/plug/registry.h>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdx/taskController.h>
#include <pxr/imaging/hgi/texture.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/bboxCache.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usdImaging/usdImaging/delegate.h>
#include <thread>
PXR_NAMESPACE_USING_DIRECTIVE
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: lava_usd_smoke scene.usd [cameraPath] [output.ppm]\n";
    return 1;
  }
  std::unique_ptr<LavaTestGlContext> glWindow;
  bool testInterop = std::getenv("LAVA_TEST_GL_INTEROP");
  if (testInterop)
    glWindow = std::make_unique<LavaTestGlContext>();
  PlugRegistry::GetInstance().RegisterPlugins(LAVA_HOUDINI_PLUGIN_PATH);
  std::cerr << "Opening stage\n";
  auto stage = UsdStage::Open(argv[1]);
  if (!stage)
    return 2;
  stage->SetEditTarget(stage->GetSessionLayer());
  SdfPath cameraPath(argc > 2 ? argv[2] : "/__testCamera");
  if (argc <= 2) {
    UsdGeomBBoxCache bounds(UsdTimeCode::Default(),
                            {UsdGeomTokens->default_, UsdGeomTokens->render},
                            true);
    auto range =
        bounds.ComputeWorldBound(stage->GetPseudoRoot()).ComputeAlignedRange();
    auto center = range.GetMidpoint();
    double extent = range.GetSize().GetLength();
    auto camera = UsdGeomCamera::Define(stage, cameraPath);
    auto gf = camera.GetCamera(UsdTimeCode::Default());
    GfMatrix4d view(1);
    view.SetLookAt(center + GfVec3d(extent * .4, extent * .25, extent * .6),
                   center, GfVec3d(0, 1, 0));
    gf.SetTransform(view.GetInverse());
    gf.SetClippingRange(GfRange1f(.01f, float(extent * 10)));
    camera.SetFromCamera(gf, UsdTimeCode::Default());
  }
  auto &registry = HdRendererPluginRegistry::GetInstance();
  auto *plugin = registry.GetRendererPlugin(TfToken("LavaRendererPlugin"));
  if (!plugin)
    return 3;
  auto *renderer = plugin->CreateRenderDelegate();
  renderer->SetRenderSetting(TfToken("lava:samples"), VtValue(4));
  if (const char *mode = std::getenv("LAVA_TEST_TEXTURE_MODE"))
    renderer->SetRenderSetting(TfToken("lava:textures"),
                               VtValue(std::string(mode)));
  auto *index = HdRenderIndex::New(renderer, {});
  int result = 0;
  {
    std::cerr << "Populating USD delegate\n";
    UsdImagingDelegate scene(index, SdfPath("/usd"));
    scene.SetSceneMaterialsEnabled(true);
    scene.Populate(stage->GetPseudoRoot());
    scene.SetTime(UsdTimeCode::Default());
    std::cerr << "Creating task controller: " << index->GetRprimIds().size()
              << " rprims\n";
    HdxTaskController controller(index, SdfPath("/tasks"), false);
    controller.SetEnablePresentation(false);
    controller.SetEnableSelection(false);
    controller.SetViewportRenderOutput(TfToken());
    controller.SetCollection(HdRprimCollection(
        HdTokens->geometry, HdReprSelector(HdReprTokens->hull)));
    controller.SetRenderTags({HdRenderTagTokens->geometry});
    controller.SetRenderOutputs({HdAovTokens->color, HdAovTokens->depth});
    controller.SetRenderBufferSize(GfVec2i(320, 180));
    controller.SetRenderViewport(GfVec4d(0, 0, 320, 180));
    auto indexedCamera = scene.ConvertCachePathToIndexPath(cameraPath);
    std::cerr << "Requested camera " << indexedCamera << ", available:";
    for (auto const &id : index->GetSprimSubtree(HdPrimTypeTokens->camera,
                                                 SdfPath::AbsoluteRootPath()))
      std::cerr << " " << id;
    std::cerr << "\n";
    controller.SetCameraPath(indexedCamera);
    HdEngine engine;
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(120);
    bool first = true;
    do {
      auto tasks = controller.GetRenderingTasks();
      if (first)
        std::cerr << "Executing " << tasks.size() << " tasks\n";
      engine.Execute(index, &tasks);
      if (first) {
        std::cerr << "First Execute completed\n";
        first = false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (!controller.IsConverged() &&
             std::chrono::steady_clock::now() < deadline);
    if (std::getenv("LAVA_TEST_NORMALS")) {
      unsigned imported = 0;
      for (auto const &id : index->GetRprimIds()) {
        if (id.GetName() == "missing")
          continue;
        auto topology = scene.GetMeshTopology(id);
        auto importedNormals = lava::ReadMeshNormals(&scene, id, topology);
        auto const &indices = importedNormals.indices;
        auto const &value = importedNormals.value;
        if (importedNormals.interpolation == HdInterpolationCount) {
          std::cerr << "Authored normal interpolation missing: " << id << "\n";
          result = 14;
        }
        if (!value.IsHolding<VtVec3fArray>() ||
            value.UncheckedGet<VtVec3fArray>().empty()) {
          std::cerr << "Authored normals missing from Hydra: " << id << "\n";
          result = 14;
        } else {
          ++imported;
          auto const &normals = value.UncheckedGet<VtVec3fArray>();
          if (id.GetName() == "primvar" && normals[0] != GfVec3f(8, 0, 1)) {
            std::cerr << "primvars:normals precedence failed\n";
            result = 14;
          }
          std::cout << "Normals " << id << ": " << normals.size() << " values, "
                    << indices.size() << " indices\n";
        }
      }
      if (imported != 9)
        result = 14;
      if (result == 0)
        std::cout << "PASS: authored normals reached native Hydra for all "
                     "interpolation rates\n";
    }
    if (std::getenv("LAVA_TEST_LIGHT_PROPERTIES")) {
      auto meshId = scene.ConvertCachePathToIndexPath(SdfPath("/lights/mesh"));
      auto meshFlag = scene.Get(meshId, TfToken("lava:geometryLight"));
      if (!meshFlag.IsHolding<bool>() || !meshFlag.UncheckedGet<bool>()) {
        std::cerr << "Mesh light flag did not reach Hydra: " << meshFlag
                  << "\n";
        result = 13;
      }

      for (auto const &id : index->GetSprimSubtree(
               HdPrimTypeTokens->rectLight, SdfPath::AbsoluteRootPath())) {
        auto sided =
            scene.GetLightParamValue(id, TfToken("primvars:lava:singleSided"));
        auto geometry = scene.GetLightParamValue(
            id, TfToken("primvars:lava:renderLightGeometry"));
        if (!sided.IsHolding<bool>() || sided.UncheckedGet<bool>() ||
            !geometry.IsHolding<bool>() || !geometry.UncheckedGet<bool>()) {
          std::cerr << "Light primvars did not reach Hydra: " << sided << " / "
                    << geometry << "\n";
          result = 13;
        } else
          std::cout << "PASS: custom light flags reached Hydra\n";
      }
    }
    if (testInterop && std::getenv("LAVA_TEST_RESIZE")) {
      auto readbacks = lava_image_readback_count();
      auto started = std::chrono::steady_clock::now();
      for (unsigned step = 0; step < 60; ++step) {
        unsigned width = step % 11 == 0 ? 1 : 96 + (step * 37) % 400;
        unsigned height = step % 11 == 0 ? 1 : 64 + (step * 23) % 240;
        controller.SetRenderBufferSize(GfVec2i(width, height));
        controller.SetRenderViewport(GfVec4d(0, 0, width, height));
        auto presentations = lava_gpu_presentation_count();
        auto until =
            std::chrono::steady_clock::now() + std::chrono::seconds(10);
        do {
          auto tasks = controller.GetRenderingTasks();
          engine.Execute(index, &tasks);
          for (auto aov : {HdAovTokens->color, HdAovTokens->depth}) {
            auto resource = controller.GetRenderOutput(aov)->GetResource(false);
            if (resource.IsHolding<HgiTextureHandle>()) {
              auto texture = resource.UncheckedGet<HgiTextureHandle>();
              if (texture && texture->GetDescriptor().dimensions !=
                                 GfVec3i(width, height, 1))
                result = 17;
            }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (lava_gpu_presentation_count() == presentations &&
                 std::chrono::steady_clock::now() < until);
        if (lava_gpu_presentation_count() == presentations) {
          std::cerr << "Resize stalled at " << width << "x" << height << "\n";
          result = 17;
          break;
        }
      }
      if (lava_image_readback_count() != readbacks)
        result = 17;
      controller.SetRenderBufferSize(GfVec2i(320, 180));
      controller.SetRenderViewport(GfVec4d(0, 0, 320, 180));
      auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      do {
        auto tasks = controller.GetRenderingTasks();
        engine.Execute(index, &tasks);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      } while (!controller.IsConverged() &&
               std::chrono::steady_clock::now() < until);
      if (!controller.IsConverged())
        result = 17;
      if (result == 0)
        std::cout << "PASS: 60 viewport resizes including 1x1, correct GPU AOV "
                     "extents and zero readbacks, "
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - started)
                         .count()
                  << " seconds\n";
    }
    if (testInterop && std::getenv("LAVA_TEST_CAMERA_MOTION")) {
      auto presentations = lava_gpu_presentation_count();
      auto readbacks = lava_image_readback_count();
      auto camera = UsdGeomCamera(stage->GetPrimAtPath(cameraPath));
      auto originalTransform =
          camera.GetCamera(UsdTimeCode::Default()).GetTransform();
      auto motionOp = UsdGeomXformable(camera).MakeMatrixXform();
      float focal = 50;
      camera.GetFocalLengthAttr().Get(&focal);
      auto movingUntil =
          std::chrono::steady_clock::now() + std::chrono::seconds(1);
      unsigned moves = 0;
      do {
        auto transform = originalTransform;
        auto translation = transform.ExtractTranslation();
        translation[0] += .2 * std::sin(double(moves) * .05);
        transform.SetTranslateOnly(translation);
        motionOp.Set(transform);
        camera.GetFocalLengthAttr().Set(focal + float(moves % 20) * .05f);
        scene.ApplyPendingUpdates();
        auto tasks = controller.GetRenderingTasks();
        engine.Execute(index, &tasks);
        controller.GetRenderOutput(HdAovTokens->color)->GetResource(false);
        controller.GetRenderOutput(HdAovTokens->depth)->GetResource(false);
        ++moves;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      } while (std::chrono::steady_clock::now() < movingUntil);
      auto presented = lava_gpu_presentation_count() - presentations;
      if (presented < 3 || lava_image_readback_count() != readbacks) {
        std::cerr << "Continuous camera motion starved GPU display: " << moves
                  << " camera updates, " << presented << " images\n";
        result = 10;
      } else {
        std::cout << "PASS: " << moves << " continuous camera updates, "
                  << presented
                  << " GPU images during motion, no image readback\n";
      }
      deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
      do {
        auto tasks = controller.GetRenderingTasks();
        engine.Execute(index, &tasks);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      } while (!controller.IsConverged() &&
               std::chrono::steady_clock::now() < deadline);
    }
    if (std::getenv("LAVA_TEST_PRINCIPLED_EDITS")) {
      auto snapshot = [&]() {
        auto *b = controller.GetRenderOutput(HdAovTokens->color);
        auto *p = static_cast<float *>(b->Map());
        std::vector<float> data(p, p + b->GetWidth() * b->GetHeight() * 4);
        b->Unmap();
        return data;
      };
      auto before = snapshot();
      auto shader = stage->GetPrimAtPath(
          SdfPath("/materials/principled/principled_surface"));
      shader.GetAttribute(TfToken("inputs:rough_monoChannel")).Set(3);
      scene.ApplyPendingUpdates();
      auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      do {
        auto tasks = controller.GetRenderingTasks();
        engine.Execute(index, &tasks);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      } while (!controller.IsConverged() &&
               std::chrono::steady_clock::now() < until);
      auto after = snapshot();
      float difference = 0;
      for (size_t i = 0; i < before.size(); i++)
        difference = std::max(difference, std::abs(after[i] - before[i]));
      if (!controller.IsConverged() || difference < 1e-3f) {
        std::cerr << "Live Principled channel edit did not change rendering\n";
        result = 16;
      } else
        std::cout
            << "PASS: live Principled channel edit synchronized, image delta "
            << difference << "\n";
    }
    if (std::getenv("LAVA_TEST_SUBD_NORMAL_EDITS")) {
      auto mesh = UsdGeomMesh(stage->GetPrimAtPath(SdfPath("/World/constant")));
      VtVec3fArray original;
      mesh.GetNormalsAttr().Get(&original);
      auto converge = [&]() {
        scene.ApplyPendingUpdates();
        auto until =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        do {
          auto tasks = controller.GetRenderingTasks();
          engine.Execute(index, &tasks);
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } while (!controller.IsConverged() &&
                 std::chrono::steady_clock::now() < until);
        return controller.IsConverged();
      };
      auto panel = [&]() {
        auto *b = controller.GetRenderOutput(HdAovTokens->color);
        auto *image = static_cast<float *>(b->Map());
        int x = int((-4.8f / 14.f + .5f) * b->GetWidth());
        int y = int((.5f + 1.2f / 7.875f) * b->GetHeight());
        float value = image[(y * b->GetWidth() + x) * 4];
        b->Unmap();
        return value;
      };
      for (auto scheme :
           {TfToken("none"), TfToken("catmullClark"), TfToken("bilinear")}) {
        mesh.GetSubdivisionSchemeAttr().Set(scheme);
        if (!converge() || panel() < .33f || panel() > .40f) {
          std::cerr << "Scheme toggle lost authored normals: " << scheme
                    << "\n";
          result = 14;
        }
        auto topology = scene.GetMeshTopology(
            scene.ConvertCachePathToIndexPath(mesh.GetPath()));
        if (topology.GetFaceVertexCounts() != VtIntArray({3, 3}) ||
            topology.GetFaceVertexIndices() != VtIntArray({0, 1, 2, 0, 2, 3}))
          result = 14;
      }
      mesh.GetNormalsAttr().Set(VtVec3fArray({GfVec3f(0, 0, 1)}));
      if (!converge() || panel() < .78f) {
        std::cerr << "Live normal edit was not synchronized on a "
                     "subdivision-tagged cage\n";
        result = 14;
      }
      mesh.GetNormalsAttr().Set(original);
      mesh.GetSubdivisionSchemeAttr().Set(TfToken("catmullClark"));
      if (!converge() || panel() < .33f || panel() > .40f)
        result = 14;
      if (result == 0)
        std::cout << "PASS: live subdivision-tag toggles and normal edits "
                     "preserve polygon cage shading\n";
    }
    auto *buffer = controller.GetRenderOutput(HdAovTokens->color);
    if (!buffer || !buffer->IsConverged()) {
      std::cerr << "USD render did not converge\n";
      result = 4;
    } else {
      buffer->Resolve();
      VtValue initialColorResource, initialDepthResource;
      if (testInterop) {
        initialColorResource = buffer->GetResource(false);
        initialDepthResource =
            controller.GetRenderOutput(HdAovTokens->depth)->GetResource(false);
      }
      auto *pixels = static_cast<float *>(buffer->Map());
      size_t count = size_t(buffer->GetWidth()) * buffer->GetHeight();
      float lo = 1e30f, hi = -1e30f;
      bool finite = true;
      size_t redMaterialPixels = 0, blueMaterialPixels = 0;
      for (size_t i = 0; i < count; i++) {
        float v = pixels[i * 4];
        finite &= std::isfinite(v);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
        redMaterialPixels += v > 0.05f && v > pixels[i * 4 + 1] * 2.f;
        blueMaterialPixels +=
            pixels[i * 4 + 2] > 0.05f && pixels[i * 4 + 2] > v * 2.f;
      }
      if (std::getenv("LAVA_TEST_NORMALS")) {
        auto panel = [&](float worldX, float worldY) {
          int x = int((worldX / 14.f + .5f) * buffer->GetWidth());
          int y = int((.5f + worldY / 7.875f) * buffer->GetHeight());
          float average = 0;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              average += pixels[((y + dy) * buffer->GetWidth() + x + dx) * 4];
          return average / 9.f;
        };
        for (auto center : {GfVec2f(-4.8f, 1.2f), GfVec2f(-2.4f, -1.2f),
                            GfVec2f(0, -1.2f), GfVec2f(2.4f, -1.2f)}) {
          float value = panel(center[0], center[1]);
          if (value < .33f || value > .40f) {
            std::cerr << "Transformed shading normal irradiance wrong at "
                      << center << ": " << value << "\n";
            result = 14;
          }
        }
        float fallback = panel(4.8f, -1.2f);
        if (fallback < .78f || fallback > .84f) {
          std::cerr << "Missing normals lost flat fallback: " << fallback
                    << "\n";
          result = 14;
        }
        if (result == 0)
          std::cout
              << "PASS: native normal transforms, indexed precedence, mirrored "
                 "orientation and flat fallback produced expected irradiance\n";
      }
      if (std::getenv("LAVA_TEST_TEXTURES")) {
        for (auto sample : {GfVec2f(-1.4f, 0), GfVec2f(-.5f, 1),
                            GfVec2f(.5f, 0), GfVec2f(1.4f, 1)}) {
          int x = int((sample[0] / 4.f + .5f) * buffer->GetWidth());
          auto *p =
              pixels + ((buffer->GetHeight() / 2) * buffer->GetWidth() + x) * 4;
          bool blue = sample[1] > 0;
          // PreviewSurface also contributes a white dielectric specular
          // highlight.
          float wanted = blue ? p[2] : p[0], other = blue ? p[0] : p[2];
          if (wanted < .1f || wanted < other * 3.f ||
              (sample[0] > 0 && other > .05f)) {
            std::cerr << "Native texture/UV sampling failed at x=" << sample[0]
                      << ": " << p[0] << "," << p[1] << "," << p[2] << "\n";
            result = 15;
          }
        }
        if (result == 0)
          std::cout << "PASS: native PreviewSurface and OpenPBR texture/UV "
                       "bindings sampled expected red/blue regions\n";
      }
      if (std::getenv("LAVA_TEST_MATERIALS")) {
        if (redMaterialPixels < 10 || blueMaterialPixels < 10) {
          std::cerr << "Native surface materials fell back to gray\n";
          result = 11;
        } else
          std::cout << "PASS: native OpenPBR material color regions "
                    << redMaterialPixels << "/" << blueMaterialPixels
                    << " pixels\n";
      }
      if (argc > 3) {
        std::ofstream file(argv[3], std::ios::binary);
        file << "P6\n"
             << buffer->GetWidth() << " " << buffer->GetHeight() << "\n255\n";
        for (int y = int(buffer->GetHeight()) - 1; y >= 0; --y)
          for (unsigned x = 0; x < buffer->GetWidth(); ++x)
            for (int c = 0; c < 3; ++c) {
              float v = pixels[(size_t(y) * buffer->GetWidth() + x) * 4 + c];
              unsigned char b = static_cast<unsigned char>(
                  std::pow(std::max(0.f, v) / (1.f + std::max(0.f, v)),
                           1.f / 2.2f) *
                  255);
              file.write(reinterpret_cast<char *>(&b), 1);
            }
      }
      buffer->Unmap();
      auto *depthBuffer = controller.GetRenderOutput(HdAovTokens->depth);
      if (!depthBuffer || !depthBuffer->IsConverged()) {
        std::cerr << "depth AOV missing or not converged\n";
        result = 7;
      }
      if (testInterop) {
        auto colorResource = initialColorResource;
        auto depthResource = initialDepthResource;
        if (!colorResource.IsHolding<HgiTextureHandle>() ||
            !depthResource.IsHolding<HgiTextureHandle>()) {
          std::cerr << "GPU textures not exposed\n";
          result = 8;
        } else {
          auto colorTexture = colorResource.UncheckedGet<HgiTextureHandle>();
          auto depthTexture = depthResource.UncheckedGet<HgiTextureHandle>();
          std::vector<float> gpuColor(count * 4), gpuDepth(count);
          glGetTextureImage(colorTexture->GetRawResource(), 0, GL_RGBA,
                            GL_FLOAT, gpuColor.size() * sizeof(float),
                            gpuColor.data());
          glGetTextureImage(depthTexture->GetRawResource(), 0, GL_RED, GL_FLOAT,
                            gpuDepth.size() * sizeof(float), gpuDepth.data());
          auto *cpu = static_cast<float *>(buffer->Map());
          float error = 0;
          for (size_t i = 0; i < gpuColor.size(); ++i)
            error = std::max(error, std::abs(gpuColor[i] - cpu[i]));
          buffer->Unmap();
          auto *z = static_cast<float *>(depthBuffer->Map());
          for (size_t i = 0; i < gpuDepth.size(); ++i)
            error = std::max(error, std::abs(gpuDepth[i] - z[i]));
          depthBuffer->Unmap();
          if (error > 1e-5f || glGetError() != GL_NO_ERROR) {
            std::cerr << "GPU/CPU AOV mismatch " << error << "\n";
            result = 9;
          } else
            std::cout << "PASS: Vulkan external-memory/semaphore -> GL color "
                         "and depth textures, pixel error "
                      << error << "\n";
        }
      }
      std::cout << "USD image red range " << lo << ".." << hi << "; "
                << index->GetRprimIds().size() << " rprims\n";
      if (!finite || hi - lo < .01f)
        result = 5;
    }
  }
  delete index;
  plugin->DeleteRenderDelegate(renderer);
  registry.ReleasePlugin(plugin);
  return result;
}
