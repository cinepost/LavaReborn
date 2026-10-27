#include "gl_context.h"
#include "lava.h"
// Exercise Solaris' HUSD imaging/AOV interface, not only upstream Hdx tasks.
class UT_StringHolder;
#include <HUSD/HUSD_Compositor.h>
#include <HUSD/HUSD_DataHandle.h>
#include <HUSD/HUSD_Imaging.h>
#include <HUSD/HUSD_Overrides.h>
#include <HUSD/HUSD_PostLayers.h>
#include <UT/UT_Options.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <pxr/base/gf/frustum.h>
#include <pxr/base/plug/registry.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <thread>
#include <vector>
PXR_NAMESPACE_USING_DIRECTIVE
class Capture : public HUSD_Compositor {
public:
  int cpuColor = 0, cpuDepth = 0, gpuColor = 0, gpuDepth = 0, clears = 0;
  void updateColorBuffer(void *data, PXL_DataFormat, int channels) override {
    if (data && channels > 0)
      ++cpuColor;
    else
      ++clears;
  }
  void updateDepthBuffer(void *data, PXL_DataFormat, int channels) override {
    if (data && channels > 0)
      ++cpuDepth;
    else
      ++clears;
  }
  void updateColorTexture(int) override { ++gpuColor; }
  void updateDepthTexture(int) override { ++gpuDepth; }
  void updatePrimIDBuffer(void *, PXL_DataFormat, bool = false,
                          bool = false) override {}
  void updateInstanceIDBuffer(void *, PXL_DataFormat, bool = false,
                              bool = false) override {}
  const PXL_Raster *primID() const override { return nullptr; }
  const PXL_Raster *instanceID() const override { return nullptr; }
};
extern "C" int lava_houdini_smoke(const char *path, const char *cameraPath) {
  int argc = 3;
  const char *argv[] = {"test", path, cameraPath};
  if (argc < 3)
    return 1;
  PlugRegistry::GetInstance().RegisterPlugins(LAVA_HOUDINI_PLUGIN_PATH);
  auto stage = UsdStage::Open(argv[1]);
  if (!stage)
    return 2;
  auto camera = UsdGeomCamera(stage->GetPrimAtPath(SdfPath(argv[2])));
  if (!camera)
    return 3;
  auto gf = camera.GetCamera(UsdTimeCode::Default());
  auto view = gf.GetTransform().GetInverse();
  auto projection = gf.GetFrustum().ComputeProjectionMatrix();
  UT_Matrix4D v, p;
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 4; ++col) {
      v(row, col) = view[row][col];
      p(row, col) = projection[row][col];
    }
  LavaTestGlContext gl;
  std::cerr << "Created Houdini offscreen GL context\n";
  HUSD_DataHandle data{UT_StringHolder(argv[1])};
  auto baseline = lava_image_readback_count();
  Capture capture;
  HUSD_Imaging imaging;
  imaging.setAOVCompositor(&capture);
  imaging.setStage(data, {}, {});
  imaging.setCameraPath(argv[2]);
  imaging.setFrame(1);
  imaging.showPurposeRender(true);
  imaging.showPurposeProxy(true);
  imaging.setDefaultLights(true, false);
  imaging.setLighting(true);
  imaging.setMaterials(true);
  if (const char *settings = std::getenv("LAVA_TEST_RENDER_SETTINGS"))
    imaging.setRenderSettings(UT_StringHolder(settings));
  UT_Options options;
  auto setting = [](const char *name, int fallback) {
    const char *value = std::getenv(name);
    return value ? std::max(1, std::atoi(value)) : fallback;
  };
  const int width = setting("LAVA_TEST_WIDTH", 320);
  const int height = setting("LAVA_TEST_HEIGHT", 180);
  options.setOptionI("lava:samples", setting("LAVA_TEST_SAMPLES", 4));
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  do {
    bool result = imaging.render(v, p, UT_DimRect(0, 0, width, height),
                                 "LavaRendererPlugin", &options, true);
    if (!result) {
      std::cerr << "HUSD render failed\n";
      return 5;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  } while (!imaging.isConverged() &&
           std::chrono::steady_clock::now() < deadline);
  std::cerr << "HUSD converged=" << imaging.isConverged()
            << " AOV mode=" << imaging.hasAOVBuffers() << " planes=";
  for (auto const &name : imaging.rendererPlanes())
    std::cerr << name << " ";
  std::cerr << "\n";
  auto b = imaging.getAOVBuffer("color");
  std::cerr << "color buffer valid=" << b.isValid();
  if (b)
    std::cerr << " size=" << b.xres() << "x" << b.yres();
  std::cerr << "\n";
  std::cerr << "Compositor delivery: GPU color=" << capture.gpuColor
            << ", depth=" << capture.gpuDepth
            << " CPU color=" << capture.cpuColor
            << ", depth=" << capture.cpuDepth
            << ", empty clear callbacks=" << capture.clears << "\n";
  auto transfers = lava_image_readback_count() - baseline;
  std::cerr << "Actual Lava GPU-to-CPU frame readbacks: " << transfers << "\n";
  return b && imaging.isConverged() && capture.gpuColor > 0 &&
                 capture.gpuDepth > 0 && capture.cpuColor == 0 &&
                 capture.cpuDepth == 0
             ? 0
             : 6;
}
