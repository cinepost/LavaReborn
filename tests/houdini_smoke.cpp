#include "../hydra/texturePaths.h"
#include "gl_context.h"
#include "lava.h"
// Exercise Solaris' HUSD imaging/AOV interface, not only upstream Hdx tasks.
class UT_StringHolder;
#include <HUSD/HUSD_Compositor.h>
#include <HUSD/HUSD_DataHandle.h>
#include <HUSD/HUSD_Imaging.h>
#include <HUSD/HUSD_Overrides.h>
#include <HUSD/HUSD_PostLayers.h>
#include <HUSD/XUSD_RenderSettings.h>
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
  double brightness = 0.;
  bool brightnessValid = false;
  std::vector<float> brightnessPixels;
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
  void updateColorTexture(int texture) override {
    ++gpuColor;
    if (!std::getenv("LAVA_TEST_BRIGHTNESS") &&
        !std::getenv("LAVA_TEST_SAMPLE_COUNT") &&
        !std::getenv("LAVA_TEST_ALPHA"))
      return;
    GLint width = 0, height = 0;
    glGetTextureLevelParameteriv(texture, 0, GL_TEXTURE_WIDTH, &width);
    glGetTextureLevelParameteriv(texture, 0, GL_TEXTURE_HEIGHT, &height);
    if (width <= 0 || height <= 0)
      return;
    static bool warned = false;
    if (!warned) {
      std::cerr << "Lava test [WARNING]: diagnostic GPU color readback enabled "
                   "for compositor image checks\n";
      warned = true;
    }
    std::vector<float> pixels(size_t(width) * height * 4);
    glGetTextureImage(texture, 0, GL_RGBA, GL_FLOAT,
                      pixels.size() * sizeof(float), pixels.data());
    brightness = 0.;
    size_t count = 0;
    for (int y = height / 4; y < 3 * height / 4; y++)
      for (int x = width / 4; x < 3 * width / 4; x++) {
        auto *p = &pixels[(y * width + x) * 4];
        brightness += p[0] * .2126 + p[1] * .7152 + p[2] * .0722;
        ++count;
      }
    brightness /= std::max(size_t(1), count);
    brightnessValid = true;
    brightnessPixels = std::move(pixels);
  }
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
  if (std::getenv("LAVA_TEST_SAMPLE_COUNT")) {

    struct Context : XUSD_RenderSettingsContext {
      TfToken renderer() const override {
        return TfToken("LavaRendererPlugin");
      }
      GfVec2i defaultResolution() const override { return GfVec2i(320, 180); }
      fpreal startFrame() const override { return 1; }
      UsdTimeCode evalTime() const override { return UsdTimeCode(1); }
    } context;
    XUSD_RenderSettings settings;
    bool initialized = settings.init(stage, SdfPath("/Render/Lava"), context);
    bool resolved = settings.resolveProducts(stage, context, false);
    TfTokenVector aovs;
    HdAovDescriptorList descs;
    bool collected = settings.collectAovs(aovs, descs);
    if (!initialized || !resolved || !collected ||
        aovs != TfTokenVector{TfToken("color"), TfToken("sampleCount")})
      return 12;
    std::cerr << "Diagnostic USD settings: initialized=" << initialized
              << " resolved=" << resolved << " collected=" << collected
              << " AOVs=";
    for (auto const &a : aovs)
      std::cerr << a << " ";
    std::cerr << "\n";
  }
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
  if (std::getenv("LAVA_TEST_SAMPLE_COUNT"))
    imaging.setOutputPlane("sampleCount");
  UT_Options options;
  auto setting = [](const char *name, int fallback) {
    const char *value = std::getenv(name);
    return value ? std::max(1, std::atoi(value)) : fallback;
  };
  const int width = setting("LAVA_TEST_WIDTH", 320);
  const int height = setting("LAVA_TEST_HEIGHT", 180);
  options.setOptionI("lava:samples", setting("LAVA_TEST_SAMPLES", 4));
  if (std::getenv("LAVA_TEST_SAMPLE_COUNT")) {
    imaging.render(v, p, UT_DimRect(0, 0, width, height), "LavaRendererPlugin",
                   &options, true);
    imaging.setRenderSettings(
        UT_StringHolder(std::getenv("LAVA_TEST_RENDER_SETTINGS")));
    imaging.setOutputPlane("sampleCount");
    std::cerr << "Available planes after initialization: ";
    for (auto const &name : imaging.rendererPlanes())
      std::cerr << name << " ";
    std::cerr << "\n";
  }
  if (std::getenv("LAVA_TEST_ALPHA"))
    options.setOptionI("lava:headlight", 0);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  do {
    bool result = imaging.render(v, p, UT_DimRect(0, 0, width, height),
                                 "LavaRendererPlugin", &options, true);
    if (!result) {
      std::cerr << "HUSD render failed\n";
      return 5;
    }
    if (std::getenv("LAVA_TEST_BRIGHTNESS") && capture.brightnessValid)
      std::cerr << "HUSD brightness converged=" << imaging.isConverged()
                << " mean=" << capture.brightness << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  } while (!imaging.isConverged() &&
           std::chrono::steady_clock::now() < deadline);
  if (std::getenv("LAVA_TEST_BRIGHTNESS")) {
    double final = capture.brightness;
    auto finalPixels = capture.brightnessPixels;
    for (int i = 0; i < 4; i++)
      imaging.render(v, p, UT_DimRect(0, 0, width, height),
                     "LavaRendererPlugin", &options, true);
    if (!capture.brightnessValid || capture.brightnessPixels != finalPixels ||
        std::abs(final - capture.brightness) > 1e-7) {
      std::cerr << "FAIL: compositor brightness changed after completion\n";
      return 9;
    }
    std::cerr << "PASS: compositor brightness stable after completion mean="
              << final << "\n";
  }
  if (std::getenv("LAVA_TEST_SAMPLE_COUNT")) {
    for (int i = 0; i < 4; ++i)
      imaging.render(v, p, UT_DimRect(0, 0, width, height),
                     "LavaRendererPlugin", &options, true);
    std::cerr << "Sample count output plane=" << imaging.outputPlane()
              << " current AOV=" << imaging.currentAov() << "\n";
    if (!capture.brightnessValid || capture.brightnessPixels.empty() ||
        capture.cpuColor || !capture.gpuColor)
      return 10;
    for (size_t i = 0; i < capture.brightnessPixels.size(); i += 4) {
      const float *pixel = &capture.brightnessPixels[i];
      if (pixel[0] != .5f || pixel[1] != 0.f || pixel[2] != 0.f ||
          pixel[3] != 1.f) {
        std::cerr << "FAIL heat map pixel=" << pixel[0] << "," << pixel[1]
                  << "," << pixel[2] << "," << pixel[3] << "\n";
        return 11;
      }
    }
    std::cerr << "PASS: uniform completed sampleCount heat map delivered "
                 "through GPU compositor\n";
  }
  if (std::getenv("LAVA_TEST_ALPHA")) {
    for (int i = 0; i < 4; ++i)
      imaging.render(v, p, UT_DimRect(0, 0, width, height),
                     "LavaRendererPlugin", &options, true);
    unsigned transparent = 0, partial = 0;
    for (size_t i = 0; i < capture.brightnessPixels.size(); i += 4) {
      auto *pixel = &capture.brightnessPixels[i];
      if (pixel[3] == 0.f) {
        ++transparent;
        if (pixel[0] != 0.f || pixel[1] != 0.f || pixel[2] != 0.f)
          return 13;
      } else if (pixel[3] > 0.f && pixel[3] < 1.f) {
        ++partial;
        if (std::abs(pixel[0] - 2.f * pixel[3]) > 1e-5 ||
            std::abs(pixel[1] - pixel[3]) > 1e-5 ||
            std::abs(pixel[2] - .5f * pixel[3]) > 1e-5)
          return 14;
      }
    }
    if (!transparent || !partial || capture.cpuColor || !capture.gpuColor)
      return 15;
    std::cerr << "PASS: GPU compositor preserves transparent background and "
                 "premultiplied cutout alpha: transparent="
              << transparent << " partial=" << partial << "\n";
  }
  std::vector<float> lightResponse;
  auto recordLightResponse = [&]() {
    if (!std::getenv("LAVA_TEST_LIGHT_RESPONSE"))
      return;
    auto image = imaging.getAOVBuffer("color");
    auto *pixels = static_cast<const float *>(image.map());
    if (pixels)
      lightResponse.push_back(
          pixels[(image.yres() / 2 * image.xres() + image.xres() / 2) * 4]);
    image.unmap();
  };
  recordLightResponse();
  const int animationFrames = setting("LAVA_TEST_ANIMATION_FRAMES", 1);
  for (int frame = 2; frame <= animationFrames; ++frame) {
    imaging.setFrame(frame);
    auto started = std::chrono::steady_clock::now();
    deadline = started + std::chrono::seconds(120);
    do {
      if (!imaging.render(v, p, UT_DimRect(0, 0, width, height),
                          "LavaRendererPlugin", &options, true))
        return 5;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } while (!imaging.isConverged() &&
             std::chrono::steady_clock::now() < deadline);
    std::cerr << "HUSD animation frame=" << frame
              << " converged=" << imaging.isConverged() << " elapsed_ms="
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - started)
                     .count()
              << "\n";
    if (!imaging.isConverged())
      return 7;
    recordLightResponse();
  }
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
  bool texturePixels = true;
  if (b && std::getenv("LAVA_TEST_TEXTURES")) {
    auto *pixels = static_cast<const float *>(b.map());
    if (!pixels)
      texturePixels = false;
    else
      for (auto sample : {GfVec2f(-1.4f, 0), GfVec2f(-.5f, 1), GfVec2f(.5f, 0),
                          GfVec2f(1.4f, 1)}) {
        int x = int((sample[0] / 4.f + .5f) * b.xres());
        auto *p = pixels + ((b.yres() / 2) * b.xres() + x) * 4;
        float wanted = sample[1] > 0 ? p[2] : p[0],
              other = sample[1] > 0 ? p[0] : p[2];
        if (wanted < .1f || wanted < other * 3.f) {
          texturePixels = false;
          std::cerr << "HUSD texture sample failed at " << sample[0] << ": "
                    << p[0] << "," << p[1] << "," << p[2] << "\n";
        }
      }
    b.unmap();
    if (texturePixels)
      std::cout << "PASS: Solaris sampled expected UDIM red/blue tiles\n";
  }
  if (b && std::getenv("LAVA_TEST_CURVES")) {
    auto *pixels = static_cast<const float *>(b.map());
    size_t bright = 0, total = size_t(b.xres()) * b.yres();
    if (pixels)
      for (size_t i = 0; i < total; i++) {
        auto *p = pixels + i * 4;
        if (p[0] > .4f && p[0] > p[2] * 1.2f)
          ++bright;
      }
    b.unmap();
    texturePixels = texturePixels && pixels && bright > total / 200;
    std::cout << (texturePixels ? "PASS" : "FAIL")
              << ": native curved strand pixels=" << bright << "/" << total
              << "\n";
  }
  if (std::getenv("LAVA_TEST_LIGHT_RESPONSE")) {
    if (lightResponse.size() != 3)
      return 8;
    float zero = lightResponse[0], one = lightResponse[1],
          two = lightResponse[2];
    bool linear =
        one - zero > .05f && std::abs(two - zero - 2.f * (one - zero)) < .02f;
    std::cout << (linear ? "PASS" : "FAIL")
              << ": native light 0/1/2 response=" << zero << "," << one << ","
              << two << "\n";
    if (!linear)
      return 8;
  }
  auto transfers = lava_image_readback_count() - baseline;
  std::cerr << "Actual Lava GPU-to-CPU frame readbacks: " << transfers << "\n";
  return texturePixels && b && imaging.isConverged() && capture.gpuColor > 0 &&
                 capture.gpuDepth > 0 && capture.cpuColor == 0 &&
                 capture.cpuDepth == 0
             ? 0
             : 6;
}

extern "C" const char *lava_houdini_texture_path(const char *path) {
  static thread_local std::string result;
  result = lava::ExpandTexturePath(path ? path : "");
  return result.c_str();
}
