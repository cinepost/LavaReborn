#pragma once
#include "lava.h"
#include <memory>
#include <pxr/imaging/hgi/texture.h>
#include <pxr/pxr.h>
PXR_NAMESPACE_OPEN_SCOPE
// Called only with a current GL context. Imports immutable Vulkan snapshots;
// one GL compute dispatch flips rows and writes the color/depth Hgi textures.
class LavaGlInterop {
public:
  LavaGlInterop();
  ~LavaGlInterop();
  HgiTextureHandle Get(LavaSession *, uint64_t revision, unsigned plane,
                       unsigned width, unsigned height);
  static bool HasContext();
  static void Retire(LavaGlInterop *);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
PXR_NAMESPACE_CLOSE_SCOPE
