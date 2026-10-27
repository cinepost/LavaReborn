#include "interop.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <pxr/imaging/garch/glApi.h>
#include <pxr/imaging/hgiGL/hgi.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>
extern "C" const unsigned char *glGetString(unsigned int);
PXR_NAMESPACE_OPEN_SCOPE
namespace {
std::mutex cleanupMutex;
auto &cleanup() {
  static auto *jobs = new std::vector<LavaGlInterop *>;
  return *jobs;
}
void drain() {
  std::vector<LavaGlInterop *> pending;
  {
    std::lock_guard<std::mutex> l(cleanupMutex);
    pending.swap(cleanup());
  }
  if (!pending.empty()) {
    glFinish();
    for (auto *p : pending)
      delete p;
  }
}
bool extension(char const *name) {
  GLint n = 0;
  glGetIntegerv(GL_NUM_EXTENSIONS, &n);
  for (GLint i = 0; i < n; ++i)
    if (std::strcmp(
            reinterpret_cast<char const *>(glGetStringi(GL_EXTENSIONS, i)),
            name) == 0)
      return true;
  return false;
}
GLuint program() {
  const char *source = R"GLSL(#version 430
layout(local_size_x=8,local_size_y=8) in;
layout(std430,binding=0) readonly buffer Input {vec4 pixels[];};
layout(rgba32f,binding=0) writeonly uniform image2D colorOut;
layout(r32f,binding=1) writeonly uniform image2D depthOut;
uniform uvec2 dims;
void main(){uvec2 p=gl_GlobalInvocationID.xy;if(any(greaterThanEqual(p,dims)))return;
 uint i=p.y*dims.x+p.x;ivec2 dst=ivec2(p.x,dims.y-1u-p.y);
 imageStore(colorOut,dst,pixels[i]);imageStore(depthOut,dst,pixels[dims.x*dims.y+i]);}
)GLSL";
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    glDeleteShader(shader);
    throw std::runtime_error(log);
  }
  GLuint p = glCreateProgram();
  glAttachShader(p, shader);
  glLinkProgram(p);
  glDeleteShader(shader);
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    glDeleteProgram(p);
    throw std::runtime_error("interop compute program link failed");
  }
  return p;
}
struct Import {
  GLuint memory = 0, buffer = 0, semaphore = 0, returned = 0;
  GLsync fence = nullptr;
  LavaGpuLease *lease = nullptr;
  void destroy() {
    if (fence)
      glDeleteSync(fence);
    if (buffer)
      glDeleteBuffers(1, &buffer);
    if (semaphore)
      glDeleteSemaphoresEXT(1, &semaphore);
    if (returned)
      glDeleteSemaphoresEXT(1, &returned);
    if (memory)
      glDeleteMemoryObjectsEXT(1, &memory);
    lava_release_gpu_frame(lease);
  }
};
struct GlState {
  GLint program = 0, ssbo = 0, general = 0;
  struct Image {
    GLint name, level, layered, layer, access, format;
  };
  std::array<Image, 2> images;
  GlState() {
    glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &general);
    glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 0, &ssbo);
    for (unsigned i = 0; i < 2; ++i) {
      auto &v = images[i];
      glGetIntegeri_v(GL_IMAGE_BINDING_NAME, i, &v.name);
      glGetIntegeri_v(GL_IMAGE_BINDING_LEVEL, i, &v.level);
      glGetIntegeri_v(GL_IMAGE_BINDING_LAYERED, i, &v.layered);
      glGetIntegeri_v(GL_IMAGE_BINDING_LAYER, i, &v.layer);
      glGetIntegeri_v(GL_IMAGE_BINDING_ACCESS, i, &v.access);
      glGetIntegeri_v(GL_IMAGE_BINDING_FORMAT, i, &v.format);
    }
  }
  ~GlState() {
    glUseProgram(program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, general);
    for (unsigned i = 0; i < 2; ++i) {
      auto &v = images[i];
      glBindImageTexture(i, v.name, v.level, v.layered, v.layer, v.access,
                         v.format);
    }
  }
};
} // namespace
struct LavaGlInterop::Impl {
  HgiGL hgi;
  HgiTextureHandle color, depth;
  GLuint shader = 0;
  uint64_t revision = 0;
  uint32_t sample = 0;
  uint64_t serial = 0;
  unsigned width = 0, height = 0;
  bool announced = false;
  std::vector<Import> imports;
  Impl() {
    if (!extension("GL_EXT_memory_object_fd") ||
        !extension("GL_EXT_semaphore_fd") || !glNamedBufferStorageMemEXT ||
        !glGetUnsignedBytei_vEXT)
      throw std::runtime_error(
          "GL external memory/semaphore FD extensions unavailable");
    shader = program();
  }
  ~Impl() {
    glFinish();
    for (auto &i : imports)
      i.destroy();
    if (color)
      hgi.DestroyTexture(&color);
    if (depth)
      hgi.DestroyTexture(&depth);
    if (shader)
      glDeleteProgram(shader);
  }
  void collect() {
    for (auto it = imports.begin(); it != imports.end();) {
      auto status = glClientWaitSync(it->fence, 0, 0);
      if (status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED) {
        it->destroy();
        it = imports.erase(it);
      } else
        ++it;
    }
  }
  void resize(unsigned w, unsigned h) {
    if (width != w || height != h) {
      if (color)
        hgi.DestroyTexture(&color);
      if (depth)
        hgi.DestroyTexture(&depth);
      HgiTextureDesc d;
      d.debugName = "Lava shared color";
      d.dimensions = GfVec3i(w, h, 1);
      d.format = HgiFormatFloat32Vec4;
      d.usage = HgiTextureUsageBitsShaderRead | HgiTextureUsageBitsShaderWrite;
      color = hgi.CreateTexture(d);
      d.debugName = "Lava shared depth";
      d.format = HgiFormatFloat32;
      depth = hgi.CreateTexture(d);
      width = w;
      height = h;
      float zero[4] = {0, 0, 0, 0}, far = 1e30f;
      glClearTexImage(color->GetRawResource(), 0, GL_RGBA, GL_FLOAT, zero);
      glClearTexImage(depth->GetRawResource(), 0, GL_RED, GL_FLOAT, &far);
    }
  }
  void update(LavaSession *session, uint64_t r) {
    collect();
    if (revision != r) {
      revision = r;
      sample = 0;
    }
    LavaGpuInfo info{};
    auto *lease = lava_gpu_frame(session, r, serial, &info);
    if (!lease)
      return;
    GLint devices = 0;
    glGetIntegerv(GL_NUM_DEVICE_UUIDS_EXT, &devices);
    bool match = false;
    for (GLint i = 0; i < devices; ++i) {
      GLubyte uuid[16];
      glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT, i, uuid);
      match |= std::memcmp(uuid, info.uuid, 16) == 0;
    }
    if (!match) {
      close(info.memory_fd);
      close(info.ready_fd);
      close(info.returned_fd);
      lava_release_gpu_frame(lease);
      throw std::runtime_error("Vulkan/OpenGL physical-device UUID mismatch");
    }
    Import imported;
    imported.lease = lease;
    glCreateMemoryObjectsEXT(1, &imported.memory);
    GLint dedicated = GL_TRUE;
    glMemoryObjectParameterivEXT(imported.memory,
                                 GL_DEDICATED_MEMORY_OBJECT_EXT, &dedicated);
    glImportMemoryFdEXT(imported.memory, info.allocation_size,
                        GL_HANDLE_TYPE_OPAQUE_FD_EXT, info.memory_fd);
    if (glGetError() != GL_NO_ERROR) {
      close(info.memory_fd);
      close(info.ready_fd);
      close(info.returned_fd);
      imported.destroy();
      throw std::runtime_error("GL memory FD import failed");
    }
    glCreateBuffers(1, &imported.buffer);
    glNamedBufferStorageMemEXT(imported.buffer, info.allocation_size,
                               imported.memory, 0);
    glGenSemaphoresEXT(1, &imported.semaphore);
    glImportSemaphoreFdEXT(imported.semaphore, GL_HANDLE_TYPE_OPAQUE_FD_EXT,
                           info.ready_fd);
    if (glGetError() != GL_NO_ERROR) {
      close(info.ready_fd);
      close(info.returned_fd);
      imported.destroy();
      throw std::runtime_error("GL semaphore FD import failed");
    }
    resize(info.width, info.height);
    glGenSemaphoresEXT(1, &imported.returned);
    glImportSemaphoreFdEXT(imported.returned, GL_HANDLE_TYPE_OPAQUE_FD_EXT,
                           info.returned_fd);
    glWaitSemaphoreEXT(imported.semaphore, 1, &imported.buffer, 0, nullptr,
                       nullptr);
    {
      GlState saved;
      glUseProgram(shader);
      glUniform2ui(glGetUniformLocation(shader, "dims"), width, height);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, imported.buffer);
      glBindImageTexture(0, color->GetRawResource(), 0, GL_FALSE, 0,
                         GL_WRITE_ONLY, GL_RGBA32F);
      glBindImageTexture(1, depth->GetRawResource(), 0, GL_FALSE, 0,
                         GL_WRITE_ONLY, GL_R32F);
      glDispatchCompute((width + 7) / 8, (height + 7) / 8, 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                      GL_TEXTURE_FETCH_BARRIER_BIT);
    }
    glSignalSemaphoreEXT(imported.returned, 1, &imported.buffer, 0, nullptr,
                         nullptr);
    lava_gpu_frame_gl_submitted(imported.lease);
    imported.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
    imports.push_back(imported);
    sample = info.samples;
    serial = info.serial;
    if (!announced) {
      std::fprintf(stderr, "Lava viewport: Vulkan/OpenGL interop active (color "
                           "+ depth, no image readback)\n");
      announced = true;
    }
    if (std::getenv("LAVA_TRACE"))
      std::fprintf(stderr,
                   "Lava interop: GPU-only color/depth, %ux%u sample %u, no "
                   "image readback\n",
                   width, height, sample);
  }
};
bool LavaGlInterop::HasContext() {
  return ::glGetString(GL_VERSION) != nullptr;
}
LavaGlInterop::LavaGlInterop() {
  GarchGLApiLoad();
  drain();
  impl = std::make_unique<Impl>();
}
LavaGlInterop::~LavaGlInterop() = default;
HgiTextureHandle LavaGlInterop::Get(LavaSession *session, uint64_t revision,
                                    bool depth, unsigned width,
                                    unsigned height) {
  impl->hgi.StartFrame();
  try {
    impl->resize(width, height);
    if (session)
      impl->update(session, revision);
  } catch (...) {
    impl->hgi.EndFrame();
    throw;
  }
  impl->hgi.EndFrame();
  return depth ? impl->depth : impl->color;
}
void LavaGlInterop::Retire(LavaGlInterop *p) {
  if (!p)
    return;
  std::lock_guard<std::mutex> l(cleanupMutex);
  cleanup().push_back(p);
}
PXR_NAMESPACE_CLOSE_SCOPE
