#ifndef LAVA_REBORN_H
#define LAVA_REBORN_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct LavaSession LavaSession;
bool lava_is_supported(void);
LavaSession *lava_create(void);
void lava_destroy(LavaSession *);
const char *lava_error(const LavaSession *);
/* Caller owns width*height*4 float output. Serial access per session. */
int32_t lava_render(LavaSession *, const char *json, uint32_t width,
                    uint32_t height, bool final_mode, uint32_t samples,
                    float exposure, float *output);
uint64_t lava_submit(LavaSession *, const char *json, uint32_t width,
                     uint32_t height, bool final_mode, uint32_t samples,
                     float exposure);
/* Interactive mesh snapshot, retaining curves from the last full submission. */
uint64_t lava_update_meshes(LavaSession *, const char *json, uint32_t width,
                            uint32_t height, uint32_t samples, float exposure);
/* Interactive camera/light delta: omit geometry and retain the current
 * snapshot. */
uint64_t lava_update_view(LavaSession *, const char *json, uint32_t width,
                          uint32_t height, uint32_t samples, float exposure);
int64_t lava_poll(LavaSession *, uint64_t revision, float *output,
                  size_t capacity);
/* Linux immutable Vulkan -> OpenGL opaque-FD snapshots. No CPU image mapping.
 */
typedef struct LavaGpuLease LavaGpuLease;
typedef struct LavaGpuInfo {
  int32_t memory_fd, ready_fd, returned_fd;
  uint64_t allocation_size;
  uint32_t width, height, samples;
  uint64_t serial;
  uint8_t uuid[16];
} LavaGpuInfo;
void lava_enable_interop(LavaSession *, bool);
uint64_t lava_image_readback_count(void);
uint64_t lava_gpu_presentation_count(void);
int32_t lava_frame_transport(const LavaSession *, uint64_t revision);
int64_t lava_status(LavaSession *, uint64_t revision);
int64_t lava_poll_depth(LavaSession *, uint64_t revision, float *,
                        size_t capacity);
LavaGpuLease *lava_gpu_frame(LavaSession *, uint64_t revision,
                             uint64_t after_serial, LavaGpuInfo *);
void lava_release_gpu_frame(LavaGpuLease *);
void lava_gpu_frame_gl_submitted(const LavaGpuLease *);
#ifdef __cplusplus
}
#endif
#endif
