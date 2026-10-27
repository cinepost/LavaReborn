#include "gl_context.h"
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <stdexcept>
struct LavaTestGlContext::Impl {
  Display *display = nullptr;
  GLXContext context = nullptr;
  GLXPbuffer drawable = 0;

public:
  Impl() {
    display = XOpenDisplay(nullptr);
    if (!display)
      throw std::runtime_error("Cannot open test X display");
    int attributes[] = {GLX_X_RENDERABLE,
                        True,
                        GLX_DRAWABLE_TYPE,
                        GLX_PBUFFER_BIT,
                        GLX_RENDER_TYPE,
                        GLX_RGBA_BIT,
                        GLX_RED_SIZE,
                        8,
                        GLX_GREEN_SIZE,
                        8,
                        GLX_BLUE_SIZE,
                        8,
                        None};
    int count = 0;
    auto *configurations =
        glXChooseFBConfig(display, DefaultScreen(display), attributes, &count);
    if (!configurations || !count)
      throw std::runtime_error("No GLX pbuffer configuration");
    int size[] = {GLX_PBUFFER_WIDTH, 320, GLX_PBUFFER_HEIGHT, 180, None};
    drawable = glXCreatePbuffer(display, configurations[0], size);
    context = glXCreateNewContext(display, configurations[0], GLX_RGBA_TYPE,
                                  nullptr, True);
    XFree(configurations);
    if (!context ||
        !glXMakeContextCurrent(display, drawable, drawable, context))
      throw std::runtime_error("Cannot make GLX test context current");
    GarchGLApiLoad();
  }
  ~Impl() {
    glXMakeContextCurrent(display, 0, 0, nullptr);
    glXDestroyContext(display, context);
    glXDestroyPbuffer(display, drawable);
    XCloseDisplay(display);
  }
};

LavaTestGlContext::LavaTestGlContext() : impl(std::make_unique<Impl>()) {}
LavaTestGlContext::~LavaTestGlContext() = default;
