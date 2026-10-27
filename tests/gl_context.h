#pragma once
// This header precedes any other GL headers; GLX declarations stay in the .cpp.
#include <pxr/imaging/garch/glApi.h>
#include <memory>
class LavaTestGlContext {
    struct Impl;std::unique_ptr<Impl> impl;
public:
    LavaTestGlContext();
    ~LavaTestGlContext();
};
