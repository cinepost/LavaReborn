#!/usr/bin/env python3
"""Vulkan end-to-end checks via the C ABI; build the debug cdylib first."""
import ctypes as C
import json
import math
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lib = C.CDLL(str(root / 'target/debug/liblava_reborn.so'))
lib.lava_create.restype = C.c_void_p
lib.lava_destroy.argtypes = [C.c_void_p]
lib.lava_error.argtypes = [C.c_void_p]
lib.lava_error.restype = C.c_char_p
lib.lava_render.argtypes = [C.c_void_p, C.c_char_p, C.c_uint32, C.c_uint32, C.c_bool, C.c_uint32, C.c_float, C.POINTER(C.c_float)]
lib.lava_submit.argtypes = lib.lava_render.argtypes[:-1]
lib.lava_submit.restype = C.c_uint64
lib.lava_poll.argtypes = [C.c_void_p, C.c_uint64, C.POINTER(C.c_float), C.c_size_t]
lib.lava_poll.restype = C.c_int64
session = lib.lava_create()
width, height = 64, 48
buffer = (C.c_float * (width * height * 4))()
scene = json.loads((root / 'examples/studio.json').read_text())

def render(mode, samples=17):
    status = lib.lava_render(session, json.dumps(scene).encode(), width, height, mode, samples, 0, buffer)
    assert status == 0, lib.lava_error(session)
    assert all(math.isfinite(v) for v in buffer)
    return list(buffer)

try:
    interactive = render(False,256)
    final = render(True,256)
    energy=lambda image:sum(v for i,v in enumerate(image) if i%4!=3)
    assert abs(energy(interactive)-energy(final))/energy(final)<0.03, 'mode changes mean energy beyond sampling tolerance'
    scene.setdefault('settings',{})['visibility']='raster'
    raster=render(True,256)
    assert abs(energy(raster)-energy(final))/energy(final)<0.005, 'raster visibility changes mean energy'
    assert max(final) > 0.1
    scene['lights'] = []
    dark = render(True, 1)
    assert sum(final) > sum(dark) + 1, 'lights have no effect'
    # An empty Hydra stage should render a background, not fail a BLAS build.
    scene['meshes'] = []
    empty = render(True, 1)
    assert max(empty[::4]) - min(empty[::4]) < 1e-5
    scene = json.loads((root / 'examples/studio.json').read_text())
    old = lib.lava_submit(session, json.dumps(scene).encode(), width, height, False, 100, 0)
    scene['camera']['transform'][12] = 1
    new = lib.lava_submit(session, json.dumps(scene).encode(), width, height, False, 3, 0)
    assert new > old > 0
    deadline = time.monotonic() + 30
    while True:
        done = lib.lava_poll(session, new, buffer, len(buffer))
        assert done >= 0, lib.lava_error(session)
        if done == 3:
            break
        assert time.monotonic() < deadline, 'async worker timeout'
        time.sleep(0.01)
    scene['camera']['near'] = -1
    assert lib.lava_submit(session, json.dumps(scene).encode(), width, height, False, 1, 0) == 0
    print('PASS: sampler energy agreement, raster/ray agreement, lights, empty stage, async latest revision, validation')
finally:
    lib.lava_destroy(session)
