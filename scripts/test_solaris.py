#!/usr/bin/env python3
"""Run with Houdini hython; requires an X display and the built test module."""
import argparse
import ctypes
from pathlib import Path
import hou  # Initializes Houdini before invoking its HUSD imaging interface.

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('scene', type=Path)
parser.add_argument('camera')
parser.add_argument('--library', type=Path, default=Path(__file__).resolve().parents[1] / 'build/liblava_houdini_smoke.so')
args = parser.parse_args()
library = ctypes.CDLL(str(args.library.resolve()))
library.lava_houdini_smoke.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
code = library.lava_houdini_smoke(str(args.scene.resolve()).encode(), args.camera.encode())
print(f'Solaris compositor test exit: {code}')
raise SystemExit(code)
