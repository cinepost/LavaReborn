#!/usr/bin/env hython
"""Export a read-only HIP snapshot and reproducible fitted camera for GPU profiling."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import hou
from pxr import Gf, Usd, UsdGeom, UsdRender
from render_usd import extract


def export(args):
    digest = hashlib.sha256(args.hip.read_bytes()).hexdigest()
    hou.hipFile.load(str(args.hip.resolve()), suppress_save_prompt=True, ignore_load_warnings=True)
    node = hou.node(args.lop) if args.lop else hou.node('/stage').displayNode()
    if node is None:
        raise RuntimeError('No displayed Solaris node; specify --lop')
    stage = node.stage()
    if node.errors():
        raise RuntimeError(node.errors())
    frame = hou.frame()
    settings = UsdRender.Settings(stage.GetPrimAtPath(args.settings))
    if not settings:
        raise RuntimeError('No render settings at ' + args.settings)
    camera_targets = settings.GetCameraRel().GetTargets()
    camera_path = str(camera_targets[0]) if camera_targets else ''
    fitted = args.fit or not stage.GetPrimAtPath(camera_path)
    bounds = None
    if fitted:
        cache = UsdGeom.BBoxCache(Usd.TimeCode(frame), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bounds = Gf.Range3d()
        for prim in stage.Traverse():
            imageable = UsdGeom.Imageable(prim)
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.BasisCurves)):
                continue
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue
            bounds.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedRange())
        if bounds.IsEmpty():
            raise RuntimeError('No visible render geometry to fit')
        center = bounds.GetMidpoint()
        view = Gf.Vec3d(1.3, .55, 1.8).GetNormalized()
        right = Gf.Cross(Gf.Vec3d(0, 1, 0), view).GetNormalized()
        up = Gf.Cross(view, right)
        distance = 0.
        for mask in itertools.product((0, 1), repeat=3):
            offset = Gf.Vec3d(*[(bounds.GetMax() if mask[i] else bounds.GetMin())[i] for i in range(3)]) - center
            distance = max(distance, Gf.Dot(offset, view) + abs(Gf.Dot(offset, right)) / .36,
                           Gf.Dot(offset, view) + abs(Gf.Dot(offset, up)) / .2025)
        eye = center + view * distance * 1.10
        matrix = Gf.Matrix4d(1)
        matrix.SetLookAt(eye, center, Gf.Vec3d(0, 1, 0))
        stage.SetEditTarget(stage.GetSessionLayer())
        camera_path = '/LavaProfileCamera'
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(50)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.001, max(1000., distance * 4)))
        UsdGeom.Xformable(camera).AddTransformOp().Set(matrix.GetInverse())
    scene = extract(stage, camera_path, frame)
    prim = settings.GetPrim()
    for source, target in [('diffuseDepth', 'diffuse_depth'), ('reflectionDepth', 'reflection_depth'),
                           ('refractionDepth', 'refraction_depth'), ('rouletteStart', 'roulette_start')]:
        attribute = prim.GetAttribute('lava:' + source)
        if attribute and attribute.Get() is not None:
            scene.setdefault('settings', {})[target] = int(attribute.Get())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'scene.json').write_text(json.dumps(scene, separators=(',', ':'), allow_nan=False))
    metadata = dict(hip=str(args.hip.resolve()), hip_sha256=digest, frame=frame, lop=node.path(),
                    resolution=list(settings.GetResolutionAttr().Get()), camera=scene['camera'], fitted_camera=fitted,
                    settings=scene.get('settings', {}), mesh_objects=len(scene['meshes']),
                    curve_objects=len(scene.get('curves', [])),
                    strands=sum(len(c['counts']) for c in scene.get('curves', [])),
                    control_points=sum(len(c['points']) for c in scene.get('curves', [])))
    if bounds is not None:
        metadata['bounds'] = [list(bounds.GetMin()), list(bounds.GetMax())]
    assert hashlib.sha256(args.hip.read_bytes()).hexdigest() == digest, 'Source HIP changed'
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hip', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--lop')
    parser.add_argument('--settings', default='/Render/Lava')
    parser.add_argument('--fit', action='store_true')
    export(parser.parse_args())
