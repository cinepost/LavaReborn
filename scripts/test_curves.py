#!/usr/bin/env hython
"""Synthetic curve scene and USD-to-JSON validation; never edits user HIP files."""
import json,sys
from pathlib import Path
from pxr import Usd,UsdGeom,UsdShade,UsdRender,Sdf
from render_usd import extract
path=Path(sys.argv[1]).resolve();stage=Usd.Stage.CreateNew(str(path))
camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateProjectionAttr('orthographic');camera.CreateHorizontalApertureAttr(40);camera.CreateVerticalApertureAttr(22.5)
UsdGeom.Xformable(camera).AddTranslateOp().Set((0,0,5))
material=UsdShade.Material.Define(stage,'/Materials/Curves');shader=UsdShade.Shader.Define(stage,'/Materials/Curves/OpenPBR');shader.CreateIdAttr('ND_open_pbr_surface_surfaceshader')
for name,value in [('base_weight',0.),('specular_weight',0.),('emission_luminance',2.)]:shader.CreateInput(name,Sdf.ValueTypeNames.Float).Set(value)
shader.CreateInput('emission_color',Sdf.ValueTypeNames.Color3f).Set((1,.5,.25));material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'out')
examples=[('linear',[(-1.5,-.8,0),(-1.5,.8,0)]),('bezier',[(-.8,-.8,0),(-1.2,-.2,0),(-.2,.2,0),(-.8,.8,0)]),('catmullRom',[(-.2,-1.5,0),(-.2,-.8,0),(.2,.8,0),(.2,1.5,0)]),('bspline',[(.8,-1.5,0),(.5,-.8,0),(1.1,0,0),(.5,.8,0),(.8,1.5,0)])]
for basis,points in examples:
    curve=UsdGeom.BasisCurves.Define(stage,'/World/'+basis)
    curve.CreateTypeAttr('linear' if basis=='linear' else 'cubic');curve.CreateBasisAttr('bezier' if basis=='linear' else basis);curve.CreateWrapAttr('nonperiodic')
    curve.CreatePointsAttr(points);curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreateWidthsAttr([.08]);curve.SetWidthsInterpolation('constant')
    if basis=='linear':UsdGeom.PrimvarsAPI(curve).CreatePrimvar('radius',Sdf.ValueTypeNames.FloatArray,'vertex').Set([.02,.08])
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
settings=UsdRender.Settings.Define(stage,'/Render/Lava');settings.CreateCameraRel().SetTargets([camera.GetPath()]);settings.GetPrim().CreateAttribute('lava:visibility',Sdf.ValueTypeNames.String,custom=True).Set('raster')
stage.GetRootLayer().Save();scene=extract(stage,'/World/Camera',1)
assert not scene['meshes'] and len(scene['curves'])==4
by_basis={c['basis']:c for c in scene['curves']}
assert by_basis['linear']['radius_interpolation']=='vertex' and max(abs(a-b) for a,b in zip(by_basis['linear']['radius'],[.02,.08]))<1e-6
assert abs(by_basis['bezier']['radius'][0]-.04)<1e-6
assert all(c['material']['openpbr']['emission_luminance']==2 for c in scene['curves'])
path.with_suffix('.json').write_text(json.dumps(scene,indent=2))
print('PASS: linear, Bezier, Catmull-Rom, B-spline, custom radius override, USD widths, OpenPBR binding and JSON export')
