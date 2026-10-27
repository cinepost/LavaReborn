#!/usr/bin/env hython
"""Exercise USD normal interpolation, indexing, transforms and JSON lowering."""
import sys
from pathlib import Path
from pxr import Usd, UsdGeom, UsdLux, UsdRender, Gf, Sdf
from render_usd import extract
output=Path(sys.argv[1]).resolve()
stage=Usd.Stage.CreateNew(str(output))
camera=UsdGeom.Camera.Define(stage,'/World/Camera')
camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
camera.CreateHorizontalApertureAttr(140.)
camera.CreateVerticalApertureAttr(78.75)
UsdGeom.Xformable(camera).AddTranslateOp().Set((0,0,5))
light=UsdLux.DistantLight.Define(stage,'/World/Light');light.CreateIntensityAttr(3.14159265)
rates=[UsdGeom.Tokens.constant,UsdGeom.Tokens.uniform,UsdGeom.Tokens.vertex,UsdGeom.Tokens.varying,UsdGeom.Tokens.faceVarying,'indexed','primvar','mirrored','leftHanded','missing']
for i,rate in enumerate(rates):
 mesh=UsdGeom.Mesh.Define(stage,'/World/'+rate)
 mesh.CreatePointsAttr([(-.4,-.4,0),(.4,-.4,0),(.4,.4,0),(-.4,.4,0)])
 mesh.CreateFaceVertexCountsAttr([3,3]);mesh.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
 mesh.CreateSubdivisionSchemeAttr(sys.argv[2] if len(sys.argv)>2 else UsdGeom.Tokens.none)
 xform=UsdGeom.Xformable(mesh)
 xform.AddTranslateOp().Set(((i%5-2)*2.4,1.2-(i//5)*2.4,0))
 xform.AddScaleOp().Set((-2 if rate=='mirrored' else 2,1,.5))
 mesh.CreateDisplayColorAttr([(0.8,0.8,0.8)])
 if rate=='missing':continue
 if rate=='leftHanded':mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
 if rate in ('indexed','primvar'):
  mesh.CreateNormalsAttr([(0,1,0)]*4);mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
  pv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('normals',Sdf.ValueTypeNames.Normal3fArray,UsdGeom.Tokens.faceVarying)
  pv.Set([(8,0,1),(0,0,1)]);pv.SetIndices([0,0,0,1,1,1] if rate=='indexed' else [0]*6)
 else:
  interp=UsdGeom.Tokens.vertex if rate in ('mirrored','leftHanded') else rate
  count={UsdGeom.Tokens.constant:1,UsdGeom.Tokens.uniform:2,UsdGeom.Tokens.vertex:4,UsdGeom.Tokens.varying:4,UsdGeom.Tokens.faceVarying:6}[interp]
  mesh.CreateNormalsAttr([(8,0,1)]*count);mesh.SetNormalsInterpolation(interp)
settings=UsdRender.Settings.Define(stage,'/Render/Lava')
settings.CreateCameraRel().SetTargets([camera.GetPath()])
settings.GetPrim().CreateAttribute('lava:visibility',Sdf.ValueTypeNames.String,custom=True).Set('raster')
stage.GetRootLayer().Save()
scene=extract(stage,'/World/Camera',1)
assert len(scene['meshes'])==len(rates)
expected=Gf.Vec3d(4,0,2).GetNormalized()
for rate,mesh in zip(rates,scene['meshes']):
 if rate=='missing':assert not mesh.get('normals');continue
 assert abs(mesh['normals'][0][0]-(-expected[0] if rate=='mirrored' else expected[0]))<1e-6
 assert abs(mesh['normals'][0][2]-expected[2])<1e-6
 if rate in ('constant','uniform','faceVarying','indexed','primvar'):assert len(mesh['normal_indices'])==6
 assert mesh['reverse_orientation']==(rate in ('mirrored','leftHanded'))
# The lowered scene must remain valid finite JSON.
import json
output.with_suffix('.json').write_text(json.dumps(scene,allow_nan=False,indent=2))
print('PASS: USD authored normals, all interpolation rates, indexed primvar precedence, inverse transpose, mirrored/left-handed orientation and absent fallback')
