#!/usr/bin/env hython
"""Check USD light export and create native Solaris animated light test scenes."""
import sys
from pathlib import Path
from pxr import Usd, UsdGeom, UsdLux, UsdShade, UsdRender, Sdf, Gf
from render_usd import extract
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
def camera(stage):
    cam=UsdGeom.Camera.Define(stage,'/World/Camera')
    cam.CreateProjectionAttr('orthographic');cam.CreateHorizontalApertureAttr(20);cam.CreateVerticalApertureAttr(11.25)
    cam.AddTranslateOp().Set(Gf.Vec3d(0,0,.5));cam.CreateClippingRangeAttr(Gf.Vec2f(.001,100))
    settings=UsdRender.Settings.Define(stage,'/Render/Lava');settings.CreateCameraRel().SetTargets([cam.GetPath()])
    for name,typ,value in [('mode',Sdf.ValueTypeNames.String,'interactive'),('samples',Sdf.ValueTypeNames.Int,1),('headlight',Sdf.ValueTypeNames.Bool,False),('visibility',Sdf.ValueTypeNames.String,'ray_query'),('reflectionDepth',Sdf.ValueTypeNames.Int,1)]:
        settings.GetPrim().CreateAttribute('lava:'+name,typ,custom=True).Set(value)
def quad(stage,path,z,metal=False,emissive=False):
    mesh=UsdGeom.Mesh.Define(stage,path);mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreatePointsAttr([(-3,-3,z),(3,-3,z),(3,3,z),(-3,3,z)])
    mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,3,2,1] if emissive else [0,1,2,3])
    mat=UsdShade.Material.Define(stage,path+'Material');shader=UsdShade.Shader.Define(stage,path+'Material/Shader')
    shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set((.8,.8,.8))
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(metal));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(0. if metal else .4)
    if emissive:shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set((3.,3.,3.))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh
stage=Usd.Stage.CreateInMemory();camera(stage);quad(stage,'/World/Floor',0)
for name,schema in [('Point',UsdLux.SphereLight),('Spot',UsdLux.SphereLight),('Area',UsdLux.RectLight),('Sun',UsdLux.DistantLight)]:
    light=schema.Define(stage,'/World/'+name);api=UsdLux.LightAPI(light.GetPrim())
    api.CreateDiffuseAttr(.25);api.CreateSpecularAttr(1.75)
    if name=='Spot':UsdLux.ShapingAPI.Apply(light.GetPrim()).CreateShapingConeAngleAttr(30)
    if name=='Sun':light.CreateAngleAttr(9.)
mesh=quad(stage,'/World/Emitter',1,emissive=True);UsdLux.MeshLightAPI.Apply(mesh.GetPrim())
api=UsdLux.LightAPI(mesh.GetPrim());api.CreateDiffuseAttr(.3);api.CreateSpecularAttr(1.4)
result=extract(stage,'/World/Camera',1)
assert len(result['lights'])==4
for light in result['lights']:assert light['diffuse']==.25 and light['specular']==1.75
sun=next(l for l in result['lights'] if l['type']=='directional');assert sun['angle']==9.
emitter=next(m for m in result['meshes'] if m['geometry_light']);assert abs(emitter['light_response'][0]-.3)<1e-6 and abs(emitter['light_response'][1]-1.4)<1e-6
for kind in ['sun','mesh']:
    path=out/(kind+'.usda')
    if path.exists():path.unlink()
    stage=Usd.Stage.CreateNew(str(path.resolve()));camera(stage);quad(stage,'/World/Floor',0,metal=kind=='mesh')
    if kind=='sun':
        sun=UsdLux.DistantLight.Define(stage,'/World/Sun');sun.CreateAngleAttr(20.)
        api=UsdLux.LightAPI(sun.GetPrim());api.CreateIntensityAttr(3.);api.CreateSpecularAttr(0.)
        attr=api.CreateDiffuseAttr()
    else:
        mesh=quad(stage,'/World/Emitter',1,emissive=True);UsdLux.MeshLightAPI.Apply(mesh.GetPrim())
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('lava:geometryLight',Sdf.ValueTypeNames.Bool).Set(True)
        api=UsdLux.LightAPI(mesh.GetPrim());api.CreateDiffuseAttr(0.)
        attr=api.CreateSpecularAttr()
    for frame,gain in [(1,0.),(2,1.),(3,2.)]:attr.Set(gain,frame)
    stage.GetRootLayer().Save()
print('PASS: USD angle, analytic/mesh diffuse/specular translation; native fixtures created')

# Solaris reads renderer settings as constant opinions; animate the light response
# to verify accumulation resets with adaptive sampling enabled.
stage=Usd.Stage.Open(str((out/'sun.usda').resolve()))
prim=stage.GetPrimAtPath('/Render/Lava')
prim.GetAttribute('lava:samples').Set(64)
for name,typ,value in [('adaptiveSampling',Sdf.ValueTypeNames.Bool,True),('adaptiveThreshold',Sdf.ValueTypeNames.Float,.03),('adaptiveMinSamples',Sdf.ValueTypeNames.Int,16)]:
    prim.CreateAttribute('lava:'+name,typ,custom=True).Set(value)
stage.GetRootLayer().Export(str((out/'sun_adaptive.usda').resolve()))
