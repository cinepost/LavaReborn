#!/usr/bin/env hython
"""Generate a native USD texture regression scene and verify material lowering."""
from pathlib import Path
import json, struct, sys, zlib
from pxr import Usd, UsdGeom, UsdShade, UsdRender, Gf, Sdf
from render_usd import extract
output=Path(sys.argv[1]).resolve();output.parent.mkdir(parents=True,exist_ok=True)
texture=output.parent/'material-texture.png'
def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
rows=b''.join(b'\0'+b''.join(bytes((255,0,0,255)) if x<4 else bytes((0,0,255,255)) for x in range(8)) for y in range(4))
texture.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',8,4,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))
stage=Usd.Stage.CreateNew(str(output))
camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic);camera.CreateHorizontalApertureAttr(40.);camera.CreateVerticalApertureAttr(22.5);UsdGeom.Xformable(camera).AddTranslateOp().Set((0,0,5))
for name,center,openpbr in [('Preview',-.95,False),('OpenPBR',.95,True)]:
 mesh=UsdGeom.Mesh.Define(stage,'/World/'+name)
 mesh.CreatePointsAttr([(center-.85,-.85,0),(center+.85,-.85,0),(center+.85,.85,0),(center-.85,.85,0)])
 mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateSubdivisionSchemeAttr('none')
 UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([(0,0),(1,0),(1,1),(0,1)])
 material=UsdShade.Material.Define(stage,'/Materials/'+name)
 reader=UsdShade.Shader.Define(stage,material.GetPath().AppendChild('uv'));reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.String).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
 image=UsdShade.Shader.Define(stage,material.GetPath().AppendChild('image'));image.CreateIdAttr('UsdUVTexture');image.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture.name));image.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw');image.CreateInput('lava:compressionQuality',Sdf.ValueTypeNames.String).Set('high' if openpbr else 'fast');image.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result');image.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
 shader=UsdShade.Shader.Define(stage,material.GetPath().AppendChild('surface'))
 shader.CreateIdAttr('OpenPBRSurface' if openpbr else 'UsdPreviewSurface')
 if openpbr:
  for field,value in [('base_weight',0),('specular_weight',0),('emission_luminance',1)]:shader.CreateInput(field,Sdf.ValueTypeNames.Float).Set(value)
  shader.CreateInput('emission_color',Sdf.ValueTypeNames.Color3f).Set((.1,.2,.4))
 shader.CreateInput('emission_color' if openpbr else 'diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(),'rgb')
 shader.CreateOutput('surface',Sdf.ValueTypeNames.Token);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
settings=UsdRender.Settings.Define(stage,'/Render/Lava');settings.CreateCameraRel().SetTargets([camera.GetPath()]);settings.CreateResolutionAttr().Set(Gf.Vec2i(320,180))
for name,value in [('textures','compressed'),('visibility','raster')]:settings.GetPrim().CreateAttribute('lava:'+name,Sdf.ValueTypeNames.String,custom=True).Set(value)
stage.GetRootLayer().Save()
scene=extract(stage,'/World/Camera',1)
assert len(scene['meshes'])==2
for m in scene['meshes']:
 assert m['material']['textures']
 binding=next(iter(m['material']['textures'].values()))
 assert binding['file']==str(texture) and binding['color_space']=='raw'
 assert binding['compression']['quality'] in ('fast','high')
 assert len(m['texcoords'])==4
output.with_suffix('.json').write_text(json.dumps(scene,indent=2))
print('PASS: authored connected PreviewSurface/OpenPBR textures, mesh UVs and texture mode settings')
