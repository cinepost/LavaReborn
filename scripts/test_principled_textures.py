#!/usr/bin/env hython
"""Real Houdini Principled texture regression; creates temporary USD, never edits HIPs."""
import json,sys,struct,zlib
from pathlib import Path
import hou
from pxr import Usd,UsdGeom,UsdShade,Sdf
from render_usd import extract
out=Path(sys.argv[1]).resolve()
lib=hou.node('/stage').createNode('materiallibrary')
m=lib.createNode('principledshader::2.0','principled')
def png(path,pixel):
 def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
 rows=b''.join(b'\0'+b''.join(bytes(pixel(x,y)) for x in range(16)) for y in range(16))
 path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',16,16,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))
color=out.parent/'principled-texture.png';normal=out.parent/'principled-normal.png';bump=out.parent/'principled-bump.png'
png(color,lambda x,y:(51+x*10,153,230,255));png(normal,lambda x,y:(204,128,230,255));png(bump,lambda x,y:(x*17,x*17,x*17,255))
texture=str(color)
for name,value in [('rough',.35),('metallic',.7),('basecolor_useTexture',1),('rough_useTexture',1),('metallic_useTexture',1),('rough_monoChannel',2),('metallic_monoChannel',3),('baseBumpAndNormal_enable',1),('baseNormal_scale',.4),('basecolor_texture',texture),('rough_texture',texture),('metallic_texture',texture),('baseNormal_texture',str(normal))]:m.parm(name).set(value)
m.parmTuple('basecolor').set((.4,.5,.6));lib.parm('matnode1').set('*')
for kind in ['normal','bump','luminance']:
 if kind=='bump':
  m.parm('baseBumpAndNormal_type').set('bump');m.parm('baseBump_bumpTexture').set(str(bump));m.parm('baseBump_bumpScale').set(.08)
 if kind=='luminance':m.parm('rough_monoChannel').set(0);m.parm('metallic_monoChannel').set(1)
 path=out.with_name(out.stem+'-'+kind+'.usda');lib.stage().Export(str(path));stage=Usd.Stage.Open(str(path))
 camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateProjectionAttr('orthographic');camera.CreateHorizontalApertureAttr(40);camera.CreateVerticalApertureAttr(22.5);UsdGeom.Xformable(camera).AddTranslateOp().Set((0,0,5))
 mesh=UsdGeom.Mesh.Define(stage,'/World/Plane');mesh.CreatePointsAttr([(-2,-2,0),(2,-2,0),(2,2,0),(-2,2,0)]);mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateSubdivisionSchemeAttr('none')
 UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set([(0,0),(1,0),(1,1),(0,1)])
 UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath('/materials/principled')));stage.GetRootLayer().Save()
 scene=extract(stage,'/World/Camera',1);b=scene['meshes'][0]['material']['textures']
 assert b['specular_roughness']['channel']==('luminance' if kind=='luminance' else 'g'),b
 assert b['base_metalness']['channel']==('r' if kind=='luminance' else 'b')
 assert abs(b['specular_roughness']['scale'][0]-.35)<1e-6
 assert abs(b['base_metalness']['scale'][2]-.7)<1e-6
 assert max(abs(a-c) for a,c in zip(b['base_color']['scale'][:3],[.4,.5,.6]))<1e-6
 if kind=='normal':assert abs(b['normal']['strength']-.4)<1e-6 and b['normal']['signed']
 else:assert abs(b['bump']['strength']-.08)<1e-6
 path.with_suffix('.json').write_text(json.dumps(scene,indent=2));print('PASS: Principled channels, multipliers and shading frame:',kind)

# After rendering these stages with LAVA_TRACE=1, optionally check the actual
# native import against the independently lowered bindings above.
if '--check-native-logs' in sys.argv:
 def matches(actual,expected):
  if isinstance(expected,dict):return all(k in actual and matches(actual[k],v) for k,v in expected.items())
  if isinstance(expected,list):return len(actual)==len(expected) and all(matches(a,b) for a,b in zip(actual,expected))
  if isinstance(expected,(int,float)):return abs(actual-expected)<1e-5
  return actual==expected
 for kind in ['normal','bump','luminance']:
  expected=json.loads(out.with_name(out.stem+'-'+kind+'.json').read_text())['meshes'][0]['material']['textures']
  log=(out.parent/('principled-native-'+kind+'.log')).read_text()
  imports=[json.loads(line.split(': ',1)[1]) for line in log.splitlines() if line.startswith('Lava textures (Principled): ')]
  for data in imports:
   for binding in data.values():binding.setdefault('signed',False)
  assert imports and matches(imports[0],expected),(kind,imports,expected)
  assert 'PASS: Vulkan external-memory/semaphore' in log
  if kind=='normal':assert 'PASS: live Principled channel edit synchronized' in log and imports[-1]['specular_roughness']['channel']=='b'
  print('PASS: native Principled bindings match authored controls:',kind)
