#!/usr/bin/env hython
"""Verify installed native node interfaces and their authored USD opinions."""
from pathlib import Path
import sys
import hou
from pxr import UsdGeom, UsdLux, Sdf
from render_usd import extract
root=Path(sys.argv[1]).resolve()
for path in root.glob('*.hda'):hou.hda.installFile(str(path))
stage=hou.node('/stage')
source=stage.createNode('pythonscript','source')
source.parm('python').set('''from pxr import UsdGeom, UsdLux
s=hou.pwd().editableStage()
UsdGeom.Camera.Define(s,'/cameras/camera1')
UsdLux.RectLight.Define(s,'/lights/rect')
UsdLux.SphereLight.Define(s,'/lights/point')
m=UsdGeom.Mesh.Define(s,'/lights/mesh')
m.CreatePointsAttr([(-1,-1,-3),(1,-1,-3),(0,1,-3)])
m.CreateFaceVertexCountsAttr([3])
m.CreateFaceVertexIndicesAttr([0,1,2])
UsdLux.MeshLightAPI.Apply(m.GetPrim())
UsdGeom.Mesh.Define(s,'/World/ordinary')
''')
r=stage.createNode('lava::render_settings::1.0');r.setInput(0,source)
l=stage.createNode('lava::light_properties::1.0');l.setInput(0,r)
s=l.stage();assert not l.errors(),l.errors()
p=s.GetPrimAtPath('/Render/Lava')
assert p.GetTypeName()=='RenderSettings'
assert p.GetAttribute('lava:samples').Get()==64
samples_template=r.parm('samples').parmTemplate()
assert samples_template.minValue()==1 and samples_template.maxValue()==1024
assert samples_template.minIsStrict() and samples_template.maxIsStrict()
r.parm('samples').set(1024)
assert r.stage().GetPrimAtPath('/Render/Lava').GetAttribute('lava:samples').Get()==1024
r.parm('samples').set(64)
s=l.stage();p=s.GetPrimAtPath('/Render/Lava')
assert p.GetAttribute('lava:visibility').Get()=='ray_query'
assert p.GetAttribute('lava:textures').Get()=='original'
assert p.GetAttribute('lava:uploadChunkMiB').Get()==16
assert r.parm('upload_chunk_mib').parmTemplate().menuItems()==('16','32','64')
assert r.parm('upload_chunk_mib').eval()=='16'
assert p.GetAttribute('lava:reduceAnalyticCaustics').Get() is False
assert r.parm('reduce_analytic_caustics').eval()==0
assert r.parm('textures').parmTemplate().menuItems()==('original','compressed','virtual')
assert p.GetRelationship('camera').GetTargets()==[Sdf.Path('/cameras/camera1')]
def pv(path,name):return UsdGeom.PrimvarsAPI(s.GetPrimAtPath(path)).GetPrimvar('lava:'+name)
assert pv('/lights/rect','singleSided').Get() is True
assert pv('/lights/rect','renderLightGeometry').Get() is False
assert pv('/lights/mesh','renderLightGeometry').Get() is False
assert not pv('/lights/mesh','singleSided')
assert not pv('/lights/point','renderLightGeometry')
assert not pv('/World/ordinary','renderLightGeometry')
for size in (16,32,64):
 r.parm('upload_chunk_mib').set(str(size))
 cooked=r.stage()
 assert cooked.GetPrimAtPath('/Render/Lava').GetAttribute('lava:uploadChunkMiB').Get()==size
 assert extract(cooked,'/cameras/camera1',1)['settings']['upload_chunk_mib']==size
l.parm('single_sided').set(False);l.parm('render_light_geometry').set(True)
r.parm('textures').set('compressed')
r.parm('reduce_analytic_caustics').set(True)
r.parm('samples').set(17);r.parm('diffuse_depth').set(1);r.parm('visibility').set('raster')
s=l.stage();assert not l.errors(),l.errors()
assert pv('/lights/rect','singleSided').Get() is False
assert pv('/lights/mesh','renderLightGeometry').Get() is True
assert s.GetPrimAtPath('/Render/Lava').GetAttribute('lava:samples').Get()==17
assert s.GetPrimAtPath('/Render/Lava').GetAttribute('lava:textures').Get()=='compressed'
assert s.GetPrimAtPath('/Render/Lava').GetAttribute('lava:reduceAnalyticCaustics').Get() is True
assert s.GetPrimAtPath('/Render/Lava').GetAttribute('lava:visibility').Get()=='raster'
assert [p.label() for p in r.parmTemplateGroup().entries() if isinstance(p,hou.FolderParmTemplate)]==['Render Settings Prim','Sampling','Limits','Textures','Image','Advanced']
s.Export(str(root.parent/'lava_light_properties.usda'))
# Serialize/reload proves the nodes cook as locked reusable assets.
l.setDisplayFlag(True)
hou.hipFile.save(str(root.parent/'lava_nodes_test.hip'),save_to_recent_files=False)
light_path=l.path()
hou.hipFile.load(str(root.parent/'lava_nodes_test.hip'),suppress_save_prompt=True)
assert hou.node(light_path).stage().GetPrimAtPath('/Render/Lava').GetAttribute('lava:samples').Get()==17
assert hou.node(light_path).stage().GetPrimAtPath('/Render/Lava').GetAttribute('lava:uploadChunkMiB').Get()==64
print('PASS: native Lava assets, tabs, defaults, supported-light filtering, edits, USD authoring and HIP reload')
