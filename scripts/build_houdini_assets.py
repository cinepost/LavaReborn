#!/usr/bin/env hython
"""Build Lava Solaris HDAs with the installed target Houdini."""
from pathlib import Path
import argparse
import hou
from expose_hip_settings import COOK

# The editable stage belongs to the internal Python LOP; controls belong to the HDA.
RENDER_COOK = COOK.replace("node=hou.pwd()", "cook=hou.pwd()\nnode=cook.parent()").replace('stage=node.editableStage()', 'stage=cook.editableStage()')
LIGHT_COOK = '''from pxr import UsdGeom, UsdLux, Sdf
cook=hou.pwd()
node=cook.parent()
rule=hou.LopSelectionRule()
rule.setPathPattern(node.evalParm('primpattern'))
paths=rule.expandedPaths(cook.inputs()[0])
stage=cook.editableStage()
for path in paths:
 prim=stage.GetPrimAtPath(path)
 area=any(prim.IsA(t) for t in (UsdLux.RectLight,UsdLux.DiskLight,UsdLux.CylinderLight))
 mesh=prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdLux.MeshLightAPI)
 if not (area or mesh):continue
 api=UsdGeom.PrimvarsAPI(prim)
 if area:api.CreatePrimvar('lava:singleSided',Sdf.ValueTypeNames.Bool,UsdGeom.Tokens.constant).Set(bool(node.evalParm('single_sided')))
 api.CreatePrimvar('lava:renderLightGeometry',Sdf.ValueTypeNames.Bool,UsdGeom.Tokens.constant).Set(bool(node.evalParm('render_light_geometry')))
 if mesh:api.CreatePrimvar('lava:geometryLight',Sdf.ValueTypeNames.Bool,UsdGeom.Tokens.constant).Set(True)
'''
def string(name,label,default,**kw):
 return hou.StringParmTemplate(name,label,1,default_value=(default,),**kw)
def integer(name,label,default,maximum,minimum=0,help=''):
 return hou.IntParmTemplate(name,label,1,default_value=(default,),min=minimum,max=maximum,min_is_strict=True,max_is_strict=True,help=help)
def folder(name,label,parms,kind=hou.folderType.Tabs):
 return hou.FolderParmTemplate(name,label,parm_templates=parms,folder_type=kind)
def asset(path,name,label,parms,code,help):
 n=hou.node('/stage').createNode('subnet')
 script=n.createNode('pythonscript','author_properties')
 script.setInput(0,n.indirectInputs()[0]);script.parm('python').set(code)
 n.node('output0').setInput(0,script)
 n=n.createDigitalAsset(name=name,hda_file_name=str(path),description=label,min_num_inputs=1,max_num_inputs=1)
 d=n.type().definition();d.setParmTemplateGroup(hou.ParmTemplateGroup(parms))
 d.addSection('Help',help)
 d.addSection('Tools.shelf','''<?xml version="1.0" encoding="UTF-8"?>
<shelfDocument><tool name="%s" label="%s"><toolMenuContext name="network"><contextNetType>LOP</contextNetType></toolMenuContext><toolSubmenu>Lava</toolSubmenu><script scriptType="python"><![CDATA[import loptoolutils
loptoolutils.genericTool(kwargs, '%s')]]></script></tool></shelfDocument>'''%(name,label,name))
 d.setIcon('LOP_renderproperties' if 'render' in name else 'LOP_light')
 n.destroy()
def build(output):
 output.mkdir(parents=True,exist_ok=True)
 render=output/'lava_render_settings.hda';light=output/'lava_light_properties.hda'
 for p in (render,light):
  if p.exists():p.unlink()
 prim=string('settings_path','RenderSettings Primitive Path','/Render/Lava');prim.setTags({'usdprimpath':'1'})
 camera=string('camera_path','Camera','/cameras/camera1',help='USD camera path. Configure lens and depth of field on the Camera LOP.');camera.setTags({'usdprimpath':'1'})
 asset(render,'lava::render_settings::1.0','Lava Render Settings',[
  folder('settings_prim','Render Settings Prim',[prim],hou.folderType.Collapsible),camera,
  hou.IntParmTemplate('resolution','Resolution',2,default_value=(1920,1080),min=1,max=16384,min_is_strict=True),
  folder('sampling','Sampling',[
   string('mode','Render Mode','interactive',menu_items=('interactive','final'),menu_labels=('Interactive','Final')),
   string('visibility','Visibility Generator','ray_query',menu_items=('ray_query','raster'),menu_labels=('Hardware Ray Queries','Hardware Rasterisation')),
   integer('samples','Samples',64,1048576,1,help='Interactive images update after each sample. Final rendering uses this fixed sample budget.'),
   hou.ToggleParmTemplate('headlight','Automatic Headlight',default_value=True)]),
  folder('limits','Limits',[integer('diffuse_depth','Diffuse Limit',3,32),integer('reflection_depth','Reflection Limit',4,32),integer('refraction_depth','Refraction Limit',8,32),integer('roulette_start','Russian Roulette Start Depth',3,96)]),
  folder('textures_tab','Textures',[string('textures','Texture Mode','original',menu_items=('original','compressed','virtual'),menu_labels=('Original','Compressed (BC4/BC7)','Virtual (inactive)'),help='Original decodes the source. Compressed uses cached BC4 for grayscale and BC7 for color. Virtual is reserved and currently leaves texture inputs inactive.')]),
  folder('image','Image',[hou.FloatParmTemplate('exposure','Exposure',1,default_value=(0,),min=-8,max=8,help='Display exposure in stops. +1 doubles brightness.')])
 ],RENDER_COOK,'Lava Render Settings\n\nConnect after the scene. Select /Render/Lava in the viewport Render Settings Prim selector. Only implemented renderer controls are exposed. Camera optics and depth of field are controlled by the Camera LOP.')
 asset(light,'lava::light_properties::1.0','Lava Light Properties',[
  string('primpattern','Primitives','/lights/*',help='Solaris selection pattern. Area lights and meshes with MeshLightAPI only.'),
  folder('light','Lava',[
   hou.ToggleParmTemplate('single_sided','Single Sided',default_value=True,help='Area lights only. Emit from the front side; disable for emission from both sides.'),
   hou.ToggleParmTemplate('render_light_geometry','Render Light Geometry',default_value=False,help='Area and mesh lights. Show emitting geometry to camera rays. Lighting and secondary rays remain active.')])
 ],LIGHT_COOK,'Lava Light Properties\n\nConnect after Light creation. Single Sided applies to area lights. Render Light Geometry applies to area lights and meshes carrying MeshLightAPI. Other primitives are untouched. Lava currently renders rectangular area lights; other area schemas and mesh-light next-event sampling are not implemented.')
 print('Built',render,light)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);build(p.parse_args().output.resolve())
