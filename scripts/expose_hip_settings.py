#!/usr/bin/env python3
"""Prepare saved Lava controls in a HIP, using installed Houdini's hython."""
import argparse,json,hashlib
from pathlib import Path
import hou
from pxr import UsdGeom
COOK = '''from pxr import UsdRender, Sdf, Gf
node=hou.pwd()
path=node.evalParm('settings_path')
camera=node.evalParm('camera_path')
resolution=node.evalParmTuple('resolution')
values={
 'lava:visibility':(Sdf.ValueTypeNames.String,node.evalParm('visibility')),
 'lava:textures':(Sdf.ValueTypeNames.String,node.evalParm('textures') if node.parm('textures') else 'original'),
 'lava:mode':(Sdf.ValueTypeNames.String,node.evalParm('mode')),
 'lava:samples':(Sdf.ValueTypeNames.Int,node.evalParm('samples')),
 'lava:exposure':(Sdf.ValueTypeNames.Float,node.evalParm('exposure')),
 'lava:headlight':(Sdf.ValueTypeNames.Bool,bool(node.evalParm('headlight'))),
 'lava:diffuseDepth':(Sdf.ValueTypeNames.Int,node.evalParm('diffuse_depth')),
 'lava:reflectionDepth':(Sdf.ValueTypeNames.Int,node.evalParm('reflection_depth')),
 'lava:refractionDepth':(Sdf.ValueTypeNames.Int,node.evalParm('refraction_depth')),
 'lava:rouletteStart':(Sdf.ValueTypeNames.Int,node.evalParm('roulette_start')),
}
stage=node.editableStage()
settings=UsdRender.Settings.Define(stage,path)
settings.CreateCameraRel().SetTargets([Sdf.Path(camera)])
settings.CreateResolutionAttr().Set(Gf.Vec2i(*resolution))
settings.CreatePixelAspectRatioAttr().Set(1.)
for name,(kind,value) in values.items():
 settings.GetPrim().CreateAttribute(name,kind,custom=True).Set(value)
'''
def build(args):
    source_digest=hashlib.sha256(args.input.read_bytes()).hexdigest()
    hou.hipFile.load(str(args.input),suppress_save_prompt=True,ignore_load_warnings=True)
    network=hou.node('/stage');upstream=network.displayNode()
    if upstream is None:raise RuntimeError('No displayed Solaris node')
    if network.node('LAVA_RENDER_SETTINGS'):raise RuntimeError('Lava controls already exist')
    source=upstream.stage()
    before=sum(p.IsA(UsdGeom.Mesh) for p in source.Traverse())
    cameras=[str(p.GetPath()) for p in source.Traverse() if p.IsA(UsdGeom.Camera)]
    if not cameras:raise RuntimeError('No camera in displayed scene')
    camera='/cameras/camera1' if '/cameras/camera1' in cameras else cameras[0]
    n=network.createNode('pythonscript','LAVA_RENDER_SETTINGS');n.setInput(0,upstream)
    n.setPosition(upstream.position()+hou.Vector2(0,-2));n.setColor(hou.Color((.95,.4,.12)))
    n.setComment('Select this node to edit Lava settings.\nUSD settings: /Render/Lava\nIn the viewport, choose /Render/Lava under Render Settings Prim if local overrides are active.')
    n.setGenericFlag(hou.nodeFlag.DisplayComment,True)
    group=n.parmTemplateGroup()
    for name in ('python','maintainstate'):
        p=group.find(name);p.hide(True);group.replace(name,p)
    folder=hou.FolderParmTemplate('lava_controls','Lava Render Settings')
    folder.addParmTemplate(hou.StringParmTemplate('visibility','Visibility Generator',1,default_value=('ray_query',),menu_items=('ray_query','raster'),menu_labels=('Hardware Ray Queries','Hardware Rasterisation')))
    folder.addParmTemplate(hou.StringParmTemplate('textures','Textures',1,default_value=('original',),menu_items=('original','compressed','virtual'),menu_labels=('Original','Compressed (BC4/BC7)','Virtual (inactive)')))
    folder.addParmTemplate(hou.StringParmTemplate('mode','Render Mode',1,default_value=('interactive',),menu_items=('interactive','final'),menu_labels=('Interactive — viewport','Final — offline frame')))
    folder.addParmTemplate(hou.IntParmTemplate('samples','Samples',1,default_value=(64,),min=1,max=1024,min_is_strict=True,help='Target after movement stops. Interactive images update before convergence.'))
    folder.addParmTemplate(hou.FloatParmTemplate('exposure','Exposure (stops)',1,default_value=(0.,),min=-8,max=8,help='+1 doubles image brightness.'))
    folder.addParmTemplate(hou.ToggleParmTemplate('headlight','Automatic Headlight',default_value=True,help='Interactive headlight when the scene has no supported lights.'))
    depths=hou.FolderParmTemplate('path_depths','Path Depth Limits',folder_type=hou.folderType.Simple)
    for name,label,default in [('diffuse_depth','Diffuse Bounces',3),('reflection_depth','Reflection Bounces',4),('refraction_depth','Refraction / Glass Bounces',8)]:
        depths.addParmTemplate(hou.IntParmTemplate(name,label,1,default_value=(default,),min=0,max=32,min_is_strict=True,max_is_strict=True,help='Independent scattering limit; zero disables this lobe.'))
    depths.addParmTemplate(hou.IntParmTemplate('roulette_start','Russian Roulette Start Depth',1,default_value=(3,),min=0,max=96,min_is_strict=True,max_is_strict=True,help='Start after this many total bounces. Set above the sum of limits to disable roulette.'))
    folder.addParmTemplate(depths)
    output=hou.FolderParmTemplate('scene_output','Camera and Output',folder_type=hou.folderType.Simple)
    output.addParmTemplate(hou.StringParmTemplate('camera_path','USD Camera',1,default_value=(camera,)))
    output.addParmTemplate(hou.IntParmTemplate('resolution','Output Resolution',2,default_value=(1920,1080),min=1,max=8192,min_is_strict=True))
    output.addParmTemplate(hou.StringParmTemplate('settings_path','USD Settings Primitive',1,default_value=('/Render/Lava',)))
    folder.addParmTemplate(output);group.append(folder);n.setParmTemplateGroup(group)
    n.parm('python').set(COOK);n.setDisplayFlag(True);n.setSelected(True,clear_all_selected=True);n.setCurrent(True,clear_all_selected=True)
    stage=n.stage()
    if n.errors():raise RuntimeError(n.errors())
    after=sum(p.IsA(UsdGeom.Mesh) for p in stage.Traverse())
    assert before==after and after>0
    assert stage.GetPrimAtPath('/Render/Lava').GetTypeName()=='RenderSettings'
    assert stage.GetPrimAtPath(camera)
    n.parm('samples').set(7);n.parm('diffuse_depth').set(2)
    stage=n.stage();settings=stage.GetPrimAtPath('/Render/Lava')
    assert settings.GetAttribute('lava:samples').Get()==7
    assert settings.GetAttribute('lava:diffuseDepth').Get()==2
    if args.test_export:stage.Export(str(args.test_export))
    n.parm('samples').set(64);n.parm('diffuse_depth').set(3)
    assert n.stage().GetPrimAtPath('/Render/Lava').GetAttribute('lava:samples').Get()==64
    hou.hipFile.save(str(args.output),save_to_recent_files=False)
    report=dict(source_sha256=source_digest,input=str(args.input),prepared=str(args.output),node=n.path(),upstream=upstream.path(),settings_prim='/Render/Lava',camera=camera,meshes_before=before,meshes_after=after,controls_verified=True)
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--report',type=Path);p.add_argument('--test-export',type=Path)
    build(p.parse_args())
