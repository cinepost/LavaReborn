#!/usr/bin/env hython
"""Verify Houdini $HIP and printf UDIM paths without modifying a user HIP."""
import ctypes,json,shutil,sys
from pathlib import Path
import hou
from pxr import Usd,UsdShade,Sdf
from render_usd import extract
root=Path(sys.argv[1]).resolve()
hou.hipFile.setName(str(root/'udim-path-test.hip'))
module=ctypes.CDLL(str(root/'liblava_houdini_smoke.so'))
module.lava_houdini_texture_path.argtypes=[ctypes.c_char_p];module.lava_houdini_texture_path.restype=ctypes.c_char_p
raw='$HIP/Library/Assets/bar_island_sink/Textures/bar_island_sink_original/bar_island_sink_%(UDIM)d_Roughness.exr'
expected=str(root/'Library/Assets/bar_island_sink/Textures/bar_island_sink_original/bar_island_sink_<UDIM>_Roughness.exr')
actual=module.lava_houdini_texture_path(raw.encode()).decode();assert actual==expected,(actual,expected)
assert module.lava_houdini_texture_path(raw.replace('$HIP','${HIP}').encode()).decode()==expected
print('PASS: native Houdini globals expand exact $HIP + %(UDIM)d roughness path')
# Verify Principled's authored path and standalone lowering preserve the pattern.
lib=hou.node('/stage').createNode('materiallibrary');shader=lib.createNode('principledshader::2.0','path_test')
shader.parm('rough_useTexture').set(1);shader.parm('rough_texture').set(raw);shader.parm('rough').set(.4);lib.parm('matnode1').set('*')
principled_stage=lib.stage();material=UsdShade.Material(principled_stage.GetPrimAtPath('/materials/path_test'))
from render_usd import principled_textures
binding=principled_textures(material,Usd.TimeCode.Default())['specular_roughness'];assert binding['file']==expected,binding
print('PASS: Principled roughness $HIP + printf UDIM binding lowers correctly')
# A renderable USD fixture retains a literal $HIP/printf path to exercise HUSD.
stage=Usd.Stage.Open(str(root/'material-udim.usda'));stage.SetEditTarget(stage.GetSessionLayer())
for file in (root/'udim').glob('albedo.*.png'):
 number=file.name.split('.')[1];shutil.copy2(file,root/'udim'/f'albedo_{number}.png')
for prim in stage.Traverse():
 shader=UsdShade.Shader(prim)
 if shader and shader.GetIdAttr().Get()=='UsdUVTexture':shader.GetInput('file').Set(Sdf.AssetPath('$HIP/udim/albedo_%(UDIM)d.png'))
output=root/'material-udim-printf.usda';stage.Export(str(output))
scene=extract(stage,'/World/Camera',1)
for mesh in scene['meshes']:assert next(iter(mesh['material']['textures'].values()))['file']==str(root/'udim/albedo_<UDIM>.png')
output.with_suffix('.json').write_text(json.dumps(scene,indent=2))
module.lava_houdini_smoke.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
code=module.lava_houdini_smoke(str(output).encode(),b'/World/Camera');assert code==0,code
print('PASS: raw $HIP/printf UDIM paths render through Solaris')
