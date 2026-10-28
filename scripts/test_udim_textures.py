#!/usr/bin/env hython
"""Author native PreviewSurface/OpenPBR UDIM bindings with layer-relative paths."""
from pathlib import Path
import runpy,sys,struct,zlib,json
from pxr import Usd,UsdShade,UsdGeom,Sdf
from render_usd import extract
runpy.run_path(str(Path(__file__).with_name('test_textures.py')),run_name='__main__')
output=Path(sys.argv[1]).resolve();directory=output.parent/'udim';directory.mkdir(exist_ok=True)
def png(path,width,height,color,gray=False):
 def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
 rows=(b'\0'+bytes(color)*width)*height
 path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,0 if gray else 6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))
png(directory/'albedo.1001.png',8,4,[255,0,0,255]);png(directory/'albedo.1002.png',5,7,[0,0,255,255]);png(directory/'albedo.1011.png',7,3,[128],True)
stage=Usd.Stage.Open(str(output))
for prim in stage.Traverse():
 shader=UsdShade.Shader(prim)
 if shader and shader.GetIdAttr().Get()=='UsdUVTexture':shader.GetInput('file').Set(Sdf.AssetPath('udim/albedo.<UDIM>.png'))
 mesh=UsdGeom.Mesh(prim)
 if mesh:UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Set([(0,0),(2,0),(2,.9),(0,.9)])
stage.GetRootLayer().Save()
scene=extract(stage,'/World/Camera',1)
for mesh in scene['meshes']:
 binding=next(iter(mesh['material']['textures'].values()))
 assert binding['file']==str(directory/'albedo.<UDIM>.png'),binding
output.with_suffix('.json').write_text(json.dumps(scene,indent=2))
print('PASS: layer-relative PreviewSurface/OpenPBR UDIM patterns survive lowering')
