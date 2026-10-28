#!/usr/bin/env hython
"""Real Principled refraction controls: constants, texture toggles and native import."""
import json, os, struct, subprocess, sys, zlib
from pathlib import Path
import hou
from pxr import Usd, UsdGeom, UsdShade, Sdf
from render_usd import extract

root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
texture=root/'refraction.png'
def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
texture.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',4,4,8,6,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+bytes([64,128,192,255])*4)*4))+chunk(b'IEND',b''))
lib=hou.node('/stage').createNode('materiallibrary')
m=lib.createNode('principledshader::2.0','principled');lib.parm('matnode1').set('*')
m.parm('rough').set(.1);m.parm('transparency').set(.6)
m.parmTuple('transcolor').set((.3,.5,.8))
cases=[('constant',False,str(texture),False,str(texture),0,.6),
       ('textured',True,str(texture),True,str(texture),0,.6),
       ('green',True,str(texture),True,str(texture),2,.6),
       ('opaque',True,str(texture),True,str(texture),0,0.),
       ('empty',True,'',True,'',0,.6)]
for name,amount_on,amount_file,tint_on,tint_file,channel,amount in cases:
    for parm,value in [('transparency',amount),('transparency_useTexture',amount_on),('transparency_texture',amount_file),('transparency_monoChannel',channel),('transcolor_useTexture',tint_on),('transcolor_texture',tint_file),('transcolor_textureColorSpace','linear')]:m.parm(parm).set(value)
    path=root/(name+'.usda');lib.stage().Export(str(path));stage=Usd.Stage.Open(str(path))
    camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateProjectionAttr('orthographic');camera.CreateHorizontalApertureAttr(40);camera.CreateVerticalApertureAttr(22.5);UsdGeom.Xformable(camera).AddTranslateOp().Set((0,0,5))
    mesh=UsdGeom.Mesh.Define(stage,'/World/Plane');mesh.CreatePointsAttr([(-2,-2,0),(2,-2,0),(2,2,0),(-2,2,0)]);mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set([(0,0),(1,0),(1,1),(0,1)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath('/materials/principled')));stage.GetRootLayer().Save()
    material=extract(stage,'/World/Camera',1)['meshes'][0]['material'];surface=material['openpbr'];bindings=material.get('textures',{})
    assert abs(surface['transmission_weight']-amount)<1e-6,material
    assert max(abs(a-b) for a,b in zip(surface['transmission_color'],[.3,.5,.8]))<1e-6
    assert surface['geometry_opacity']==1.,surface
    for field,active in [('transmission_weight',amount_on and bool(amount_file)),('transmission_color',tint_on and bool(tint_file))]:assert (field in bindings)==active,(name,field,bindings)
    if 'transmission_weight' in bindings:
        assert bindings['transmission_weight']['channel']==('g' if channel==2 else 'luminance')
        assert all(abs(x-amount)<1e-6 for x in bindings['transmission_weight']['scale'])
    if 'transmission_color' in bindings:
        assert bindings['transmission_color']['channel']=='rgb'
        assert max(abs(a-b) for a,b in zip(bindings['transmission_color']['scale'],[.3,.5,.8,1]))<1e-6
    if '--native' in sys.argv:
        env=dict(os.environ,LAVA_TRACE='1',LAVA_TEST_SAMPLES='4',LAVA_TEST_WIDTH='64',LAVA_TEST_HEIGHT='36')
        result=subprocess.run([str(Path(hou.getenv('HFS'))/'bin/hython'),str(Path(__file__).with_name('test_solaris.py')),str(path),'/World/Camera'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (root/(name+'.log')).write_text(result.stdout)
        assert result.returncode==0,result.stdout
        imports=[json.loads(line.split(': ',1)[1]) for line in result.stdout.splitlines() if line.startswith('Lava textures (Principled): ')]
        assert imports,(name,result.stdout)
        surfaces=[json.loads(line.split(': ',1)[1]) for line in result.stdout.splitlines() if line.startswith('Lava OpenPBR (Principled): ')]
        assert surfaces,(name,result.stdout)
        assert abs(surfaces[0]['transmission_weight']-amount)<1e-6
        assert max(abs(a-b) for a,b in zip(surfaces[0]['transmission_color'],[.3,.5,.8]))<1e-6
        assert surfaces[0]['geometry_opacity']==1.
        for field in ['transmission_weight','transmission_color']:
            assert (field in imports[0])==(field in bindings),(name,imports,bindings)
            if field in bindings:
                assert imports[0][field]['channel']==bindings[field]['channel']
                assert max(abs(a-b) for a,b in zip(imports[0][field]['scale'],bindings[field]['scale']))<1e-6
    print('PASS: Principled refraction amount, tint, opacity and bindings:',name)
