#!/usr/bin/env python3
"""Rebuild the native OpenPBR JSON/USD material showcase (no external assets)."""
import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=json.loads((root/'examples/transport.json').read_text())
s['camera']['transform'][14]=7.5
s['meshes']=s['meshes'][:2]
for m in s['meshes']:m['material']={'openpbr':{'base_color':m['color'],'specular_roughness':.65}}
materials=[
 {'base_color':[.65,.035,.015],'specular_roughness':.25,'coat_weight':1.,'coat_roughness':.12},
 {'base_color':[.9,.65,.28],'base_metalness':1.,'specular_roughness':.28,'specular_roughness_anisotropy':.65},
 {'base_color':[1,1,1],'transmission_weight':1.,'specular_roughness':.08,'specular_ior':1.5},
 {'base_color':[.025,.12,.25],'specular_roughness':.5,'fuzz_weight':.7,'fuzz_color':[.15,.6,.9],'fuzz_roughness':.4,'thin_film_weight':.5,'thin_film_thickness':.45},
]
for index,material in enumerate(materials):
 radius=.55;cx=(index-1.5)*1.25;cy=.56;n=64;rings=32
 points=[[cx,cy+radius,0]];uv=[[.5,1.]]
 for j in range(1,rings):
  theta=math.pi*j/rings
  for i in range(n):
   phi=2*math.pi*i/n
   points.append([cx+radius*math.sin(theta)*math.cos(phi),cy+radius*math.cos(theta),radius*math.sin(theta)*math.sin(phi)])
   uv.append([i/n,1-j/rings])
 points.append([cx,cy-radius,0]);uv.append([.5,0]);bottom=len(points)-1
 faces=[]
 for i in range(n):faces.append([0,1+(i+1)%n,1+i])
 for j in range(rings-2):
  for i in range(n):
   a=1+j*n+i;d=1+j*n+(i+1)%n;b=a+n;c=d+n
   faces.extend([[a,d,c],[a,c,b]])
 for i in range(n):faces.append([1+(rings-2)*n+i,1+(rings-2)*n+(i+1)%n,bottom])
 s['meshes'].append({'points':points,'face_counts':[3]*len(faces),'indices':sum(faces,[]),'texcoords':uv,'color':material['base_color'],'material':{'openpbr':material}})
s['settings']={'diffuse_depth':3,'reflection_depth':4,'refraction_depth':8,'roulette_start':3}
(root/'examples/openpbr.json').write_text(json.dumps(s,separators=(',',':'))+'\n')
def usd(v):
 if isinstance(v,list):return '('+', '.join(str(x) for x in v)+')'
 if isinstance(v,bool):return str(v).lower()
 return str(v)
lines=['#usda 1.0','(\n defaultPrim="World"\n upAxis="Y"\n metersPerUnit=1\n)','def Xform "World" {']
c=s['camera'];matrix=[c['transform'][i:i+4] for i in range(0,16,4)]
lines+=['def Camera "Camera" {','float focalLength = 50','float horizontalAperture = 36','float verticalAperture = 24','float2 clippingRange = (.01,1000)','matrix4d xformOp:transform = ('+', '.join(usd(row) for row in matrix)+')','uniform token[] xformOpOrder=["xformOp:transform"]','}']
for i,m in enumerate(s['meshes']):
 lines += [f'def Material "Material{i}" {{',f'token outputs:surface.connect = </World/Material{i}/Surface.outputs:surface>','def Shader "Surface" {','uniform token info:id="OpenPBRSurface"']
 for name,value in m['material']['openpbr'].items():lines += [f'{"color3f" if isinstance(value,list) else "bool" if isinstance(value,bool) else "float"} inputs:{name} = {usd(value)}']
 lines += ['token outputs:surface','}','}',f'def Mesh "Mesh{i}" (prepend apiSchemas=["MaterialBindingAPI"]) {{','point3f[] points=['+', '.join(usd(p) for p in m['points'])+']','int[] faceVertexCounts='+str(m['face_counts']),'int[] faceVertexIndices='+str(m['indices']),f'rel material:binding=</World/Material{i}>','uniform token subdivisionScheme="none"']
 if 'texcoords' in m:lines += ['texCoord2f[] primvars:st=['+', '.join(usd(p) for p in m['texcoords'])+'] (interpolation="vertex")']
 lines += ['}']
for i,l in enumerate(s['lights']):
 kind='RectLight' if l['type']=='area' else 'SphereLight';x,y,z=l['position']
 lines += [f'def {kind} "Light{i}" {{',f'color3f inputs:color={usd(l["color"])}',f'float inputs:intensity={l["intensity"]}',f'double3 xformOp:translate=({x},{y},{z})']
 if kind=='RectLight':lines += ['float inputs:width=2.5','float inputs:height=2.5','float3 xformOp:rotateXYZ=(-90,0,0)','uniform token[] xformOpOrder=["xformOp:translate","xformOp:rotateXYZ"]']
 else:lines += ['uniform token[] xformOpOrder=["xformOp:translate"]']
 lines += ['}']
lines+=['}'];(root/'examples/openpbr.usda').write_text('\n'.join(lines)+'\n')
