#!/usr/bin/env python3
"""Run with Houdini's Python (pxr required). Flatten one USD frame into the prototype ABI."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Gf, Sdf


def expand_texture_path(path):
    import re
    path=path.replace('%(UDIM)d','<UDIM>')
    try:
        import hou
        lookup=hou.getenv
    except ImportError:
        import os
        lookup=os.environ.get
    def replace(match):
        value=lookup(match.group(1) or match.group(2))
        return match.group(0) if value is None else value
    return re.sub(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)',replace,path)

def texture_asset_path(asset):
    return asset.path if '<UDIM>' in asset.path or '%(UDIM)d' in asset.path else (asset.resolvedPath or asset.path)

def anchor_udim(attr,path):
    path=expand_texture_path(path)
    if '<UDIM>' not in path:return path
    stack=attr.GetPropertyStack()
    if not stack:return path
    if not path.startswith('/') and ':' not in path and not path.startswith('./'):path='./'+path
    return Sdf.ComputeAssetPathRelativeToLayer(stack[0].layer,path)

def texture_input(shader, name, field):
    input_=shader.GetInput(name)
    source=input_.GetConnectedSource() if input_ else None
    if not source:return None
    texture=UsdShade.Shader(source[0].GetPrim())
    if not texture or texture.GetIdAttr().Get()!='UsdUVTexture':
        print(f"Lava [WARNING]: unsupported connected material input {name}; using constant")
        return None
    def value(name, fallback):
        attr=texture.GetInput(name)
        v=attr.Get() if attr else None
        return fallback if v is None else v
    asset=value('file',None)
    if not asset:return None
    path=texture_asset_path(asset)
    path=anchor_udim(texture.GetInput('file').GetAttr(),path)
    space=str(value('sourceColorSpace','auto'))
    if space=='sRGB':space='srgb'
    quality=str(value('lava:compressionQuality','balanced'))
    if quality not in ('fast','balanced','high'):quality='balanced'
    channel=str(source[1])
    if channel not in ('r','g','b','a'):channel='rgb'
    result={'file':path,'channel':channel,'color_space':space if space in ('srgb','raw') else 'auto',
            'scale':list(value('scale',Gf.Vec4f(1))), 'bias':list(value('bias',Gf.Vec4f(0))),
            'compression':{'quality':quality}}
    if field=='normal':result['signed']=True
    for name,key in [('wrapS','wrap_s'),('wrapT','wrap_t')]:
        mode=str(value(name,'repeat'));result[key]=mode if mode in ('repeat','clamp','mirror','black') else 'repeat'
    return result

def principled_source(material):
    output=material.GetSurfaceOutput('vex')
    source=output.GetConnectedSource() if output else None
    if not source:return None
    prim=source[0].GetPrim()
    asset=prim.GetAttribute('info:sourceAsset').Get()
    if not asset or 'principledshader' not in asset.path:return None
    return prim

def principled_value(prim,name,time,default):
    a=prim.GetAttribute('inputs:'+name)
    v=a.Get(time) if a else None
    if v is None and a:v=a.GetCustomDataByKey('HoudiniPreviewTags:default_value')
    return default if v is None else v

def principled_textures(material,time):
    prim=principled_source(material)
    if not prim:return None
    def value(name,default):
        return principled_value(prim,name,time,default)
    result={}
    def add(field,file,wrap,space,scale,bias,channel='rgb',strength=1,signed=False):
        asset=value(file,None)
        if not asset:return
        colorspace=str(value(space,'auto' if field in ('base_color','transmission_color') else 'raw'))
        if colorspace=='linear':colorspace='raw'
        mode=str(value(wrap,'repeat'))
        result[field]={'file':anchor_udim(prim.GetAttribute('inputs:'+file),texture_asset_path(asset)),'color_space':colorspace if colorspace in ('raw','srgb') else 'auto',
                       'scale':scale,'bias':bias,'channel':channel,'strength':strength,'signed':signed,
                       'wrap_s':mode,'wrap_t':mode}
    if value('basecolor_useTexture',0):
        color=value('basecolor',[.2]*3);intensity=value('basecolor_textureIntensity',1)
        add('base_color','basecolor_texture','basecolor_textureWrap','basecolor_textureColorSpace',[c*intensity for c in color]+[1],[0]*4)
    for name,field,default in [('rough','specular_roughness',.3),('metallic','base_metalness',0),('transparency','transmission_weight',0)]:
        if value(name+'_useTexture',0):
            channel=['luminance','r','g','b'][max(0,min(3,int(value(name+'_monoChannel',0))))]
            add(field,name+'_texture',name+'_textureWrap',name+'_textureColorSpace',[value(name,default)]*4,[0]*4,channel)
    if value('transcolor_useTexture',0):
        add('transmission_color','transcolor_texture','transcolor_textureWrap','transcolor_textureColorSpace',list(value('transcolor',[1]*3))+[1],[0]*4)
    if value('baseBumpAndNormal_enable',0):
        if value('baseBumpAndNormal_type','normal')=='bump':
            add('bump','baseBump_bumpTexture','baseBump_wrap','baseBump_colorSpace',[1]*4,[0]*4,'luminance',value('baseBump_bumpScale',.05))
        elif value('baseNormal_vectorSpace','uvtangent')=='uvtangent':
            x=-1 if value('baseNormal_flipX',0) else 1;y=-1 if value('baseNormal_flipY',0) else 1
            add('normal','baseNormal_texture','baseNormal_wrap','baseNormal_colorspace',[2*x,2*y,2,1],[-x,-y,-1,0],strength=value('baseNormal_scale',1),signed=True)
    return result

def bind_material(target,prim,time):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material:
        shader, _, _ = material.ComputeSurfaceSource()
        if shader and shader.GetIdAttr().Get(time) == "UsdPreviewSurface":
            def constant(name, fallback):
                value = shader.GetInput(name).Get(time)
                return fallback if value is None else value
            target["color"] = list(constant("diffuseColor", target["color"]))
            target["material"] = {"openpbr": {
                "base_color": target["color"],
                "base_metalness": min(1, max(0, float(constant("metallic", 0)))),
                "transmission_weight": min(1, max(0, float(constant("transmission", 0)))),
                "geometry_opacity": min(1, max(0, float(constant("opacity", 1)))),
                "specular_roughness": min(1, max(0, float(constant("roughness", .5)))),
                "specular_ior": min(4, max(1, float(constant("ior", 1.5)))),
                "emission_luminance": 1,
                "emission_color": list(constant("emissiveColor", [0,0,0])),
            }}
            source=principled_source(material)
            if source:
                target['material']['openpbr']['transmission_weight']=min(1,max(0,float(principled_value(source,'transparency',time,0))))
                target['material']['openpbr']['transmission_color']=list(principled_value(source,'transcolor',time,[1]*3))
                target['material']['openpbr']['geometry_opacity']=min(1,max(0,float(principled_value(source,'opac',time,1))))
        elif shader and shader.GetIdAttr().Get(time) in ("ND_open_pbr_surface_surfaceshader", "open_pbr_surface", "OpenPBRSurface"):
            # Native constants only; connected MaterialX graph generation is a future frontend.
            import json
            inputs={}
            for input_ in shader.GetInputs():
                value=input_.Get(time)
                if value is not None:
                    inputs[input_.GetBaseName()]=list(value) if isinstance(value,(Gf.Vec3f,Gf.Vec3d)) else value
            target["material"]={"openpbr":inputs}
    if material and shader:
        texture_inputs={}
        preview=shader.GetIdAttr().Get(time)=='UsdPreviewSurface'
        names={'diffuseColor':'base_color','metallic':'base_metalness','roughness':'specular_roughness','transmission':'transmission_weight','transmissionColor':'transmission_color','ior':'specular_ior','opacity':'geometry_opacity','emissiveColor':'emission_color','normal':'normal'}
        for input_ in shader.GetInputs():
            name=input_.GetBaseName()
            if not input_.HasConnectedSource():continue
            field=names.get(name) if preview else name
            if not field:continue
            binding=texture_input(shader,name,field)
            if binding:texture_inputs[field]=binding
        if preview:
            principled=principled_textures(material,time)
            if principled is not None:
                texture_inputs.pop("normal",None)
                texture_inputs.pop("transmission_weight",None)
                texture_inputs.pop("transmission_color",None)
                texture_inputs.pop("geometry_opacity",None)
                texture_inputs.update(principled)
        if texture_inputs:
            target.setdefault('material',{})['textures']=texture_inputs
            if preview and 'emission_color' in texture_inputs:
                target['material']['openpbr']['emission_luminance']=1

def extract(stage, camera_path, frame):
    time = Usd.TimeCode(frame)
    cache = UsdGeom.XformCache(time)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    if not camera:
        raise ValueError(f"No camera at {camera_path}")
    c = camera.GetCamera(time)
    transform = cache.GetLocalToWorldTransform(camera.GetPrim())
    # USD focal length/aperture attributes are tenths of a world unit.
    scene = {"camera": {
        "transform": [float(transform[r][col]) for r in range(4) for col in range(4)],
        "focal_length": float(c.focalLength) * 0.1,
        "horizontal_aperture": float(c.horizontalAperture) * 0.1,
        "vertical_aperture": float(c.verticalAperture) * 0.1,
        "aperture_offset": [float(c.horizontalApertureOffset) * 0.1, float(c.verticalApertureOffset) * 0.1],
        "near": float(c.clippingRange.min), "far": float(c.clippingRange.max),
        "orthographic": c.projection == Gf.Camera.Orthographic,
        "f_stop": float(c.fStop), "focus_distance": max(0.001, float(c.focusDistance)),
    }, "meshes": [], "lights": []}
    for prim in stage.Traverse():
        if prim.GetTypeName()=="RenderSettings":
            attr=prim.GetAttribute("lava:visibility")
            if attr and attr.Get(time) is not None:scene.setdefault("settings",{})["visibility"]=str(attr.Get(time))
            caustics=prim.GetAttribute("lava:reduceAnalyticCaustics")
            if caustics and caustics.Get(time) is not None:scene.setdefault("settings",{})["reduce_analytic_caustics"]=bool(caustics.Get(time))
            mode=prim.GetAttribute("lava:textures")
            if mode and mode.Get(time) is not None:scene.setdefault("settings",{})["textures"]=str(mode.Get(time))
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility(time) == UsdGeom.Tokens.invisible:
            continue
        if imageable and imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        if prim.IsA(UsdGeom.BasisCurves):
            curve=UsdGeom.BasisCurves(prim)
            points=curve.GetPointsAttr().Get(time) or []
            counts=list(curve.GetCurveVertexCountsAttr().Get(time) or [])
            basis='linear' if curve.GetTypeAttr().Get(time)=='linear' else str(curve.GetBasisAttr().Get(time))
            wrap=str(curve.GetWrapAttr().Get(time))
            if basis not in ('linear','bezier','catmullRom','bspline') or wrap=='pinned':
                print(f'Lava [WARNING]: unsupported curve basis/wrap on {prim.GetPath()}');continue
            pv=UsdGeom.PrimvarsAPI(prim).GetPrimvar('radius')
            custom=pv.ComputeFlattened(time) if pv else prim.GetAttribute('radius').Get(time)
            is_radius=custom is not None
            radii=custom if is_radius else curve.GetWidthsAttr().Get(time)
            if radii is None:radii=[.01];is_radius=True
            if isinstance(radii,(int,float)):radii=[radii]
            rate=str(pv.GetInterpolation()) if pv and is_radius else ('vertex' if is_radius else str(curve.GetWidthsInterpolation()))
            scale=max(matrix.TransformDir(Gf.Vec3d(*axis)).GetLength() for axis in [(1,0,0),(0,1,0),(0,0,1)])*(1 if is_radius else .5)
            colors=curve.GetDisplayColorAttr().Get(time)
            target={'points':[list(matrix.Transform(Gf.Vec3d(p))) for p in points], 'counts':counts,
                    'basis':basis,'periodic':wrap=='periodic','radius':[float(r)*scale for r in radii],
                    'radius_interpolation':rate,'color':list(colors[0]) if colors else [.5]*3}
            bind_material(target,prim,time)
            scene.setdefault('curves',[]).append(target)
        elif prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            colors = mesh.GetDisplayColorAttr().Get(time)
            counts = list(mesh.GetFaceVertexCountsAttr().Get(time) or [])
            indices = list(mesh.GetFaceVertexIndicesAttr().Get(time) or [])
            holes = set(mesh.GetHoleIndicesAttr().Get(time) or [])
            kept_counts, kept_indices, offset = [], [], 0
            kept_corners, kept_faces = [], []
            for face, count in enumerate(counts):
                if face not in holes:
                    kept_counts.append(count)
                    kept_indices.extend(indices[offset:offset + count])
                    kept_corners.extend(range(offset,offset+count))
                    kept_faces.extend([face]*count)
                offset += count
            scene["meshes"].append({
                "points": [list(matrix.Transform(Gf.Vec3d(p))) for p in (mesh.GetPointsAttr().Get(time) or [])],
                "face_counts": kept_counts, "indices": kept_indices,
                "color": list(colors[0]) if colors else [0.5, 0.5, 0.5],
            })
            def mesh_flag(name, fallback=False):
                pv=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('lava:'+name)
                value=pv.Get(time) if pv else None
                return fallback if value is None else bool(value)
            scene["meshes"][-1].update(geometry_light=mesh_flag('geometryLight',prim.HasAPI(UsdLux.MeshLightAPI)),render_light_geometry=mesh_flag('renderLightGeometry'))
            target=scene["meshes"][-1]
            target["reverse_orientation"]=(mesh.GetOrientationAttr().Get(time)==UsdGeom.Tokens.leftHanded) != (matrix.GetDeterminant3()<0)
            # primvars:normals takes precedence over the built-in normals attribute.
            normals_pv=UsdGeom.PrimvarsAPI(mesh).GetPrimvar("normals")
            normals=normals_pv.ComputeFlattened(time) if normals_pv else mesh.GetNormalsAttr().Get(time)
            rate=normals_pv.GetInterpolation() if normals_pv else mesh.GetNormalsInterpolation()
            if normals:
                normal_matrix=matrix.GetInverse().GetTranspose()
                values=[normal_matrix.TransformDir(Gf.Vec3d(n)) for n in normals]
                import math
                if abs(matrix.GetDeterminant3())<=1e-20 or any(not math.isfinite(x) for n in values for x in n):
                    print(f"Lava [WARNING]: unusable authored normals on {prim.GetPath()}; using geometric normals")
                else:
                    target['normals']=[list(n.GetNormalized()) if n.GetLength()>1e-20 else [0,0,0] for n in values]
                    if rate==UsdGeom.Tokens.faceVarying:target['normal_indices']=kept_corners
                    elif rate==UsdGeom.Tokens.uniform:target['normal_indices']=kept_faces
                    elif rate==UsdGeom.Tokens.constant:target['normal_indices']=[0]*len(kept_indices)
            st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            values = st.ComputeFlattened(time) if st else None
            if values:
                target=scene["meshes"][-1]
                target["texcoords"]=[list(v) for v in values]
                interpolation=st.GetInterpolation()
                if interpolation==UsdGeom.Tokens.faceVarying:
                    target["texcoord_indices"]=kept_corners
                elif interpolation==UsdGeom.Tokens.uniform:
                    target["texcoord_indices"]=kept_faces
                elif interpolation==UsdGeom.Tokens.constant:
                    target["texcoord_indices"]=[0]*len(kept_indices)
            bind_material(scene["meshes"][-1],prim,time)
        elif prim.HasAPI(UsdLux.LightAPI):
            api = UsdLux.LightAPI(prim)
            def attr(name, fallback):
                a = prim.GetAttribute(name)
                value = a.Get(time) if a else None
                return fallback if value is None else value
            light = {"color": list(api.GetColorAttr().Get(time) or Gf.Vec3f(1)),
                     "intensity": float(attr("inputs:intensity", 1)) * 2 ** float(attr("inputs:exposure", 0))}
            position = list(matrix.Transform(Gf.Vec3d(0)))
            direction = list(matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized())
            cone = float(attr("inputs:shaping:cone:angle", 180))
            if prim.IsA(UsdLux.DistantLight):
                light.update(type="directional", direction=direction)
            elif prim.IsA(UsdLux.RectLight):
                light.update(type="area", position=position,
                             single_sided=bool(attr("primvars:lava:singleSided", True)),
                             render_light_geometry=bool(attr("primvars:lava:renderLightGeometry", False)),
                             u=list(matrix.TransformDir(Gf.Vec3d(float(attr("inputs:width", 1)), 0, 0))),
                             v=list(matrix.TransformDir(Gf.Vec3d(0, -float(attr("inputs:height", 1)), 0))))
            elif prim.IsA(UsdLux.SphereLight):
                if 0 < cone < 90:
                    light.update(type="spot", position=position, direction=direction,
                                 cone_degrees=cone, softness=float(attr("inputs:shaping:cone:softness", 0)))
                else:
                    light.update(type="point", position=position)
            else:
                raise ValueError(f"Unsupported light {prim.GetPath()} ({prim.GetTypeName()})")
            scene["lights"].append(light)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--frame", type=float, default=1)
    parser.add_argument("--lava", type=Path, default=Path(__file__).resolve().parents[1] / "target/release/lava")
    parser.add_argument("--export", type=Path, help="Only export JSON; no GPU needed")
    args, render_args = parser.parse_known_args()
    stage = Usd.Stage.Open(str(args.usd))
    if not stage:
        raise ValueError(f"Cannot open {args.usd}")
    scene = extract(stage, args.camera, args.frame)
    if args.export:
        args.export.write_text(json.dumps(scene, indent=2, allow_nan=False))
        return
    with tempfile.TemporaryDirectory(prefix="lava-usd-") as directory:
        path = Path(directory) / "scene.json"
        path.write_text(json.dumps(scene, allow_nan=False))
        subprocess.run([str(args.lava.resolve()), "render", str(path), *render_args], check=True)


if __name__ == "__main__":
    main()
