// Generated lowering from native material IR to the reference OpenPBR inputs.
#extension GL_EXT_control_flow_attributes : require
#include "textures.glsl"
OpenPBR_ResolvedInputs materialInputs(uint id, vec3 normal, vec2 uv, float footprint) {
 MaterialRecord m=materials[id];
 [[dont_unroll]] for(uint field=1u;field<38u;field++) {if(textureSlots[id*40u+field].info.x!=0u)m.values[field]=materialValue(id,field,uv,footprint); }
 OpenPBR_ResolvedInputs p=openpbr_make_default_resolved_inputs();
 p.base_weight=clamp(m.values[1].x,0.,1.);
 p.base_color=max(m.values[2].xyz,vec3(0));
 p.base_diffuse_roughness=max(m.values[3].x,0.);
 p.base_metalness=clamp(m.values[4].x,0.,1.);
 p.specular_weight=max(m.values[5].x,0.);
 p.specular_color=max(m.values[6].xyz,vec3(0));
 p.specular_roughness=clamp(m.values[7].x,0.,1.);
 p.specular_roughness_anisotropy=clamp(m.values[8].x,0.,1.);
 p.specular_ior=max(1.,m.values[9].x);
 p.coat_weight=clamp(m.values[10].x,0.,1.);
 p.coat_color=max(m.values[11].xyz,vec3(0));
 p.coat_roughness=clamp(m.values[12].x,0.,1.);
 p.coat_roughness_anisotropy=clamp(m.values[13].x,0.,1.);
 p.coat_ior=max(1.,m.values[14].x);
 p.coat_darkening=max(m.values[15].x,0.);
 p.fuzz_weight=clamp(m.values[16].x,0.,1.);
 p.fuzz_color=max(m.values[17].xyz,vec3(0));
 p.fuzz_roughness=clamp(m.values[18].x,0.,1.);
 p.transmission_weight=clamp(m.values[19].x,0.,1.);
 p.transmission_color=max(m.values[20].xyz,vec3(0));
 p.transmission_depth=max(m.values[21].x,0.);
 p.transmission_scatter=max(m.values[22].xyz,vec3(0));
 p.transmission_scatter_anisotropy=max(m.values[23].x,0.);
 p.transmission_dispersion_scale=max(m.values[24].x,0.);
 p.transmission_dispersion_abbe_number=max(m.values[25].x,0.);
 p.thin_film_weight=clamp(m.values[26].x,0.,1.);
 p.thin_film_thickness=max(m.values[27].x,0.);
 p.thin_film_ior=max(1.,m.values[28].x);
 p.emission_luminance=max(m.values[29].x,0.);
 p.emission_color=max(m.values[30].xyz,vec3(0));
 p.geometry_thin_walled=m.values[32].x!=0.;
 p.subsurface_weight=max(m.values[33].x,0.);
 p.subsurface_color=max(m.values[34].xyz,vec3(0));
 p.subsurface_radius=max(m.values[35].x,0.);
 p.subsurface_radius_scale=max(m.values[36].xyz,vec3(0));
 p.subsurface_scatter_anisotropy=max(m.values[37].x,0.);
 p.geometry_basis=openpbr_make_basis(normal);
 p.geometry_coat_basis=p.geometry_basis;
 return p;
}
