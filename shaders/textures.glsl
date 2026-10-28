#ifndef LAVA_TEXTURE_SUBSYSTEM
#define LAVA_TEXTURE_SUBSYSTEM
#extension GL_EXT_nonuniform_qualifier : require
// Shared TextureHandle/TextureTile ABI with src/texture/mod.rs. No material dependency.
const uint TEXTURE_PLAIN=1u,TEXTURE_UDIM=2u,TEXTURE_VIRTUAL=4u,TEXTURE_SOLID=8u;
struct TextureHandle {uint flags;uint tableOffset;uint tileCount;uint reserved;};
struct TextureTile {uvec4 info;vec4 rangeScale;vec4 rangeBias;};
layout(set=0,binding=18) uniform sampler2D textureImages[128];
layout(std430,set=0,binding=20) readonly buffer TextureTileTable {TextureTile textureTiles[];};
bool textureResolve(TextureHandle h,vec2 uv,out TextureTile tile,out vec2 localUv) {
 localUv=uv;
 if((h.flags&(TEXTURE_SOLID|TEXTURE_VIRTUAL))!=0u||h.tileCount==0u||any(isnan(uv))||any(isinf(uv)))return false;
 uint offset=h.tableOffset;
 if((h.flags&TEXTURE_UDIM)!=0u) {
  vec2 cell=floor(uv);
  if(cell.x<0.||cell.x>=10.||cell.y<0.||cell.y>=900.)return false;
  uint number=1001u+uint(cell.x)+10u*uint(cell.y);
  if(number>9999u)return false;
  uint lo=0u,hi=h.tileCount;
  while(lo<hi) {uint mid=lo+(hi-lo)/2u;if(textureTiles[offset+mid].info.x<number)lo=mid+1u;else hi=mid;}
  if(lo>=h.tileCount||textureTiles[offset+lo].info.x!=number)return false;
  offset+=lo;localUv=uv-cell;
 }
 tile=textureTiles[offset];return tile.info.y!=0u;
}
bool textureAvailable(TextureHandle h,vec2 uv) {TextureTile tile;vec2 localUv;return textureResolve(h,uv,tile,localUv);}
vec2 textureSizeAt(TextureHandle h,vec2 uv) {
 TextureTile tile;vec2 localUv;
 return textureResolve(h,uv,tile,localUv)?vec2(textureSize(textureImages[nonuniformEXT(tile.info.y)],0)):vec2(1);
}
// Missing/placeholder/virtual handles return false; the pass supplies its own solid fallback.
bool textureSample(TextureHandle h,vec2 uv,vec2 footprint,uvec2 wrapModes,out vec4 value) {
 TextureTile tile;vec2 localUv;
 if(!textureResolve(h,uv,tile,localUv))return false;
 uint index=tile.info.y;vec2 size=vec2(textureSize(textureImages[nonuniformEXT(index)],0));
 if((h.flags&TEXTURE_UDIM)==0u) {
  for(uint c=0u;c<2u;c++) {
   uint mode=wrapModes[c];
   if(mode==3u&&(localUv[c]<0.||localUv[c]>1.)){value=vec4(0);return true;}
   if(mode==1u)localUv[c]=clamp(localUv[c],.5/size[c],1.-.5/size[c]);
   if(mode==2u)localUv[c]=1.-abs(mod(localUv[c],2.)-1.);
  }
 }
 localUv.y=1.-localUv.y;
 float lod=max(0.,log2(max(max(footprint.x*size.x,footprint.y*size.y),1.)));
 vec4 sampled;
 if((h.flags&TEXTURE_UDIM)!=0u) {
  // Clamp each physical mip separately: the shared image sampler also serves
  // plain repeating textures. Never interpolate the opposite tile edge.
  lod=min(lod,float(textureQueryLevels(textureImages[nonuniformEXT(index)])-1));
  int a=int(floor(lod)),b=int(ceil(lod));
  vec2 sa=vec2(textureSize(textureImages[nonuniformEXT(index)],a));
  sampled=textureLod(textureImages[nonuniformEXT(index)],clamp(localUv,.5/sa,1.-.5/sa),float(a));
  if(b!=a) {
   vec2 sb=vec2(textureSize(textureImages[nonuniformEXT(index)],b));
   sampled=mix(sampled,textureLod(textureImages[nonuniformEXT(index)],clamp(localUv,.5/sb,1.-.5/sb),float(b)),fract(lod));
  }
 } else sampled=textureLod(textureImages[nonuniformEXT(index)],localUv,lod);
 value=sampled*tile.rangeScale+tile.rangeBias;
 return true;
}
#endif
