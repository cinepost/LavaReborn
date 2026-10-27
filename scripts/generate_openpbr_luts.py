#!/usr/bin/env python3
"""Pack unmodified Apache-2.0 OpenPBR tables into an immutable GPU data blob.
Rows follow upstream array layout. Renderer sampling uses normalized, clamped
bilinear/trilinear interpolation via OpenPBR's supplied lookup callback ABI.
"""
from pathlib import Path
import re,struct
names=['ideal_dielectric_energy_complement','ideal_dielectric_avg_energy_complement','ideal_dielectric_reflection_ratio','opaque_dielectric_energy_complement','opaque_dielectric_avg_energy_complement','ideal_metal_energy_complement','ideal_metal_avg_energy_complement','ltc']
blob=bytearray();offsets=[]
for i,name in enumerate(names):
 text=Path(f'third_party/openpbr/impl/data/openpbr_{name}_data.h').read_text()
 text=re.sub(r'//[^\n]*','',text)
 numbers=[float(n) for n in re.findall(r'-?\d+(?:\.\d+)?',text.replace('vec3',''))]
 offsets.append(len(blob)//16)
 if i==7:
  assert len(numbers)==3072
  values=[(*numbers[j:j+3],0.) for j in range(0,len(numbers),3)]
 else:
  assert len(numbers)==(32768 if i in (0,3) else 32 if i==6 else 1024)
  values=[(n/65535.,0.,0.,0.) for n in numbers]
 for value in values:blob.extend(struct.pack('<4f',*value))
Path('src/material/openpbr_luts.bin').write_bytes(blob)
Path('shaders/materials/lut_offsets.glsl').write_text('// Generated from upstream data; see scripts/generate_openpbr_luts.py.\nconst uint materialLutOffsets[8]=uint[8]('+','.join(str(i)+'u' for i in offsets)+');\n')
print(f'{len(blob)} bytes of device-resident OpenPBR lookup data')
