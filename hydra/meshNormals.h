#pragma once
#include <pxr/base/vt/array.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usdImaging/usdImaging/delegate.h>
#include <pxr/usdImaging/usdImaging/meshAdapter.h>
#include <pxr/usdImaging/usdImaging/primvarUtils.h>

PXR_NAMESPACE_OPEN_SCOPE
namespace lava {
struct MeshNormals {
  VtValue value;
  VtIntArray indices;
  HdInterpolation interpolation = HdInterpolationCount;
};
// USD 24.03's mesh adapter hides normals (including their descriptors) whenever
// subdivisionScheme != none. Lava currently shades the unrefined polygon cage.
// Use the adapter's protected source-prim accessor via a local read-only bridge;
// no adapter registration, stage edits, schema overrides or refinement occur.
class MeshSourceReader final : public UsdImagingMeshAdapter {
public:
  UsdPrim Source(UsdImagingDelegate *delegate, SdfPath const &id) {
    SetDelegate(delegate);
    return _GetPrim(delegate->GetScenePrimPath(id, 0));
  }
};
inline MeshNormals ReadMeshNormals(HdSceneDelegate *delegate, SdfPath const &id,
                                  HdMeshTopology const &topology) {
  MeshNormals result;
  result.value =
      delegate->GetIndexedPrimvar(id, HdTokens->normals, &result.indices);
  for (auto rate : {HdInterpolationConstant, HdInterpolationUniform,
                    HdInterpolationVertex, HdInterpolationVarying,
                    HdInterpolationFaceVarying})
    for (auto const &descriptor : delegate->GetPrimvarDescriptors(id, rate))
      if (descriptor.name == HdTokens->normals)
        result.interpolation = rate;
  if (topology.GetScheme() == TfToken("none"))
    return result;
  // Prefer authored data for USD subdivision cages even if scene-index filters
  // retained/generated a normal array when the subdivision flag changed.
  auto *imaging = dynamic_cast<UsdImagingDelegate *>(delegate);
  if (!imaging) {
    auto *source = delegate->GetRenderIndex().GetSceneDelegateForRprim(id);
    // Resolve the original delegate behind scene-index emulation.
    imaging = dynamic_cast<UsdImagingDelegate *>(source);
  }
  if (!imaging)
    return result;
  MeshSourceReader reader;
  auto prim = reader.Source(imaging, id);
  if (!prim)
    return result;
  auto time = imaging->GetTime();
  auto pv = UsdGeomPrimvarsAPI(prim).FindPrimvarWithInheritance(HdTokens->normals);
  MeshNormals authored;
  if (pv) {
    pv.Get(&authored.value, time);
    pv.GetIndices(&authored.indices, time);
    authored.interpolation = UsdImagingUsdToHdInterpolation(pv.GetInterpolation());
  } else {
    UsdGeomMesh mesh(prim);
    if (mesh) {
      mesh.GetNormalsAttr().Get(&authored.value, time);
      authored.interpolation =
          UsdImagingUsdToHdInterpolation(mesh.GetNormalsInterpolation());
    }
  }
  return authored;
}
} // namespace lava
PXR_NAMESPACE_CLOSE_SCOPE
