"""Read the original USD offline. Never creates SimulationApp or runs PhysX."""

import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    # Load bundled USD Python bindings in a child with their native library paths.
    # This is USD stage inspection only; no import of isaacsim or AppLauncher.
    if '--usd-child' not in sys.argv:
        site = Path(sys.executable).resolve().parents[1] / 'lib/python3.11/site-packages'
        sdk = next((site / 'isaacsim/extscache').glob('omni.usd.libs-*'))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(sdk) + os.pathsep + env.get('PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(sdk / 'bin'), str(site.parents[1]), env.get('LD_LIBRARY_PATH', '')])
        subprocess.run([sys.executable, __file__, '--usd-child'], env=env, check=True)
        return
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    import numpy as np
    import trimesh
    import importlib.util
    from common import REPO, TASK_DIR, versions
    spec = importlib.util.spec_from_file_location('geometry', TASK_DIR / 'sac_geometry.py')
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    path = REPO / 'data/Isaac/Healthcare/0.5.0/132c82d/Props/PegBlock/block.usd'
    stage = Usd.Stage.Open(str(path))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh, transform = UsdGeom.Mesh(prim), UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    v = np.array([transform.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()]) * g.ASSET_SCALE
    np.testing.assert_allclose(v.min(0), g.PEG_MIN, atol=1e-8)
    np.testing.assert_allclose(v.max(0), g.PEG_MAX, atol=1e-8)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdGeom.GetStageUpAxis(stage) == 'Z'
    assert all(c == 3 for c in mesh.GetFaceVertexCountsAttr().Get())
    tri = trimesh.Trimesh(v, np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=True)
    section = tri.section(plane_origin=[0, 0, -0.006], plane_normal=[0, 0, 1])
    contours = []
    for points in section.discrete:
        xy = points[:, :2]
        center = (xy.min(0) + xy.max(0)) / 2
        contours.append(dict(points=len(points), bounds=[xy.min(0).tolist(), xy.max(0).tolist()],
                             center=center.tolist(), min_radius=float(np.linalg.norm(xy - center, axis=1).min())))
    result = dict(versions=versions(), usd=str(path), units=UsdGeom.GetStageMetersPerUnit(stage), up_axis='Z',
                  transformed_bounds_m=[v.min(0).tolist(), v.max(0).tolist()],
                  original_collision=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                  watertight=tri.is_watertight, contours_at_z_minus_6mm=contours,
                  proxy_sectors=len(g.collider_sectors()), proxy_hole_radius=g.HOLE_RADIUS,
                  thread_tolerance=g.THREAD_TOLERANCE, reset_z=g.RESET_Z, rest_z=g.REST_Z,
                  clear_z=g.CLEAR_Z, physics_validated=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
