"""Explicit compound collider; never convexify the entire peg and fill its hole."""

from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone

from .sac_geometry import ASSET_SCALE, collider_sectors


@clone
def spawn_hollow_peg(prim_path, cfg, translation=None, orientation=None, **kwargs):
    visual_cfg = cfg.copy()
    visual_cfg.collision_props.collision_enabled = False
    root = sim_utils.spawn_from_usd(prim_path, visual_cfg, translation, orientation, **kwargs)
    stage = root.GetStage()
    # The visual mesh retains its original appearance but is not the collision mesh.
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for i, points in enumerate(collider_sectors()):
        mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Collider_{i:02d}")
        mesh.CreatePointsAttr([Gf.Vec3f(*(v / ASSET_SCALE for v in p)) for p in points])
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        api.CreateContactOffsetAttr(0.0001)
        api.CreateRestOffsetAttr(0.0)
    # Explicit mass, rather than an unknown density inferred from a scaled asset.
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.005)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateEnableCCDAttr(True)
    return root
