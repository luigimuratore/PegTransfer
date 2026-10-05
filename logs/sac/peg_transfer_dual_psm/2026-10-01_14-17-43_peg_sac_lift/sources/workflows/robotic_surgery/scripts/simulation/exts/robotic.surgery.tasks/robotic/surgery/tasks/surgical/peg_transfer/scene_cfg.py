# SPDX-License-Identifier: BSD-3-Clause

# Derived from the local i4h handover scene (BSD-3-Clause).
"""Board geometry, static posts, peg reset and two-PSM scene."""

from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg, CylinderCfg
from isaaclab.utils import configclass
from simulation.utils.assets import TABLE_USD

BOARD_X = 0.05
BOARD_Y = 0.0
BOARD_Z = 0.005
BOARD_SIZE = (0.25, 0.10, 0.01)

Z_CYL = 0.023
RAD_CYL = 0.002
H_CYL = 0.025
POST_TOP_Z = Z_CYL + H_CYL / 2

# The USD mesh is centered about 5 mm to the right of its root.
# Place the mesh center on the post axis.
PEG_OFFSET_X = -0.005
PEG_OFFSET_Y = 0.0
# The USD mesh extends 14.6 mm below its root. At z=25 mm its underside
# starts just above the board top (10 mm), avoiding penetration at reset.
PEG_Z = 0.025
PEG_BOTTOM_BELOW_ROOT = 0.0146
POST_CLEARANCE_ROOT_Z = POST_TOP_Z + PEG_BOTTOM_BELOW_ROOT + 0.002

# Randomization only between L5 and L6
POST_L5 = (0.000,  0.000, Z_CYL)
POST_L6 = (0.000, -0.030, Z_CYL)

LEFT_POST_POSITIONS = [
    POST_L5,
    POST_L6,
]


def _board_collision_cfg() -> CollisionPropertiesCfg:
    return CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.0005, rest_offset=0.0)


def _post_collision_cfg() -> CollisionPropertiesCfg:
    return CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.0005, rest_offset=0.0)


def reset_peg_random_l5_l6(
    env,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset peg randomly on L5 or L6 only.

    Board and cylinders remain fixed.
    Peg position = selected post center + fixed x/y offset.
    Peg z is fixed.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    n = len(env_ids)

    chosen = torch.randint(
        low=0,
        high=len(LEFT_POST_POSITIONS),
        size=(n,),
        device=env.device,
    )

    peg_pos_local = torch.zeros((n, 3), device=env.device)

    for i, post_pos in enumerate(LEFT_POST_POSITIONS):
        mask = chosen == i
        if torch.any(mask):
            post_tensor = torch.tensor(post_pos, device=env.device)
            peg_pos_local[mask, :] = post_tensor

    peg_pos_local[:, 0] += PEG_OFFSET_X
    peg_pos_local[:, 1] += PEG_OFFSET_Y
    peg_pos_local[:, 2] = PEG_Z

    peg_pos_world = peg_pos_local + env.scene.env_origins[env_ids]

    quat = torch.zeros((n, 4), device=env.device)
    quat[:, 0] = 1.0

    root_pose = torch.cat([peg_pos_world, quat], dim=-1)

    obj = env.scene[object_cfg.name]
    obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    root_vel = torch.zeros((n, 6), device=env.device)
    obj.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


##
# Scene
##

@configclass
class PegTransferSceneCfg(InteractiveSceneCfg):
    """Two PSMs + table + peg board + posts + one peg object."""

    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING

    ee_1_frame: FrameTransformerCfg = MISSING
    ee_2_frame: FrameTransformerCfg = MISSING

    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=TABLE_USD),
    )

    peg_board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(BOARD_X, BOARD_Y, BOARD_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=CuboidCfg(
            size=BOARD_SIZE,
            collision_props=_board_collision_cfg(),
        ),
    )

    # Left posts
    post_l1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_L1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.042, 0.030, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_l2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_L2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.042, 0.000, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_l3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_L3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.042, -0.030, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_l4 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_L4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.000, 0.030, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_l5 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_L5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.000, 0.000, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_l6 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_L6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.000, -0.030, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    # Right posts
    post_r1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_R1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.042, -0.015, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_r2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_R2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.042, 0.015, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_r3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_R3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.084, 0.035, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_r4 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_R4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.084, -0.035, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_r5 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_R5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.126, -0.015, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    post_r6 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Post_R6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.126, 0.015, Z_CYL)),
        spawn=CylinderCfg(radius=RAD_CYL, height=H_CYL, collision_props=_post_collision_cfg()),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
