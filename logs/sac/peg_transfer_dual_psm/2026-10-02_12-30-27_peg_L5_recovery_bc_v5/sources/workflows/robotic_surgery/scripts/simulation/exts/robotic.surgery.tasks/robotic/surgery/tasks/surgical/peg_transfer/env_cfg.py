# SPDX-License-Identifier: BSD-3-Clause

"""Standalone dual-PSM peg-transfer environment and PPO-facing configuration."""

from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from robotic.surgery.assets.psm import PSM_HIGH_PD_CFG
from simulation.utils.assets import BLOCK_USD

from . import mdp
from .scene_cfg import PEG_OFFSET_X, PEG_OFFSET_Y, PEG_Z, PegTransferSceneCfg, reset_peg_random_l5_l6

_OBJECT = SceneEntityCfg("object")
_EE_1 = SceneEntityCfg("ee_1_frame")
_EE_2 = SceneEntityCfg("ee_2_frame")
_PSM_1_GRIPPER = SceneEntityCfg("robot_1", joint_names=["psm_tool_gripper.*_joint"])
_PSM_2_GRIPPER = SceneEntityCfg("robot_2", joint_names=["psm_tool_gripper.*_joint"])
_ARM_JOINTS = [
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
]


@configclass
class ActionsCfg:
    body_1_joint_pos: DifferentialInverseKinematicsActionCfg = MISSING
    finger_1_joint_pos: BinaryJointPositionActionCfg = MISSING
    body_2_joint_pos: DifferentialInverseKinematicsActionCfg = MISSING
    finger_2_joint_pos: BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_1 = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        joint_vel_1 = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        joint_pos_2 = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2")})
        joint_vel_2 = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2")})
        object_position_1 = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot_1"), "object_cfg": _OBJECT},
        )
        object_position_2 = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot_2"), "object_cfg": _OBJECT},
        )
        object_to_target = ObsTerm(func=mdp.object_to_target)
        ee_1_to_grasp = ObsTerm(func=mdp.ee_to_psm1_grasp)
        ee_2_to_object = ObsTerm(func=mdp.ee_to_object, params={"ee_frame_cfg": _EE_2})
        object_velocity = ObsTerm(func=mdp.object_velocity, params={"object_cfg": _OBJECT})
        transfer_phase = ObsTerm(func=mdp.phase_observation)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    pass


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=reset_peg_random_l5_l6,
        mode="reset",
        params={"object_cfg": _OBJECT},
    )
    reset_transfer_state = EventTerm(func=mdp.reset_transfer_state, mode="reset")
    assisted_grasp = EventTerm(
        func=mdp.update_assisted_grasp, mode="interval",
        interval_range_s=(0.02, 0.02), is_global_time=True,
    )


@configclass
class PegTransferRewardsCfg:
    # Progress terms pay only for reducing distance, not for waiting close to the block.
    psm1_reach = RewTerm(func=mdp.ProgressReward, params={"kind": "psm1_reach"}, weight=3.0)
    psm1_precision = RewTerm(func=mdp.ProgressReward, params={"kind": "psm1_precision"}, weight=4.0)
    psm1_open_approach = RewTerm(func=mdp.ProgressReward, params={"kind": "psm1_open_approach"}, weight=1.5)
    psm1_grasp = RewTerm(func=mdp.psm1_grasp_reward, weight=0.10)
    first_grasp_event = RewTerm(func=mdp.first_grasp_milestone, weight=125.0)
    psm1_lift = RewTerm(func=mdp.ProgressReward, params={"kind": "psm1_lift"}, weight=6.0)
    lift_event = RewTerm(func=mdp.lift_milestone, weight=300.0)
    psm1_stable_hold = RewTerm(func=mdp.stable_hold_reward, weight=10.0)
    stable_lift_event = RewTerm(func=mdp.stable_lift_milestone, weight=125.0)
    handover_position = RewTerm(func=mdp.ProgressReward, params={"kind": "handover_position"}, weight=5.0)
    psm2_ready = RewTerm(func=mdp.psm2_ready_reward, weight=0.05)
    psm2_reach = RewTerm(func=mdp.ProgressReward, params={"kind": "psm2_reach"}, weight=4.0)
    psm2_grasp = RewTerm(func=mdp.psm2_grasp_reward, weight=0.10)
    receiver_grasp_event = RewTerm(func=mdp.receiver_grasp_milestone, weight=300.0)
    handover_event = RewTerm(func=mdp.handover_milestone, weight=500.0)
    transport = RewTerm(func=mdp.ProgressReward, params={"kind": "transport"}, weight=6.0)
    placement = RewTerm(func=mdp.ProgressReward, params={"kind": "placement"}, weight=8.0)
    success = RewTerm(func=mdp.peg_transfer_success, weight=1500.0)
    elapsed_time = RewTerm(func=mdp.elapsed_time, weight=-0.30)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-5e-4)


@configclass
class PegTransferTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": -0.02, "asset_cfg": _OBJECT},
    )
    success = DoneTerm(func=mdp.peg_transfer_success)


@configclass
class PegTransferCurriculumCfg:
    pass


@configclass
class PegTransferEnvCfg(ManagerBasedRLEnvCfg):
    """Pick from L5/L6 with PSM1, receive with PSM2, and place on R2."""

    scene: PegTransferSceneCfg = PegTransferSceneCfg(num_envs=64, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: PegTransferRewardsCfg = PegTransferRewardsCfg()
    terminations: PegTransferTerminationsCfg = PegTransferTerminationsCfg()
    curriculum: PegTransferCurriculumCfg = PegTransferCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        self.viewer.eye = (0.15, 0.5, 0.25)
        self.viewer.lookat = (0.04, 0.0, 0.04)

        for index, position in ((1, (-0.07, 0.0, 0.15)), (2, (0.07, 0.0, 0.15))):
            robot = PSM_HIGH_PD_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/Robot_{index}")
            robot.init_state.pos = position
            robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
            # Start open: the upstream PSM defaults to ±0.09 rad, almost closed.
            robot.init_state.joint_pos["psm_tool_gripper1_joint"] = -0.5
            robot.init_state.joint_pos["psm_tool_gripper2_joint"] = 0.5
            # Open/close remains a real gripper action; assisted grasp handles attachment.
            robot.actuators["psm_tool"].effort_limit = None
            robot.actuators["psm_tool"].effort_limit_sim = 1.0
            setattr(self.scene, f"robot_{index}", robot)
            setattr(self.actions, f"body_{index}_joint_pos", _arm_action(f"robot_{index}"))
            setattr(self.actions, f"finger_{index}_joint_pos", _finger_action(f"robot_{index}"))

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(PEG_OFFSET_X, PEG_OFFSET_Y, PEG_Z), rot=(1.0, 0.0, 0.0, 0.0)
            ),
            spawn=UsdFileCfg(
                usd_path=BLOCK_USD,
                scale=(0.011, 0.011, 0.011),
                collision_props=CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.0005, rest_offset=0.0
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=3.0,
                    max_linear_velocity=0.5,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        for index in (1, 2):
            frame = FrameTransformerCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot_{index}/psm_base_link",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/Robot_{index}/psm_tool_tip_link",
                        name="end_effector",
                    ),
                ],
            )
            setattr(self.scene, f"ee_{index}_frame", frame)


@configclass
class PegTransferEnvCfg_PLAY(PegTransferEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False


def _arm_action(robot_name: str) -> DifferentialInverseKinematicsActionCfg:
    return DifferentialInverseKinematicsActionCfg(
        asset_name=robot_name,
        joint_names=_ARM_JOINTS,
        body_name="psm_tool_tip_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(0.006, 0.006, 0.006, 0.015, 0.015, 0.015),
    )


def _finger_action(robot_name: str) -> BinaryJointPositionActionCfg:
    return BinaryJointPositionActionCfg(
        asset_name=robot_name,
        joint_names=["psm_tool_gripper.*_joint"],
        open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
        close_command_expr={"psm_tool_gripper1_joint": -0.07, "psm_tool_gripper2_joint": 0.07},
    )
