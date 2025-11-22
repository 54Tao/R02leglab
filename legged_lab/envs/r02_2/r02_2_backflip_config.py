"""Configuration for R02_2 quadruped robot backflip task."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.r02_2 import R02_2_CFG
from legged_lab.envs.base.base_env_config import (
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# 定义平地地形配置（后空翻需要平坦表面）
FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,  # 100% flat ground
            noise_range=(0.0, 0.01),  # Minimal noise
            noise_step=0.01,
            border_width=0.0
        )
    },
)

from isaaclab_rl.rsl_rl import RslRlRndCfg


@configclass
class R02_2BackflipRewardCfg(RewardCfg):
    """Reward configuration for R02_2 backflip task."""

    # ========== 5-Stage Curriculum for Backflip ==========
    # Stage 0: STAND - Maintain upright posture
    # Stage 1: CROUCH - Prepare for jump
    # Stage 2: JUMP - Launch into air  
    # Stage 3: AIRBORNE - Perform flip
    # Stage 4: LANDING - Recover to standing

    # COM高度奖励 (5个阶段的目标不同)
    com_height = RewTerm(
        func=mdp.track_com_height,  # 需要在mdp中实现
        weight=2.0,
        params={"stage_targets": [0.09, 0.05, 0.15, 0.15, 0.09]}  # Stand, Crouch, Jump, Air, Land
    )

    # 身体平衡奖励
    body_balance = RewTerm(
        func=mdp.track_body_orientation,  # 需要在mdp中实现
        weight=1.0,
        params={
            "up_direction": [0, 0, 1],  # 向上方向
            "target_angles": [[0, 0, 0], [0, 0, 0], [-0.3, 0, 0], [-0.3, 0, 0], [0, 0, 0]]  # 不同阶段的角度目标
        }
    )

    # 脚接触控制
    foot_contact_control = RewTerm(
        func=mdp.foot_contact_control,  # 需要在mdp中实现
        weight=1.0,
        params={
            "contact_targets": [1.0, 1.0, 0.0, 0.0, 1.0],  # 脚接触目标：完全接触、完全接触、离地、离地、完全接触
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*3_.*"])
        }
    )

    # 角速度奖励（用于翻转）
    angular_velocity = RewTerm(
        func=mdp.track_angular_velocity,  # 需要在mdp中实现
        weight=1.5,
        params={
            "target_vel_y": [0, 0, 0, 3.0, 0],  # Stage 3: 目标pitch旋转速度
            "std": 0.5
        }
    )

    # 能量消耗（所有阶段）
    energy = RewTerm(func=mdp.energy, weight=-0.0001)

    # 动作平滑性
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 不期望的身体接触
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-500.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[
                "body",  # 躯干
                ".*1_.*",  # 腿根部（yaw关节）
                ".*2_.*",  # 膝盖（pitch关节）
            ]),
            "threshold": 0.1
        },
    )

    # 惩罚四条腿同时腾空（但不能在翻转阶段）
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*3_.*"]),
            "threshold": 1.0,
            "exclude_stages": [2, 3]  # 排除跳跃和翻转阶段
        },
    )

    # 翻转检测奖励（stage 3特定）
    flip_detection = RewTerm(
        func=mdp.flip_detection,  # 需要在mdp中实现
        weight=5.0,
        params={
            "stage": 3,
            "flip_threshold": 0.5  # 身体倒过来
        }
    )


@configclass
class R02_2BackflipEnvCfg(BaseEnvCfg):
    """Environment configuration for R02_2 backflip task."""

    scene_cfg = None
    observations_cfg = None
    actions_cfg = None
    rewards_cfg = R02_2BackflipRewardCfg()
    terminations_cfg = None
    events_cfg = None

    # 环境参数
    num_envs = 1000
    env_spacing = 2.0
    episode_length_s = 10.0

    # 仿真参数
    decimation = 10
    control_frequency_inv = decimation  # 20ms control step
    sim_dt = 0.0005  # 0.5ms physics step

    # 观察历史
    history_len = 10

    # 奖励缩放
    reward_scale = 1.0

    # 课程学习参数
    curriculum_enabled = True
    stage_progression = {
        "stage_0_duration": 1000,      # 迭代数
        "stage_1_trigger": "height_threshold",  # 转换条件
        "stage_1_threshold": 0.08,
        # ... 其他阶段转换条件
    }


@configclass
class R02_2BackflipAgentCfg(BaseAgentCfg):
    """Agent configuration for R02_2 backflip task."""

    observation_dim = 140  # 根据实际观察空间调整
    action_dim = 12
    hidden_size = 512
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95

    # RslRl配置
    rsl_rl_cfg = RslRlRndCfg(
        class_name="PPO",
        init_noise_std=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        entropy_coeff=0.01,
        clip_ratio=0.2,
    )
