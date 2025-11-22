"""Configuration for R02_2 quadruped robot environments - 优化版本."""

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

# 定义平地地形配置
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
            proportion=1.0,  # 100%平地
            noise_range=(0.0, 0.01),  # 极小噪声，0-1cm
            noise_step=0.01,  # 1cm步长
            border_width=0.0
        )
    },
)
from isaaclab_rl.rsl_rl import RslRlRndCfg


@configclass
class R02_2RewardCfg(RewardCfg):
    """Reward configuration for R02_2 robot - 基于walk-these-ways和legged_gym优化."""

    # ========== 速度跟踪奖励==========
    # 参考walk-these-ways: tracking_lin_vel=1.0, tracking_sigma=0.25
    # 参考legged_gym: tracking_lin_vel=1.0, tracking_sigma=0.25
    # 参考anymal_c_flat: feet_air_time=2.0（平地上步态更重要）
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,  # 速度跟踪是最核心的奖励
        params={"std": 0.25}  # 使用walk-these-ways的sigma值，更紧的跟踪
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.5,  # 角速度权重是线速度的一半（walk-these-ways的比例）
        params={"std": 0.25}  # 同样使用0.25的sigma
    )

    # 惩罚不期望的速度
    # 参考legged_gym: lin_vel_z=-2.0, ang_vel_xy=-0.05
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)  # 与legged_gym一致
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # 与legged_gym一致

    # 能量消耗
    # 参考walk-these-ways: torques=-0.0001（平衡能耗和性能）
    energy = RewTerm(func=mdp.energy, weight=-0.0001)

    # 动作平滑性
    # 参考legged_gym: dof_acc=-2.5e-7, action_rate=-0.01
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # legged_gym标准值
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # legged_gym标准值
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)

    # 不期望的接触（任何非足端连杆接触地面）
    # 参考legged_gym: collision=-1.0
    # 特别加强对膝盖（pitch2关节）着地的惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-500.0,  # 极强惩罚（从-100增加到-500）
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[
                "body",  # 躯干
                ".*1_.*",  # 所有yaw关节连杆（腿根部）
                ".*2_.*",  # 所有pitch2关节连杆（膝盖/中间关节）- 重点惩罚！
            ]),
            "threshold": 0.1  # 极度灵敏，任何轻微接触都惩罚
        },
    )

    # 惩罚四条腿同时腾空
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*3_.*"]),
            "threshold": 1.0
        },
    )

    # 身体姿态保持水平
    # 参考walk-these-ways: orientation=-5.0
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # 终止惩罚
    # 参考walk-these-ways: termination=0（不额外惩罚，让自然reset）
    # 但我们保持一定惩罚以避免过早终止
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 关节限位惩罚
    # 参考walk-these-ways: dof_pos_limits=-10.0
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # ========== 新增：关节初始角度偏离惩罚 ==========
    # 惩罚驱动关节（yaw和pitch）偏离初始位置
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,  # 适中惩罚，防止关节偏离过大
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint_yaw", ".*_joint_pitch"])
        }
    )

    # 躯干上4个yaw关节偏离初始角度的额外惩罚（着重惩罚）
    yaw_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,  # 大幅增加惩罚，保持躯干关节稳定
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint_yaw"])
        }
    )

    # ========== 步态奖励 - 三阶段课程学习 ==========
    # 课程学习策略（在base_env.py中实现）：
    #   阶段1 (0-200轮): 权重从0.2逐渐增加到2.0 - 先学站立
    #   阶段2 (200-500轮): 保持2.0 - 强化步态
    #   阶段3 (500-1200轮): 从2.0衰减到0.2 - 速度跟踪主导
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_quadruped,
        weight=0.2,  # 初始权重0.2（很小），让机器人先学站立，课程学习会逐渐提高到2.0
        # 注意：权重会在BaseEnv中通过curriculum动态调整
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*3_.*"]),
            "threshold": 0.4,
        }
    )

    # ========== 新增：基于步态相位的奖励（walk-these-ways方法）==========
    # 这些奖励更精准，权重保持不变
    # 惩罚摆动相时有接触力
    gait_contact_force = RewTerm(
        func=mdp.gait_reward_contact_force,
        weight=-1.5,  # 强惩罚摆动相接触
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*3_.*"]),
            "force_threshold": 1.0,
            "sigma": 100.0,  # walk-these-ways使用gait_force_sigma=50
        }
    )

    # 惩罚支撑相时足端滑动
    gait_foot_velocity = RewTerm(
        func=mdp.gait_reward_foot_velocity,
        weight=-2.0,  # 强惩罚支撑相滑动
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*3_.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*3_.*"]),
            "sigma": 0.5,  # walk-these-ways使用gait_vel_sigma=0.5
        }
    )

    # 奖励摆动相足端离地高度
    gait_foot_clearance = RewTerm(
        func=mdp.gait_reward_foot_clearance,
        weight=-0.5,  # 使用负权重，因为函数返回height_error
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*3_.*"]),
            "target_height": 0.09,  # walk-these-ways: footswing_height=0.09
            "sigma": 0.02,
        }
    )

    # ========== 新增：IMU稳定性奖励 ==========
    # 惩罚躯体角速度过大（抖动、翻滚）
    base_angular_velocity = RewTerm(
        func=mdp.base_angular_velocity_penalty,
        weight=-0.2,  # 惩罚躯体不稳定
        params={
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # ========== 新增：膝盖高度奖励（防止膝盖着地）==========
    knee_height = RewTerm(
        func=mdp.knee_height_reward,
        weight=-10.0,  # 强惩罚膝盖过低
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*2_.*"]),  # 所有膝盖连杆
            "min_height": 0.08,  # 膝盖必须保持至少8cm高度
        }
    )

    # ========== 新增：小腿关节偏离惩罚（防止小腿过度弯曲）==========
    shank_joint_deviation = RewTerm(
        func=mdp.shank_joint_deviation_penalty,
        weight=-5.0,  # 较大惩罚，超过10度弯曲会受到惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*3_joint_pitch"]),  # 所有小腿pitch3关节
            "threshold_deg": 10.0,  # 允许10度偏离，超过部分会被惩罚
        }
    )

    # 移除原来的feet_air_time奖励（不再使用）
    # feet_air_time = RewTerm(...)  # 已删除

    # 足端滑动惩罚 - 鼓励稳定步态
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*3_.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*3_.*"]),
        },
    )


@configclass
class R02_2FlatEnvCfg(BaseEnvCfg):
    """Environment configuration for R02_2 on flat terrain."""

    reward = R02_2RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # 设置R02_2机器人配置
        self.scene.robot = R02_2_CFG
        self.scene.height_scanner.prim_body_name = "body"
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = FLAT_TERRAINS_CFG  # 使用完全平整的地面

        # 设置终止条件
        self.robot.terminate_contacts_body_names = ["body", ".*2_.*"]  # 躯干和膝盖接触都终止
        self.robot.feet_body_names = [".*3_.*"]

        # ========== 优化：观测历史长度（参考walk-these-ways）==========
        # walk-these-ways使用15帧观测历史用于adaptation module
        # 更长的历史帮助网络学习环境动态（摩擦、质量等）
        self.robot.actor_obs_history_length = 15  # walk-these-ways标准值
        self.robot.critic_obs_history_length = 1   # critic只需当前观测+特权观测

        # ========== 修改：速度命令范围 - 强制主要前进方向 ==========
        # 关闭heading命令，使用纯速度控制
        self.commands.heading_command = False  # 关键：禁用朝向命令
        self.commands.rel_heading_envs = 0.0   # 0%环境使用heading

        # 速度命令：主要前进
        self.commands.ranges.lin_vel_x = (0.5, 1.5)  # 强制前进，最低0.5 m/s
        self.commands.ranges.lin_vel_y = (-0.2, 0.2)  # 很小的侧向
        self.commands.ranges.ang_vel_z = (-0.8, 0.8)  # 小幅转向

        # 增加静止环境比例（学习站立）
        self.commands.rel_standing_envs = 0.1  # 10%环境练习站立

        # 显示速度命令箭头
        self.commands.debug_vis = True

        # ========== 增强：域随机化参数（参考walk-these-ways）==========
        # 质量随机化
        # walk-these-ways: added_mass_range=[-1, 3], 我们body质量1.77kg
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["body"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # 摩擦力随机化
        # walk-these-ways: friction_range=[0.5, 1.25] (更大范围[0.05, 4.5]用于粗糙地形)
        # legged_gym: friction_range=[0.5, 1.25]
        self.domain_rand.events.physics_material.params["static_friction_range"] = (0.5, 1.25)
        self.domain_rand.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.25)

        # 初始状态随机化
        # walk-these-ways: yaw_init_range=0（平地训练不旋转）
        # 但我们保持小范围yaw以增强鲁棒性
        self.domain_rand.events.reset_base.params["pose_range"]["x"] = (-0.5, 0.5)
        self.domain_rand.events.reset_base.params["pose_range"]["y"] = (-0.5, 0.5)
        self.domain_rand.events.reset_base.params["pose_range"]["yaw"] = (-0.3, 0.3)
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)

        # 推力扰动
        # walk-these-ways: push_robots=False（平地不推）
        # legged_gym: push_interval_s=15, max_push_vel_xy=1.0
        # 我们使用中等推力增强鲁棒性
        self.domain_rand.events.push_robot.interval_range_s = (10.0, 15.0)
        self.domain_rand.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0)
        }


@configclass
class R02_2FlatAgentCfg(BaseAgentCfg):
    """Agent configuration for R02_2 on flat terrain."""

    experiment_name: str = "r02_2_flat"
    wandb_project: str = "r02_2_flat"

    def __post_init__(self):
        super().__post_init__()

        # ========== PPO算法参数优化（参考优秀项目）==========
        # walk-these-ways和legged_gym的标准配置

        # 学习率（参考legged_gym: 1e-3）
        self.algorithm.learning_rate = 1e-3

        # 熵系数（参考legged_gym: 0.01）
        self.algorithm.entropy_coef = 0.01

        # KL divergence目标（参考legged_gym: 0.01）
        self.algorithm.desired_kl = 0.01

        # 初始噪声（参考legged_gym: 1.0）
        self.policy.init_noise_std = 1.0

        # ========== Rollout长度优化 ==========
        # walk-these-ways使用24 steps per env
        # legged_gym使用24 steps per env
        # 较短的rollout适合fast-paced locomotion任务
        self.num_steps_per_env = 24  # 从256降低到24（walk-these-ways标准）

        # ========== 网络架构 ==========
        # walk-these-ways: [512, 256, 128]
        # legged_gym anymal_c_flat: [128, 64, 32]（平地用小网络）
        # 我们使用标准的[512, 256, 128]以获得更强的表达能力
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class R02_2RoughEnvCfg(R02_2FlatEnvCfg):
    """Environment configuration for R02_2 on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        # 启用高度扫描用于粗糙地形
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        # 使用历史观测
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1


@configclass
class R02_2RoughAgentCfg(BaseAgentCfg):
    """Agent configuration for R02_2 on rough terrain."""

    experiment_name: str = "r02_2_rough"
    wandb_project: str = "r02_2_rough"
