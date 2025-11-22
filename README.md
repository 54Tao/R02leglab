# R02 Velocity Tracking

RO2 四足机器人速度跟随强化学习训练项目，基于 IsaacLab 和 RSL-RL 框架。

## 项目简介

本项目实现了 RO2 四足机器人的速度跟随控制训练。主要特点：

- 基于 PPO 算法的速度跟随训练
- 支持平地和复杂地形训练
- 多种机器人支持（RO2、Unitree Go2 等）
- 可配置的步态生成器

## 文件结构

```
R02leglab/
├── README.md
├── setup.py
└── legged_lab/
    ├── __init__.py
    ├── assets/                     # 机器人资产配置
    │   ├── r02_2/                  # RO2 机器人
    │   ├── unitree/                # Unitree 系列
    │   └── fftai/                  # FFTAI 机器人
    ├── envs/                       # 环境定义
    │   ├── base/                   # 基础环境类
    │   │   ├── base_env.py
    │   │   ├── base_env_config.py
    │   │   └── gait_generator.py   # 步态生成器
    │   └── r02_2/                  # RO2 环境配置
    │       ├── r02_2_config.py     # 速度跟随配置
    │       └── r02_2_backflip_config.py
    ├── mdp/                        # MDP 组件
    │   ├── __init__.py
    │   └── rewards.py              # 奖励函数
    ├── scripts/                    # 脚本
    │   ├── train.py                # 训练脚本
    │   └── play.py                 # 可视化脚本
    ├── terrains/                   # 地形配置
    │   ├── ray_caster.py
    │   └── terrain_generator_cfg.py
    └── utils/                      # 工具函数
        ├── cli_args.py
        ├── task_registry.py
        └── keyboard.py
```

## 环境要求

- Ubuntu 22.04
- CUDA 11.8+
- Python 3.10
- Isaac Sim 4.5+
- IsaacLab
- RSL-RL

## 使用方法

### 1. 安装依赖

```bash
# 安装 IsaacLab（参考官方文档）
# https://isaac-sim.github.io/IsaacLab/

# 安装 RSL-RL
pip install rsl-rl

# 安装本项目
cd R02leglab
pip install -e .
```

### 2. 准备机器人模型

修改 `legged_lab/assets/r02_2/r02_2.py` 中的 USD 路径：

```python
usd_path="/path/to/your/robot.usd",
```

### 3. 训练

```bash
cd /path/to/IsaacLab

# 平地训练
./isaaclab.sh -p /path/to/R02leglab/legged_lab/scripts/train.py \
    --task r02_2_flat \
    --headless \
    --num_envs 4096 \
    --max_iterations 1500 \
    --logger tensorboard

# 地形训练
./isaaclab.sh -p /path/to/R02leglab/legged_lab/scripts/train.py \
    --task r02_2_rough \
    --headless \
    --num_envs 4096 \
    --max_iterations 3000 \
    --logger tensorboard
```

### 4. 可视化测试

```bash
./isaaclab.sh -p /path/to/R02leglab/legged_lab/scripts/play.py \
    --task r02_2_flat \
    --num_envs 4 \
    --load_run /path/to/logs/experiment_name \
    --checkpoint model_1000.pt
```

### 5. TensorBoard 监控

```bash
tensorboard --logdir=/path/to/logs --port=6006
```

## 训练配置说明

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 4096 | 并行环境数 |
| `max_episode_length_s` | 20.0 | 单次训练时长 |
| `stiffness` | 20.0 | PD控制器刚度 |
| `damping` | 0.5 | PD控制器阻尼 |
| `action_scale` | 0.25 | 动作缩放 |

### 速度命令范围

| 命令 | 范围 | 说明 |
|------|------|------|
| `lin_vel_x` | [-1.0, 1.0] | 前后速度 (m/s) |
| `lin_vel_y` | [-0.5, 0.5] | 侧向速度 (m/s) |
| `ang_vel_z` | [-1.0, 1.0] | 转向速度 (rad/s) |

### 奖励函数

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `track_lin_vel_xy_exp` | +1.5 | 线速度跟踪 |
| `track_ang_vel_z_exp` | +0.75 | 角速度跟踪 |
| `feet_air_time` | +0.125 | 足端滞空时间 |
| `action_rate_l2` | -0.01 | 动作平滑性 |
| `dof_torques_l2` | -1e-5 | 力矩惩罚 |

## 坐标系说明

RO2 机器人坐标系：

| 轴 | 方向 |
|----|------|
| Y | 前进方向 |
| X | 侧向（左） |
| Z | 垂直向上 |

## License

BSD-3-Clause
