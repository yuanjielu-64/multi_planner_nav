# VLM-DPT Adaptive Planner Selection System

基于视觉语言模型(VLM)的自适应规划器选择系统，结合强化学习和监督学习方法，在不同场景下智能切换最优导航规划算法。

## 1. 项目概述

本项目实现了一个智能的自适应导航系统，主要包含以下功能：

- **自适应规划器选择**: 根据场景特征自动切换最优规划算法（DWA/TEB/MPPI/DDP）
- **强化学习决策**: 基于TD3算法学习场景到规划器的映射策略
- **视觉语言模型集成**: Qwen2.5-VL + DPT Head用于场景理解和规划器选择
- **强化学习微调(RLFT)**: 将VLM预训练模型与TD3结合进行微调
- **多规划器支持**: 集成DWA、TEB、MPPI、DDP四种主流规划算法
- **仿真环境**: 基于Gazebo的BARN Challenge测试环境

**应用场景**:
- BARN (Benchmark Autonomous Robot Navigation) Challenge
- 室内外复杂环境的自主导航
- 动态场景下的最优规划器选择
- 不同环境特征（狭窄通道、开阔空间、密集障碍物等）的自适应导航

## 2. 系统架构

### 主要模块

```
src/
├── ros_jackal/                    # 强化学习训练和VLM集成
│   ├── td3/                       # TD3强化学习算法
│   ├── rlft/                      # VLM+DPT强化学习微调
│   ├── envs/                      # 导航环境定义
│   └── script/                    # 训练脚本和配置
│
├── dynamics_planner_nav/          # 多规划器导航包
│   ├── scripts/                   # 运行脚本（DDP/MPPI/DWA/TEB）
│   ├── launch/                    # ROS launch文件
│   ├── config/                    # 各规划器配置
│   └── params/                    # 各规划器参数
│
├── qwen_dpt/                      # Qwen视觉模型微调  (目前用不上)
│   └── lmms-finetune-qwen/        # VLM+DPT训练框架
│       ├── models/                # 模型定义
│       ├── trainers/              # 训练器
│       └── configs/               # 训练配置
│
├── jackal_setup/                  # Jackal机器人底层电机配置 (目前用不上)
│   ├── jackal_robot/              # 机器人硬件接口
│   ├── jackal_desktop/            # 可视化工具
│   └── jackal/                    # 导航配置
│
└── Imitation_Learning/            # 模仿学习 (目前用不上)
```

### 系统流程

```
传感器数据 → Costmap生成/LaserScan生成 → 场景分析(VLM/CNN) → 规划器选择决策
                                                              ↓
                                                  [DWA | TEB | MPPI | DDP]
                                                              ↓
                                                          运动规划 → 执行
                                                              ↓
              强化学习反馈(成功率/效率) ←─────────────────────────┘
```

**规划器选择策略**:
- **DWA**:  
- **TEB**: 
- **MPPI**: 
- **DDP**:  

## 3. 环境要求

### 操作系统
- Ubuntu 20.04
- ROS Noetic

### Python环境
- Python 3.8+
- PyTorch >= 1.10.1
- transformers >= 4.30.0

### GPU要求
- **强化学习训练**: 单GPU即可 (推荐 >= 8GB VRAM)
- **VLM微调**: 推荐 >= 24GB VRAM (或使用4-bit量化)
- **RLFT训练**: 推荐 >= 40GB VRAM (A100或多卡)

## 4. 核心模块详解

**📖 新手阅读指南**:

如果你是第一次接触这个项目，建议按以下顺序阅读：

```
第1步：了解"工具"（4个规划器是什么）
   ↓
   👉 先看 4.1 dynamics_planner_nav

第2步：了解"训练框架"（怎么学习规划器选择）
   ↓
   👉 再看 4.2 ros_jackal 的整体介绍

第3步：深入"环境实现"（Gym环境和切换机制）
   ↓
   👉 然后看 4.2.2 envs环境定义

第4步：学习"运行脚本"（怎么测试和训练）
   ↓
   👉 最后看 4.2.1 script脚本使用
```

---

### 4.1 dynamics_planner_nav - 多规划器导航包

**这个模块是干什么的？**

这个模块提供了**4个不同的导航规划器**，每个规划器适合不同的场景。

**类比理解**: 就像你有4种不同的交通工具：
- 🚗 **DWA** = 小轿车（快速，适合高速公路）
- 🚐 **TEB** = 面包车（灵活，适合狭窄街道）
- 🚙 **MPPI** = SUV（鲁棒，适合复杂路况）
- 🚛 **DDP** = 卡车（精确，适合特殊场景）

**本项目的核心创新**: 让AI自动学会在不同场景下选择最合适的"交通工具"！

---

#### 4.1.1 四种规划器详解

**🚗 DWA (Dynamic Window Approach) - 快速规划器**

- **优点**: 计算速度快、实时性好
- **缺点**: 容易陷入局部最优、难以处理狭窄通道
- **适用场景**: 宽敞环境、低动态障碍物、需要快速反应
- **特点**: 在速度空间中采样，选择最优速度组合

**🚐 TEB (Timed Elastic Band) - 平滑规划器**

- **优点**: 轨迹平滑、考虑时间优化、能处理狭窄通道
- **缺点**: 计算复杂度较高
- **适用场景**: 狭窄通道、需要平滑轨迹、有时间约束
- **特点**: 生成弹性带轨迹，自动调整以避障

**🚙 MPPI (Model Predictive Path Integral) - 鲁棒规划器**

- **优点**: 鲁棒性强、处理复杂约束、随机采样避免局部最优
- **缺点**: 需要大量采样、计算开销大
- **适用场景**: 复杂环境、多约束条件、需要高鲁棒性
- **特点**: 随机采样轨迹，加权选择最优路径

**🚛 DDP (Dynamics Dynamic Programming) - 精确规划器**

- **优点**: 全局最优、高精度轨迹
- **缺点**: 计算开销大、需要精确模型、可能不适合实时
- **适用场景**: 高精度要求、离线规划、已知环境模型
- **特点**: 迭代优化轨迹，考虑动力学约束

---

#### 4.1.2 规划器配置文件

**配置文件位置**:
```
dynamics_planner_nav/params/
├── base_local_planner_params.yaml  # 基础局部规划器参数
├── costmap_common_params.yaml      # 通用Costmap配置
├── global_planner_params.yaml      # 全局规划器参数
├── move_base_params.yaml           # MoveBase参数
└── odom_nav_params/                # 里程计导航参数
    ├── local_costmap_params.yaml
    └── global_costmap_params.yaml
```

**DWA关键参数**:
- `max_vel_x`: 最大线速度
- `max_vel_theta`: 最大角速度
- `vx_samples` / `vtheta_samples`: 速度采样数
- `path_distance_bias`: 路径距离权重
- `goal_distance_bias`: 目标距离权重
- `occdist_scale`: 障碍物距离权重

**TEB关键参数**:
- `min_obstacle_dist`: 最小障碍物距离
- `weight_kinematics_nh`: 非完整约束权重
- `weight_optimaltime`: 时间优化权重
- `enable_homotopy_class_planning`: 多路径规划

---

### 4.2 ros_jackal - 强化学习训练框架

**这个模块是干什么的？**

这个模块让机器人能够：
- 在Gazebo仿真环境中导航
- **自动学习在不同场景下选择最优规划器**（DWA/TEB/MPPI/DDP）
- 评估不同规划器的性能

**类比理解**: 就像训练一个司机学习在不同路况下选择最佳交通工具：
- 看到高速公路 → 选择小轿车（DWA）
- 看到狭窄街道 → 选择面包车（TEB）
- 看到复杂路况 → 选择SUV（MPPI）

---

#### 4.2.1 script目录结构

```
script/
├── applr/                                    # 基础生成baseline脚本
│   ├── configs/                              # 各规划器配置文件
│   │   ├── DWA.yaml                          # DWA规划器配置
│   │   ├── TEB.yaml                          # TEB规划器配置
│   │   ├── MPPI.yaml                         # MPPI规划器配置
│   │   └── DDP.yaml                          # DDP规划器配置
│   ├── evaluate_applr_single.py              # 单环境评估脚本
│   └── tmux_eval_applr_clients.sh            # 通过tmux批量评估脚本
|   └── eval_batch_worlds_singularity.sh      # 开启单个进程开始生成指定多个环境指定规划器的脚本 (重要！)
|   └── eval_single_worlds_singularity.sh     # 开启单个进程开始生成指定一个环境指定规划器的脚本 (重要！)(用于测试)
│
├── qwen/                          # Qwen VLM推理服务 (目前用不上)
│   ├── qwen_server.py             # VLM推理服务器
│   ├── qwen_client.py             # VLM客户端
│   ├── evaluate_qwen_single.py    # 评估VLM脚本
│   └── start_qwen_service.sh      # 启动服务脚本
│
├── ft_qwen/                        # RLFT训练脚本   (目前用不上)
│   ├── configs/                    # RLFT配置
│   ├── qwen_server.py             # RLFT推理服务器
│   ├── qwen_client.py             # RLFT客户端
│   ├── run_ftrl.sh                # RLFT训练启动脚本
│   └── evaluate_ftrl_single.py    # RLFT评估脚本
│
├── IL/                             # 模仿学习脚本（可选）  (目前用不上)
└── extract_lora_from_checkpoint.py # LoRA权重提取工具
```

#### APPLR - 基础框架+强化学习 (`script/applr/`)

使用TD3学习各规划器的最优参数。

**配置文件说明** (`configs/`):

每个配置文件定义了：
1. **环境配置** (`env_config`):
   - `env_id`: 环境类型（如`dwa_param-v0`）
   - `action_type`: 规划器类型（`dwa_local`, `teb_local`, `mppi_local`, `ddp_local`）
   - `param_list`: 要学习的参数列表
   - `param_init`: 参数初始值

2. **训练配置** (`training_config`):
   - `network`: 网络类型（`mlp`或`cnn`）
   - `actor_lr` / `critic_lr`: 学习率
   - `max_step`: 总训练步数

**测试/评估单个规划器 (Ubuntu 20.04原生)**:

直接在Ubuntu 20.04系统上运行评估脚本。

```bash
cd src/ros_jackal/script/applr/

python evaluate_applr_single.py \
  --world_id 0 \                    # 指定BARN环境ID (0-299)
  --policy_name ddp \                # 规划器类型: dwa/teb/mppi/ddp
  --buffer_path ../../buffer/ \     # 数据保存路径
  --world_path ../../jackal_helper/worlds/BARN1/ \  # Gazebo世界文件路径
  --ros_port 11311 \                # ROS Master端口 (多进程时需不同端口)
  --mode auto \                     # 运行模式 (auto: 自动运行)
  --save_image False \              # 是否保存每步的观测图像
  --algorithm_name STATIC \         # 算法名称 (STATIC: 使用默认参数，不更新)
  --num_trials 3                    # 每个环境运行次数
```

**参数说明**:
- `--world_id`: BARN环境编号，范围0-299（静态环境）或300+（动态环境）
- `--policy_name`: 规划器名称，决定使用哪个规划器和配置文件
- `--buffer_path`: 评估数据（轨迹、奖励等）保存位置
- `--ros_port`: ROS Master端口，并行运行时必须使用不同端口（如11311, 11313, 11315...）
- `--algorithm_name`:
  - `STATIC`: 使用默认参数，不进行学习
  - `TD3`/`SAC`: 使用强化学习算法
- `--save_image`: 是否保存Costmap/LaserScan图像用于后续分析
- `--num_trials`: 重复次数，用于统计平均性能

---

**批量评估多个环境 (Ubuntu 24.04容器)**:

在Ubuntu 24.04上通过Apptainer容器运行Ubuntu 20.04环境。

```bash
cd src/ros_jackal/script/applr

bash eval_batch_worlds_singularity.sh \
  --id 2 \              # 进程ID，决定端口号 (ros_port = 11311 + id*2)
  --start 200 \         # 起始环境ID
  --end 299 \           # 结束环境ID
  --policy teb          # 规划器类型: dwa/teb/mppi/ddp
```

**参数说明**:
- `--id`: 进程标识符，用于计算ROS端口（避免冲突）
  - 例如: `id=0` → `ros_port=11311`, `id=2` → `ros_port=11315`
  - 并行运行时，每个进程必须使用不同的ID
- `--start` / `--end`: 批量评估的环境范围
  - 示例: `--start 0 --end 99` 评估前100个环境
  - 示例: `--start 200 --end 299` 评估后100个环境
- `--policy`: 规划器名称，与配置文件对应（`configs/{policy}.yaml`）

**容器说明**:
- 脚本内部调用`singularity_run.sh`启动容器
- 容器自动挂载工作目录和数据目录
- 评估结果保存在宿主机的`buffer/`目录

**典型使用场景**:

1. **单环境快速测试** (Ubuntu 20.04):
```bash
python evaluate_applr_single.py --world_id 0 --policy_name dwa --num_trials 1
```

2. **批量评估全部环境** (Ubuntu 24.04):
```bash
# 分多个进程并行评估
bash eval_batch_worlds_singularity.sh --id 0 --start 0 --end 99 --policy dwa &
bash eval_batch_worlds_singularity.sh --id 1 --start 100 --end 199 --policy dwa &
bash eval_batch_worlds_singularity.sh --id 2 --start 200 --end 299 --policy dwa &
```

3. **对比不同规划器**:
```bash
for planner in dwa teb mppi ddp; do
  python evaluate_applr_single.py --world_id 0 --policy_name $planner --num_trials 5
done
```

---

#### 4.2.2 envs环境定义 - 机器人如何与Gazebo交互

**这个模块是干什么的？**

这个模块是**机器人和仿真环境之间的桥梁**，它定义了：
- 机器人能做什么（action）：选择规划器或调整参数
- 机器人能看到什么（observation）：激光雷达数据、地图等
- 机器人得到什么反馈（reward）：成功/失败/碰撞的奖励

**类比理解**:
- 就像游戏的API接口，定义了玩家的操作（选择武器）、游戏画面（视野）、得分规则（奖励）
- `envs/`就是连接"AI大脑"和"Gazebo世界"的接口

**为什么使用OpenAI Gym？**
- Gym是强化学习的标准接口，让我们可以用统一的方式训练AI
- 任何RL算法（TD3、SAC、PPO等）都可以直接使用这个环境

**目录结构**:
```
envs/
├── registration.py              # 环境注册入口
├── wrappers.py                  # 环境包装器
├── DWA/                         # DWA规划器环境
│   ├── dwa_base_envs.py        # 基础环境类
│   └── parameter_tuning_envs.py # 参数调优环境
├── TEB/                         # TEB规划器环境
├── MPPI/                        # MPPI规划器环境
├── DDP/                         # DDP规划器环境
├── utils/                       # 工具类（核心）
│   ├── gazebo_simulation.py    # Gazebo控制接口
│   ├── Jackal_ros.py           # Jackal机器人状态管理
│   ├── DWA_move_base.py        # DWA MoveBase接口
│   ├── TEB_move_base.py        # TEB MoveBase接口
│   ├── MPPI_move_base.py       # MPPI MoveBase接口
│   └── DDP_move_base.py        # DDP MoveBase接口
```

---

---

##### 📌 快速理解：envs模块的三层结构

```
┌─────────────────────────────────────────────────────────┐
│  1. registration.py - 注册环境（入口）                    │
│     → 告诉Python: "我有DWA/TEB/MPPI/DDP这4个环境"        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  2. {planner}_base_envs.py - 基础环境（核心逻辑）          │
│     → 启动Gazebo、加载地图、控制机器人移动                   │
│     → reset(): 重置环境（新的起点和终点）                   │
│     → step(): 执行一步（更新规划器、获取奖励）               │
│     → 定义reward: 成功+10, 碰撞-5, 每步-0.01              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  3. parameter_tuning_envs.py - 定义AI接口                │
│     → 定义action: 选择哪个规划器 (0/1/2/3)                 │
│     → 定义observation: 激光雷达数据 (720个点)              │
└─────────────────────────────────────────────────────────┘
```

---

##### 1️⃣ 环境注册 (`registration.py`) - 让Gym认识我们的环境

**作用**: 像"注册账号"一样，告诉OpenAI Gym我们有哪些环境可用。

**已注册的环境**:
```python
'dwa_param-v0'   → DWA规划器环境
'teb_param-v0'   → TEB规划器环境
'mppi_param-v0'  → MPPI规划器环境
'ddp_param-v0'   → DDP规划器环境
```

**如何使用** (就像创建游戏角色):
```python
import gym

# 创建一个DWA环境（相当于选择"简单模式"）
env = gym.make("dwa_param-v0",
               world_name="world_0.world",  # 选择地图
               gui=False)                   # 不显示GUI

# 重置环境，开始新游戏
obs = env.reset()  # obs是机器人"看到"的激光雷达数据
```

---

##### 2️⃣ 基础环境类 (`{planner}_base_envs.py`) - Gazebo控制中心

**作用**: 这是环境的"大脑"，负责启动和控制整个仿真系统。

**类比**: 就像游戏的游戏引擎，负责加载地图、生成角色、处理物理碰撞。

---

**1. `__init__()` - 启动仿真环境**

当你创建环境时（`gym.make()`），这个函数会：
```python
def __init__(...):
    # 步骤1: 启动Gazebo仿真器（3D物理世界）
    self.launch_gazebo(world_name="world_0.world", gui=False)

    # 步骤2: 启动ROS导航系统（MoveBase）
    self.launch_move_base(planner="DWA")  # 使用DWA规划器

    # 步骤3: 创建3个工具类（下面会详细介绍）
    self.gazebo_sim = GazeboSimulation()  # 控制Gazebo
    self.jackal_ros = JackalRos()         # 获取机器人状态
    self.move_base = DWA_MoveBase()       # 控制导航
```

**简单理解**:
- 像启动一个游戏：先加载3D引擎（Gazebo），再加载角色AI（MoveBase），最后连接控制器（工具类）

---

**2. `reset()` - 开始新一轮**

每次训练新的episode时调用，相当于"重新开始游戏"：
```python
def reset(self):
    # 1. 暂停物理引擎（freeze画面）
    self.gazebo_sim.pause()

    # 2. 把机器人传送回起点
    self.gazebo_sim.reset(position=[-2, 3, 1.57])

    # 3. 清空地图缓存
    self.move_base.clear_costmap()

    # 4. 设置新目标点
    self.move_base.send_goal([0, 10, 0])

    # 5. 恢复物理引擎（继续游戏）
    self.gazebo_sim.unpause()

    # 6. 返回初始观测（激光雷达数据）
    obs = self.jackal_ros.get_laser_scan()
    return obs
```

**简单理解**:
- 就像按"重新开始"按钮：机器人回到起点，重新设定目标，开始新的导航任务

---

**3. `step(action)` - 执行一个动作**

这是最核心的函数，AI每一步都会调用它：
```python
def step(self, action):
    # 1. 执行AI的决策
    if action == 0:
        self.switch_planner("DWA")   # 切换到DWA规划器
    elif action == 1:
        self.switch_planner("TEB")   # 切换到TEB规划器
    # ...

    # 2. 等待0.5秒（让规划器运行）
    time.sleep(0.5)

    # 3. 暂停，读取结果
    self.gazebo_sim.pause()
    obs = self.jackal_ros.get_laser_scan()      # 看到了什么
    reward = self._calculate_reward()            # 得到了多少分
    done = self.jackal_ros.reached_goal()        # 是否完成
    self.gazebo_sim.unpause()

    return obs, reward, done, {}
```

**简单理解**:
- 就像游戏的每一帧：AI做决策（选武器） → 等待执行 → 查看结果（得分/血量/是否过关）

---

**环境的3个"工具人"** (重要成员变量):

| 工具类 | 作用 | 类比 |
|--------|------|------|
| `self.gazebo_sim` | 控制Gazebo（暂停/播放/重置） | 游戏的"时间控制器" |
| `self.jackal_ros` | 获取机器人状态（位置/激光/速度） | 游戏的"视野系统" |
| `self.move_base` | 控制导航（切换规划器/发送目标） | 游戏的"AI控制器" |

---

##### 3️⃣ AI接口定义 (`parameter_tuning_envs.py`) - 定义AI能做什么

**作用**: 这个文件定义了AI的"游戏规则"——它能做什么、能看到什么、怎么得分。

**类比**: 就像游戏设计文档，定义了：
- 玩家操作（键盘按键 → AI的action）
- 画面显示（屏幕分辨率 → observation的维度）
- 得分规则（击杀+100分 → reward函数）

---

**本项目的两种玩法** (可以二选一):

---

**🎯 模式1: 规划器选择** (本项目核心 - 推荐)

**目标**: 让AI学会"看场景选策略"，就像司机看路况选车道。

**定义1: AI能做什么？** (Action空间)
```python
# 4选1：选择使用哪个规划器
self.action_space = Discrete(4)

# AI每步输出一个数字：
# 0 → 使用DWA  (快速，适合开阔空间)
# 1 → 使用TEB  (平滑，适合狭窄通道)
# 2 → 使用MPPI (鲁棒，适合复杂环境)
# 3 → 使用DDP  (精确，适合高精度场景)
```

**定义2: AI能看到什么？** (Observation空间)
```python
# 激光雷达数据 + 地面摩擦系数：720 + 1 = 721维
self.observation_space = Box(
    low=np.array([0.0]*720 + [0.0]),     # 激光范围0-10米，摩擦系数0-2
    high=np.array([10.0]*720 + [2.0]),
    shape=(721,),                         # 720个激光点 + 1个摩擦系数
    dtype=np.float32
)

# 例如: obs = [2.5, 2.3, 2.1, ..., 5.8, 6.0, 0.8]
#       → 前方2.5米有障碍物，左边5.8米有墙...
#       → 最后一维0.8 = 地面摩擦系数 (冰面≈0.1, 普通地面≈0.8, 橡胶≈1.5)
```

**为什么要加入摩擦系数？**
- 不同地面条件（冰面、湿地、沙地、正常路面）对规划器性能影响很大
- DWA在高摩擦力地面反应快，低摩擦力（冰面）容易失控
- TEB在低摩擦力地面更稳定（轨迹平滑）
- AI可以根据地面条件选择最合适的规划器

**类比理解**:
- 就像开车前看路况：看到结冰路面就切换到"雪地模式"
- 摩擦系数 = 路面的"抓地力"信号

**如何在代码中实现？**

在 `{planner}_base_envs.py` 中添加摩擦系数：

```python
class DWABaseLaser(DWABase):
    def __init__(self, laser_clip=4, friction_coeff=0.8, **kwargs):
        super().__init__(**kwargs)
        self.friction_coeff = friction_coeff  # 初始化摩擦系数

        self.observation_space = Box(
            low=0, high=laser_clip,
            shape=(722,),  # 720(laser) + 1(goal) + 1(friction)
            dtype=np.float32
        )

    def _get_observation(self):
        laser_scan = self._get_laser_scan()       # 获取激光数据
        local_goal = self._get_local_goal()       # 获取局部目标
        friction = np.array([self.friction_coeff]) # 添加摩擦系数

        # 拼接成完整的observation
        obs = np.concatenate([laser_scan, local_goal, friction])
        return obs
```

**如何设置不同的摩擦系数？**

在创建环境时传入 `friction_coeff` 参数：

```python
import gym

# 冰面环境 (低摩擦)
env_ice = gym.make("dwa_param-v0", friction_coeff=0.1)

# 普通地面
env_normal = gym.make("dwa_param-v0", friction_coeff=0.8)

# 高摩擦地面 (橡胶)
env_rubber = gym.make("dwa_param-v0", friction_coeff=1.5)
```

**在Gazebo中设置地面摩擦力**

修改 `.world` 文件中的地面材质：

```xml
<surface>
  <friction>
    <ode>
      <mu>0.8</mu>   <!-- 摩擦系数，范围0-2 -->
      <mu2>0.8</mu2>
    </ode>
  </friction>
</surface>
```

然后在创建环境时，将这个摩擦系数同步传给observation。

---

**定义3: AI如何执行动作？** (_take_action函数)
```python
def _take_action(self, action):
    # 把数字映射到实际规划器
    planner_names = {
        0: "DWA规划器",
        1: "TEB规划器",
        2: "MPPI规划器",
        3: "DDP规划器"
    }

    # 切换到选定的规划器
    self.move_base.switch_planner(planner_names[action])

    # 类比: 就像切换汽车的驾驶模式（运动/舒适/越野）
```

**定义4: AI如何获得奖励？** (Reward函数)
```python
def _get_reward(self):
    reward = 0

    # 1. 基础奖励
    if 到达目标:
        reward = +10.0        # 成功！给大奖励
    elif 撞到障碍物:
        reward = -5.0         # 失败！扣分
    else:
        reward = -0.01        # 每步小扣分（鼓励快速完成）

    # 2. 效率奖励（鼓励选快的规划器）
    if 成功 and 用时少:
        reward += 时间奖励    # 20步完成比50步完成奖励更多

    # 3. 平滑度奖励（鼓励选轨迹好的规划器）
    if 路径平滑:
        reward += 0.5         # 路径不拐弯太急，奖励

    return reward

# 类比：游戏得分 = 通关奖励 + 速通奖励 + 无伤奖励
```

**为什么这样设计？**
- Action是离散的（4选1）→ 搜索空间小，容易学习
- Observation是激光雷达 → 包含足够的场景信息（障碍物位置、通道宽度）
- Reward鼓励成功、快速、平滑 → AI会学会在不同场景选最优规划器

---

**⚙️ 模式2: 参数优化** (传统APPLR - 用于对比)

**目标**: 微调单个规划器的参数，就像调车的悬挂硬度、油门灵敏度。

**AI能做什么？**
```python
# 7个连续参数（以DWA为例）
self.action_space = Box(
    low  = [0.1, 0.5,  3, 10, 0.1, 0.1, 0.2],  # 最小值
    high = [1.0, 3.0, 20, 40, 5.0, 5.0, 1.0],  # 最大值
    shape=(7,)
)

# AI每步输出7个数字：
# [0.5, 1.2, 10, 20, 1.5, 2.0, 0.4]
# ↓
# 最大速度=0.5m/s, 最大转速=1.2rad/s, ...
```

**为什么不推荐？**
- Action空间太大（7维连续） → 难学习
- 只能优化单个规划器 → 局限性大
- 需要大量样本（5M steps） → 训练慢

**用途**: 作为Baseline对比，证明"规划器选择"比"参数优化"更好

---

**📊 两种模式对比**:

| 对比维度 | 🎯 模式1: 规划器选择 | ⚙️ 模式2: 参数优化 |
|---------|---------------------|-------------------|
| **Action** | 离散4选1 | 连续7维 |
| **搜索空间** | 4种选择 | 10^7种组合 |
| **学习难度** | 简单 ⭐ | 困难 ⭐⭐⭐ |
| **样本需求** | <1M steps | >5M steps |
| **泛化能力** | 强（4个规划器互补） | 弱（单一规划器） |
| **适用场景** | 所有场景自动适配 | 只适合特定场景 |
| **本项目定位** | ✅ 核心创新 | ❌ Baseline |

**类比总结**:
- 模式1 = 学会"换车"：高速开跑车，越野开吉普
- 模式2 = 学会"调车"：只有一辆车，调悬挂调油门

显然模式1更灵活、更高效！

---

##### 4️⃣ 工具类 (`utils/`) - 核心支持模块

**a) `gazebo_simulation.py` - Gazebo控制**

提供Gazebo物理引擎的控制接口。

**核心方法**:
```python
class GazeboSimulation:
    def __init__(self, init_position):
        # ROS服务代理
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # 碰撞监听
        self._collision_sub = rospy.Subscriber("/collision", Bool, self.collision_monitor)

    def pause(self):
        """暂停物理仿真（用于数据采集）"""
        self._pause()

    def unpause(self):
        """恢复物理仿真"""
        self._unpause()

    def reset(self, position):
        """重置机器人位置"""
        model_state = create_model_state(position[0], position[1], position[2])
        self._reset(model_state)

    def get_hard_collision(self):
        """获取碰撞状态"""
        return self.collision_count > 0
```

**关键点**:
- `pause()/unpause()`用于freeze仿真，便于安全地读取状态
- 支持碰撞检测、速度监控

---

**b) `Jackal_ros.py` - 机器人状态管理**

获取机器人的各种状态信息（位置、速度、传感器等）。

**核心功能**:
```python
class JackalRos:
    def __init__(self):
        # 订阅各种ROS话题
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap',
                                           OccupancyGrid, self.costmap_callback)
        self.path_sub = rospy.Subscriber('/move_base/TrajectoryPlannerROS/local_plan',
                                        Path, self.path_callback)

    # 获取状态方法
    def get_position(self):
        """获取机器人位置 (x, y, yaw)"""
        return self.position

    def get_velocity(self):
        """获取机器人速度 (vx, vy, omega)"""
        return self.velocity

    def get_laser_scan(self):
        """获取激光雷达数据"""
        return self.laser_ranges  # shape: (720,)

    def get_local_costmap(self):
        """获取局部costmap"""
        return self.costmap  # shape: (160, 160)

    def get_local_plan(self):
        """获取规划的局部路径"""
        return self.local_path
```

**支持的传感器**:
- **Odometry**: 机器人位姿 (x, y, θ)
- **LaserScan**: 720点激光雷达数据
- **Costmap**: 局部代价地图 (用于训练CNN)
- **Path**: 规划器输出的局部路径

**预定义参数** (PLANNER_PARAMS):
```python
PLANNER_PARAMS = {
    "DWA": ["max_vel_x", "max_vel_theta", "vx_samples", "vtheta_samples",
            "path_distance_bias", "goal_distance_bias", "inflation"],

    "TEB": ["max_vel_x", "max_vel_theta", "min_obstacle_dist",
            "weight_kinematics", "weight_obstacle", "inflation"],

    "MPPI": ["num_samples", "horizon_length", "temperature",
             "max_vel_x", "inflation"],

    "DDP": ["iterations", "horizon", "max_vel_x",
            "regularization", "inflation"]
}
```

---

**c) `{PLANNER}_move_base.py` - MoveBase接口**

每个规划器对应一个MoveBase接口类，用于与ROS导航栈交互。

**核心方法**:
```python
class dwa_MoveBase:
    def __init__(self):
        # MoveBase Action Client
        self.move_base_client = actionlib.SimpleActionClient(
            'move_base', MoveBaseAction
        )

        # Dynamic Reconfigure Client (动态参数更新)
        self.dwa_client = dynamic_reconfigure.client.Client(
            '/move_base/TrajectoryPlannerROS',
            timeout=5
        )

    def send_goal(self, goal_position):
        """发送导航目标"""
        goal = _create_MoveBaseGoal(goal_position[0], goal_position[1], goal_position[2])
        self.move_base_client.send_goal(goal)

    def update_params(self, params):
        """动态更新DWA参数"""
        # params: {'max_vel_x': 0.5, 'max_vel_theta': 1.0, ...}
        self.dwa_client.update_configuration(params)

    def get_state(self):
        """获取MoveBase状态 (ACTIVE, SUCCEEDED, ABORTED, ...)"""
        return self.move_base_client.get_state()

    def clear_costmap(self):
        """清除costmap"""
        rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
```

**支持的操作**:
- 发送/取消导航目标
- 动态更新规划器参数
- 清除costmap
- 获取导航状态

---

##### 5️⃣ 传感器接口 (`sensors/`)

**`laser.py`**: 激光雷达数据处理
- 订阅 `/scan` 话题
- 提供距离数据、点云转换

**`camera.py`**: 摄像头接口（目前未使用）

---

##### 6️⃣ 使用流程示例

```python
import gym
from envs import registration

# 1. 创建环境
env = gym.make("dwa_param-v0",
               world_name="world_0.world",
               gui=False,
               init_position=[-2, 3, 1.57],
               goal_position=[0, 10, 0],
               max_step=100,
               ros_port=11311)

# 2. 重置环境
obs = env.reset()
# obs: Costmap图像 (84, 84, 1) 或 LaserScan (720,)

# 3. 执行动作
action = [0.5, 1.0, 10, 20, 1.0, 1.0, 0.3]  # DWA参数
next_obs, reward, done, info = env.step(action)

# 4. 如果action=None，使用默认参数
next_obs, reward, done, info = env.step(None)  # 不更新参数

# 5. 关闭环境
env.close()
```

---

##### 7️⃣ 关键设计点

1. **Pause/Unpause机制**:
   - 在读取状态前`pause()`，确保数据一致性
   - 读取完成后`unpause()`，继续仿真

2. **动态参数更新**:
   - 通过`dynamic_reconfigure`实时更新规划器参数
   - 无需重启导航栈

3. **多进程支持**:
   - 通过`ros_port`和`gazebo_port`支持并行训练
   - 每个环境独立的ROS Master

4. **观测模式**:
   - **Costmap**: CNN输入 (84x84图像)
   - **LaserScan**: MLP输入 (720维向量)

5. **奖励设计** (在`_get_reward()`中):
   - 成功到达: `+success_reward`
   - 碰撞: `+collision_reward` (负值)
   - 时间步惩罚: `+slack_reward` (负值)
   - 平滑度奖励: 基于速度变化

---
