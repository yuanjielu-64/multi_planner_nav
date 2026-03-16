# Multi-Planner Navigation System

A C++-based navigation framework and learning-based simulation system for autonomous robot navigation.

## 1. Project Overview

This project implements an intelligent adaptive navigation system with the following main components:

- **dynamics_planner_nav**: A C++-based navigation framework designed to replace move_base, supporting multiple planning algorithms (DWA/TEB/MPPI/DDP)
- **ros_jackal**: A learning-based framework for simulation training and evaluation, implementing reinforcement learning algorithms for planner selection
- **Adaptive Planner Selection**: Automatically switches between optimal planning algorithms based on scene characteristics
- **Multi-Planner Support**: Integrated DWA, TEB, MPPI, DDP planning algorithms
- **Simulation Environment**: BARN Challenge testing environment based on Gazebo

**Application Scenarios**:
- BARN (Benchmark Autonomous Robot Navigation) Challenge
- Indoor and outdoor complex environment autonomous navigation
- Optimal planner selection in dynamic scenes
- Adaptive navigation for different environmental features (narrow corridors, open spaces, dense obstacles, etc.)

## 2. System Architecture

### Main Modules

```
src/
├── ros_jackal/                    # Reinforcement learning training framework (learning-based)
│   ├── td3/                       # TD3 reinforcement learning algorithm
│   ├── envs/                      # Navigation environment definition
│   └── script/                    # Training scripts and configurations
│
└── dynamics_planner_nav/          # Multi-planner navigation package (C++-based, replaces move_base)
    ├── scripts/                   # Launch scripts (DDP/MPPI/DWA/TEB)
    ├── launch/                    # ROS launch files
    ├── config/                    # Planner configurations
    └── params/                    # Planner parameters
```

### System Workflow

```
Sensor Data → Costmap/LaserScan Generation → Scene Analysis (CNN/MLP) → Planner Selection Decision
                                                              ↓
                                                  [DWA | TEB | MPPI | DDP]
                                                              ↓
                                                      Motion Planning → Execution
                                                              ↓
                         Reinforcement Learning Feedback (Success Rate/Efficiency) ←─────────┘
```

**Planner Selection Strategy**:
- **DWA**: Fast planner for open spaces
- **TEB**: Smooth planner for narrow corridors
- **MPPI**: Robust planner for complex environments
- **DDP**: Precise planner for high-accuracy scenarios

## 3. Environment Requirements

### Operating System
- Ubuntu 20.04
- ROS Noetic

### Python Environment
- Python 3.8+
- PyTorch >= 1.10.1

### GPU Requirements
- **Reinforcement Learning Training**: Single GPU (recommended >= 8GB VRAM)

## 4. Core Module Details

**📖 Reading Guide for Beginners**:

If you are new to this project, it is recommended to read in the following order:

```
Step 1: Understand the "Tools" (What are the 4 planners?)
   ↓
   👉 Read 4.1 dynamics_planner_nav first

Step 2: Understand the "Training Framework" (How to learn planner selection?)
   ↓
   👉 Read 4.2 ros_jackal overview

Step 3: Deep dive into "Environment Implementation" (Gym environment and switching mechanism)
   ↓
   👉 Read 4.2.2 envs environment definition

Step 4: Learn "Running Scripts" (How to test and train?)
   ↓
   👉 Read 4.2.1 script usage
```

---

### 4.1 dynamics_planner_nav - Multi-Planner Navigation Package

**What does this module do?**

This module provides **4 different navigation planners**, each suitable for different scenarios. It is a C++-based navigation framework designed to replace the traditional move_base package.

**Analogy**: Like having 4 different vehicles:
- 🚗 **DWA** = Sports car (fast, suitable for highways)
- 🚐 **TEB** = Van (flexible, suitable for narrow streets)
- 🚙 **MPPI** = SUV (robust, suitable for complex road conditions)
- 🚛 **DDP** = Truck (precise, suitable for special scenarios)

**Core Innovation of this Project**: Let AI automatically learn to choose the most suitable "vehicle" in different scenarios!

---

#### 4.1.1 Four Planners Explained

**🚗 DWA (Dynamic Window Approach) - Fast Planner**

- **Advantages**: Fast computation, good real-time performance
- **Disadvantages**: Easy to fall into local optima, difficult to handle narrow corridors
- **Application Scenarios**: Spacious environments, low dynamic obstacles, requires fast response
- **Characteristics**: Samples in velocity space, selects optimal velocity combination

**🚐 TEB (Timed Elastic Band) - Smooth Planner**

- **Advantages**: Smooth trajectory, considers time optimization, can handle narrow corridors
- **Disadvantages**: Higher computational complexity
- **Application Scenarios**: Narrow corridors, requires smooth trajectories, has time constraints
- **Characteristics**: Generates elastic band trajectory, automatically adjusts to avoid obstacles

**🚙 MPPI (Model Predictive Path Integral) - Robust Planner**

- **Advantages**: Strong robustness, handles complex constraints, random sampling avoids local optima
- **Disadvantages**: Requires large number of samples, high computational cost
- **Application Scenarios**: Complex environments, multiple constraints, requires high robustness
- **Characteristics**: Random trajectory sampling, weighted selection of optimal path

**🚛 DDP (Dynamics Dynamic Programming) - Precise Planner**

- **Advantages**: Global optimal, high-precision trajectory
- **Disadvantages**: High computational cost, requires accurate model, may not be suitable for real-time
- **Application Scenarios**: High precision requirements, offline planning, known environment model
- **Characteristics**: Iterative trajectory optimization, considers dynamic constraints

---

#### 4.1.2 Planner Configuration Files

**Configuration File Locations**:
```
dynamics_planner_nav/params/
├── base_local_planner_params.yaml  # Base local planner parameters
├── costmap_common_params.yaml      # Common Costmap configuration
├── global_planner_params.yaml      # Global planner parameters
├── move_base_params.yaml           # MoveBase parameters
└── odom_nav_params/                # Odometry navigation parameters
    ├── local_costmap_params.yaml
    └── global_costmap_params.yaml
```

**DWA Key Parameters**:
- `max_vel_x`: Maximum linear velocity
- `max_vel_theta`: Maximum angular velocity
- `vx_samples` / `vtheta_samples`: Velocity sampling numbers
- `path_distance_bias`: Path distance weight
- `goal_distance_bias`: Goal distance weight
- `occdist_scale`: Obstacle distance weight

**TEB Key Parameters**:
- `min_obstacle_dist`: Minimum obstacle distance
- `weight_kinematics_nh`: Non-holonomic constraint weight
- `weight_optimaltime`: Time optimization weight
- `enable_homotopy_class_planning`: Multi-path planning

---

### 4.2 ros_jackal - Reinforcement Learning Training Framework (Learning-based)

**What does this module do?**

This module enables the robot to:
- Navigate in the Gazebo simulation environment
- **Automatically learn to select the optimal planner in different scenarios** (DWA/TEB/MPPI/DDP)
- Evaluate the performance of different planners

**Analogy**: Like training a driver to learn to choose the best vehicle for different road conditions:
- See highway → Choose sports car (DWA)
- See narrow street → Choose van (TEB)
- See complex road conditions → Choose SUV (MPPI)

---

#### 4.2.1 script Directory Structure

```
script/
└── applr/                                    # Basic baseline generation scripts
    ├── configs/                              # Planner configuration files
    │   ├── DWA.yaml                          # DWA planner config
    │   ├── TEB.yaml                          # TEB planner config
    │   ├── MPPI.yaml                         # MPPI planner config
    │   └── DDP.yaml                          # DDP planner config
    ├── evaluate_applr_single.py              # Single environment evaluation script
    ├── tmux_eval_applr_clients.sh            # Batch evaluation via tmux
    ├── eval_batch_worlds_singularity.sh      # Launch single process for multiple worlds with specified planner (Important!)
    └── eval_single_worlds_singularity.sh     # Launch single process for single world with specified planner (Important!) (for testing)
```

#### APPLR - Basic Framework + Reinforcement Learning (`script/applr/`)

Uses TD3 to learn optimal parameters for each planner.

**Configuration File Explanation** (`configs/`):

Each configuration file defines:
1. **Environment Configuration** (`env_config`):
   - `env_id`: Environment type (e.g., `dwa_param-v0`)
   - `action_type`: Planner type (`dwa_local`, `teb_local`, `mppi_local`, `ddp_local`)
   - `param_list`: List of parameters to learn
   - `param_init`: Initial parameter values

2. **Training Configuration** (`training_config`):
   - `network`: Network type (`mlp` or `cnn`)
   - `actor_lr` / `critic_lr`: Learning rates
   - `max_step`: Total training steps

**Test/Evaluate Single Planner (Ubuntu 20.04 Native)**:

Run evaluation scripts directly on Ubuntu 20.04 system.

```bash
cd src/ros_jackal/script/applr/

python evaluate_applr_single.py \
  --world_id 0 \                    # Specify BARN environment ID (0-299)
  --policy_name ddp \                # Planner type: dwa/teb/mppi/ddp
  --buffer_path ../../buffer/ \     # Data save path
  --world_path ../../jackal_helper/worlds/BARN1/ \  # Gazebo world file path
  --ros_port 11311 \                # ROS Master port (different ports for multi-process)
  --mode auto \                     # Run mode (auto: automatic run)
  --save_image False \              # Whether to save observation images per step
  --algorithm_name STATIC \         # Algorithm name (STATIC: use default params, no updates)
  --num_trials 3                    # Number of runs per environment
```

**Parameter Explanation**:
- `--world_id`: BARN environment number, range 0-299 (static) or 300+ (dynamic)
- `--policy_name`: Planner name, determines which planner and config file to use
- `--buffer_path`: Evaluation data (trajectories, rewards, etc.) save location
- `--ros_port`: ROS Master port, must use different ports for parallel runs (e.g., 11311, 11313, 11315...)
- `--algorithm_name`:
  - `STATIC`: Use default parameters, no learning
  - `TD3`/`SAC`: Use reinforcement learning algorithms
- `--save_image`: Whether to save Costmap/LaserScan images for subsequent analysis
- `--num_trials`: Repetition count for statistical average performance

---

**Batch Evaluation of Multiple Environments (Ubuntu 24.04 Container)**:

Run Ubuntu 20.04 environment via Apptainer container on Ubuntu 24.04.

```bash
cd src/ros_jackal/script/applr

bash eval_batch_worlds_singularity.sh \
  --id 2 \              # Process ID, determines port number (ros_port = 11311 + id*2)
  --start 200 \         # Start environment ID
  --end 299 \           # End environment ID
  --policy teb          # Planner type: dwa/teb/mppi/ddp
```

**Parameter Explanation**:
- `--id`: Process identifier, used to calculate ROS port (avoid conflicts)
  - Example: `id=0` → `ros_port=11311`, `id=2` → `ros_port=11315`
  - When running in parallel, each process must use different IDs
- `--start` / `--end`: Range of environments for batch evaluation
  - Example: `--start 0 --end 99` evaluates first 100 environments
  - Example: `--start 200 --end 299` evaluates last 100 environments
- `--policy`: Planner name, corresponds to config file (`configs/{policy}.yaml`)

**Container Explanation**:
- Script internally calls `singularity_run.sh` to start container
- Container automatically mounts working directory and data directory
- Evaluation results are saved in host machine's `buffer/` directory

**Typical Use Cases**:

1. **Single Environment Quick Test** (Ubuntu 20.04):
```bash
python evaluate_applr_single.py --world_id 0 --policy_name dwa --num_trials 1
```

2. **Batch Evaluate All Environments** (Ubuntu 24.04):
```bash
# Split into multiple parallel processes
bash eval_batch_worlds_singularity.sh --id 0 --start 0 --end 99 --policy dwa &
bash eval_batch_worlds_singularity.sh --id 1 --start 100 --end 199 --policy dwa &
bash eval_batch_worlds_singularity.sh --id 2 --start 200 --end 299 --policy dwa &
```

3. **Compare Different Planners**:
```bash
for planner in dwa teb mppi ddp; do
  python evaluate_applr_single.py --world_id 0 --policy_name $planner --num_trials 5
done
```

---

#### 4.2.2 envs Environment Definition - How Robot Interacts with Gazebo

**What does this module do?**

This module is the **bridge between the robot and the simulation environment**, defining:
- What the robot can do (action): Select planner or adjust parameters
- What the robot can see (observation): LiDAR data, maps, etc.
- What feedback the robot receives (reward): Success/failure/collision rewards

**Analogy**:
- Like a game's API interface, defining player operations (select weapon), game screen (view), scoring rules (rewards)
- `envs/` is the interface connecting the "AI brain" and "Gazebo world"

**Why use OpenAI Gym?**
- Gym is the standard interface for reinforcement learning, allowing us to train AI in a unified way
- Any RL algorithm (TD3, SAC, PPO, etc.) can directly use this environment

**Directory Structure**:
```
envs/
├── registration.py              # Environment registration entry
├── wrappers.py                  # Environment wrappers
├── DWA/                         # DWA planner environment
│   ├── dwa_base_envs.py        # Base environment class
│   └── parameter_tuning_envs.py # Parameter tuning environment
├── TEB/                         # TEB planner environment
├── MPPI/                        # MPPI planner environment
├── DDP/                         # DDP planner environment
├── utils/                       # Utility classes (core)
│   ├── gazebo_simulation.py    # Gazebo control interface
│   ├── Jackal_ros.py           # Jackal robot state management
│   ├── DWA_move_base.py        # DWA MoveBase interface
│   ├── TEB_move_base.py        # TEB MoveBase interface
│   ├── MPPI_move_base.py       # MPPI MoveBase interface
│   └── DDP_move_base.py        # DDP MoveBase interface
```

---

##### 📌 Quick Understanding: Three-Layer Structure of envs Module

```
┌─────────────────────────────────────────────────────────┐
│  1. registration.py - Register environments (entry)      │
│     → Tell Python: "I have DWA/TEB/MPPI/DDP environments"│
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  2. {planner}_base_envs.py - Base environment (core)     │
│     → Start Gazebo, load map, control robot movement    │
│     → reset(): Reset environment (new start/goal)        │
│     → step(): Execute one step (update planner, get reward)│
│     → Define reward: Success +10, Collision -5, Step -0.01│
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  3. parameter_tuning_envs.py - Define AI interface       │
│     → Define action: Which planner to select (0/1/2/3)  │
│     → Define observation: LiDAR data (720 points)        │
└─────────────────────────────────────────────────────────┘
```

---

##### 1️⃣ Environment Registration (`registration.py`) - Let Gym Recognize Our Environments

**Purpose**: Like "account registration", tell OpenAI Gym what environments we have available.

**Registered Environments**:
```python
'dwa_param-v0'   → DWA planner environment
'teb_param-v0'   → TEB planner environment
'mppi_param-v0'  → MPPI planner environment
'ddp_param-v0'   → DDP planner environment
```

**How to Use** (like creating a game character):
```python
import gym

# Create a DWA environment (like selecting "easy mode")
env = gym.make("dwa_param-v0",
               world_name="world_0.world",  # Select map
               gui=False)                   # No GUI

# Reset environment, start new game
obs = env.reset()  # obs is what the robot "sees" - LiDAR data
```

---

##### 2️⃣ Base Environment Class (`{planner}_base_envs.py`) - Gazebo Control Center

**Purpose**: This is the "brain" of the environment, responsible for starting and controlling the entire simulation system.

**Analogy**: Like a game engine, responsible for loading maps, generating characters, handling physics collisions.

---

**1. `__init__()` - Start Simulation Environment**

When you create an environment (`gym.make()`), this function will:
```python
def __init__(...):
    # Step 1: Start Gazebo simulator (3D physics world)
    self.launch_gazebo(world_name="world_0.world", gui=False)

    # Step 2: Start ROS navigation system (MoveBase)
    self.launch_move_base(planner="DWA")  # Use DWA planner

    # Step 3: Create 3 utility classes (detailed below)
    self.gazebo_sim = GazeboSimulation()  # Control Gazebo
    self.jackal_ros = JackalRos()         # Get robot state
    self.move_base = DWA_MoveBase()       # Control navigation
```

**Simple Understanding**:
- Like starting a game: first load 3D engine (Gazebo), then load character AI (MoveBase), finally connect controller (utility classes)

---

**2. `reset()` - Start New Round**

Called for each new training episode, like "restart game":
```python
def reset(self):
    # 1. Pause physics engine (freeze screen)
    self.gazebo_sim.pause()

    # 2. Teleport robot back to start
    self.gazebo_sim.reset(position=[-2, 3, 1.57])

    # 3. Clear map cache
    self.move_base.clear_costmap()

    # 4. Set new goal
    self.move_base.send_goal([0, 10, 0])

    # 5. Resume physics engine (continue game)
    self.gazebo_sim.unpause()

    # 6. Return initial observation (LiDAR data)
    obs = self.jackal_ros.get_laser_scan()
    return obs
```

**Simple Understanding**:
- Like pressing "restart" button: robot returns to start, set new goal, begin new navigation task

---

**3. `step(action)` - Execute One Action**

This is the core function, AI calls it every step:
```python
def step(self, action):
    # 1. Execute AI's decision
    if action == 0:
        self.switch_planner("DWA")   # Switch to DWA planner
    elif action == 1:
        self.switch_planner("TEB")   # Switch to TEB planner
    # ...

    # 2. Wait 0.5 seconds (let planner run)
    time.sleep(0.5)

    # 3. Pause, read results
    self.gazebo_sim.pause()
    obs = self.jackal_ros.get_laser_scan()      # What did it see
    reward = self._calculate_reward()            # How many points earned
    done = self.jackal_ros.reached_goal()        # Is it done
    self.gazebo_sim.unpause()

    return obs, reward, done, {}
```

**Simple Understanding**:
- Like each frame in a game: AI makes decision (select weapon) → wait for execution → check results (score/health/level complete)

---

**Environment's 3 "Utility Tools"** (important member variables):

| Utility Class | Purpose | Analogy |
|--------|------|------|
| `self.gazebo_sim` | Control Gazebo (pause/play/reset) | Game's "time controller" |
| `self.jackal_ros` | Get robot state (position/laser/velocity) | Game's "vision system" |
| `self.move_base` | Control navigation (switch planner/send goal) | Game's "AI controller" |

---

##### 3️⃣ AI Interface Definition (`parameter_tuning_envs.py`) - Define What AI Can Do

**Purpose**: This file defines the AI's "game rules" - what it can do, what it can see, how it scores.

**Analogy**: Like game design documentation, defining:
- Player operations (keyboard keys → AI's action)
- Screen display (screen resolution → observation dimensions)
- Scoring rules (kill +100 points → reward function)

---

**This Project's Two Modes** (choose one):

---

**🎯 Mode 1: Planner Selection** (Core of this project - Recommended)

**Goal**: Let AI learn "see scene, select strategy", like a driver choosing lanes based on road conditions.

**Definition 1: What can AI do?** (Action space)
```python
# Choose 1 of 4: select which planner to use
self.action_space = Discrete(4)

# AI outputs one number each step:
# 0 → Use DWA  (fast, suitable for open spaces)
# 1 → Use TEB  (smooth, suitable for narrow corridors)
# 2 → Use MPPI (robust, suitable for complex environments)
# 3 → Use DDP  (precise, suitable for high-precision scenarios)
```

**Definition 2: What can AI see?** (Observation space)
```python
# LiDAR data + ground friction coefficient: 720 + 1 = 721 dimensions
self.observation_space = Box(
    low=np.array([0.0]*720 + [0.0]),     # Laser range 0-10m, friction 0-2
    high=np.array([10.0]*720 + [2.0]),
    shape=(721,),                         # 720 laser points + 1 friction coefficient
    dtype=np.float32
)

# Example: obs = [2.5, 2.3, 2.1, ..., 5.8, 6.0, 0.8]
#       → Obstacle at 2.5m ahead, wall at 5.8m to left...
#       → Last dimension 0.8 = ground friction coefficient (ice≈0.1, normal≈0.8, rubber≈1.5)
```

**Why add friction coefficient?**
- Different ground conditions (ice, wet ground, sand, normal road) greatly affect planner performance
- DWA reacts fast on high friction surfaces, but easily loses control on low friction (ice)
- TEB is more stable on low friction surfaces (smooth trajectory)
- AI can select the most suitable planner based on ground conditions

**Analogy**:
- Like checking road conditions before driving: switch to "snow mode" when seeing icy road
- Friction coefficient = road's "traction" signal

**How to implement in code?**

Add friction coefficient in `{planner}_base_envs.py`:

```python
class DWABaseLaser(DWABase):
    def __init__(self, laser_clip=4, friction_coeff=0.8, **kwargs):
        super().__init__(**kwargs)
        self.friction_coeff = friction_coeff  # Initialize friction coefficient

        self.observation_space = Box(
            low=0, high=laser_clip,
            shape=(722,),  # 720(laser) + 1(goal) + 1(friction)
            dtype=np.float32
        )

    def _get_observation(self):
        laser_scan = self._get_laser_scan()       # Get laser data
        local_goal = self._get_local_goal()       # Get local goal
        friction = np.array([self.friction_coeff]) # Add friction coefficient

        # Concatenate into complete observation
        obs = np.concatenate([laser_scan, local_goal, friction])
        return obs
```

**How to set different friction coefficients?**

Pass `friction_coeff` parameter when creating environment:

```python
import gym

# Ice surface environment (low friction)
env_ice = gym.make("dwa_param-v0", friction_coeff=0.1)

# Normal ground
env_normal = gym.make("dwa_param-v0", friction_coeff=0.8)

# High friction ground (rubber)
env_rubber = gym.make("dwa_param-v0", friction_coeff=1.5)
```

**Set ground friction in Gazebo**

Modify ground material in `.world` file:

```xml
<surface>
  <friction>
    <ode>
      <mu>0.8</mu>   <!-- Friction coefficient, range 0-2 -->
      <mu2>0.8</mu2>
    </ode>
  </friction>
</surface>
```

Then when creating environment, synchronize this friction coefficient to observation.

---

**Definition 3: How does AI execute actions?** (_take_action function)
```python
def _take_action(self, action):
    # Map number to actual planner
    planner_names = {
        0: "DWA Planner",
        1: "TEB Planner",
        2: "MPPI Planner",
        3: "DDP Planner"
    }

    # Switch to selected planner
    self.move_base.switch_planner(planner_names[action])

    # Analogy: like switching car's driving mode (sport/comfort/off-road)
```

**Definition 4: How does AI get rewards?** (Reward function)
```python
def _get_reward(self):
    reward = 0

    # 1. Basic reward
    if reached_goal:
        reward = +10.0        # Success! Big reward
    elif collision:
        reward = -5.0         # Failure! Penalty
    else:
        reward = -0.01        # Small penalty per step (encourages fast completion)

    # 2. Efficiency reward (encourages fast planners)
    if success and low_time:
        reward += time_bonus    # Completing in 20 steps gets more reward than 50 steps

    # 3. Smoothness reward (encourages planners with good trajectories)
    if smooth_path:
        reward += 0.5         # Reward for path not turning too sharply

    return reward

# Analogy: Game score = completion reward + speed-run reward + no-damage reward
```

**Why designed this way?**
- Action is discrete (choose 1 of 4) → Small search space, easy to learn
- Observation is LiDAR → Contains sufficient scene information (obstacle positions, corridor width)
- Reward encourages success, speed, smoothness → AI learns to select optimal planner in different scenarios

---

**⚙️ Mode 2: Parameter Optimization** (Traditional APPLR - for comparison)

**Goal**: Fine-tune parameters of a single planner, like adjusting car's suspension stiffness, throttle sensitivity.

**What can AI do?**
```python
# 7 continuous parameters (DWA example)
self.action_space = Box(
    low  = [0.1, 0.5,  3, 10, 0.1, 0.1, 0.2],  # Minimum values
    high = [1.0, 3.0, 20, 40, 5.0, 5.0, 1.0],  # Maximum values
    shape=(7,)
)

# AI outputs 7 numbers each step:
# [0.5, 1.2, 10, 20, 1.5, 2.0, 0.4]
# ↓
# max_velocity=0.5m/s, max_rotation=1.2rad/s, ...
```

**Why not recommended?**
- Action space too large (7D continuous) → Hard to learn
- Can only optimize single planner → Limited
- Requires many samples (5M steps) → Slow training

**Use**: As baseline comparison, proving "planner selection" is better than "parameter optimization"

---

**📊 Comparison of Two Modes**:

| Comparison Dimension | 🎯 Mode 1: Planner Selection | ⚙️ Mode 2: Parameter Optimization |
|---------|---------------------|-------------------|
| **Action** | Discrete 4-choice | Continuous 7D |
| **Search Space** | 4 choices | 10^7 combinations |
| **Learning Difficulty** | Easy ⭐ | Hard ⭐⭐⭐ |
| **Sample Requirements** | <1M steps | >5M steps |
| **Generalization** | Strong (4 planners complement) | Weak (single planner) |
| **Application Scenarios** | All scenarios auto-adapt | Only suitable for specific scenarios |
| **Project Positioning** | ✅ Core innovation | ❌ Baseline |

**Analogy Summary**:
- Mode 1 = Learn to "change vehicles": sports car for highway, jeep for off-road
- Mode 2 = Learn to "tune vehicle": only one car, adjust suspension and throttle

Obviously Mode 1 is more flexible and efficient!

---

##### 4️⃣ Utility Classes (`utils/`) - Core Support Modules

**a) `gazebo_simulation.py` - Gazebo Control**

Provides control interface for Gazebo physics engine.

**Core Methods**:
```python
class GazeboSimulation:
    def __init__(self, init_position):
        # ROS service proxies
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Collision listener
        self._collision_sub = rospy.Subscriber("/collision", Bool, self.collision_monitor)

    def pause(self):
        """Pause physics simulation (for data collection)"""
        self._pause()

    def unpause(self):
        """Resume physics simulation"""
        self._unpause()

    def reset(self, position):
        """Reset robot position"""
        model_state = create_model_state(position[0], position[1], position[2])
        self._reset(model_state)

    def get_hard_collision(self):
        """Get collision status"""
        return self.collision_count > 0
```

**Key Points**:
- `pause()/unpause()` used to freeze simulation for safe state reading
- Supports collision detection, velocity monitoring

---

**b) `Jackal_ros.py` - Robot State Management**

Gets various robot state information (position, velocity, sensors, etc.).

**Core Functions**:
```python
class JackalRos:
    def __init__(self):
        # Subscribe to various ROS topics
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap',
                                           OccupancyGrid, self.costmap_callback)
        self.path_sub = rospy.Subscriber('/move_base/TrajectoryPlannerROS/local_plan',
                                        Path, self.path_callback)

    # Get state methods
    def get_position(self):
        """Get robot position (x, y, yaw)"""
        return self.position

    def get_velocity(self):
        """Get robot velocity (vx, vy, omega)"""
        return self.velocity

    def get_laser_scan(self):
        """Get LiDAR data"""
        return self.laser_ranges  # shape: (720,)

    def get_local_costmap(self):
        """Get local costmap"""
        return self.costmap  # shape: (160, 160)

    def get_local_plan(self):
        """Get planned local path"""
        return self.local_path
```

**Supported Sensors**:
- **Odometry**: Robot pose (x, y, θ)
- **LaserScan**: 720-point LiDAR data
- **Costmap**: Local cost map (for training CNN)
- **Path**: Planner output local path

**Predefined Parameters** (PLANNER_PARAMS):
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

**c) `{PLANNER}_move_base.py` - MoveBase Interface**

Each planner has a corresponding MoveBase interface class for interacting with ROS navigation stack.

**Core Methods**:
```python
class dwa_MoveBase:
    def __init__(self):
        # MoveBase Action Client
        self.move_base_client = actionlib.SimpleActionClient(
            'move_base', MoveBaseAction
        )

        # Dynamic Reconfigure Client (dynamic parameter update)
        self.dwa_client = dynamic_reconfigure.client.Client(
            '/move_base/TrajectoryPlannerROS',
            timeout=5
        )

    def send_goal(self, goal_position):
        """Send navigation goal"""
        goal = _create_MoveBaseGoal(goal_position[0], goal_position[1], goal_position[2])
        self.move_base_client.send_goal(goal)

    def update_params(self, params):
        """Dynamically update DWA parameters"""
        # params: {'max_vel_x': 0.5, 'max_vel_theta': 1.0, ...}
        self.dwa_client.update_configuration(params)

    def get_state(self):
        """Get MoveBase state (ACTIVE, SUCCEEDED, ABORTED, ...)"""
        return self.move_base_client.get_state()

    def clear_costmap(self):
        """Clear costmap"""
        rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
```

**Supported Operations**:
- Send/cancel navigation goals
- Dynamically update planner parameters
- Clear costmap
- Get navigation state

---

##### 5️⃣ Sensor Interface (`sensors/`)

**`laser.py`**: LiDAR data processing
- Subscribe to `/scan` topic
- Provide distance data, point cloud conversion

**`camera.py`**: Camera interface (currently not used)

---

##### 6️⃣ Usage Flow Example

```python
import gym
from envs import registration

# 1. Create environment
env = gym.make("dwa_param-v0",
               world_name="world_0.world",
               gui=False,
               init_position=[-2, 3, 1.57],
               goal_position=[0, 10, 0],
               max_step=100,
               ros_port=11311)

# 2. Reset environment
obs = env.reset()
# obs: Costmap image (84, 84, 1) or LaserScan (720,)

# 3. Execute action
action = [0.5, 1.0, 10, 20, 1.0, 1.0, 0.3]  # DWA parameters
next_obs, reward, done, info = env.step(action)

# 4. If action=None, use default parameters
next_obs, reward, done, info = env.step(None)  # Don't update parameters

# 5. Close environment
env.close()
```

---

##### 7️⃣ Key Design Points

1. **Pause/Unpause Mechanism**:
   - `pause()` before reading state to ensure data consistency
   - `unpause()` after reading to continue simulation

2. **Dynamic Parameter Update**:
   - Update planner parameters in real-time via `dynamic_reconfigure`
   - No need to restart navigation stack

3. **Multi-Process Support**:
   - Support parallel training via `ros_port` and `gazebo_port`
   - Independent ROS Master for each environment

4. **Observation Modes**:
   - **Costmap**: CNN input (84x84 image)
   - **LaserScan**: MLP input (720D vector)

5. **Reward Design** (in `_get_reward()`):
   - Success: `+success_reward`
   - Collision: `+collision_reward` (negative)
   - Time step penalty: `+slack_reward` (negative)
   - Smoothness reward: Based on velocity changes
