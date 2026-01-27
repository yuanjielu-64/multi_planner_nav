"""
FTRL训练脚本 - VLM+DPT + TD3 (Condor离线模式)

优化版本:
1. 自动启动/关闭 tmux VLM 服务
2. 深度配置合并（buffer config + VLM config）
3. 服务健康检查
4. GPU 智能分配
5. 完整的信号处理和清理

纯Python 3.10环境，不依赖gym/ROS
完全模仿td3/train.py的Condor模式，使用InfoEnv + CondorCollector
"""
# ============ GPU选择必须在import torch之前 ============
import sys
import os

def _early_gpu_setup():
    """在import torch之前解析--gpu参数并设置CUDA_VISIBLE_DEVICES"""
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"[GPU] 设置 CUDA_VISIBLE_DEVICES={gpu_id} (在import torch之前)")
            return gpu_id
    return None

_EARLY_GPU = _early_gpu_setup()
# ======================================================

import argparse
import GPUtil
import yaml
import numpy as np
from datetime import datetime
from os.path import join, dirname, abspath, exists
import shutil
import logging
import collections
import time
from pprint import pformat
import signal

import torch
from tensorboardX import SummaryWriter

# 添加路径
sys.path.append(dirname(dirname(abspath(__file__))))

# RLFT模块 (VLM专用)
from rlft.vlm_net import VLM_DPT_FeatureExtractor, VLM_DPT_Actor, VLM_DPT_Critic
from rlft.rl import TD3
from rlft.collector import VLMCondorCollector, VLMReplayBuffer

torch.set_num_threads(8)

import psutil
import subprocess
import atexit

# 全局变量：跟踪启动的 tmux 服务
TMUX_SERVICES_STARTED = []
PLANNER = "ddp"  # 默认值，会在 main 中更新


def start_tmux_services(config, policy_checkpoint_path=None):
    """
    根据配置自动启动 tmux VLM 服务

    Args:
        config: 包含 ftrl_config 的配置字典
        policy_checkpoint_path: 直接指定 policy checkpoint 路径（与 rl.py 保持一致）

    Returns:
        list: 启动的 tmux session 名称列表
    """
    global TMUX_SERVICES_STARTED, PLANNER

    ftrl_config = config.get("ftrl_config", {})
    training_config = config.get("training_config", {})

    # 检查是否自动启动
    if not ftrl_config.get("auto_start_services", False):
        print("    >>>> auto_start_services=False, 跳过自动启动服务")
        return []

    # 获取配置参数
    server_urls = ftrl_config.get("server_urls", [])
    num_services = len(server_urls) if server_urls else ftrl_config.get("num_services", 1)
    start_port = ftrl_config.get("start_port", 7000)
    base_model = ftrl_config.get("base_model", "Qwen/Qwen2.5-VL-3B-Instruct")

    # lora_path：优先使用传入的路径（与 rl.py save/load 保持一致）
    if policy_checkpoint_path:
        lora_path = policy_checkpoint_path
        print(f"    >>>> 使用传入的 checkpoint 路径: {lora_path}")
    elif 'BUFFER_PATH' in globals() and BUFFER_PATH:
        # 回退：从全局变量构建路径
        lora_path = join(BUFFER_PATH, "policy")
        print(f"    >>>> [回退] 使用全局 BUFFER_PATH 构建: {lora_path}")
    else:
        print("[ERROR] 未提供 checkpoint 路径且 BUFFER_PATH 未设置")
        return []

    # 验证 checkpoint 是否存在
    if not exists(lora_path):
        print(f"[ERROR] Checkpoint 目录不存在: {lora_path}")
        print(f"[ERROR] 请确保先运行 policy.save() 保存模型")
        return []

    # 检查必需文件并显示详情
    print(f"    >>>> 验证 checkpoint 内容:")
    checkpoint_files = {
        "LoRA": join(lora_path, "adapter_model.safetensors"),
        "LoRA Config": join(lora_path, "adapter_config.json"),
        "Regression Head": join(lora_path, "regression_head", "pytorch_model.bin"),
        "History Config": join(lora_path, "history_config.json"),
        "History Encoder": join(lora_path, "history_encoder", "pytorch_model.bin"),
        "Normalization": join(lora_path, "normalization", "param_mean.npy"),
    }

    all_ok = True
    for name, filepath in checkpoint_files.items():
        if exists(filepath):
            if name == "LoRA":
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                print(f"         ✓ {name}: {size_mb:.1f} MB")
            else:
                print(f"         ✓ {name}")
        else:
            print(f"         ✗ {name} (缺失)")
            if name in ["LoRA", "Regression Head", "History Config", "Normalization"]:
                all_ok = False

    if not all_ok:
        print(f"[ERROR] Checkpoint 缺少必需文件，无法启动服务")
        return []

    # 使用全局 PLANNER（已在 main 中设置）
    planner = PLANNER

    print("\n" + "=" * 60)
    print("  启动 FTRL VLM 服务 (tmux)")
    print("=" * 60)
    print(f"  Planner: {planner}")
    print(f"  Number of services: {num_services}")
    print(f"  Start port: {start_port}")
    print(f"  Base model: {base_model}")
    print(f"  LoRA path: {lora_path}")

    # 获取脚本目录
    script_dir = dirname(dirname(abspath(__file__)))
    ft_qwen_dir = join(script_dir, "script", "ft_qwen")
    server_script = join(ft_qwen_dir, "qwen_server.py")

    # 检查 server 脚本是否存在
    if not exists(server_script):
        print(f"[ERROR] qwen_server.py not found: {server_script}")
        return []

    # 检查 LoRA checkpoint 是否存在
    if not exists(lora_path):
        print(f"[ERROR] LoRA checkpoint not found: {lora_path}")
        return []

    # ===== 新的GPU分配逻辑：支持每个服务单独指定GPU =====
    # 优先使用 gpu_assignments（列表），否则使用 gpu_strategy（全局策略）
    gpu_array = []

    if "gpu_assignments" in ftrl_config:
        # 方式1: 每个服务单独指定GPU（推荐）
        gpu_assignments = ftrl_config["gpu_assignments"]

        if not isinstance(gpu_assignments, list):
            print(f"[ERROR] gpu_assignments must be a list, got {type(gpu_assignments)}")
            return []

        if len(gpu_assignments) != num_services:
            print(f"[ERROR] gpu_assignments length ({len(gpu_assignments)}) must match num_services ({num_services})")
            return []

        gpu_array = [int(g) for g in gpu_assignments]
        print(f"  GPU assignments (per service):")
        for i, gpu in enumerate(gpu_array):
            print(f"    Service {i} -> GPU {gpu}")

    elif "gpu_strategy" in ftrl_config:
        # 方式2: 使用全局策略（向后兼容）
        gpu_strategy = str(ftrl_config["gpu_strategy"])

        if gpu_strategy == "auto":
            # 自动递增分配
            try:
                available_gpus = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.5, maxMemory=0.5)
                if len(available_gpus) < num_services:
                    print(f"  [WARN] Only {len(available_gpus)} GPUs available, requested {num_services}")
                    # 循环使用可用 GPU
                    gpu_array = [available_gpus[i % len(available_gpus)] for i in range(num_services)]
                else:
                    gpu_array = available_gpus[:num_services]
                print(f"  Auto-detected GPUs: {gpu_array}")
            except Exception as e:
                print(f"  [ERROR] Failed to auto-detect GPUs: {e}")
                print(f"  Falling back to GPU 0")
                gpu_array = [0] * num_services
        elif "," in gpu_strategy:
            gpu_array = [int(g.strip()) for g in gpu_strategy.split(",")]
            if len(gpu_array) != num_services:
                print(f"[ERROR] GPU count ({len(gpu_array)}) must match num_services ({num_services})")
                return []
            print(f"  GPU strategy: {gpu_array}")
        else:
            # 单个GPU，所有服务共享
            gpu_array = [int(gpu_strategy)] * num_services
            print(f"  All services will share GPU {gpu_strategy}")
    else:
        # 默认: 所有服务使用GPU 0
        gpu_array = [0] * num_services
        print(f"  Default: All services will use GPU 0")

    print("=" * 60)

    # Conda Python 路径 (根据环境调整)
    conda_python_candidates = [
        "/common/home/yl2832/miniconda3/envs/lmms-finetune-qwen/bin/python",
        "/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python",
        os.path.expanduser("~/miniconda3/envs/lmms-finetune-qwen/bin/python"),
        os.path.expanduser("~/miniforge3/envs/lmms-finetune-qwen/bin/python"),
    ]
    conda_python = None
    for candidate in conda_python_candidates:
        if exists(candidate):
            conda_python = candidate
            break

    if conda_python is None:
        print("[ERROR] Cannot find lmms-finetune-qwen conda environment")
        print("  Tried:", conda_python_candidates)
        return []

    print(f"  Using Python: {conda_python}")

    started_sessions = []

    for i in range(num_services):
        gpu = gpu_array[i]
        port = start_port + i
        session_name = f"ftrl_{planner}_{i}"

        print(f"\n  Starting service {i+1}/{num_services}...")
        print(f"    Name: {session_name}")
        print(f"    GPU: {gpu}")
        print(f"    Port: {port}")

        # 检查端口是否被占用
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                print(f"    [WARN] Port {port} is in use, killing process...")
                subprocess.run(["kill", "-9"] + result.stdout.strip().split(), timeout=5)
                time.sleep(1)
        except Exception as e:
            pass  # lsof 可能不存在或端口未被占用

        # 检查 tmux session 是否已存在
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                print(f"    [WARN] tmux session '{session_name}' already exists, killing it...")
                subprocess.run(["tmux", "kill-session", "-t", session_name], timeout=5)
                time.sleep(1)
        except Exception as e:
            pass

        # 构建启动命令
        cmd = f"""export CUDA_VISIBLE_DEVICES={gpu} && \
{conda_python} {server_script} \
    --base_model {base_model} \
    --lora_path {lora_path} \
    --algorithm {planner.upper()} \
    --port {port} \
    --device cuda:0 \
    --load_in_4bit; \
echo 'Service stopped. Press Enter to exit.'; read"""

        # 创建 tmux session
        try:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name, cmd],
                check=True, timeout=10
            )
            started_sessions.append(session_name)
            print(f"    [OK] Started in tmux session '{session_name}'")
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] Failed to start tmux session: {e}")
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] Timeout starting tmux session")

        # 等待一小段时间，避免同时启动太多进程
        time.sleep(3)

    TMUX_SERVICES_STARTED = started_sessions

    print("\n" + "=" * 60)
    print(f"  Started {len(started_sessions)}/{num_services} service(s)")
    print("=" * 60)

    # 等待服务启动完成
    if started_sessions:
        print("\n  等待服务启动...")
        wait_time = 15  # 等待30秒让服务初始化
        for i in range(wait_time):
            print(f"\r  等待中... {wait_time - i}s ", end="", flush=True)
            time.sleep(1)
        print("\n  服务应该已经启动完成")

    return started_sessions


def stop_tmux_services(session_names=None, planner=None):
    """
    关闭 tmux VLM 服务

    Args:
        session_names: 要关闭的 session 名称列表，None 则关闭所有 TMUX_SERVICES_STARTED
        planner: 如果指定，关闭所有匹配 ftrl_{planner}_* 的 session
    """
    global TMUX_SERVICES_STARTED, PLANNER

    if session_names is None:
        session_names = TMUX_SERVICES_STARTED.copy()

    if planner is None:
        planner = PLANNER

    if not session_names and not planner:
        print("    >>>> 没有需要关闭的 tmux 服务")
        return

    print("\n" + "=" * 60)
    print("  关闭 FTRL VLM 服务 (tmux)")
    print("=" * 60)

    closed_count = 0

    # 关闭指定的 session
    for session_name in session_names:
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                subprocess.run(
                    ["tmux", "kill-session", "-t", session_name],
                    timeout=5
                )
                print(f"  [OK] Closed tmux session: {session_name}")
                closed_count += 1
            else:
                print(f"  [SKIP] Session not found: {session_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to close {session_name}: {e}")

    # 额外检查：关闭所有 ftrl_{planner}_* 的 session
    if planner:
        try:
            result = subprocess.run(
                ["tmux", "ls"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line and f"ftrl_{planner}_" in line:
                        session = line.split(":")[0]
                        if session not in session_names:
                            try:
                                subprocess.run(
                                    ["tmux", "kill-session", "-t", session],
                                    timeout=5
                                )
                                print(f"  [OK] Closed extra tmux session: {session}")
                                closed_count += 1
                            except:
                                pass
        except:
            pass

    TMUX_SERVICES_STARTED.clear()

    print(f"\n  总共关闭了 {closed_count} 个 tmux session")
    print("=" * 60)


def restart_gazebo():
    print(">>>>>>>> 正在重启Gazebo...")

    gazebo_processes = ['gazebo', 'gzserver', 'gzclient', 'roslaunch']

    for proc_name in gazebo_processes:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == proc_name or \
                        (proc.info['cmdline'] and any(proc_name in cmd for cmd in proc.info['cmdline'])):
                    print(f"    >>>> 正在杀死进程: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass


def deep_merge(base_dict, override_dict, path="", verbose=True):
    """
    深度合并两个字典，override_dict 中的值会覆盖 base_dict 中的相同键

    Args:
        base_dict: 基础字典（会被修改）
        override_dict: 覆盖字典
        path: 当前路径（用于日志输出）
        verbose: 是否打印详细日志

    Returns:
        合并后的字典（就是修改后的 base_dict）
    """
    for key, value in override_dict.items():
        current_path = f"{path}.{key}" if path else key

        if key in base_dict:
            if isinstance(base_dict[key], dict) and isinstance(value, dict):
                # 两个都是字典，递归合并
                deep_merge(base_dict[key], value, current_path, verbose)
            elif base_dict[key] != value:
                # 值不同，用 override 覆盖
                old_value = base_dict[key]
                base_dict[key] = value
                if verbose:
                    # 只打印重要的配置变更
                    if not isinstance(value, (list, dict)) or len(str(value)) < 100:
                        print(f"    >>>> Override {current_path}: {old_value} -> {value}")
                    else:
                        print(f"    >>>> Override {current_path}: [complex value]")
        else:
            # base 中没有这个键，直接添加
            base_dict[key] = value
            if verbose and (not isinstance(value, (list, dict)) or len(str(value)) < 100):
                print(f"    >>>> Add {current_path}: {value}")

    return base_dict


def initialize_config(config_path, save_path, override_config_path=None):
    """
    加载配置文件，支持用另一个配置覆盖

    Args:
        config_path: 基础配置文件路径
        save_path: 保存路径
        override_config_path: 覆盖配置路径（会覆盖 config_path 中的同名参数）
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config is None:
        print(f"[ERROR] 配置文件为空或格式错误: {config_path}")
        sys.exit(1)

    if override_config_path and exists(override_config_path):
        with open(override_config_path, 'r') as f:
            override_config = yaml.load(f, Loader=yaml.FullLoader)
        if override_config:
            deep_merge(config, override_config, verbose=False)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config


def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    save_path = join(
        env_config["save_path"],
        env_config["env_id"],
        training_config['algorithm'],
        dt_string
    )
    print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)

    writer = SummaryWriter(save_path)

    shutil.copyfile(
        env_config["config_path"],
        join(save_path, "config.yaml")
    )

    return save_path, writer


def initialize_envs(config):
    """
    初始化环境 - Condor模式使用InfoEnv (不需要真实gym环境)

    完全模仿td3/train.py的逻辑，但移除gym依赖
    """
    env_config = config["env_config"]
    env_id = env_config["env_id"]

    print("    >>>> Using condor mode (offline training)")

    # 简化的Space类 - 替代gym.spaces.Box
    class SimpleSpace:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    # 创建InfoEnv - 只提供space信息，不实际运行仿真
    class InfoEnv:
        def __init__(self, config):
            self.config = config
            env_id = config["env_config"]["env_id"]

            # 根据env_id确定action space (参数 + next_linear_vel + next_angular_vel)
            if "dwa" in env_id.lower():
                action_low = np.array([0.1, 0.314, 3, 10, 0.01, 0.01, 0.1, -0.5, -3.14])
                action_high = np.array([2.0, 3.14, 50, 50, 10.0, 10.0, 0.6, 2.0, 3.14])
            elif "teb" in env_id.lower():
                action_low = np.array([0.1, 0.0, 0.314, 0.1, 0.05, 0.05, 0.1, -0.5, -3.14])
                action_high = np.array([2.0, 1.0, 3.14, 1.0, 0.5, 0.5, 0.6, 2.0, 3.14])
            elif "ddp" in env_id.lower():
                action_low = np.array([0.1, 0.314, 400, 0.01, 0.01, 0.1, -0.5, -3.14])
                action_high = np.array([2.0, 3.14, 800, 0.4, 0.15, 0.6, 2.0, 3.14])
            elif "mppi" in env_id.lower():
                action_low = np.array([0.1, 0.314, 400, 10, 0.01, 0.01, 0.1, 0.1, -0.5, -3.14])
                action_high = np.array([2.0, 3.14, 800, 40, 0.5, 0.5, 2.0, 0.6, 2.0, 3.14])
            else:
                raise ValueError(f"Unknown env_id: {env_id}")

            self.action_space = SimpleSpace(
                low=action_low,
                high=action_high,
                shape=(len(action_low),),
                dtype=np.float32
            )

            # VLM使用图像，但这里不需要实际使用
            self.observation_space = SimpleSpace(
                low=np.array([0]),
                high=np.array([255]),
                shape=(480, 480, 3),  # RGB costmap
                dtype=np.uint8
            )

    return InfoEnv(config)


def seed(config):
    env_config = config["env_config"]

    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])


def initialize_policy(config, env, device_override=None):
    training_config = config["training_config"]

    state_dim = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high

    if device_override is not None:
        device = device_override
        print(f"    >>>> Using manually specified device: {device}")
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"    >>>> CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, using device: {device}")
    else:
        devices = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.8, maxMemory=0.8,
                                      includeNan=False, excludeID=[], excludeUUID=[])
        device = "cuda:%d" % (devices[0]) if len(devices) > 0 else "cpu"
        print(f"    >>>> Auto-selected device: {device}")

    vlm_checkpoint = training_config["vlm_checkpoint_path"]
    print(f"    >>>> Loading VLM+DPT from {vlm_checkpoint}")

    # 获取算法类型 (DWA/TEB/MPPI/DDP)
    algorithm = PLANNER.upper()
    print(f"    >>>> Algorithm: {algorithm}")

    # Actor
    actor_feature_extractor = VLM_DPT_FeatureExtractor(
        checkpoint_path=vlm_checkpoint,
        freeze_vlm=training_config.get("freeze_vlm_actor", True),
        freeze_dpt=training_config.get("freeze_dpt_actor", False),
        freeze_history=training_config.get("freeze_history_actor", None),  # None表示跟随freeze_dpt
        device=device,
        use_4bit=True,  # 使用4-bit量化
        algorithm=algorithm  # 传递算法类型
    )
    actor = VLM_DPT_Actor(
        feature_extractor=actor_feature_extractor,
        action_dim=action_dim,
        algorithm=algorithm  # 传递给Actor
    ).to(device)

    actor_optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, actor.parameters()),
        lr=training_config['actor_lr']
    )

    print("    >>>> Critic sharing Actor's feature extractor with detach protection")
    critic = VLM_DPT_Critic(
        feature_extractor=actor_feature_extractor,  # 共享，VLM只运行一次
        action_dim=action_dim,
        detach_features=True  # 默认True，保护Actor的VLM更新
    ).to(device)

    critic_params = []
    for name, param in critic.named_parameters():
        if param.requires_grad and not name.startswith('feature_extractor.'):
            critic_params.append(param)

    critic_optim = torch.optim.Adam(critic_params, lr=training_config['critic_lr'])

    # 打印参数统计
    actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    actor_total = sum(p.numel() for p in actor.parameters())
    print(f"    >>>> Actor: {actor_trainable:,} / {actor_total:,} trainable ({100*actor_trainable/actor_total:.2f}%)")

    critic_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    critic_total = sum(p.numel() for p in critic.parameters())
    print(f"    >>>> Critic: {critic_trainable:,} / {critic_total:,} trainable ({100*critic_trainable/critic_total:.2f}%)")

    # 加载参数归一化统计（从VLM checkpoint）
    param_mean, param_std = None, None
    normalization_dir = os.path.join(vlm_checkpoint, "normalization")
    if os.path.exists(normalization_dir):
        mean_path = os.path.join(normalization_dir, "param_mean.npy")
        std_path = os.path.join(normalization_dir, "param_std.npy")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            param_mean = np.load(mean_path)
            param_std = np.load(std_path)
            print(f"    >>>> Loaded parameter normalization from {normalization_dir}")
            print(f"         Mean: {param_mean}")
            print(f"         Std:  {param_std}")

            # 验证维度匹配
            if len(param_mean) != action_dim or len(param_std) != action_dim:
                print(f"    >>>> WARNING: Normalization dim mismatch!")
                print(f"         Expected {action_dim}, got mean={len(param_mean)}, std={len(param_std)}")
                print(f"         Disabling normalization")
                param_mean, param_std = None, None
        else:
            print(f"    >>>> WARNING: Normalization files not found in {normalization_dir}")
    else:
        print(f"    >>>> INFO: No normalization directory found, using raw parameter space")

    # TD3 policy
    policy = TD3(
        actor, actor_optim,
        critic, critic_optim,
        action_range=[action_space_low, action_space_high],
        device=device,
        param_mean=param_mean,
        param_std=param_std,
        **training_config["policy_args"]
    )

    # VLM专用ReplayBuffer
    buffer = VLMReplayBuffer(
        state_dim, action_dim,
        training_config['buffer_size'],
        device=device,
        image_size=(400, 600)  # (height, width)
    )

    return policy, buffer


def train(env, policy, buffer, config):
    """训练主循环 - Condor模式（完全模仿td3/train.py）"""
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)
    training_args = training_config["training_args"]

    # 传递 ftrl_config 给 collector
    collector = VLMCondorCollector(
        policy, env, buffer, BUFFER_PATH,
        ftrl_config=config.get("ftrl_config", {})  # 传递服务器配置
    )

    # 注册信号处理器，确保 Ctrl+C 时清理子进程和 tmux 服务
    def signal_handler(sig, frame):
        print("\n\n" + "=" * 80)
        print("🛑 收到退出信号 (Ctrl+C)，正在清理...")
        print("=" * 80)

        # 1. 停止数据收集容器
        try:
            collector.stop_collection_containers()
            print("✓ 所有数据收集容器已停止")
        except Exception as e:
            print(f"⚠️ 清理数据收集容器时出现错误: {e}")

        # 2. 停止 tmux VLM 服务（可配置）
        ftrl_config = config.get("ftrl_config", {})
        auto_stop_services = ftrl_config.get("auto_stop_services", True)  # 默认关闭

        try:
            if TMUX_SERVICES_STARTED and auto_stop_services:
                stop_tmux_services()
                print("✓ 所有 tmux VLM 服务已停止")
            elif TMUX_SERVICES_STARTED and not auto_stop_services:
                print("ℹ️ tmux VLM 服务保持运行 (auto_stop_services=False)")
                print(f"   手动停止: tmux kill-session -t vlm_server_*")
            else:
                print("ℹ️ 没有需要停止的 tmux 服务")
        except Exception as e:
            print(f"⚠️ 清理 tmux 服务时出现错误: {e}")

        print("=" * 80)
        print("退出训练")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill 命令

    # 显示信号处理器配置
    ftrl_config = config.get("ftrl_config", {})
    auto_stop = ftrl_config.get("auto_stop_services", True)
    print(f"    >>>> 信号处理器已注册 (Ctrl+C 将清理容器，tmux服务={'自动关闭' if auto_stop else '保持运行'})")

    # 启动数据收集容器（后台持续运行）
    # 优先级：命令行参数 > 配置文件 > 默认值(False)
    start_containers = ftrl_config.get("start_containers", False)
    if args.start_containers is not None:  # 命令行覆盖
        start_containers = args.start_containers

    if start_containers:
        print("\n" + "=" * 80)
        print("启动数据收集容器...")
        print("=" * 80)
        collector.start_collection_containers(mode='train')
        print("=" * 80)
        print("✓ 数据收集容器已启动，将在后台持续收集数据")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("⚠️  跳过启动数据收集容器 (start_containers=False)")
        print("    将从已有的 buffer 数据进行训练")
        print("=" * 80 + "\n")

    print("    >>>> Pre-collect experience (Condor mode)")
    print(f"    >>>> Target buffer size: {training_config['pre_collect']} steps")

    # Condor collect：从actor目录加载轨迹
    collector.collect(n_steps=training_config['pre_collect'], status='train')

    print(f"    >>>> Buffer filled! Current size: {buffer.size} steps")
    print(f"    >>>> Starting policy training...")
    print("=" * 80)

    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=BUFFER_SIZE)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=30))
    t0 = time.time()

    tensorboard_step = 0

    best_episode_length = float('inf')
    best_episode_nav = float('-inf')

    print(f"\n{'='*60}")
    print(f"开始训练循环 | max_step: {training_args['max_step']} | batch_size: {training_args['batch_size']}")
    print(f"update_per_step: {training_args['update_per_step']} | collect_per_step: {training_args['collect_per_step']}")
    print(f"{'='*60}\n")

    while n_steps < training_args["max_step"]:
        iter_start_time = time.time()
        print(f"\n[Iter {n_iter}] 开始 | 总步数: {n_steps}/{training_args['max_step']}")

        # Linear decaying exploration noise
        policy.exploration_noise = \
            - (training_config["exploration_noise_start"] - training_config["exploration_noise_end"]) \
            * n_steps / training_args["max_step"] + training_config["exploration_noise_start"]

        print(f"[Iter {n_iter}] 收集训练数据...")
        collect_start = time.time()
        steps, epinfo = collector.collect(training_args["collect_per_step"], status = 'train')
        collect_time = time.time() - collect_start
        print(f"[Iter {n_iter}] 收集完成: {steps} steps, {len(epinfo)} episodes, 耗时: {collect_time:.1f}s")

        n_steps += steps
        n_iter += 1
        n_ep += len(epinfo)
        epinfo_buf.extend(epinfo)

        for d in epinfo:
            world = d["world"].split("/")[-1]
            world_ep_buf[world].append(d)

        actor_grad_norms = []
        critic_grad_norms = []
        actor_losses = []
        critic_losses = []

        print(f"[Iter {n_iter}] 开始策略更新 ({training_args['update_per_step']} 次)...")
        update_start = time.time()
        for update_idx in range(training_args["update_per_step"]):
            update_iter_start = time.time()
            actor_grad_norm, critic_grad_norm, actor_loss, critic_loss = policy.train(buffer,
                                                                                      training_args["batch_size"])
            update_iter_time = time.time() - update_iter_start

            if actor_loss is not None:
                actor_grad_norms.append(actor_grad_norm)
                actor_losses.append(actor_loss)

            critic_grad_norms.append(critic_grad_norm)
            critic_losses.append(critic_loss)

            # 每4次更新打印一次进度（偶数次能看到actor_loss）
            if (update_idx + 1) % 4 == 0 or update_idx == 0:
                print(f"  更新 {update_idx+1}/{training_args['update_per_step']} | "
                      f"critic_loss: {critic_loss:.4f} | "
                      f"actor_loss: {actor_loss if actor_loss else 'N/A'} | "
                      f"耗时: {update_iter_time:.2f}s")

        update_time = time.time() - update_start
        print(f"[Iter {n_iter}] 策略更新完成, 总耗时: {update_time:.1f}s")

        t1 = time.time()
        
        print(f"[Iter {n_iter}] 收集测试数据...")
        test_start = time.time()
        test_steps, test_epinfo = collector.collect(n_steps=training_args['collect_per_step'], status='test')
        test_time = time.time() - test_start
        print(f"[Iter {n_iter}] 测试完成: {test_steps} steps, {len(test_epinfo)} episodes, 耗时: {test_time:.1f}s")

        status_counts = {"success": 0, "flip": 0, "timeout": 0}
        total_episodes = len(epinfo_buf)

        for epinfo in epinfo_buf:
            status = epinfo.get("ep_status", "unknown")
            if status in status_counts:
                status_counts[status] += 1

        success_rate = 100.0 * status_counts["success"] / total_episodes if total_episodes > 0 else 0.0
        flip_rate = 100.0 * status_counts["flip"] / total_episodes if total_episodes > 0 else 0.0
        timeout_rate = 100.0 * status_counts["timeout"] / total_episodes if total_episodes > 0 else 0.0

        nav_metric_score = np.mean([epinfo["nav_metric"] for epinfo in epinfo_buf])

        nav_metrics = [ep['nav_metric'] for ep in test_epinfo]
        avg_nav_metric = sum(nav_metrics) / len(nav_metrics) if nav_metrics else 0

        ep_times = [ep['ep_time'] for ep in test_epinfo]
        avg_ep_time = sum(ep_times) / len(ep_times) if ep_times else 0

        ep_time_steps = [ep['ep_len'] for ep in test_epinfo]
        avg_ep_len = sum(ep_time_steps) / len(ep_time_steps) if ep_time_steps else 0

        log = {
            "Episode_reward": np.mean([epinfo["ep_rew"] for epinfo in epinfo_buf]),
            "Episode_length": np.mean([epinfo["ep_len"] for epinfo in epinfo_buf]),
            "Episode_nav_metric": nav_metric_score,
            "Test_nav_metric": avg_nav_metric,
            "Test_time" : avg_ep_time,
            "Test_length": avg_ep_len,
            "Test_counts": len(nav_metrics),
            "Success_rate": success_rate,
            "Flip_rate": flip_rate,
            "Timeout_rate": timeout_rate,
            "Status_counts": total_episodes,
            "Time": np.mean([epinfo["ep_time"] for epinfo in epinfo_buf]),
            "Collision": np.mean([epinfo["collision"] for epinfo in epinfo_buf]),
            "Actor_grad_norm": np.mean(actor_grad_norms),
            "Critic_grad_norm": np.mean(critic_grad_norms),
            "Actor_loss": np.mean(actor_losses),
            "Critic_loss": np.mean(critic_losses),
            "fps": n_steps / (t1 - t0),
            "n_episode": n_ep,
            "Steps": n_steps,
            "Exploration_noise": policy.exploration_noise
        }

        logging.info(pformat(log))

        if len(epinfo_buf) >= 0:
            current_episode_reward = log["Episode_reward"]
            current_episode_length = log["Test_length"]
            current_episode_nav_metric = log["Test_nav_metric"]

            if (current_episode_nav_metric > best_episode_nav):
                # 保存到 policy_step_XXX（历史记录）
                policy_name = f"policy_step_{tensorboard_step}"
                policy.save(save_path, policy_name)

                # 保存到 best_model（固定目录，方便查找最佳模型）
                policy.save(save_path, "best_model")
                print(f"    >>>> 💾 新的最佳模型! Test_nav_metric: {current_episode_nav_metric:.3f} (之前: {best_episode_nav:.3f})")
                print(f"    >>>> 已保存到: {join(save_path, 'best_model')}")

                # 记录最佳性能信息
                with open(join(save_path, f"best_performance_step_{tensorboard_step}.txt"), 'w') as f:
                    f.write(f"Best Performance at TensorBoard Step {tensorboard_step}:\n")
                    f.write(f"Training Step: {n_steps}\n")
                    f.write(f"Episode Reward: {current_episode_reward:.3f}\n")
                    f.write(f"Episode Length: {current_episode_length:.3f}\n")
                    f.write(f"Success Rate: {success_rate:.3f}%\n")
                    f.write(f"Test_nav_metric: {current_episode_nav_metric:.3f}\n")

                # 更新 best_model 的元信息
                with open(join(save_path, "best_model", "best_info.txt"), 'w') as f:
                    f.write(f"Best Model Info\n")
                    f.write(f"===============\n")
                    f.write(f"TensorBoard Step: {tensorboard_step}\n")
                    f.write(f"Training Step: {n_steps}\n")
                    f.write(f"Test_nav_metric: {current_episode_nav_metric:.3f}\n")
                    f.write(f"Episode Reward: {current_episode_reward:.3f}\n")
                    f.write(f"Episode Length: {current_episode_length:.3f}\n")
                    f.write(f"Success Rate: {success_rate:.3f}%\n")
                    f.write(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                best_episode_nav = current_episode_nav_metric
                best_episode_length = current_episode_length

            if n_iter % training_config["log_intervals"] == 0:
                for k in log.keys():
                    writer.add_scalar('train/' + k, log[k], global_step=tensorboard_step)

                for k in world_ep_buf.keys():
                    writer.add_scalar(k + "/Episode_reward", np.mean([epinfo["ep_rew"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Episode_length", np.mean([epinfo["ep_len"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Episode_nav_metric", np.mean([epinfo["nav_metric"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Success_rate", success_rate,
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Flip_rate", flip_rate,
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Timeout_rate", timeout_rate,
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Status_counts", total_episodes,
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Time", np.mean([epinfo["ep_time"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Collision", np.mean([epinfo["collision"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)

            tensorboard_step += steps

    # Condor模式：清理临时文件（可选）
    # shutil.rmtree(BUFFER_PATH, ignore_errors=True)
    print("    >>>> Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Start training')
    parser.add_argument('--config_path', dest='config_path', default="../script/ft_qwen/configs/")
    parser.add_argument('--config_file', dest='config_file', default="ftrl_vlm_ddp")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--model_path', dest='model_path', default="../model_rlft/",
                        help='Path to load checkpoints (e.g., ../model_ftrl/)')
    parser.add_argument('--logging_path', dest='logging_path', default="../logging/")
    parser.add_argument('--buffer_size', dest='buffer_size', default= 200)
    parser.add_argument('--device', dest='device', default="cuda:0")
    parser.add_argument('--gpu', dest='gpu', type=int, default=None,
                        help='GPU ID to use (e.g., --gpu 1). Sets CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--policy_name', dest='policy_name', default="ddp_rlft",
                        help='Policy name for buffer directory (e.g., dwa_ftrl)')
    parser.add_argument('--planner', dest='planner', default=None,
                        help='Planner name for checkpoint directory (e.g., DWA, TEB, DDP, MPPI)')
    parser.add_argument('--start_containers', dest='start_containers', default=None, type=lambda x: x.lower() == 'true',
                        help='Start collection containers (default: from config file, see ftrl_config.start_containers)')
    parser.add_argument('--start_services', dest='start_services', default=None, type=lambda x: x.lower() == 'true',
                        help='Auto start tmux VLM services (default: from config file, see ftrl_config.auto_start_services)')

    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()

    # ===== GPU 设置 =====
    # 注意: CUDA_VISIBLE_DEVICES 已在文件开头 import torch 之前设置
    if args.gpu is not None:
        args.device = "cuda:0"  # CUDA_VISIBLE_DEVICES 后，cuda:0 就是指定的 GPU
        print(f"[GPU] 使用 GPU {args.gpu} (实际设备: cuda:0)")

    BUFFER_BASE_PATH = args.buffer_path   # e.g., ../buffer/
    MODEL_PATH = args.model_path           # e.g., ../model_rlft/
    BUFFER_SIZE = args.buffer_size
    SAVE_PATH = args.logging_path          # e.g., ../logging/
    policy_name = args.policy_name
    VLM_CONFIG_PATH = args.config_path
    VLM_CONFIG_NAME = args.config_file

    # ===== 检查参数和加载配置 =====
    if not policy_name:
        print("[ERROR] --policy_name 是必须参数")
        sys.exit(1)

    BUFFER_PATH = join(BUFFER_BASE_PATH, policy_name)
    vlm_config_file = join(VLM_CONFIG_PATH, VLM_CONFIG_NAME + ".yaml")
    buffer_config_file = join(BUFFER_PATH, "config.yaml")

    if not exists(vlm_config_file):
        print(f"[ERROR] VLM 配置不存在: {vlm_config_file}")
        sys.exit(1)

    if not exists(BUFFER_PATH):
        os.makedirs(BUFFER_PATH)

    # buffer 必须有 config，用 VLM config 覆盖
    if not exists(buffer_config_file):
        print(f"[ERROR] Buffer 配置不存在: {buffer_config_file}")
        sys.exit(1)

    config = initialize_config(buffer_config_file, SAVE_PATH, override_config_path=vlm_config_file)

    ACTION_TYPE = config["env_config"].get("action_type", "DDP")
    PLANNER = args.planner.lower() if args.planner else ACTION_TYPE.split('_')[0].lower()

    print(f"[Config] policy={policy_name}, planner={PLANNER}, buffer={BUFFER_PATH}")

    # ===== Step 2: 初始化环境和策略 =====
    print("\n" + "=" * 80)
    print("Step 2: 初始化环境和策略")
    print("=" * 80)
    seed(config)
    print("    >>>> Creating the environments (Condor mode)")
    env = initialize_envs(config)

    print("    >>>> Initializing the policy")
    policy, buffer = initialize_policy(config, env, device_override=args.device)

    # 加载 RLFT checkpoint（如果有）
    policy.load(BUFFER_PATH, "policy")

    # 立即保存到 buffer，确保 VLM 服务器有完整的 checkpoint 可用
    print("    >>>> Saving policy to buffer (确保 VLM 服务器可访问)")
    policy.save(BUFFER_PATH, "policy")

    # 计算 policy checkpoint 路径（与 rl.py 中的 save/load 逻辑一致）
    policy_checkpoint_path = join(BUFFER_PATH, "policy")
    print(f"         Checkpoint 已保存到: {policy_checkpoint_path}")

    # ===== Step 4: 启动 tmux VLM 服务 =====
    # 优先级：命令行参数 > 配置文件 > 默认值(True)
    start_services = config.get("ftrl_config", {}).get("auto_start_services", True)
    if args.start_services is not None:  # 命令行覆盖
        start_services = args.start_services

    if start_services:
        print("\n" + "=" * 80)
        print("Step 4: 启动 tmux VLM 服务")
        print("=" * 80)
        # 传递实际的 checkpoint 路径（与 rl.py 保持一致）
        started_services = start_tmux_services(config, policy_checkpoint_path=policy_checkpoint_path)
        if started_services:
            print(f"    >>>> ✓ Started {len(started_services)} tmux VLM services")
            # 注册 atexit 清理函数，确保程序退出时关闭服务
            atexit.register(stop_tmux_services)
        else:
            print("    >>>> ✗ No tmux services started (check config or errors above)")
    else:
        print("\n" + "=" * 80)
        print("Step 4: 跳过启动 tmux VLM 服务 (--start_services=False)")
        print("=" * 80)

    # ===== Step 5: 开始训练 =====
    print("\n" + "=" * 80)
    print("Step 5: 开始训练")
    print("=" * 80)
    train(env, policy, buffer, config)
