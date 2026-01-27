#!/bin/bash
# 启动Qwen2.5-VL推理服务

# bash start_qwen_service.sh --gpu 0 --port 5000
#
# 并行运行示例 (4个终端，每个用2个GPU):
#   终端1: bash start_qwen_service.sh --gpu 0,1 --port 5000
#   终端2: bash start_qwen_service.sh --gpu 2,3 --port 5001
#   终端3: bash start_qwen_service.sh --gpu 4,5 --port 5002
#   终端4: bash start_qwen_service.sh --gpu 6,7 --port 5003

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 🔍 智能检测 appvlm_ws 路径
# 方法1: 从脚本路径向上查找 appvlm_ws
APPVLM_WS="$(cd "${SCRIPT_DIR}" && while [[ "$PWD" != "/" ]]; do
  if [[ "$(basename "$PWD")" == "appvlm_ws" ]]; then
    echo "$PWD";
    break;
  fi;
  cd ..;
done)"

# 方法2: 如果方法1失败，尝试常见路径
if [[ -z "$APPVLM_WS" ]]; then
  for candidate in \
    "/home/yuanjielu/robot_navigation/noetic/appvlm_ws" \
    "/data/local/yl2832/appvlm_ws" \
    "$HOME/robot_navigation/noetic/appvlm_ws" \
    "$HOME/appvlm_ws"; do
    if [[ -d "$candidate" ]]; then
      APPVLM_WS="$candidate"
      break
    fi
  done
fi

# 验证找到了有效路径
if [[ -z "$APPVLM_WS" ]]; then
  echo "❌ Error: Cannot find appvlm_ws directory!"
  echo "Please set APPVLM_WS environment variable or run from within appvlm_ws"
  exit 1
fi

echo "📁 Detected appvlm_ws: $APPVLM_WS"

# 解析命令行参数
GPU_ID=""
PORT_ARG=""
ALGORITHM_ARG=""
NUM_PARAMS_ARG=""
LORA_PATH_ARG=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --port)
      PORT_ARG="$2"
      shift 2
      ;;
    --algorithm)
      ALGORITHM_ARG="$2"
      shift 2
      ;;
    --num_params)
      NUM_PARAMS_ARG="$2"
      shift 2
      ;;
    --lora_path)
      LORA_PATH_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--gpu GPU_ID] [--port PORT] [--algorithm ALG] [--num_params N] [--lora_path PATH]"
      echo "Example: $0 --gpu 0 --port 5000 --algorithm DWA --num_params 9 --lora_path /path/to/lora"
      echo "Example: $0 --gpu 0,1 --port 5001    # GPU 0和1, 端口5001"
      echo "Example: $0                          # 所有GPU, 端口5000 (默认)"
      exit 1
      ;;
  esac
done

# 配置参数（命令行参数优先，否则使用默认值）
BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"  # ✅ 使用3B模型（与训练时一致）
# 默认 LoRA 路径 - 使用自动检测的 appvlm_ws 路径
DEFAULT_LORA_PATH="${APPVLM_WS}/src/ros_jackal/model/ddp/checkpoint-5000"
LORA_PATH="${LORA_PATH_ARG:-$DEFAULT_LORA_PATH}"

# 验证 LORA_PATH 存在
if [[ ! -d "$LORA_PATH" ]]; then
  echo "⚠️  Warning: LORA_PATH does not exist: $LORA_PATH"
  echo "Available checkpoints:"
  ls -d "${APPVLM_WS}/src/ros_jackal/model/"*"/checkpoint-"* 2>/dev/null || echo "  (none found)"
fi
HEAD_TYPE="dpt"       # ✅ DPT head
NUM_PARAMS="${NUM_PARAMS_ARG:-9}"          # ✅ DDP: 8,  DWA: 9, TEB: 9, MPPI: 10
DEVICE_MAP="auto"     # 使用auto让模型自动分配
ALGORITHM="${ALGORITHM_ARG:-DWA}"       # 命令行参数优先，默认DWA
PORT="${PORT_ARG:-5000}"  # 命令行参数优先，默认5000
STARTUP_WARMUP=true
STARTUP_TOKENS=16
LOAD_IN_4BIT=true
LOAD_IN_8BIT=false

# 🔍 性能分析配置
ENABLE_PROFILER=false  # 设为true启用详细profiler (会增加10-20%开销)
CUDA_TIMING=true       # 使用CUDA事件精确计时

# 🚀 性能优化配置（默认启用安全优化）
ENABLE_OPTIMIZATIONS=true      # 总开关
USE_FLASH_ATTENTION=true       # FlashAttention-2/SDPA (推荐)
OPTIMIZE_MEMORY=true           # 内存优化 (推荐)

# Conda环境Python解释器
CONDA_PYTHON="/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python"
#CONDA_PYTHON="/common/home/yl2832/miniconda3/envs/lmms-finetune-qwen/bin/python"

# Qwen服务脚本路径
QWEN_SERVER="${SCRIPT_DIR}/qwen_server.py"

# 🎮 设置GPU（必须在CUDA操作之前设置）
if [ -n "${GPU_ID}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  GPU_INFO="GPU ${GPU_ID}"
else
  GPU_INFO="All available GPUs"
fi

# 🔇 抑制不重要的警告（设置环境变量）
export TRANSFORMERS_VERBOSITY=error  # 只显示错误，隐藏警告
export TOKENIZERS_PARALLELISM=false  # 避免tokenizer警告
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"

# 🛡️ 禁用PyTorch编译器优化（避免CUDA Graphs内存错误）
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0  # 保持异步执行以提高性能

echo "=================================================="
echo "  Starting Qwen2.5-VL Navigation Service"
echo "=================================================="
echo "🎮 GPU:         ${GPU_INFO}"
echo "Base Model:    ${BASE_MODEL}"
echo "LoRA Path:     ${LORA_PATH}"
echo "Head Type:     ${HEAD_TYPE}"
echo "Num Params:    ${NUM_PARAMS}"
echo "Device Map:    ${DEVICE_MAP}"
echo "Algorithm:     ${ALGORITHM}"
echo "Port:          ${PORT}"
echo "4-bit Quant:   ${LOAD_IN_4BIT}"
echo "8-bit Quant:   ${LOAD_IN_8BIT}"
echo "Startup Warm:  ${STARTUP_WARMUP}"
echo "Profiler:      ${ENABLE_PROFILER}"
echo "CUDA Timing:   ${CUDA_TIMING}"
echo ""
echo "🚀 Performance Optimizations:"
echo "  FlashAttn:   ${USE_FLASH_ATTENTION}"
echo "  Memory Opt:  ${OPTIMIZE_MEMORY}"
echo "=================================================="

# 检查文件是否存在
if [ ! -f "${QWEN_SERVER}" ]; then
    echo "Error: qwen_server.py not found at ${QWEN_SERVER}"
    exit 1
fi

# 构建基础命令
CMD=(
    "${CONDA_PYTHON}" "${QWEN_SERVER}"
    --base_model "${BASE_MODEL}"
    --lora_path "${LORA_PATH}"
    --head_type "${HEAD_TYPE}"
    --num_params ${NUM_PARAMS}
    --algorithm "${ALGORITHM}"
    --port ${PORT}
    --max_new_tokens 30  # ✅ 优化：只需输出7个数字，30 tokens足够
)

# 添加可选参数
if [ -n "${DEVICE_MAP}" ]; then
  CMD+=( --device_map "${DEVICE_MAP}" )
fi

if [ "${LOAD_IN_4BIT}" = true ]; then
  CMD+=( --load_in_4bit )
fi

if [ "${LOAD_IN_8BIT}" = true ]; then
  CMD+=( --load_in_8bit )
fi

if [ "${STARTUP_WARMUP}" = true ]; then
  CMD+=( --startup_warmup --startup_tokens ${STARTUP_TOKENS} )
fi

# 🔍 性能分析选项
if [ "${ENABLE_PROFILER}" = true ]; then
  CMD+=( --enable_profiler )
fi

# 🚀 性能优化选项
if [ "${ENABLE_OPTIMIZATIONS}" = true ]; then
  # FlashAttention
  if [ "${USE_FLASH_ATTENTION}" = true ]; then
    CMD+=( --use_flash_attention )
  fi

  # 内存优化
  if [ "${OPTIMIZE_MEMORY}" = true ]; then
    CMD+=( --optimize_memory )
  fi
else
  CMD+=( --no_optimizations )
fi

if [ "${CUDA_TIMING}" = false ]; then
  CMD+=( --no_cuda_timing )
fi

echo "Launching: ${CMD[*]}"
"${CMD[@]}"

echo "Qwen service stopped."
