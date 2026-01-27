#!/bin/bash
# 批量启动所有 Qwen VLM 服务
# 每个服务在独立的 tmux 会话中运行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 🔍 智能检测 appvlm_ws 路径（与 start_qwen_service.sh 保持一致）
APPVLM_WS="$(cd "${SCRIPT_DIR}" && while [[ "$PWD" != "/" ]]; do
  if [[ "$(basename "$PWD")" == "appvlm_ws" ]]; then
    echo "$PWD";
    break;
  fi;
  cd ..;
done)"

# 备用路径检测
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

if [[ -z "$APPVLM_WS" ]]; then
  echo "❌ Error: Cannot find appvlm_ws directory!"
  exit 1
fi

MODEL_BASE="${APPVLM_WS}/src/ros_jackal/model"
echo "📁 Using model base: $MODEL_BASE"

# 定义所有服务配置
# 格式: "tmux名称:GPU:端口:算法:参数数量:checkpoint路径（相对于MODEL_BASE）"
declare -a SERVICES=(
    #"ddp_2500:5:5000:DDP:8:ddp/checkpoint-2500"
    "ddp_5000:0:5001:DDP:8:ddp/checkpoint-5000"
    #"dwa_2500:5:5002:DWA:9:dwa/qwen2.5-vl-regression_lora-True_dwa_regression_3b/checkpoint-2500"
    #"dwa_5000:6:5003:DWA:9:dwa/qwen2.5-vl-regression_lora-True_dwa_regression_3b/checkpoint-5000"
    #"teb_2500:6:5004:TEB:9:teb/qwen2.5-vl-regression_lora-True_teb_regression_3b/checkpoint-2500"
    #"teb_5000:6:5005:TEB:9:teb/qwen2.5-vl-regression_lora-True_teb_regression_3b/checkpoint-4067"
    #"mppi_2500:7:5006:MPPI:10:mppi/qwen2.5-vl-regression_lora-True_mppi_regression_3b/checkpoint-2500"
    #"mppi_5000:7:5007:MPPI:10:mppi/qwen2.5-vl-regression_lora-True_mppi_regression_3b/checkpoint-5000"
    #"mppi_7500:6:5008:MPPI:10:mppi/qwen2.5-vl-regression_lora-True_mppi_regression_3b/checkpoint-7500"
)

echo "=================================================="
echo "  Starting All Qwen VLM Services"
echo "=================================================="

for service in "${SERVICES[@]}"; do
    IFS=':' read -r NAME GPU PORT ALG NUM_PARAMS LORA_REL_PATH <<< "$service"

    # 构建完整的 LoRA 路径
    LORA_PATH="${MODEL_BASE}/${LORA_REL_PATH}"

    echo ""
    echo "Starting ${NAME}..."
    echo "  GPU: ${GPU}, Port: ${PORT}, Algorithm: ${ALG}, Params: ${NUM_PARAMS}"
    echo "  LoRA: ${LORA_PATH}"

    # 验证路径存在
    if [[ ! -d "$LORA_PATH" ]]; then
        echo "  [ERROR] Checkpoint not found: $LORA_PATH"
        continue
    fi

    # 检查 tmux 会话是否已存在
    if tmux has-session -t "${NAME}" 2>/dev/null; then
        echo "  [SKIP] tmux session '${NAME}' already exists"
        continue
    fi

    # 创建 tmux 会话并启动服务
    tmux new-session -d -s "${NAME}" \
        "cd ${SCRIPT_DIR} && bash start_qwen_service.sh \
            --gpu ${GPU} \
            --port ${PORT} \
            --algorithm ${ALG} \
            --num_params ${NUM_PARAMS} \
            --lora_path ${LORA_PATH}; \
         echo 'Service stopped. Press Enter to exit.'; read"

    echo "  [OK] Started in tmux session '${NAME}'"

    # 等待一小段时间，避免同时启动太多进程
    sleep 2
done

echo ""
echo "=================================================="
echo "  All services started!"
echo "=================================================="
echo ""
echo "查看所有 tmux 会话:  tmux ls"
echo "进入某个会话:        tmux attach -t ddp_2500"
echo "退出会话(不停止):    Ctrl+b d"
echo ""
