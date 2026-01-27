# RLFT (Reinforcement Learning Fine-Tuning)

VLM+DPT 与 TD3 结合的强化学习微调实现

## 📁 目录结构

```
rlft/
├── __init__.py          # 包初始化
├── vlm_net.py           # VLM+DPT网络定义
├── rl.py                # TD3算法实现 (复用自td3/)
├── train.py             # FTRL训练脚本
└── README.md            # 本文档
```

## 🔧 核心组件

### 1. VLM_DPT_FeatureExtractor
从监督学习checkpoint加载VLM+DPT作为特征提取器

**关键设计**:
- 加载预训练的Qwen2.5-VL + LoRA
- 加载预训练的DPT Head
- 支持选择性冻结（VLM/DPT独立控制）
- 只提取DPT的中间特征（256-d），不使用回归层

### 2. VLM_DPT_Actor
TD3的Actor网络

**架构**:
```
Costmap Image → VLM+DPT特征提取 → FC层 → 7个导航参数
```

**训练策略**:
- VLM: 冻结（太大，更新慢）
- DPT Head: 可训练（FTRL微调）
- FC层: 可训练

### 3. VLM_DPT_Critic
TD3的Twin Critic网络

**架构**:
```
Costmap Image → VLM+DPT特征提取 → [256-d]
                                    ↓
Action (7个参数) → MLP编码 → [64-d] ↓
                                    ↓
            Concat → Fusion MLP → Q值
```

**训练策略**:
- VLM+DPT: 全部冻结（节省显存）
- Action编码器: 可训练
- Fusion MLP: 可训练

## 🚀 快速开始

### 1. 准备checkpoint

确保你有监督学习训练好的VLM+DPT checkpoint：

```bash
checkpoint-2500/
├── adapter_config.json      # LoRA配置
├── adapter_model.bin         # LoRA权重
└── regression_head/
    └── pytorch_model.bin     # DPT Head权重
```

### 2. 修改配置文件

编辑 `script/ft_qwen/configs/ftrl_vlm_dwa.yaml`:

```yaml
training_config:
  # 修改为你的checkpoint路径
  vlm_checkpoint_path: "/path/to/your/checkpoint-2500"
```

### 3. 启动训练

**方式1: 使用启动脚本（推荐）**
```bash
cd /path/to/ros_jackal
./script/ft_qwen/run_ftrl.sh
```

**方式2: 直接运行Python**
```bash
cd /path/to/ros_jackal
python rlft/train.py \
  --config_path script/ft_qwen/configs/ \
  --config_file ftrl_vlm_dwa \
  --buffer_path buffer/ftrl_vlm \
  --logging_path logging/ftrl_vlm
```

### 4. 监控训练

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir logging/ftrl_vlm
```

**关键指标**:
- `train/Test_nav_metric`: 测试集导航性能（越高越好）
- `train/Success_rate`: 成功率
- `train/Actor_loss`: Actor损失
- `train/Critic_loss`: Critic损失

## ⚙️ 配置说明

### VLM+DPT配置

```yaml
training_config:
  # Checkpoint路径
  vlm_checkpoint_path: "/path/to/checkpoint"

  # 冻结策略
  freeze_vlm_actor: true     # Actor的VLM冻结（推荐）
  freeze_dpt_actor: false    # Actor的DPT可训练（FTRL）
  freeze_dpt_critic: true    # Critic的DPT冻结（省显存）
```

### 学习率配置

```yaml
training_config:
  # VLM微调需要更小的学习率
  actor_lr: 1.0e-5    # 比APPLR小10倍
  critic_lr: 3.0e-4
```

### TD3超参数

```yaml
training_config:
  policy_args:
    gamma: 0.99              # 折扣因子
    tau: 0.005               # 软更新系数
    policy_noise: 0.2        # 目标策略平滑噪声
    noise_clip: 0.5          # 噪声裁剪
    n_step: 4                # N-step return
    update_actor_freq: 2     # Actor延迟更新
    exploration_noise: 0.1   # 探索噪声
```

### 训练参数

```yaml
training_config:
  training_args:
    max_step: 1000000         # 总训练步数
    collect_per_step: 1000    # 每次收集步数
    update_per_step: 50       # 每次更新次数
    batch_size: 256           # 批大小
```

## 🎯 与APPLR的区别

| 方面 | APPLR (Baseline) | RLFT (本实现) |
|------|------------------|---------------|
| 特征提取 | CNN (3层Conv) | VLM+DPT (预训练) |
| 初始化 | 随机初始化 | 监督学习预训练 |
| 训练数据需求 | 5M samples | 预期<1M samples |
| Actor参数量 | ~1M | ~8B (大部分冻结) |
| 样本效率 | 低 | 高（预训练加持） |
| 训练时间 | 6小时 (500 CPU) | 待测试 |

## 💡 关键技术点

### 1. Critic为什么不能直接复用Actor的checkpoint？

**问题**: Actor和Critic的输入空间不同
- Actor: `state` → `action`
- Critic: `(state, action)` → `Q值`

**解决**: Critic使用VLM+DPT提取state特征，额外用MLP编码action，然后fusion

### 2. 为什么要冻结VLM？

**原因**:
- VLM有8B参数，RL更新太慢
- VLM的视觉理解能力已经很强，不需要继续训练
- 节省显存和计算

**FTRL策略**: 只微调DPT Head（256-d特征空间的回归）

### 3. 为什么Critic的DPT也冻结？

**原因**:
- Critic不需要直接预测参数，只需要评估好坏
- 冻结DPT可以节省大量显存（双Q网络需要2个VLM）
- Critic的fusion层已经足够学习Q值

## 🐛 常见问题

### Q1: 显存不足怎么办？

**解决方案**:
1. 减小batch_size（256 → 128 → 64）
2. 使用4-bit量化加载VLM
3. 只在Actor中使用VLM，Critic用轻量CNN

### Q2: 训练不稳定？

**解决方案**:
1. 减小actor_lr（1e-5 → 5e-6）
2. 增加pre_collect（10000 → 50000）
3. 减小exploration_noise_start（0.05 → 0.02）

### Q3: VLM加载失败？

**检查**:
1. checkpoint路径是否正确
2. 是否包含`adapter_config.json`（LoRA）
3. 是否包含`regression_head/pytorch_model.bin`（DPT）

### Q4: 性能不如监督学习？

**可能原因**:
1. RL探索破坏了预训练知识 → 减小exploration noise
2. 学习率太大 → 减小actor_lr
3. 训练步数不够 → 增加max_step

## 📊 预期性能

**监督学习baseline**:
- MAE: ~0.05-0.1（归一化后）
- 推理速度: ~100-500ms/frame

**FTRL目标**:
- 导航成功率: 超过监督学习
- 样本效率: <1M steps（vs APPLR的5M）
- 训练时间: ~24小时（单GPU）

## 🔬 实验建议

### 消融实验

1. **VLM冻结vs微调**
   - 配置: `freeze_vlm_actor: true/false`
   - 对比训练速度和性能

2. **DPT冻结vs微调**
   - 配置: `freeze_dpt_actor: true/false`
   - 验证FTRL的必要性

3. **不同学习率**
   - `actor_lr: [1e-6, 5e-6, 1e-5, 5e-5]`
   - 找最优学习率

### 对比实验

1. **FTRL vs 监督学习**
   - 在相同测试环境评估
   - 对比成功率、轨迹平滑度

2. **FTRL vs APPLR**
   - 样本效率对比
   - 性能上界对比

## 📚 参考文献

- APPLR: Adaptive Planner Parameter Learning from Reinforcement
- TD3: Twin Delayed Deep Deterministic Policy Gradient
- DPT: Dense Prediction Transformer (参考DUSt3R)
- Qwen2.5-VL: 视觉语言模型

## 🤝 贡献

如有问题或改进建议，请联系项目维护者。

---

**Happy Fine-Tuning! 🚀**
