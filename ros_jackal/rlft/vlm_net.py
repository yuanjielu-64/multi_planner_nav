import warnings
# 抑制PEFT的冗长警告
warnings.filterwarnings('ignore', category=UserWarning, module='peft')

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import sys
import os

# 添加qwen_dpt模型路径
qwen_dpt_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "qwen_dpt/lmms-finetune-qwen/models"
)
sys.path.append(qwen_dpt_path)

try:
    from qwen2_5_vl_dpt_regression import DPTHead, HistoryEncoder
except ImportError:
    print(f"Warning: Cannot import from qwen2_5_vl_dpt_regression")
    print(f"Expected path: {qwen_dpt_path}")


class VLM_DPT_FeatureExtractor(nn.Module):
    """
    VLM+DPT特征提取器 - 从checkpoint加载

    用于Actor和Critic共享视觉特征提取

    Args:
        checkpoint_path: VLM+DPT监督学习checkpoint路径
        freeze_vlm: 是否冻结VLM参数 (推荐True，因为VLM太大)
        freeze_dpt: 是否冻结DPT head (Actor建议False进行FTRL微调)
        freeze_history: 是否冻结History Encoder (如果使用历史帧)
        device: 设备
        use_4bit: 是否使用4-bit量化 (推荐True节省显存)
    """
    def __init__(
        self,
        checkpoint_path,
        freeze_vlm=True,
        freeze_dpt=False,
        freeze_history=None,  # None表示跟随freeze_dpt，也可以独立设置
        device="cuda",
        use_4bit=True,
        algorithm="DWA"  # 算法类型 (DWA/TEB/MPPI/DDP)
    ):
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.algorithm = algorithm.upper()  # 存储算法类型

        # 1. 加载base VLM (使用4-bit量化)
        # Qwen2.5-VL-3B 的 hidden_size 是 2048（与7B相同）
        print(f"[VLM_DPT_FeatureExtractor] Loading VLM from Qwen/Qwen2.5-VL-3B-Instruct...")

        if use_4bit and "cuda" in device:
            print("[VLM_DPT_FeatureExtractor] Using 4-bit quantization to save memory")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                quantization_config=bnb_config,
                device_map=device
            )
        else:
            self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map=device
            )

        # 统计base VLM参数
        base_vlm_params = sum(p.numel() for p in self.base_model.parameters())
        print(f"[VLM_DPT_FeatureExtractor] ✓ Base VLM loaded: {base_vlm_params:,} parameters ({base_vlm_params/1e9:.2f}B)")

        # 2. 加载LoRA (如果checkpoint包含LoRA)
        lora_path = checkpoint_path
        if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            print(f"[VLM_DPT_FeatureExtractor] Loading LoRA from {lora_path}...")

            # 检查 adapter_model.safetensors 是否为空
            adapter_weights_path = os.path.join(lora_path, "adapter_model.safetensors")
            if os.path.exists(adapter_weights_path):
                from safetensors.torch import load_file
                adapter_state = load_file(adapter_weights_path)
                non_empty_params = {k: v for k, v in adapter_state.items() if v.numel() > 0}

                if len(non_empty_params) == 0:
                    print(f"⚠️  WARNING: adapter_model.safetensors is EMPTY (0 parameters)!")
                    print(f"⚠️  LoRA will be randomly initialized (not using trained weights)")
                    print(f"⚠️  This is OK for current testing, but will need to be fixed for full training")
                else:
                    total_lora_params = sum(v.numel() for v in non_empty_params.values())
                    print(f"✓ LoRA adapter contains {len(non_empty_params)} keys, {total_lora_params:,} parameters ({total_lora_params/1e6:.2f}M)")

            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                is_trainable=(not freeze_vlm)  # 如果VLM可训练，保持LoRA可训练
            )

            # 关键决策：是否merge LoRA
            if freeze_vlm:
                # VLM冻结：merge LoRA以节省显存和加速推理
                self.base_model = self.base_model.merge_and_unload()
                print(f"[VLM_DPT_FeatureExtractor] ✓ LoRA merged into base model (VLM frozen)")
            else:
                # VLM可训练：保持LoRA作为独立层，以便训练和保存
                lora_params = sum(p.numel() for n, p in self.base_model.named_parameters() if 'lora' in n.lower())
                print(f"[VLM_DPT_FeatureExtractor] ✓ LoRA loaded as trainable layers: {lora_params:,} parameters ({lora_params/1e6:.2f}M)")
        else:
            print(f"[VLM_DPT_FeatureExtractor] No LoRA found, using base VLM")

        # 3. 加载DPT head
        regression_head_path = os.path.join(checkpoint_path, "regression_head", "pytorch_model.bin")
        if not os.path.exists(regression_head_path):
            # Fallback: 尝试不带子目录的路径
            regression_head_path = os.path.join(checkpoint_path, "pytorch_model.bin")

        # 3.1 读取历史帧配置
        history_config_path = os.path.join(checkpoint_path, "history_config.json")
        if os.path.exists(history_config_path):
            import json
            with open(history_config_path) as f:
                history_config = json.load(f)
                self.use_history = history_config.get('use_history', False)
                self.num_history_frames = history_config.get('num_history_frames', 2)
                self.history_dim = history_config.get('history_dim', 256)
            print(f"[VLM_DPT_FeatureExtractor] History config: use_history={self.use_history}, num_frames={self.num_history_frames}")
        else:
            self.use_history = False
            self.num_history_frames = 2
            self.history_dim = 256
            print(f"[VLM_DPT_FeatureExtractor] No history_config.json found, using use_history=False")

        # 3.2 创建DPT Head
        print(f"[VLM_DPT_FeatureExtractor] Loading DPT head from {regression_head_path}...")
        self.dpt_head = DPTHead(
            hidden_size=2048,  # Qwen2.5-VL-3B (与checkpoint匹配)
            num_params=8,      # DDP有8个参数
            feature_dim=256,
            num_layers=4,
            use_history=self.use_history,  # 从配置读取
            history_dim=self.history_dim
        )

        if os.path.exists(regression_head_path):
            state_dict = torch.load(regression_head_path, map_location='cpu')
            # 统计加载的参数数量
            total_params = sum(p.numel() for p in state_dict.values() if p.numel() > 0)
            num_keys = len([k for k, v in state_dict.items() if v.numel() > 0])

            self.dpt_head.load_state_dict(state_dict, strict=False)
            print(f"[VLM_DPT_FeatureExtractor] ✓ DPT head loaded: {num_keys} keys, {total_params:,} parameters ({total_params/1e6:.2f}M)")
        else:
            print(f"[VLM_DPT_FeatureExtractor] ⚠️  Warning: DPT head not found at {regression_head_path}")
            print("[VLM_DPT_FeatureExtractor] Using randomly initialized DPT head")

        # 🔧 关键修复：DPT head 保持 float32，不要转 bfloat16！
        # 原因：bfloat16 精度太低，在 TD3 训练的梯度更新中容易导致 NaN
        # VLM 输出的 bfloat16 hidden states 会在 forward 中自动转换为 float32
        self.dpt_head.to(device=device, dtype=torch.float32)
        print(f"[VLM_DPT_FeatureExtractor] ✓ DPT head moved to device={device}, dtype=float32 (NOT bfloat16 for stability)")

        # 3.3 加载History Encoder (如果使用)
        self.history_encoder = None
        if self.use_history:
            # 监督学习保存格式: history_encoder/pytorch_model.bin
            history_encoder_path = os.path.join(checkpoint_path, "history_encoder", "pytorch_model.bin")

            # Fallback: 尝试旧格式 history_encoder.pth
            if not os.path.exists(history_encoder_path):
                history_encoder_path = os.path.join(checkpoint_path, "history_encoder.pth")

            if os.path.exists(history_encoder_path):
                self.history_encoder = HistoryEncoder(
                    hidden_dim=self.history_dim,
                    num_frames=self.num_history_frames,
                    num_transformer_layers=2
                )
                history_state_dict = torch.load(history_encoder_path, map_location='cpu')

                # 统计参数数量
                total_history_params = sum(p.numel() for p in history_state_dict.values() if p.numel() > 0)
                num_history_keys = len([k for k, v in history_state_dict.items() if v.numel() > 0])

                self.history_encoder.load_state_dict(history_state_dict)
                # 🔧 关键修复: History encoder 保持 float32，不要转 bfloat16
                # CNN + Transformer 在 bfloat16 下数值不稳定，容易产生 NaN
                self.history_encoder.to(device=device, dtype=torch.float32)
                print(f"[VLM_DPT_FeatureExtractor] ✓ History encoder loaded: {num_history_keys} keys, {total_history_params:,} parameters ({total_history_params/1e6:.2f}M)")
                print(f"[VLM_DPT_FeatureExtractor] ✓ History encoder moved to device={device}, dtype=float32 (NOT bfloat16 for stability)")
            else:
                print(f"[VLM_DPT_FeatureExtractor] ⚠️  Warning: use_history=True but history encoder not found")
                print(f"[VLM_DPT_FeatureExtractor] Expected: {os.path.join(checkpoint_path, 'history_encoder/pytorch_model.bin')}")
                self.use_history = False  # 降级到不使用历史帧

        # 4. 冻结策略
        if freeze_vlm:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("[VLM_DPT_FeatureExtractor] ✓ VLM frozen")
        else:
            print("[VLM_DPT_FeatureExtractor] VLM trainable")

        if freeze_dpt:
            for param in self.dpt_head.parameters():
                param.requires_grad = False
            print("[VLM_DPT_FeatureExtractor] ✓ DPT head frozen")
        else:
            print("[VLM_DPT_FeatureExtractor] DPT head trainable")

        # History encoder独立冻结策略
        if self.history_encoder is not None:
            # 如果freeze_history未指定，跟随freeze_dpt
            freeze_history_final = freeze_history if freeze_history is not None else freeze_dpt

            if freeze_history_final:
                for param in self.history_encoder.parameters():
                    param.requires_grad = False
                print("[VLM_DPT_FeatureExtractor] ✓ History encoder frozen")
            else:
                print("[VLM_DPT_FeatureExtractor] History encoder trainable")

        self.feature_dim = 256  # DPT输出维度

        # 5. 加载processor (用于图像预处理)
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            use_fast=True  # 使用快速处理器，避免警告
        )

        # 6. 参数统计汇总
        print("\n" + "="*70)
        print("📊 VLM_DPT_FeatureExtractor Parameter Summary")
        print("="*70)

        # VLM Base Model
        vlm_total = sum(p.numel() for p in self.base_model.parameters())
        vlm_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        print(f"VLM Base (Qwen2.5-VL-3B):")
        print(f"  ├─ Total:      {vlm_total:>12,} parameters ({vlm_total/1e9:.2f}B)")
        print(f"  ├─ Trainable:  {vlm_trainable:>12,} parameters ({vlm_trainable/1e6:.2f}M)")
        print(f"  └─ Frozen:     {vlm_total-vlm_trainable:>12,} parameters ({(vlm_total-vlm_trainable)/1e9:.2f}B)")

        # DPT Head
        dpt_total = sum(p.numel() for p in self.dpt_head.parameters())
        dpt_trainable = sum(p.numel() for p in self.dpt_head.parameters() if p.requires_grad)
        print(f"\nDPT Head:")
        print(f"  ├─ Total:      {dpt_total:>12,} parameters ({dpt_total/1e6:.2f}M)")
        print(f"  ├─ Trainable:  {dpt_trainable:>12,} parameters ({dpt_trainable/1e6:.2f}M)")
        print(f"  └─ Frozen:     {dpt_total-dpt_trainable:>12,} parameters ({(dpt_total-dpt_trainable)/1e6:.2f}M)")

        # History Encoder (如果存在)
        if self.history_encoder is not None:
            hist_total = sum(p.numel() for p in self.history_encoder.parameters())
            hist_trainable = sum(p.numel() for p in self.history_encoder.parameters() if p.requires_grad)
            print(f"\nHistory Encoder:")
            print(f"  ├─ Total:      {hist_total:>12,} parameters ({hist_total/1e6:.2f}M)")
            print(f"  ├─ Trainable:  {hist_trainable:>12,} parameters ({hist_trainable/1e6:.2f}M)")
            print(f"  └─ Frozen:     {hist_total-hist_trainable:>12,} parameters ({(hist_total-hist_trainable)/1e6:.2f}M)")
        else:
            print(f"\nHistory Encoder: Not loaded")

        # 总计
        total_params = vlm_total + dpt_total
        total_trainable = vlm_trainable + dpt_trainable
        if self.history_encoder is not None:
            total_params += hist_total
            total_trainable += hist_trainable

        print(f"\n{'─'*70}")
        print(f"Total Feature Extractor:")
        print(f"  ├─ Total:      {total_params:>12,} parameters ({total_params/1e9:.2f}B)")
        print(f"  ├─ Trainable:  {total_trainable:>12,} parameters ({total_trainable/1e6:.2f}M)")
        print(f"  ├─ Frozen:     {total_params-total_trainable:>12,} parameters ({(total_params-total_trainable)/1e9:.2f}B)")
        print(f"  └─ Trainable%: {100*total_trainable/total_params:>11.2f}%")
        print("="*70 + "\n")

    def forward(self, images, prompt=None, history_images=None,
                linear_vels=None, angular_vels=None, algorithm="DWA"):
        """
        Args:
            images: PIL.Image list 或 [B, 3, H, W] tensor (当前帧)
            prompt: str, 如果为None使用默认导航prompt
            history_images: [B, num_frames, 3, H, W] tensor 或 List[List[PIL.Image]] (历史帧，可选)
            linear_vels: List[float] (当前线速度，可选，用于生成prompt)
            angular_vels: List[float] (当前角速度，可选，用于生成prompt)
            algorithm: str (规划算法名称，默认DWA)

        Returns:
            features: [B, 256] 特征向量
        """
        # System prompt (与训练时一致)
        SYSTEM_PROMPT = """You are a navigation scene analyzer for Clearpath Jackal robot motion planning.
Scene Representation (Costmap Visualization):
- Scale: Each grid cell represents approximately 1 meter.
- Robot: Yellow square (0.508 m x 0.430 m footprint)
- Green arrow: forward driving direction (x-axis)
- Blue line: lateral direction (y-axis)
- Obstacles: Red points from laser scanner measurements
- Global Path: Black curve showing a kinematic reference trajectory that guides direction but does not encode dynamics or feasibility.
- Background: Grey grid is for visualization only and has no obstacle or cost meaning.
Your Role:
Analyze the spatial layout to understand:
- Obstacle distribution and density around the robot
- Available free space and safe clearance margins
- Path geometry (straight corridor, sharp turn, narrow passage)
- Motion feasibility given the current velocity state

Your navigation reasoning will be used downstream to select parameters for the local motion planner."""

        # 构建完整的chat messages
        if not isinstance(images, list):
            # Tensor -> 暂时不支持，因为chat template需要PIL.Image
            raise ValueError("QwenVLMFeatureExtractor only supports PIL.Image list in forward()")

        batch_size = len(images)
        texts = []

        for i in range(batch_size):
            # User prompt (包含速度信息)
            if linear_vels is not None and angular_vels is not None:
                user_prompt = f"""Current robot state:
- Linear velocity: {linear_vels[i]:.3f} m/s
- Angular velocity: {angular_vels[i]:.3f} rad/s

Target local planner: {algorithm}

Use the scene image and robot state to perform navigation reasoning about obstacle proximity, path curvature, free-space structure, and motion constraints."""
            else:
                user_prompt = f"""Target local planner: {algorithm}

Analyze the costmap to understand obstacle distribution and path geometry."""

            # 构建messages (与qwen_server_flash_attn.py一致)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": user_prompt.strip()},
                    ],
                }
            ]

            # 应用chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # 回归任务不生成assistant回复
            )
            texts.append(text)

        # Processor处理 (与qwen_server_flash_attn.py一致)
        inputs = self.processor(
            text=texts,
            images=images,
            videos=None,
            padding=False,  # 与训练时一致
            return_tensors="pt"
        ).to(self.device)

        # VLM前向 (获取hidden states)
        # 🔧 关键：使用 self.base_model(**inputs) 而不是 self.base_model.model(**inputs)
        # - base_model (Qwen2_5_VLForConditionalGeneration) 会先通过 vision encoder 处理 pixel_values
        # - base_model.model (Qwen2_5_VLModel) 不接受 pixel_values，只接受处理后的 inputs_embeds
        # - 参考: script/qwen/qwen_server.py:180-184
        is_vlm_frozen = not any(p.requires_grad for p in self.base_model.parameters())

        with torch.no_grad() if is_vlm_frozen else torch.enable_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            multi_layer_hidden_states = outputs.hidden_states[-4:]

            # Debug: check VLM hidden states
            for i, hs in enumerate(multi_layer_hidden_states):
                if torch.isnan(hs).any() or torch.isinf(hs).any():
                    print(f"    [VLM] WARNING: hidden_state layer {-4+i} contains nan/inf!")
                    print(f"      shape: {hs.shape}, nan: {torch.isnan(hs).sum()}, inf: {torch.isinf(hs).sum()}")

        # 编码历史帧 (如果使用)
        history_feat = None
        if self.use_history and self.history_encoder is not None and history_images is not None:
            # 转换 List[List[PIL.Image]] -> [B, num_frames, 3, H, W] tensor
            if isinstance(history_images, list) and len(history_images) > 0 and isinstance(history_images[0], list):
                # VLMReplayBuffer format: List[List[PIL.Image]]
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                batch_history = []
                # 从 history_encoder 读取实际需要的帧数
                required_frames = self.num_history_frames  # 从 __init__ 读取

                for batch_idx, hist_imgs in enumerate(history_images):
                    if hist_imgs is None or len(hist_imgs) == 0:
                        # 用零填充（使用 float32，与 history_encoder 一致）
                        batch_history.append(torch.zeros(required_frames, 3, 224, 224, dtype=torch.float32))
                    else:
                        # 智能处理：截取或填充到 required_frames
                        actual_frames = len(hist_imgs)

                        if actual_frames >= required_frames:
                            # 截取最新的 required_frames 帧
                            selected_imgs = hist_imgs[:required_frames]
                        else:
                            # 填充：用最后一帧重复填充
                            selected_imgs = hist_imgs + [hist_imgs[-1]] * (required_frames - actual_frames)

                        frames = [transform(img) for img in selected_imgs]
                        batch_history.append(torch.stack(frames))  # [required_frames, 3, H, W]

                # 关键：转换为 float32（与 history_encoder 一致，不用 bfloat16）
                history_images = torch.stack(batch_history).to(device=self.device, dtype=torch.float32)  # [B, required_frames, 3, H, W]

            is_history_frozen = not any(p.requires_grad for p in self.history_encoder.parameters())

            # 🔧 DEBUG: 检查 history_encoder 输入和参数
            if torch.isnan(history_images).any():
                print(f"    [HistoryEncoder] WARNING: INPUT history_images contains NaN!")
                print(f"      shape: {history_images.shape}, nan count: {torch.isnan(history_images).sum()}")

            # 检查 history_encoder 参数是否有 NaN
            for name, param in self.history_encoder.named_parameters():
                if torch.isnan(param).any():
                    print(f"    [HistoryEncoder] WARNING: PARAM {name} contains NaN!")

            with torch.no_grad() if is_history_frozen else torch.enable_grad():
                history_feat = self.history_encoder(history_images)  # [B, history_dim]

            # 🔧 DEBUG: 检查输出
            if torch.isnan(history_feat).any():
                print(f"    [HistoryEncoder] WARNING: OUTPUT history_feat contains NaN!")
                print(f"      history_feat shape: {history_feat.shape}")
                print(f"      history_images dtype: {history_images.dtype}")
                print(f"      history_encoder dtype: {next(self.history_encoder.parameters()).dtype}")

        # DPT提取特征 (包含历史特征融合)
        is_dpt_frozen = not any(p.requires_grad for p in self.dpt_head.parameters())

        with torch.no_grad() if is_dpt_frozen else torch.enable_grad():
            features = self._extract_dpt_features(multi_layer_hidden_states, history_feat)

        return features

    def _extract_dpt_features(self, multi_layer_hidden_states, history_feat=None):
        """
        提取DPT的中间特征 (256-d pooled)，不经过最后的MLP回归层
        """
        B, seq_len, _ = multi_layer_hidden_states[0].shape

        # 🔧 关键修复：将 VLM 输出的 bfloat16 hidden states 转换为 float32
        # DPT head 现在是 float32，需要匹配的输入 dtype
        multi_layer_hidden_states = [hs.float() for hs in multi_layer_hidden_states]

        # 🔧 DEBUG: 检查 DPT 参数是否有 NaN
        dpt_has_nan = False
        for name, param in self.dpt_head.named_parameters():
            if torch.isnan(param).any():
                print(f"    [DPT] WARNING: PARAM {name} has NaN! requires_grad={param.requires_grad}")
                dpt_has_nan = True
        if dpt_has_nan:
            print(f"    [DPT] DPT head parameters have NaN!")

        # Step 1: 投影所有层到统一特征空间
        projected = []
        for i, (proj, hidden_state) in enumerate(zip(self.dpt_head.projections, multi_layer_hidden_states)):
            p = proj(hidden_state)
            if torch.isnan(p).any():
                print(f"    [DPT] WARNING: projection[{i}] output has NaN!")
            projected.append(p)

        # Step 2: 转换为Conv1d格式 [B, feature_dim, seq_len]
        projected = [p.transpose(1, 2) for p in projected]

        # Step 3: 渐进式融合 (top-down refinement)
        fused = projected[-1]
        for i in range(len(self.dpt_head.fusion_blocks) - 1, -1, -1):
            skip = projected[i]
            fused = self.dpt_head.fusion_blocks[i](fused, skip)
            if torch.isnan(fused).any():
                print(f"    [DPT] WARNING: fusion_block[{i}] output has NaN!")

        fused = fused.transpose(1, 2)  # [B, seq_len, feature_dim]

        # Step 4: Spatial attention pooling
        attention_weights = self.dpt_head.spatial_attention(fused)  # [B, seq_len, 1]
        if torch.isnan(attention_weights).any():
            print(f"    [DPT] WARNING: attention_weights has NaN!")
        pooled = (fused * attention_weights).sum(dim=1)  # [B, feature_dim]

        # Step 5: 历史特征融合 (如果使用)
        if self.use_history and history_feat is not None:
            # 🔧 DEBUG: 检查 dtype 和 NaN
            if torch.isnan(pooled).any():
                print(f"    [DPT] WARNING: pooled contains NaN BEFORE history fusion!")
            if torch.isnan(history_feat).any():
                print(f"    [DPT] WARNING: history_feat contains NaN!")
                print(f"      pooled dtype: {pooled.dtype}, history_feat dtype: {history_feat.dtype}")

            # 确保 dtype 一致（都转为 float32 以保证稳定性）
            pooled = pooled.float()
            history_feat = history_feat.float()

            # 与监督学习保持一致的融合方式
            combined = torch.cat([pooled, history_feat], dim=-1)  # [B, feature_dim + history_dim]

            # history_fusion 也需要 float32 输入
            pooled = self.dpt_head.history_fusion(combined.to(self.dpt_head.history_fusion[0].weight.dtype))  # [B, feature_dim]

            if torch.isnan(pooled).any():
                print(f"    [DPT] WARNING: pooled contains NaN AFTER history fusion!")
                # 检查 history_fusion 权重
                for i, layer in enumerate(self.dpt_head.history_fusion):
                    if hasattr(layer, 'weight') and torch.isnan(layer.weight).any():
                        print(f"      history_fusion[{i}] weight has NaN!")

        return pooled  # [B, 256]


class VLM_DPT_Actor(nn.Module):
    """
    VLM+DPT Actor - 用于FTRL

    架构: VLM+DPT特征提取 → FC层 → Action

    Args:
        feature_extractor: VLM_DPT_FeatureExtractor实例
        action_dim: 动作维度 (7个导航参数)
    """
    def __init__(self, feature_extractor, action_dim=7, algorithm="DWA"):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.action_dim = action_dim
        self.algorithm = algorithm.upper()  # 存储算法类型

        # 输出层: 256-d → action_dim
        self.fc = nn.Linear(feature_extractor.feature_dim, action_dim)

        # 初始化
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # 重要: Actor的FC层保持float32，不要转bfloat16
        # bfloat16精度太低会导致数值不稳定
        # self._sync_dtype()  # 禁用dtype同步

    def _sync_dtype(self):
        """同步FC层dtype与feature_extractor"""
        if hasattr(self.feature_extractor, 'base_model'):
            target_dtype = self.feature_extractor.base_model.dtype
            self.fc = self.fc.to(dtype=target_dtype)

    def forward(self, images, prompt=None, history_images=None):
        """
        Args:
            images: PIL.Image list 或 [B, 3, H, W] tensor (当前帧)
                    或 List[(PIL.Image, List[PIL.Image])] (VLMReplayBuffer格式)
            prompt: str, 可选
            history_images: [B, num_frames, 3, H, W] tensor (历史帧，可选)

        Returns:
            action: [B, action_dim] 归一化到[-1, 1]
        """
        # Handle VLMReplayBuffer format: List[(img, history_imgs, linear_vel, angular_vel)]
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], tuple):
            # Unpack tuple format
            imgs = [item[0] for item in images]
            hists = [item[1] for item in images]
            # Extract velocities for prompt generation
            linear_vels = [item[2] if len(item) > 2 else 0.0 for item in images]
            angular_vels = [item[3] if len(item) > 3 else 0.0 for item in images]

            features = self.feature_extractor(
                imgs, prompt, hists,
                linear_vels=linear_vels,
                angular_vels=angular_vels,
                algorithm=self.algorithm  # 使用实例属性
            )
        else:
            # Normal format: separate images and history_images
            features = self.feature_extractor(images, prompt, history_images)

        # 转换到float32以保证数值稳定性 (FC层是float32)
        features = features.float()

        # Debug: check VLM features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"    [Actor] WARNING: VLM features contain nan/inf!")
            print(f"      features range: [{features.min():.4f}, {features.max():.4f}]")
            print(f"      nan count: {torch.isnan(features).sum()}, inf count: {torch.isinf(features).sum()}")

        # Debug: check FC weights
        if torch.isnan(self.fc.weight).any() or torch.isnan(self.fc.bias).any():
            print(f"    [Actor] WARNING: FC layer weights contain nan!")

        action = torch.tanh(self.fc(features))
        return action


class VLM_DPT_Critic(nn.Module):
    """
    VLM+DPT Critic - 用于FTRL

    架构: VLM+DPT特征提取 + Action编码 → 双Q网络

    TD3的Twin Critic设计: 共享特征提取器，独立的Q-head
    (Twin指的是两个独立的Q-value估计，不是两个特征提取器)

    Args:
        feature_extractor: 共享的VLM_DPT_FeatureExtractor (与Actor共享)
        action_dim: 动作维度
        detach_features: 是否detach特征（防止Critic梯度回传到VLM）
    """
    def __init__(
        self,
        feature_extractor,
        action_dim=7,
        detach_features=True  # 默认detach，保护Actor的VLM更新
    ):
        super().__init__()

        # 共享的特征提取器 (与Actor共享，VLM只需要跑1次)
        self.feature_extractor = feature_extractor
        self.detach_features = detach_features

        feature_dim = feature_extractor.feature_dim

        # 共享的Action编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Q1网络: [256 + 64] → 1
        self.q1_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2网络: [256 + 64] → 1
        self.q2_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 初始化
        self._init_weights()

        # 重要: Critic的可训练层保持float32，不要转bfloat16
        # bfloat16精度太低，Adam更新时会导致nan
        # VLM特征会自动转换dtype，但Q-heads需要float32保证数值稳定性
        # self._sync_dtype()  # 禁用dtype同步

    def _init_weights(self):
        """权重初始化"""
        for module in [self.action_encoder, self.q1_head, self.q2_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _sync_dtype(self):
        """同步所有层的dtype与feature_extractor"""
        if hasattr(self.feature_extractor, 'base_model'):
            target_dtype = self.feature_extractor.base_model.dtype
            self.action_encoder = self.action_encoder.to(dtype=target_dtype)
            self.q1_head = self.q1_head.to(dtype=target_dtype)
            self.q2_head = self.q2_head.to(dtype=target_dtype)

    def forward(self, images, action, prompt=None, history_images=None):
        """
        Args:
            images: PIL.Image list 或 [B, 3, H, W] tensor (当前帧)
                    或 List[(PIL.Image, List[PIL.Image], float, float)] (VLMReplayBuffer格式)
            action: [B, action_dim] 动作参数
            prompt: str, 可选
            history_images: [B, num_frames, 3, H, W] tensor (历史帧，可选)

        Returns:
            q1, q2: [B, 1] 两个Q值估计
        """
        # Handle VLMReplayBuffer format: List[(img, history_imgs, linear_vel, angular_vel)]
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], tuple):
            # Unpack tuple format
            imgs = [item[0] for item in images]
            hists = [item[1] for item in images]
            linear_vels = [item[2] if len(item) > 2 else 0.0 for item in images]
            angular_vels = [item[3] if len(item) > 3 else 0.0 for item in images]

            vlm_feat = self.feature_extractor(
                imgs, prompt, hists,
                linear_vels=linear_vels,
                angular_vels=angular_vels,
                algorithm=self.feature_extractor.algorithm  # 从feature_extractor获取
            )
        else:
            # Normal format: separate images and history_images
            vlm_feat = self.feature_extractor(images, prompt, history_images)

        # 关键: detach阻止Critic的梯度回传到VLM，保护Actor的更新
        if self.detach_features:
            vlm_feat = vlm_feat.detach()

        # 转换到float32以保证数值稳定性 (Q-heads是float32)
        vlm_feat = vlm_feat.float()
        action = action.float()

        action_feat = self.action_encoder(action)
        q_input = torch.cat([vlm_feat, action_feat], dim=-1)

        # 两个独立的Q-head (TD3的Twin Critic)
        q1 = self.q1_head(q_input)
        q2 = self.q2_head(q_input)

        return q1, q2

    def Q1(self, images, action, prompt=None, history_images=None):
        """
        只计算Q1 (用于actor更新时的梯度计算)

        Args:
            images: PIL.Image list 或 [B, 3, H, W] tensor (当前帧)
                    或 List[(PIL.Image, List[PIL.Image], float, float)] (VLMReplayBuffer格式)
            action: [B, action_dim]
            prompt: str, 可选
            history_images: [B, num_frames, 3, H, W] tensor (历史帧，可选)

        Returns:
            q1: [B, 1]
        """
        # Handle VLMReplayBuffer format: List[(img, history_imgs, linear_vel, angular_vel)]
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], tuple):
            # Unpack tuple format
            imgs = [item[0] for item in images]
            hists = [item[1] for item in images]
            linear_vels = [item[2] if len(item) > 2 else 0.0 for item in images]
            angular_vels = [item[3] if len(item) > 3 else 0.0 for item in images]

            # 获取算法类型 (从feature_extractor或默认DWA)
            algorithm = getattr(self.feature_extractor, 'algorithm', 'DWA')

            vlm_feat = self.feature_extractor(
                imgs, prompt, hists,
                linear_vels=linear_vels,
                angular_vels=angular_vels,
                algorithm=algorithm
            )
        else:
            # Normal format: separate images and history_images
            vlm_feat = self.feature_extractor(images, prompt, history_images)

        # 关键: detach阻止Critic的梯度回传到VLM，保护Actor的更新
        if self.detach_features:
            vlm_feat = vlm_feat.detach()

        # 转换到float32以保证数值稳定性 (Q-heads是float32)
        vlm_feat = vlm_feat.float()
        action = action.float()

        action_feat = self.action_encoder(action)
        q1_input = torch.cat([vlm_feat, action_feat], dim=-1)
        return self.q1_head(q1_input)
