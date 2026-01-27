import copy
import os
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_preprocess, head, action_dim):
        super(Actor, self).__init__()

        self.state_preprocess = state_preprocess
        self.head = head
        self.fc = nn.Linear(self.head.feature_dim, action_dim)

    def forward(self, state):
        a = self.state_preprocess(state) if self.state_preprocess else state
        a = self.head(a)
        return torch.tanh(self.fc(a))


class Critic(nn.Module):
    def __init__(self, state_preprocess, head):
        super(Critic, self).__init__()

        # Q1 architecture
        self.state_preprocess1 = state_preprocess
        self.head1 = head
        self.fc1 = nn.Linear(self.head1.feature_dim, 1)

        # Q2 architecture
        self.state_preprocess2 = copy.deepcopy(state_preprocess)
        self.head2 = copy.deepcopy(head)
        self.fc2 = nn.Linear(self.head2.feature_dim, 1)

    def forward(self, state, action):
        state1 = self.state_preprocess1(
            state) if self.state_preprocess1 else state
        sa1 = torch.cat([state1, action], 1)

        state2 = self.state_preprocess2(
            state) if self.state_preprocess2 else state
        sa2 = torch.cat([state2, action], 1)

        q1 = self.head1(sa1)
        q1 = self.fc1(q1)

        q2 = self.head2(sa2)
        q2 = self.fc2(q2)
        return q1, q2

    def Q1(self, state, action):
        state = self.state_preprocess1(
            state) if self.state_preprocess1 else state
        sa = torch.cat([state, action], 1)

        q1 = self.head1(sa)
        q1 = self.fc1(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            actor,
            actor_optim,
            critic,
            critic_optim,
            action_range,
            device="cpu",
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            n_step=4,
            update_actor_freq=2,
            exploration_noise=0.1,
            param_mean=None,
            param_std=None
    ):

        self.actor = actor
        # 🔧 关键修复: 不能 deepcopy 4-bit 量化的 VLM
        # actor_target 共享 feature_extractor，只复制 FC 层
        self.actor_target = self._create_actor_target(actor)
        self.actor_optimizer = actor_optim

        self.critic = critic
        # 🔧 关键修复: critic_target 也共享 feature_extractor
        self.critic_target = self._create_critic_target(critic)
        self.critic_optimizer = critic_optim

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_freq = update_actor_freq
        self.exploration_noise = exploration_noise
        self.device = device
        self.n_step = n_step

        self.total_it = 0
        self.action_range = action_range
        self._action_scale = torch.tensor(
            (action_range[1] - action_range[0]) / 2.0, device=self.device)
        self._action_bias = torch.tensor(
            (action_range[1] + action_range[0]) / 2.0, device=self.device)

        # 参数归一化统计（用于与监督学习保持一致）
        if param_mean is not None and param_std is not None:
            self.param_mean = torch.tensor(param_mean, dtype=torch.float32, device=self.device)
            self.param_std = torch.tensor(param_std, dtype=torch.float32, device=self.device)
            print(f"[TD3] Using parameter normalization:")
            print(f"  Mean: {param_mean}")
            print(f"  Std:  {param_std}")
        else:
            self.param_mean = None
            self.param_std = None
            print(f"[TD3] No parameter normalization (using raw action space)")

    def _create_actor_target(self, actor):
        """
        创建 actor_target，共享 feature_extractor，只复制 FC 层

        原因: 4-bit 量化的 VLM 不能用 copy.deepcopy，会破坏量化状态导致 NaN
        """
        from vlm_net import VLM_DPT_Actor

        print(f"[TD3] Creating actor_target...")
        print(f"  actor.feature_extractor id: {id(actor.feature_extractor)}")

        # 创建新 actor，共享 feature_extractor
        actor_target = VLM_DPT_Actor(
            feature_extractor=actor.feature_extractor,  # 共享！
            action_dim=actor.action_dim,
            algorithm=actor.algorithm
        )

        print(f"  actor_target.feature_extractor id: {id(actor_target.feature_extractor)}")
        print(f"  Same feature_extractor: {actor.feature_extractor is actor_target.feature_extractor}")

        # 复制 FC 层权重
        actor_target.fc.load_state_dict(actor.fc.state_dict())

        # 移动到相同设备
        device = actor.fc.weight.device
        actor_target.fc = actor_target.fc.to(device)

        # target 不需要梯度
        for param in actor_target.fc.parameters():
            param.requires_grad = False

        # 验证 FC 层权重一致
        fc_match = torch.allclose(actor.fc.weight, actor_target.fc.weight)
        print(f"  FC weights match: {fc_match}")
        print(f"  FC device: {actor_target.fc.weight.device}")

        print(f"[TD3] Created actor_target (sharing feature_extractor, FC copied)")
        return actor_target

    def _create_critic_target(self, critic):
        """
        创建 critic_target，共享 feature_extractor，只复制 Q-head

        原因: 4-bit 量化的 VLM 不能用 copy.deepcopy
        """
        from vlm_net import VLM_DPT_Critic

        # 从 action_encoder 推断 action_dim
        action_dim = critic.action_encoder[0].in_features

        # 创建新 critic，共享 feature_extractor
        critic_target = VLM_DPT_Critic(
            feature_extractor=critic.feature_extractor,  # 共享！
            action_dim=action_dim,
            detach_features=critic.detach_features
        )

        # 复制 Q-head 权重 (action_encoder, q1_head, q2_head)
        critic_target.action_encoder.load_state_dict(critic.action_encoder.state_dict())
        critic_target.q1_head.load_state_dict(critic.q1_head.state_dict())
        critic_target.q2_head.load_state_dict(critic.q2_head.state_dict())

        # 移动到相同设备
        device = next(critic.q1_head.parameters()).device
        critic_target.action_encoder = critic_target.action_encoder.to(device)
        critic_target.q1_head = critic_target.q1_head.to(device)
        critic_target.q2_head = critic_target.q2_head.to(device)

        # target 不需要梯度
        for name, param in critic_target.named_parameters():
            # 跳过 feature_extractor 的参数（它们已经有自己的 requires_grad 设置）
            if 'feature_extractor' in name:
                continue
            param.requires_grad = False

        print(f"[TD3] Created critic_target (sharing feature_extractor, Q-heads copied)")
        return critic_target

    def _soft_update_actor(self):
        """Soft update actor_target 的 FC 层（feature_extractor 是共享的，不需要更新）"""
        for param, target_param in zip(self.actor.fc.parameters(), self.actor_target.fc.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def _soft_update_critic(self):
        """Soft update critic_target 的 Q-heads（feature_extractor 是共享的，不需要更新）"""
        # action_encoder
        for param, target_param in zip(self.critic.action_encoder.parameters(),
                                       self.critic_target.action_encoder.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        # q1_head
        for param, target_param in zip(self.critic.q1_head.parameters(),
                                       self.critic_target.q1_head.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        # q2_head
        for param, target_param in zip(self.critic.q2_head.parameters(),
                                       self.critic_target.q2_head.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state):
        """
        选择动作

        Args:
            state: PIL.Image (VLM模式) 或 np.array (传统模式)

        Returns:
            action: np.array
        """
        # 检查state类型，兼容VLM和传统模式
        from PIL import Image
        if isinstance(state, Image.Image):
            # VLM模式: PIL.Image输入
            # actor.forward接受PIL.Image
            action = self.actor([state]).cpu().data.numpy().flatten()  # List[PIL.Image]
        elif isinstance(state, list):
            # VLM模式: List[PIL.Image]
            action = self.actor(state).cpu().data.numpy().flatten()
        else:
            # 传统模式: numpy array
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state).cpu().data.numpy().flatten()

        # 添加探索噪声
        action += np.random.normal(0, self.exploration_noise, size=action.shape)

        # 缩放到动作空间
        action *= self._action_scale.cpu().data.numpy()
        action += self._action_bias.cpu().data.numpy()

        return action

    def train(self, replay_buffer, batch_size=256, verbose=False):
        import time as time_module
        self.total_it += 1

        # 每50次迭代打印详细信息
        verbose = verbose or (self.total_it % 50 == 1)

        if verbose:
            print(f"    [TD3.train] iter={self.total_it}, batch_size={batch_size}")

        t0 = time_module.time()

        # Sample replay buffer ("task" for multi-task learning)
        state, action, next_state, reward, not_done, task, ind = replay_buffer.sample(
            batch_size)

        next_state, reward, not_done, gammas = replay_buffer.n_step_return(self.n_step, ind, self.gamma)

        if verbose:
            print(f"    [TD3.train] 采样完成: {time_module.time()-t0:.2f}s")
            t1 = time_module.time()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # Actor输出[-1, 1]，映射到原始参数空间
            if verbose:
                print(f"    [TD3.train] actor_target forward (VLM)...")
            next_action_raw = self.actor_target(next_state)  # [-1, 1]
            if verbose:
                print(f"    [TD3.train] actor_target完成: {time_module.time()-t1:.2f}s")

            # Debug: check actor output
            if torch.isnan(next_action_raw).any() or torch.isinf(next_action_raw).any():
                print(f"    [TD3.train] WARNING: next_action_raw contains nan/inf!")
                print(f"      next_action_raw: {next_action_raw}")

            next_action_param = next_action_raw * self._action_scale + self._action_bias  # 原始参数

            # 在原始参数空间添加噪声（policy noise）
            noise_param = (
                torch.randn_like(next_action_param) * self.policy_noise * self.param_std
                if self.param_std is not None
                else torch.randn_like(next_action_param) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action_param = next_action_param + noise_param

            # 归一化后给Critic
            if self.param_mean is not None and self.param_std is not None:
                next_action = (next_action_param - self.param_mean) / (self.param_std + 1e-8)
            else:
                next_action = next_action_param

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Debug: check critic_target output BEFORE min
            if torch.isnan(target_Q1).any() or torch.isnan(target_Q2).any():
                print(f"    [TD3.train] WARNING: critic_target output contains nan!")
                print(f"      next_action range: [{next_action.min():.4f}, {next_action.max():.4f}]")
                print(f"      next_action has nan: {torch.isnan(next_action).any()}")

            target_Q = torch.min(target_Q1, target_Q2)

            # 转换reward, not_done, gammas到与target_Q相同的dtype
            reward = reward.to(dtype=target_Q.dtype)
            not_done = not_done.to(dtype=target_Q.dtype)
            gammas = gammas.to(dtype=target_Q.dtype)

            target_Q = reward + not_done * gammas * target_Q

            # Debug: check for nan in target_Q
            if torch.isnan(target_Q).any():
                print(f"    [TD3.train] WARNING: target_Q contains nan!")
                print(f"      reward range: [{reward.min():.4f}, {reward.max():.4f}]")
                print(f"      target_Q1 range: [{target_Q1.min():.4f}, {target_Q1.max():.4f}]")
                print(f"      target_Q2 range: [{target_Q2.min():.4f}, {target_Q2.max():.4f}]")

        # Get current Q estimates
        # 归一化action以提高训练稳定性（与监督学习保持一致）
        if self.param_mean is not None and self.param_std is not None:
            action_normalized = (action - self.param_mean) / (self.param_std + 1e-8)
        else:
            action_normalized = action

        current_Q1, current_Q2 = self.critic(state, action_normalized)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Skip update if loss is nan (prevent corrupting network)
        if torch.isnan(critic_loss):
            print(f"    [TD3.train] WARNING: critic_loss is nan, skipping update!")
            return 0.0, 0.0, None, float('nan')

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Check gradients for nan/inf BEFORE clipping
        grad_has_nan = False
        for name, p in self.critic.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                if not grad_has_nan:
                    print(f"    [TD3.train] WARNING: gradients contain nan/inf BEFORE clipping!")
                    grad_has_nan = True
                print(f"      {name}: grad_max={p.grad.abs().max():.4f}, has_nan={torch.isnan(p.grad).any()}, has_inf={torch.isinf(p.grad).any()}")

        # Skip update if gradients are bad
        if grad_has_nan:
            print(f"    [TD3.train] Skipping critic update due to bad gradients")
            return 0.0, 0.0, None, critic_loss.item()

        # Gradient clipping to prevent explosion (lower threshold for bfloat16 stability)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # Check critic weights for nan BEFORE soft update
        critic_has_nan = any(torch.isnan(p).any() for p in self.critic.parameters())
        if critic_has_nan:
            print(f"    [TD3.train] WARNING: critic weights contain nan after update!")
            # Find which layer has nan
            for name, p in self.critic.named_parameters():
                if torch.isnan(p).any():
                    print(f"      nan in: {name}, shape={p.shape}")

        # Update the target critic (every iteration)
        # 🔧 只更新非共享的参数 (Q-heads)，feature_extractor 是共享的
        self._soft_update_critic()

        actor_loss = None
        # Delayed policy updates
        if self.total_it % self.update_actor_freq == 0:

            # Compute actor loss
            # Actor输出[-1, 1]，需要映射到原始参数空间
            actor_output = self.actor(state)  # [-1, 1]
            predicted_action = actor_output * self._action_scale + self._action_bias  # 原始参数范围

            # 归一化后给Critic（与训练数据保持一致）
            if self.param_mean is not None and self.param_std is not None:
                predicted_action_normalized = (predicted_action - self.param_mean) / (self.param_std + 1e-8)
            else:
                predicted_action_normalized = predicted_action

            actor_loss = -self.critic.Q1(state, predicted_action_normalized).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Gradient clipping to prevent explosion (lower threshold for bfloat16 stability)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            # Update the target actor (delayed)
            # 🔧 只更新非共享的参数 (FC层)，feature_extractor 是共享的
            self._soft_update_actor()

        actor_loss = actor_loss.item() if actor_loss is not None else None
        critic_loss = critic_loss.item()
        return self.grad_norm(self.actor), self.grad_norm(self.critic), actor_loss, critic_loss

    def grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2).item() if p.grad is not None else 0
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def save(self, dir, filename):
        """
        保存Actor模型（VLM+DPT版本）- 统一使用标准格式

        保存到 dir/policy/ 目录：
           - adapter_model.safetensors + adapter_config.json (LoRA)
           - regression_head/pytorch_model.bin (DPT)
           - history_encoder/pytorch_model.bin
           - normalization/param_mean.npy + param_std.npy

        探索噪声单独保存到 dir/policy_noise（TD3 专用）

        策略：
        1. VLM base: 跳过（4-bit量化，从checkpoint重新加载）
        2. LoRA: 使用 PEFT 标准格式（bfloat16）
        3. Regression Head (DPT) + History: 使用 torch.save
        4. 归一化: 使用 numpy
        """
        import numpy as np
        import shutil

        # 创建 policy 目录
        policy_dir = join(dir, filename)
        os.makedirs(policy_dir, exist_ok=True)

        print(f"\n[FTRL Save] 💾 保存模型到统一目录: {policy_dir}")

        # ===== 1. 保存 LoRA =====
        from peft import PeftModel
        is_peft = isinstance(self.actor.feature_extractor.base_model, PeftModel) if hasattr(self.actor.feature_extractor, 'base_model') else False

        if is_peft:
            print(f"[FTRL Save] 1️⃣ 保存 LoRA adapters...")
            lora_params_list = [(n, p.numel(), p.requires_grad) for n, p in self.actor.feature_extractor.base_model.named_parameters() if 'lora' in n.lower()]
            total_lora_params = sum(numel for _, numel, _ in lora_params_list)
            trainable_lora = sum(1 for _, _, requires_grad in lora_params_list if requires_grad)

            print(f"    - LoRA 参数: {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")
            print(f"    - 可训练层: {trainable_lora}/{len(lora_params_list)}")

            # 转换为 bfloat16 节省空间
            params_to_restore = []
            for name, param in self.actor.feature_extractor.base_model.named_parameters():
                if 'lora' in name.lower() and param.dtype != torch.bfloat16:
                    params_to_restore.append((param, param.dtype))
                    param.data = param.data.to(torch.bfloat16)

            # 保存到 policy/ 目录
            self.actor.feature_extractor.base_model.save_pretrained(policy_dir)

            # 恢复 dtype
            for param, original_dtype in params_to_restore:
                param.data = param.data.to(original_dtype)

            # 验证
            adapter_model_path = join(policy_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_model_path):
                file_size = os.path.getsize(adapter_model_path) / 1024 / 1024
                print(f"    ✓ adapter_model.safetensors ({file_size:.2f} MB)")
                print(f"    ✓ adapter_config.json")
        else:
            print(f"[FTRL Save] ⚠️  LoRA 未找到或已 merge，跳过")

        # ===== 2. 保存 Regression Head (DPT) =====
        if hasattr(self.actor.feature_extractor, 'dpt_head'):
            print(f"[FTRL Save] 2️⃣ 保存 Regression Head (DPT)...")
            regression_dir = join(policy_dir, 'regression_head')
            os.makedirs(regression_dir, exist_ok=True)
            dpt_state_dict = self.actor.feature_extractor.dpt_head.state_dict()
            torch.save(dpt_state_dict, join(regression_dir, 'pytorch_model.bin'))
            dpt_params = sum(p.numel() for p in self.actor.feature_extractor.dpt_head.parameters())
            print(f"    ✓ regression_head/pytorch_model.bin ({dpt_params:,} params)")

            # 保存 history_config.json（用于 qwen_server.py 正确创建 DPTHead）
            import json
            use_history = self.actor.feature_extractor.use_history if hasattr(self.actor.feature_extractor, 'use_history') else False
            history_dim = 256  # 默认值，可以从 dpt_head 推断
            if hasattr(self.actor.feature_extractor.dpt_head, 'history_dim'):
                history_dim = self.actor.feature_extractor.dpt_head.history_dim

            history_config = {
                'use_history': use_history,
                'history_dim': history_dim,
                'history_image_size': 224  # 默认值
            }
            with open(join(policy_dir, 'history_config.json'), 'w') as f:
                json.dump(history_config, f, indent=2)
            print(f"    ✓ history_config.json (use_history={use_history})")

        # ===== 3. 保存 History Encoder =====
        if hasattr(self.actor.feature_extractor, 'history_encoder') and self.actor.feature_extractor.history_encoder is not None:
            print(f"[FTRL Save] 3️⃣ 保存 History Encoder...")
            history_dir = join(policy_dir, 'history_encoder')
            os.makedirs(history_dir, exist_ok=True)
            history_state_dict = self.actor.feature_extractor.history_encoder.state_dict()
            torch.save(history_state_dict, join(history_dir, 'pytorch_model.bin'))
            history_params = sum(p.numel() for p in self.actor.feature_extractor.history_encoder.parameters())
            print(f"    ✓ history_encoder/pytorch_model.bin ({history_params:,} params)")

        # ===== 4. 保存归一化参数 =====
        if self.param_mean is not None and self.param_std is not None:
            print(f"[FTRL Save] 4️⃣ 保存归一化参数...")
            norm_dir = join(policy_dir, 'normalization')
            os.makedirs(norm_dir, exist_ok=True)
            np.save(join(norm_dir, 'param_mean.npy'), self.param_mean.cpu().numpy())
            np.save(join(norm_dir, 'param_std.npy'), self.param_std.cpu().numpy())
            print(f"    ✓ normalization/param_mean.npy")
            print(f"    ✓ normalization/param_std.npy")

        # ===== 5. 保存探索噪声（TD3 专用，在 policy/ 里面）=====
        print(f"[FTRL Save] 5️⃣ 保存探索噪声 (TD3)...")
        with open(join(policy_dir, "exploration_noise.pkl"), "wb") as f:
            pickle.dump(self.exploration_noise, f)
        print(f"    ✓ exploration_noise.pkl")

        print(f"\n[FTRL Save] ✅ 完成！模型已保存到: {policy_dir}")
        print(f"[FTRL Save]    可直接用于: qwen_server.py --lora_path {policy_dir}")

    def load(self, dir, filename):
        """加载 RLFT checkpoint（统一从 policy/ 目录加载）"""
        policy_dir = join(dir, filename)

        # 检查 policy/ 目录是否存在
        if not os.path.exists(policy_dir):
            print(f"[Load] 无 RLFT checkpoint: {policy_dir}")
            print(f"[Load] 将使用 Stage 1 初始化")
            return

        print(f"\n[Load] 📂 从统一目录加载: {policy_dir}")

        # ===== 1. 加载 LoRA =====
        adapter_model_path = join(policy_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_model_path):
            print(f"[Load] 1️⃣ 加载 LoRA adapters...")
            from peft import PeftModel

            try:
                is_peft = isinstance(self.actor.feature_extractor.base_model, PeftModel)

                if is_peft:
                    # 卸载旧的 LoRA
                    print(f"    - 检测到已有 PeftModel，卸载旧适配器...")
                    if hasattr(self.actor.feature_extractor.base_model, 'unload'):
                        base_model_clean = self.actor.feature_extractor.base_model.unload()
                    elif hasattr(self.actor.feature_extractor.base_model, 'get_base_model'):
                        base_model_clean = self.actor.feature_extractor.base_model.get_base_model()
                    else:
                        base_model_clean = self.actor.feature_extractor.base_model.model

                    # 加载新 LoRA
                    self.actor.feature_extractor.base_model = PeftModel.from_pretrained(
                        base_model_clean, policy_dir, is_trainable=True
                    )
                else:
                    # 直接加载
                    self.actor.feature_extractor.base_model = PeftModel.from_pretrained(
                        self.actor.feature_extractor.base_model, policy_dir, is_trainable=True
                    )

                file_size = os.path.getsize(adapter_model_path) / 1024 / 1024
                print(f"    ✓ LoRA adapters ({file_size:.2f} MB)")
            except Exception as e:
                print(f"    ⚠️  加载 LoRA 失败: {e}")

        # ===== 2. 加载 Regression Head (DPT) =====
        regression_path = join(policy_dir, 'regression_head', 'pytorch_model.bin')
        if os.path.exists(regression_path):
            print(f"[Load] 2️⃣ 加载 Regression Head (DPT)...")
            dpt_state_dict = torch.load(regression_path, map_location=self.device)
            self.actor.feature_extractor.dpt_head.load_state_dict(dpt_state_dict, strict=False)
            print(f"    ✓ regression_head/pytorch_model.bin")

        # ===== 3. 加载 History Encoder =====
        history_path = join(policy_dir, 'history_encoder', 'pytorch_model.bin')
        if os.path.exists(history_path):
            print(f"[Load] 3️⃣ 加载 History Encoder...")
            history_state_dict = torch.load(history_path, map_location=self.device)
            if hasattr(self.actor.feature_extractor, 'history_encoder') and self.actor.feature_extractor.history_encoder is not None:
                self.actor.feature_extractor.history_encoder.load_state_dict(history_state_dict, strict=False)
                print(f"    ✓ history_encoder/pytorch_model.bin")

        # ===== 4. 加载归一化参数 =====
        param_mean_path = join(policy_dir, 'normalization', 'param_mean.npy')
        param_std_path = join(policy_dir, 'normalization', 'param_std.npy')
        if os.path.exists(param_mean_path) and os.path.exists(param_std_path):
            print(f"[Load] 4️⃣ 加载归一化参数...")
            import numpy as np
            self.param_mean = torch.tensor(np.load(param_mean_path), dtype=torch.float32, device=self.device)
            self.param_std = torch.tensor(np.load(param_std_path), dtype=torch.float32, device=self.device)
            print(f"    ✓ normalization/")

        # ===== 5. 加载探索噪声（TD3 专用）=====
        noise_path = join(policy_dir, "exploration_noise.pkl")
        # 兼容旧格式
        if not os.path.exists(noise_path):
            noise_path = join(dir, filename + "_noise")
        if os.path.exists(noise_path):
            print(f"[Load] 5️⃣ 加载探索噪声...")
            with open(noise_path, "rb") as f:
                self.exploration_noise = pickle.load(f)
            print(f"    ✓ exploration_noise.pkl")

        # ===== 6. 更新 actor_target =====
        self.actor_target = self._create_actor_target(self.actor)

        print(f"[Load] ✅ 加载完成！")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.mean, self.std = 0.0, 1.0

        self.state = np.zeros((max_size, *state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, *state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.task = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done, task):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward # (reward - 0.02478) / 6.499
        self.not_done[self.ptr] = 1. - done
        self.task[self.ptr] = task

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.ptr == 1000:
            rew = self.reward[:1000]
            self.mean, self.std = rew.mean(), rew.std()
            if np.isclose(self.std, 0, 1e-2):
                self.mean, self.std = 0.0, 1.0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.task[ind]).to(self.device),
            ind)

    def n_step_return(self, n_step, ind, gamma):
        reward = []
        not_done = []
        next_state = []
        gammas = []
        for i in ind:
            n = 0
            r = 0
            for _ in range(n_step):
                idx = (i + n) % self.size
                r += (self.reward[idx] - self.mean) / self.std * gamma ** n
                if not self.not_done[idx]:
                    break
                n = n + 1
            next_state.append(self.next_state[idx])
            not_done.append(self.not_done[idx])
            reward.append(r)
            gammas.append([gamma ** (n + 1)])
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        not_done = torch.FloatTensor(np.array(not_done)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).to(self.device)
        gammas = torch.FloatTensor(np.array(gammas)).to(self.device)
        return next_state, reward, not_done, gammas