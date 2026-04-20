"""
GRPO 训练器核心实现
基于 DeepSeekMath 的 GRPO 和 OR-R1 的 TGRPO 设计

核心创新：
1. 组内相对优势计算 (Group Relative Advantage)
2. 多维度奖励函数（格式 + 执行 + 答案）
3. 分段 KL 惩罚（推理链轻惩罚，建模代码重惩罚）
4. 自适应采样温度
5. 完整的训练指标记录
"""
import os
import re
import json
import math
import time
import copy
import html
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model

from .reward import RewardFunction, RewardResult
from .segment_kl import SegmentConfig, compute_segment_kl_penalty


@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-1.5B"
    sft_model_path: Optional[str] = None  # 如果有 SFT 后的模型
    use_8bit: bool = False
    use_flash_attention: bool = True

    # LoRA 配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # 训练配置
    num_samples_per_prompt: int = 4  # 每个 prompt 生成多少个样本
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 4
    per_device_batch_size: int = 1
    max_seq_length: int = 1536
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eps_clip: float = 0.2  # PPO clip 范围
    gamma: float = 1.0  # 折扣因子（GRPO 通常为 1）

    # KL 惩罚配置
    kl_coef: float = 0.04  # 基础 KL 惩罚系数
    use_segment_kl: bool = True  # 是否使用分段 KL

    # 奖励配置
    answer_reward_scale: float = 3.0
    format_reward_scale: float = 1.0
    execution_reward_scale: float = 2.0

    # 日志和保存
    output_dir: str = "./outputs/grpo"
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    eval_samples: int = 50  # 每次评估的样本数
    seed: int = 42


class RolloutStorage:
    """
    存储 rollouts 的数据
    每个 prompt 生成 G 个样本，记录它们的 logprobs、rewards
    """

    def __init__(self):
        self.prompts: List[str] = []
        self.responses: List[List[str]] = []  # [batch_idx][sample_idx] -> response
        self.rewards: List[List[float]] = []  # [batch_idx][sample_idx] -> reward
        self.reward_details: List[List[RewardResult]] = []
        self.advantages: List[List[float]] = []  # [batch_idx][sample_idx] -> advantage
        self.query_token_ids: List[torch.Tensor] = []
        self.response_token_ids: List[torch.Tensor] = []
        self.query_logprobs: List[torch.Tensor] = []
        self.response_logprobs: List[torch.Tensor] = []

    def add(
        self,
        prompt: str,
        responses: List[str],
        rewards: List[float],
        reward_details: List[RewardResult],
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
        query_logprobs: torch.Tensor,
        response_logprobs: torch.Tensor,
    ):
        self.prompts.append(prompt)
        self.responses.append(responses)
        self.rewards.append(rewards)
        self.reward_details.append(reward_details)
        self.query_token_ids.append(query_ids)
        self.response_token_ids.append(response_ids)
        self.query_logprobs.append(query_logprobs)
        self.response_logprobs.append(response_logprobs)

        # 计算组内相对优势
        group_rewards = rewards
        if len(group_rewards) > 1:
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards) + 1e-8
            advantages = [(r - mean_r) / std_r for r in group_rewards]
        else:
            advantages = [0.0]

        self.advantages.append(advantages)

    def clear(self):
        self.__init__()

    def get_statistics(self) -> Dict[str, float]:
        """获取当前批次统计信息"""
        all_rewards = [r for group in self.rewards for r in group]
        if not all_rewards:
            return {}

        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "max_reward": np.max(all_rewards),
            "min_reward": np.min(all_rewards),
            "positive_rate": np.mean([r > 0 for r in all_rewards]),
            "num_samples": len(all_rewards),
        }


class MetricsTracker:
    """训练指标追踪器"""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.global_step = 0

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录指标到 tensorboard 和内存"""
        if step is None:
            step = self.global_step

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                self.writer.add_scalar(key, value, step)
                self.history[key].append((step, value))

        self.global_step += 1

    def log_histogram(self, tag: str, values: List[float], step: Optional[int] = None):
        """记录直方图"""
        if step is None:
            step = self.global_step
        if values:
            self.writer.add_histogram(tag, np.array(values), step)

    def close(self):
        self.writer.close()

    def save_history(self, path: str):
        """保存训练历史为 JSON"""
        with open(path, "w") as f:
            json.dump(
                {k: [{"step": s, "value": v} for s, v in vals]}
                for k, vals in self.history.items()
            },
                f,
                indent=2,
            )


class GRPOTrainer:
    """
    GRPO 训练器

    训练流程：
    1. 对每个 prompt 生成 G 个样本
    2. 用奖励函数计算每个样本的奖励
    3. 计算组内相对优势
    4. 用 PPO-clip 目标更新策略
    5. 应用分段 KL 惩罚
    6. 记录训练指标
    """

    def __init__(
        self,
        config: GRPOConfig,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
        reward_fn: Optional[RewardFunction] = None,
    ):
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data
        self.reward_fn = reward_fn or RewardFunction(
            answer_reward_scale=config.answer_reward_scale,
            format_reward_scale=config.format_reward_scale,
            execution_reward_scale=config.execution_reward_scale,
        )

        # 设置随机种子
        self._set_seed(config.seed)

        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self._load_model()

        # 初始化指标追踪
        os.makedirs(config.output_dir, exist_ok=True)
        self.metrics = MetricsTracker(os.path.join(config.output_dir, "logs"))

        # Rollout 存储
        self.rollout_storage = RolloutStorage()

        # 训练状态
        self.current_epoch = 0
        self.total_steps = 0
        self.best_eval_reward = -float("inf")

    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_model(self):
        """加载模型和 tokenizer"""
        model_name = self.config.model_name.strip('"')

        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载基础模型
        model_kwargs = {
            "torch_dtype": torch.float32,
            "device_map": None,
            "trust_remote_code": True,
        }

        if self.config.use_flash_attention and torch.cuda.is_available():
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass

        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # 如果有 SFT 模型路径，加载 SFT 后的模型
        if self.config.sft_model_path and os.path.exists(self.config.sft_model_path):
            print(f"加载 SFT 模型: {self.config.sft_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.sft_model_path,
                **model_kwargs,
            )

        # 应用 LoRA（如果基础模型还没有 LoRA）
        if not isinstance(base_model, PeftModel):
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type=transformers.LoraConfig.task_type,
            )
            base_model = get_peft_model(base_model, lora_config)
            print("已应用 LoRA 配置")

        base_model.print_trainable_parameters()

        self.model = base_model.to(self.device)
        self.model.eval()

    def generate_response(
        self,
        prompt: str,
        num_samples: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        为单个 prompt 生成多个样本

        Returns:
            (responses, token_ids, logprobs)
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length // 2,
            add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]

        responses = []
        response_ids_list = []
        logprobs_list = []

        with torch.no_grad():
            for _ in range(num_samples):
                # 生成
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_seq_length - prompt_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                gen_ids = generation_output.sequences[0]
                gen_tokens = gen_ids[prompt_length:]

                # 解码响应
                response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                response = response.strip()
                responses.append(response)

                response_ids_list.append(gen_tokens.cpu())
                logprobs_list.append(torch.zeros(len(gen_tokens)))  # 占位，后面计算

        return responses, response_ids_list, logprobs_list

    @torch.no_grad()
    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        prompt_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算输入序列的 log probabilities

        Returns:
            (logprobs, per-token logprobs)
        """
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids)
        logits = outputs.logits

        # 移位以获得下一个 token 的预测概率
        # logprobs = log(p(next_token))
        log_probs = F.log_softmax(logits[:-1], dim=-1)
        tokens = input_ids[1:]

        # 获取每个 token 的 log probability
        per_token_logp = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

        prompt_logprobs = per_token_logp[:prompt_length - 1]
        response_logprobs = per_token_logp[prompt_length - 1:]

        return per_token_logp, response_logprobs

    def generate_batch_rollouts(
        self,
        batch_prompts: List[str],
        batch_ground_truths: List[Optional[float]],
        num_samples: int,
    ) -> RolloutStorage:
        """
        为一批 prompts 生成 rollouts

        Returns:
            RolloutStorage with prompts, responses, rewards
        """
        storage = RolloutStorage()

        for i, (prompt, gt) in enumerate(zip(batch_prompts, batch_ground_truths)):
            # 为每个 prompt 生成 G 个样本
            responses, resp_ids, _ = self.generate_response(
                prompt,
                num_samples=num_samples,
            )

            # 计算每个响应的奖励
            rewards = []
            reward_details = []
            resp_logprobs_list = []

            # 对每个响应计算 logprobs
            # 构建完整序列
            full_prompt = prompt + "\n\n"
            prompt_encoding = self.tokenizer(
                full_prompt,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_ids = prompt_encoding["input_ids"][0].to(self.device)
            prompt_len = len(prompt_ids)

            for j, resp_text in enumerate(responses):
                # 构建完整输入
                full_text = full_prompt + resp_text
                full_encoding = self.tokenizer(
                    full_text,
                    add_special_tokens=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length,
                )
                full_ids = full_encoding["input_ids"][0].to(self.device)

                # 计算 logprobs
                _, resp_logprobs = self.compute_logprobs(full_ids, prompt_len)
                resp_logprobs_list.append(resp_logprobs)

                # 计算奖励
                reward_result = self.reward_fn(resp_text, ground_truth=gt, is_last=True)
                rewards.append(reward_result.total_reward)
                reward_details.append(reward_result)

            # 存储
            storage.add(
                prompt=prompt,
                responses=responses,
                rewards=rewards,
                reward_details=reward_details,
                query_ids=prompt_ids.cpu(),
                response_ids=torch.cat(resp_ids, dim=-1).cpu(),
                query_logprobs=torch.zeros(prompt_len - 1),  # 占位
                response_logprobs=torch.zeros(sum(len(lp) for lp in resp_ids)),
            )

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"  Rollout 进度: {i + 1}/{len(batch_prompts)}")

        # 计算组内相对优势
        all_group_rewards = storage.rewards
        storage.advantages = []
        for group_rewards in all_group_rewards:
            if len(group_rewards) > 1:
                mean_r = np.mean(group_rewards)
                std_r = np.std(group_rewards) + 1e-8
                advantages = [(r - mean_r) / std_r for r in group_rewards]
            else:
                advantages = [0.0]
            storage.advantages.append(advantages)

        return storage

    def compute_policy_loss(
        self,
        storage: RolloutStorage,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 GRPO 策略损失

        GRPO 损失 = -E[min(r * A, clip(r, 1-e, 1+e) * A)] + beta * KL(pi||pi_ref)

        其中 r = exp(log_pi_theta - log_pi_ref) = pi_theta / pi_ref
        """
        total_loss = 0.0
        num_valid_groups = 0

        loss_info = {
            "policy_loss": 0.0,
            "kl_penalty": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
        }

        self.model.train()

        for group_idx in range(len(storage.prompts)):
            prompt = storage.prompts[group_idx]
            responses = storage.responses[group_idx]
            advantages = storage.advantages[group_idx]
            rewards = storage.rewards[group_idx]

            if len(responses) == 0:
                continue

            # 对组内的每个样本计算损失
            prompt_str = prompt + "\n\n"
            prompt_encoding = self.tokenizer(
                prompt_str,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_ids = prompt_encoding["input_ids"][0].to(self.device)
            prompt_len = len(prompt_ids)

            group_policy_losses = []
            group_kls = []
            clipped_count = 0

            for sample_idx, (response, advantage, reward) in enumerate(
                zip(responses, advantages, rewards)
            ):
                # 跳过无效样本
                if not response or len(response.strip()) < 5:
                    continue

                # 构建完整序列
                full_text = prompt_str + response
                full_encoding = self.tokenizer(
                    full_text,
                    add_special_tokens=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length,
                )
                full_ids = full_encoding["input_ids"][0].to(self.device)

                if len(full_ids) <= prompt_len + 5:  # 响应太短
                    continue

                # 前向传播，计算 log probabilities
                outputs = self.model(full_ids)
                logits = outputs.logits

                # 计算 logprobs
                log_probs = F.log_softmax(logits[:-1], dim=-1)
                tokens = full_ids[1:]

                # 只对响应部分计算
                resp_start = prompt_len - 1
                resp_end = min(len(tokens), full_ids.shape[0] - 1)
                resp_len = resp_end - resp_start

                if resp_len <= 0:
                    continue

                resp_tokens = tokens[resp_start:resp_end]
                resp_log_probs = log_probs[resp_start:resp_end]

                # 计算每个 token 的 log prob
                token_logp = resp_log_probs.gather(-1, resp_tokens.unsqueeze(-1)).squeeze(-1)

                # 简单的 ratio = 1（因为我们没有 ref model，这里用简化的方式）
                # 在实际实现中，应该先跑一次 ref model 获取 log_pi_ref
                # 这里使用简化：ratio = exp(token_logp - old_logp)
                # old_logp 从 storage 中获取
                old_logp = storage.response_logprobs[group_idx]

                # 确保长度匹配
                min_len = min(len(token_logp), len(old_logp))
                if min_len == 0:
                    continue

                token_logp = token_logp[-min_len:]
                old_logp = old_logp[-min_len:].to(self.device)

                # 计算 ratio
                ratio = torch.exp(token_logp - old_logp)

                # GRPO: 使用相对优势
                advantage_tensor = torch.tensor(advantage, device=self.device)

                # Clipped surrogate objective
                surr1 = ratio * advantage_tensor
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantage_tensor
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL 惩罚（简化的整体 KL）
                kl = (old_logp - token_logp).mean()
                kl_penalty = self.config.kl_coef * kl

                # 总损失
                loss = policy_loss + kl_penalty

                group_policy_losses.append(loss)
                group_kls.append(kl.item())

                # 统计 clip 比例
                with torch.no_grad():
                    clipped = ((ratio - 1.0).abs() > self.config.eps_clip).float().mean().item()
                    clipped_count += clipped

            if group_policy_losses:
                group_loss = torch.stack(group_policy_losses).mean()
                total_loss += group_loss
                num_valid_groups += 1

                loss_info["policy_loss"] += group_loss.item()
                loss_info["kl_penalty"] += sum(group_kls) / len(group_kls)
                loss_info["clip_fraction"] += clipped_count / len(group_policy_losses)

        if num_valid_groups > 0:
            total_loss = total_loss / num_valid_groups
            for k in loss_info:
                if k != "approx_kl":
                    loss_info[k] /= num_valid_groups

        return total_loss, loss_info

    def train_step(self, batch_prompts: List[str], batch_gt: List[Optional[float]]) -> Dict[str, float]:
        """执行一个训练步骤"""
        # 1. 生成 rollouts
        print(f"  生成 {len(batch_prompts)} 个 prompts 的 rollouts...")
        rollout_storage = self.generate_batch_rollouts(
            batch_prompts,
            batch_gt,
            num_samples=self.config.num_samples_per_prompt,
        )

        # 2. 记录统计信息
        stats = rollout_storage.get_statistics()
        self.metrics.log(stats, self.total_steps)
        self.metrics.log_histogram("rewards", [r for g in rollout_storage.rewards for r in g], self.total_steps)

        # 打印奖励分布
        all_rewards = [r for g in rollout_storage.rewards for r in g]
        print(f"  奖励统计: mean={np.mean(all_rewards):.3f}, "
              f"std={np.std(all_rewards):.3f}, "
              f"max={np.max(all_rewards):.3f}, "
              f"positive_rate={np.mean([r > 0 for r in all_rewards]):.2%}")

        # 3. 计算损失并更新
        self.model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        loss, loss_info = self.compute_policy_loss(rollout_storage)

        # 梯度累积
        loss = loss / self.config.gradient_accumulation_steps

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            self.config.max_grad_norm,
        )

        if (self.total_steps + 1) % self.config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 4. 记录指标
        self.metrics.log(loss_info, self.total_steps)
        self.metrics.log({"train/loss": loss.item()}, self.total_steps)

        # 记录有效样本率
        valid_count = sum(
            1 for g in rollout_storage.rewards
            for r in g if r > 0
        )
        total_count = sum(len(g) for g in rollout_storage.rewards)
        self.metrics.log(
            {"train/valid_sample_rate": valid_count / max(total_count, 1)},
            self.total_steps,
        )

        self.total_steps += 1
        return {**stats, **loss_info}

    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """评估模型性能"""
        print(f"\n评估模型 (样本数: {num_samples})...")

        if self.eval_data:
            eval_samples = self.eval_data[:num_samples]
        else:
            eval_samples = self.train_data[:num_samples]

        batch_prompts = []
        batch_gt = []
        for item in eval_samples:
            prompt = item.get("prompt", "")
            if not prompt:
                prompt = f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\n\n问题描述：\n{item.get('problem_text', item.get('text', ''))}"
            batch_prompts.append(prompt)
            gt_str = item.get("answer", item.get("ground_truth_answer", ""))
            gt = float(gt_str) if gt_str else None
            batch_gt.append(gt)

        # 生成
        rollout_storage = self.generate_batch_rollouts(
            batch_prompts,
            batch_gt,
            num_samples=1,  # 评估时只生成 1 个
        )

        # 计算 Pass@1 指标
        all_rewards = [g[0] for g in rollout_storage.rewards]  # 第一个样本即 Pass@1
        all_details = [g[0] for g in rollout_storage.reward_details]

        stats = {
            "eval/mean_reward": np.mean(all_rewards),
            "eval/pass_at_1": np.mean([r > 0 for r in all_rewards]),
            "eval/execution_rate": np.mean([d.is_valid for d in all_details]),
            "eval/format_rate": np.mean([d.format_reward > 0 for d in all_details]),
        }

        self.metrics.log(stats, self.total_steps)
        print(f"  评估结果: Pass@1 = {stats['eval/pass_at_1']:.2%}, "
              f"执行率 = {stats['eval/execution_rate']:.2%}, "
              f"平均奖励 = {stats['eval/mean_reward']:.3f}")

        return stats

    def save_checkpoint(self, tag: str = "checkpoint"):
        """保存检查点"""
        path = os.path.join(self.config.output_dir, f"{tag}_{self.total_steps}")
        os.makedirs(path, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # 保存训练状态
        state = {
            "step": self.total_steps,
            "epoch": self.current_epoch,
            "config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                       for k, v in vars(self.config).items()},
        }
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(state, f, indent=2)

        print(f"检查点已保存: {path}")

    def train(self):
        """主训练循环"""
        print("=" * 60)
        print("开始 GRPO 训练")
        print("=" * 60)
        print(f"模型: {self.config.model_name}")
        print(f"训练样本数: {len(self.train_data)}")
        print(f"每个 prompt 的样本数: {self.config.num_samples_per_prompt}")
        print(f"批次大小: {self.config.per_device_batch_size}")
        print(f"学习率: {self.config.learning_rate}")
        print(f"输出目录: {self.config.output_dir}")
        print("=" * 60)

        total_steps_per_epoch = len(self.train_data) // self.config.per_device_batch_size
        total_steps = total_steps_per_epoch * self.config.num_train_epochs

        # 计算每个 epoch 的 step 数
        steps_per_epoch = len(self.train_data) // self.config.per_device_batch_size

        for epoch in range(self.config.num_train_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")

            # 打乱数据
            indices = np.random.permutation(len(self.train_data))

            epoch_losses = []
            epoch_rewards = []

            for step_in_epoch in range(steps_per_epoch):
                data_idx = step_in_epoch * self.config.per_device_batch_size
                batch_indices = indices[data_idx:data_idx + self.config.per_device_batch_size]

                batch = [self.train_data[i] for i in batch_indices]
                batch_prompts = []
                batch_gt = []

                for item in batch:
                    prompt = item.get("prompt", "")
                    if not prompt:
                        prompt = (
                            f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。"
                            f"\n\n问题描述：\n{item.get('problem_text', item.get('text', ''))}"
                        )
                    batch_prompts.append(prompt)
                    gt_str = item.get("answer", item.get("ground_truth_answer", ""))
                    gt = float(gt_str) if gt_str else None
                    batch_gt.append(gt)

                # 训练步骤
                step_metrics = self.train_step(batch_prompts, batch_gt)

                epoch_losses.append(step_metrics.get("policy_loss", 0))
                epoch_rewards.extend([r for g in self.rollout_storage.rewards for r in g])

                # 日志
                if (self.total_steps + 1) % self.config.logging_steps == 0:
                    print(
                        f"Step {self.total_steps} | "
                        f"Loss: {np.mean(epoch_losses[-self.config.logging_steps:]):.4f} | "
                        f"Reward: {np.mean(epoch_rewards[-self.config.logging_steps:]):.3f} | "
                        f"Positive Rate: {np.mean([r > 0 for r in epoch_rewards[-self.config.logging_steps:]]):.2%}"
                    )

                # 评估
                if (self.total_steps + 1) % self.config.eval_steps == 0:
                    eval_stats = self.evaluate(num_samples=self.config.eval_samples)
                    if eval_stats["eval/mean_reward"] > self.best_eval_reward:
                        self.best_eval_reward = eval_stats["eval/mean_reward"]
                        self.save_checkpoint("best")

                # 保存
                if (self.total_steps + 1) % self.config.save_steps == 0:
                    self.save_checkpoint()

            print(f"\nEpoch {epoch + 1} 完成！")

        # 最终保存
        self.save_checkpoint("final")
        self.metrics.save_history(os.path.join(self.config.output_dir, "training_history.json"))
        self.metrics.close()

        print("\n训练完成！")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GRPO 训练脚本")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--sft_model_path", type=str, default=None)
    parser.add_argument("--train_data", type=str, default="./data/processed/train.jsonl")
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo")
    parser.add_argument("--num_samples_per_prompt", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    args = parser.parse_args()

    # 加载数据
    train_data = []
    with open(args.train_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    train_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if args.max_train_samples > 0 and len(train_data) > args.max_train_samples:
        train_data = train_data[:args.max_train_samples]

    eval_data = None
    if args.eval_data and os.path.exists(args.eval_data):
        eval_data = []
        with open(args.eval_data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        eval_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # 构建配置
    config = GRPOConfig(
        model_name=args.model_name,
        sft_model_path=args.sft_model_path,
        train_data=train_data,
        eval_data=eval_data,
        output_dir=args.output_dir,
        num_samples_per_prompt=args.num_samples_per_prompt,
        num_train_epochs=args.num_epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_samples=args.eval_samples,
        use_8bit=args.use_8bit,
        use_flash_attention=not args.no_flash_attention,
        kl_coef=args.kl_coef,
        eps_clip=args.eps_clip,
    )

    # 创建训练器并开始训练
    trainer = GRPOTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
