"""
稳健版 GRPO 训练器
使用 trl 库的 GRPO 实现，避免手动实现 PPO/GRPO 的复杂性

安装: pip install trl
"""
import os
import sys
import json
import torch
import argparse
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForCausalLMWithValueHead,
)
from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# 导入自定义奖励函数
sys.path.insert(0, str(Path(__file__).parent))
from grpo.reward import RewardFunction, RewardResult


def load_jsonl_data(file_path: str, max_samples: int = 0) -> List[Dict]:
    """加载 JSONL 数据"""
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return []

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if max_samples > 0 and len(data) > max_samples:
        import random
        random.seed(42)
        data = random.sample(data, max_samples)

    print(f"加载了 {len(data)} 条数据 from {file_path}")
    return data


def prepare_dataset(
    data: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int = 1536,
) -> "datasets.Dataset":
    """将数据转换为 HuggingFace Dataset 格式"""
    from datasets import Dataset

    prompts = []
    responses = []
    answers = []

    for item in data:
        # 构建 prompt
        problem_text = item.get("problem_text", item.get("text", ""))
        if "prompt" in item:
            prompt = item["prompt"]
        else:
            prompt = (
                f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。"
                f"\n\n问题描述：\n{problem_text}"
            )

        # 获取响应
        if "response" in item:
            response = item["response"]
        elif "code_solution" in item:
            response = f"```python\n{item['code_solution']}\n```"
        else:
            response = ""

        prompts.append(prompt)
        responses.append(response)
        answers.append(item.get("answer", item.get("ground_truth_answer", "")))

    # 创建 dataset
    def format_sample(prompt, response, answer):
        # 使用 ChatML 格式
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "answer": answer}

    formatted_data = [format_sample(p, r, a) for p, r, a in zip(prompts, responses, answers)]

    dataset = Dataset.from_list(formatted_data)

    return dataset


class RewardTracker:
    """追踪奖励变化"""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.history = []
        self.step = 0

    def log(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.writer.add_scalar(key, value, self.step)
        self.history.append((self.step, metrics))
        self.step += 1

    def close(self):
        self.writer.close()


def compute_reward(
    prompts: List[str],
    responses: List[str],
    answers: List[str],
    reward_fn: RewardFunction,
) -> List[float]:
    """计算批次样本的奖励"""
    rewards = []

    for prompt, response, answer_str in zip(prompts, responses, answers):
        # 提取 ground truth
        ground_truth = None
        if answer_str:
            try:
                ground_truth = float(answer_str)
            except (ValueError, TypeError):
                pass

        # 计算奖励
        reward_result = reward_fn(response, ground_truth=ground_truth, is_last=True)
        rewards.append(reward_result.total_reward)

    return rewards


def run_grpo_with_trl(
    model_name: str,
    sft_model_path: Optional[str],
    train_data: List[Dict],
    eval_data: Optional[List[Dict]],
    output_dir: str,
    max_samples: int = 5000,
    num_epochs: int = 2,
    learning_rate: float = 1e-5,
    num_generations: int = 4,
    max_length: int = 1536,
    kl_coef: float = 0.04,
    logging_steps: int = 10,
    save_steps: int = 200,
    seed: int = 42,
):
    """
    使用 trl 库运行 GRPO 训练

    Args:
        model_name: 基础模型
        sft_model_path: SFT 模型路径（可选）
        train_data: 训练数据
        eval_data: 验证数据
        output_dir: 输出目录
        max_samples: 最大样本数
        num_epochs: 训练轮数
        learning_rate: 学习率
        num_generations: 每个样本的生成数
        max_length: 最大序列长度
        kl_coef: KL 惩罚系数
        logging_steps: 日志步数
        save_steps: 保存步数
        seed: 随机种子
    """
    print("=" * 60)
    print("稳健版 GRPO 训练 (trl)")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"训练样本: {len(train_data)}")
    print(f"输出目录: {output_dir}")
    print(f"每样本生成数: {num_generations}")
    print("=" * 60)

    # 设置随机种子
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 加载 tokenizer
    model_name = model_name.strip('"')
    print(f"加载 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print(f"加载模型...")
    model_path = sft_model_path if sft_model_path and os.path.exists(sft_model_path) else model_name

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # 应用 LoRA
    if not hasattr(model, 'peft_type') or model.peft_type is None:
        print("应用 LoRA 配置...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 准备数据集
    print("准备数据集...")
    train_dataset = prepare_dataset(train_data, tokenizer, max_length)

    eval_dataset = None
    if eval_data:
        eval_dataset = prepare_dataset(eval_data, tokenizer, max_length)

    # 创建奖励函数
    reward_fn = RewardFunction(
        enable_format_reward=True,
        enable_execution_reward=True,
        enable_answer_reward=True,
        enable_process_reward=True,
        format_reward_scale=1.0,
        execution_reward_scale=2.0,
        answer_reward_scale=3.0,
        answer_tolerance=1e-4,
    )

    # 创建奖励追踪器
    os.makedirs(output_dir, exist_ok=True)
    tracker = RewardTracker(os.path.join(output_dir, "logs"))

    # 包装奖励函数
    def reward_function(prompts, responses, **kwargs):
        # 提取 answers
        answers = []
        for i in range(len(responses)):
            idx = kwargs.get("batch_idx", [i])[i] if isinstance(kwargs.get("batch_idx"), list) else i
            if idx < len(train_data):
                answers.append(train_data[idx].get("answer", ""))
            else:
                answers.append("")

        rewards = compute_reward(prompts, responses, answers, reward_fn)

        # 打印奖励分布
        if random.random() < 0.1:  # 10% 概率打印
            print(f"  奖励分布: mean={np.mean(rewards):.3f}, "
                  f"std={np.std(rewards):.3f}, "
                  f"max={np.max(rewards):.3f}")

        # 记录到 tensorboard
        tracker.log({
            "train/mean_reward": np.mean(rewards),
            "train/std_reward": np.std(rewards),
            "train/max_reward": np.max(rewards),
            "train/min_reward": np.min(rewards),
            "train/positive_rate": np.mean([r > 0 for r in rewards]),
        })

        return rewards

    # GRPO 配置
    grpo_config = TRLGRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        max_completion_length=max_length // 2,
        num_generations=num_generations,
        beta=kl_coef,  # KL 惩罚系数
        loss_type="grpo",
        # 生成配置
        generation_config={
            "temperature": 0.8,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        },
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=seed,
    )

    # 创建训练器
    print("创建 GRPO 训练器...")
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        reward_function=reward_function,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    print("开始 GRPO 训练...")
    trainer.train()

    # 保存最终模型
    print(f"保存模型到: {output_dir}/final")
    trainer.save_model(os.path.join(output_dir, "final"))

    tracker.close()
    print("训练完成！")

    return os.path.join(output_dir, "final")


def main():
    parser = argparse.ArgumentParser(description="稳健版 GRPO 训练 (trl)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--sft_model_path", type=str, default=None)
    parser.add_argument("--train_data", type=str, default="./data/processed/train.jsonl")
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_trl")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载数据
    train_data = load_jsonl_data(args.train_data, args.max_samples)
    eval_data = None
    if args.eval_data and os.path.exists(args.eval_data):
        eval_data = load_jsonl_data(args.eval_data)

    if not train_data:
        print("错误: 训练数据为空！")
        sys.exit(1)

    # 运行训练
    run_grpo_with_trl(
        model_name=args.model_name,
        sft_model_path=args.sft_model_path,
        train_data=train_data,
        eval_data=eval_data,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_length=args.max_length,
        kl_coef=args.kl_coef,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
