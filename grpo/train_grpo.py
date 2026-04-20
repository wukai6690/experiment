"""
GRPO 训练脚本入口
更简洁的命令行接口
"""
import os
import sys
import json
import torch
import argparse
from pathlib import Path

from grpo_trainer import GRPOTrainer, GRPOConfig
from reward import RewardFunction


def load_jsonl_data(file_path: str, max_samples: int = 0) -> list:
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


def main():
    parser = argparse.ArgumentParser(
        description="GRPO 训练脚本 - LLM4OR 运筹优化建模",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 SFT 模型进行 GRPO 训练
  python train_grpo.py --sft_model ./outputs/sft/final --train_data ./data/train.jsonl

  # 使用预训练模型从头训练
  python train_grpo.py --model_name Qwen/Qwen2.5-1.5B --train_data ./data/train.jsonl

  # 小规模快速验证
  python train_grpo.py --model_name Qwen/Qwen2.5-0.5B --train_data ./data/train.jsonl \\
    --max_train_samples 500 --num_epochs 1 --num_samples_per_prompt 2
        """
    )

    # 模型配置
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基础模型名称或路径")
    parser.add_argument("--sft_model_path", type=str, default=None,
                        help="SFT 模型路径（优先使用）")
    parser.add_argument("--use_8bit", action="store_true",
                        help="使用 8-bit 量化加载")
    parser.add_argument("--no_flash_attention", action="store_true",
                        help="禁用 Flash Attention")

    # 数据配置
    parser.add_argument("--train_data", type=str, default="./data/processed/train.jsonl",
                        help="训练数据路径")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="验证数据路径")
    parser.add_argument("--max_train_samples", type=int, default=5000,
                        help="最大训练样本数（0=全部）")
    parser.add_argument("--max_eval_samples", type=int, default=100,
                        help="最大验证样本数")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--num_samples_per_prompt", type=int, default=4,
                        help="每个 prompt 生成的样本数（G）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--max_seq_length", type=int, default=1536,
                        help="最大序列长度")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪")
    parser.add_argument("--eps_clip", type=float, default=0.2,
                        help="PPO clip 范围")
    parser.add_argument("--kl_coef", type=float, default=0.04,
                        help="KL 惩罚系数")

    # LoRA 配置
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="评估步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")
    parser.add_argument("--eval_samples", type=int, default=50,
                        help="每次评估的样本数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从检查点恢复")

    args = parser.parse_args()

    # ===== 加载数据 =====
    print("=" * 60)
    print("GRPO 训练 - LLM4OR 运筹优化建模")
    print("=" * 60)

    train_data = load_jsonl_data(args.train_data, args.max_train_samples)
    eval_data = load_jsonl_data(args.eval_data, args.max_eval_samples) if args.eval_data else None

    if not train_data:
        print("错误: 训练数据为空！")
        sys.exit(1)

    # 标准化数据格式
    for item in train_data:
        if "prompt" not in item:
            problem_text = item.get("problem_text", item.get("text", ""))
            item["prompt"] = (
                f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。"
                f"\n\n问题描述：\n{problem_text}"
            )
        if "response" not in item and "code_solution" in item:
            item["response"] = f"```python\n{item['code_solution']}\n```"

    for item in (eval_data or []):
        if "prompt" not in item:
            problem_text = item.get("problem_text", item.get("text", ""))
            item["prompt"] = (
                f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。"
                f"\n\n问题描述：\n{problem_text}"
            )

    # ===== 构建配置 =====
    config = GRPOConfig(
        model_name=args.model_name,
        sft_model_path=args.sft_model_path,
        use_8bit=args.use_8bit,
        use_flash_attention=not args.no_flash_attention,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_samples_per_prompt=args.num_samples_per_prompt,
        num_train_epochs=args.num_epochs,
        per_device_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        kl_coef=args.kl_coef,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )

    # ===== 创建奖励函数 =====
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

    # ===== 创建训练器 =====
    trainer = GRPOTrainer(
        config=config,
        train_data=train_data,
        eval_data=eval_data,
        reward_fn=reward_fn,
    )

    # ===== 开始训练 =====
    trainer.train()

    print("\n训练完成！")
    print(f"模型保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
