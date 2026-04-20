"""
主训练入口脚本
一键运行 SFT + GRPO 完整流程
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)

    # Python 版本
    print(f"Python: {sys.version}")

    # PyTorch 版本
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("错误: PyTorch 未安装")

    # Transformers
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("警告: Transformers 未安装")

    # 其他关键依赖
    for pkg in ["peft", "datasets", "pyomo", "tensorboard"]:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"{pkg}: {version}")
        except ImportError:
            print(f"警告: {pkg} 未安装")

    print("=" * 60)


def prepare_data(
    raw_data_dir: str = "./data/raw",
    processed_data_dir: str = "./data/processed",
    max_samples: int = 5000,
):
    """准备训练数据"""
    print("\n准备训练数据...")

    # 检查数据是否存在
    processed_path = Path(processed_data_dir)
    if processed_path.exists() and list(processed_path.glob("*.jsonl")):
        print(f"数据已存在: {processed_data_dir}")
        return

    # 创建目录
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # 下载数据
    print("下载数据集...")
    download_script = Path(__file__).parent / "data" / "download_data.py"
    if download_script.exists():
        subprocess.run(
            [sys.executable, str(download_script), "--output_dir", raw_data_dir],
            check=False,
        )

    # 预处理
    preprocess_script = Path(__file__).parent / "data" / "preprocess.py"
    if preprocess_script.exists():
        subprocess.run(
            [
                sys.executable, str(preprocess_script),
                "--input_dir", raw_data_dir,
                "--output_dir", processed_data_dir,
                "--max_samples", str(max_samples),
            ],
            check=False,
        )

    print("数据准备完成！")


def run_sft(
    model_name: str,
    train_data: str,
    eval_data: str = None,
    output_dir: str = "./outputs/sft",
    num_epochs: int = 3,
    max_samples: int = 5000,
    lora_r: int = 16,
    batch_size: int = 2,
):
    """运行 SFT 训练"""
    print("\n" + "=" * 60)
    print("SFT 训练")
    print("=" * 60)

    sft_script = Path(__file__).parent / "sft" / "train_sft.py"
    cmd = [
        sys.executable, str(sft_script),
        "--model_name", model_name,
        "--train_data", train_data,
        "--output_dir", output_dir,
        "--num_epochs", str(num_epochs),
        "--max_samples", str(max_samples),
        "--lora_r", str(lora_r),
        "--per_device_batch_size", str(batch_size),
    ]

    if eval_data:
        cmd.extend(["--eval_data", eval_data])

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("SFT 训练失败！")
        return None

    sft_model_path = Path(output_dir) / "final"
    if sft_model_path.exists():
        return str(sft_model_path)

    return None


def run_grpo(
    model_name: str,
    sft_model_path: str,
    train_data: str,
    eval_data: str = None,
    output_dir: str = "./outputs/grpo",
    num_epochs: int = 2,
    max_samples: int = 5000,
    num_samples_per_prompt: int = 4,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    kl_coef: float = 0.04,
    eps_clip: float = 0.2,
):
    """运行 GRPO 训练"""
    print("\n" + "=" * 60)
    print("GRPO 训练")
    print("=" * 60)

    grpo_script = Path(__file__).parent / "grpo" / "train_grpo.py"
    cmd = [
        sys.executable, str(grpo_script),
        "--model_name", model_name,
        "--sft_model_path", sft_model_path,
        "--train_data", train_data,
        "--output_dir", output_dir,
        "--num_epochs", str(num_epochs),
        "--max_train_samples", str(max_samples),
        "--num_samples_per_prompt", str(num_samples_per_prompt),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--kl_coef", str(kl_coef),
        "--eps_clip", str(eps_clip),
        "--logging_steps", "10",
        "--eval_steps", "200",
        "--save_steps", "500",
    ]

    if eval_data:
        cmd.extend(["--eval_data", eval_data])

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("GRPO 训练失败！")
        return None

    return output_dir


def run_evaluation(
    model_path: str,
    output_dir: str = "./eval_results",
    max_samples: int = 100,
):
    """运行评测"""
    print("\n" + "=" * 60)
    print("模型评测")
    print("=" * 60)

    eval_script = Path(__file__).parent / "eval" / "evaluate.py"
    cmd = [
        sys.executable, str(eval_script),
        "--model_path", model_path,
        "--benchmarks", "nl4opt", "optmath",
        "--max_samples", str(max_samples),
        "--output_dir", output_dir,
    ]

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="LLM4OR 完整训练流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 阶段选择
    parser.add_argument("--skip_sft", action="store_true",
                        help="跳过 SFT 阶段")
    parser.add_argument("--skip_grpo", action="store_true",
                        help="跳过 GRPO 阶段")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过评测阶段")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="跳过数据准备")

    # 模型配置
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-1.5B",
                        help="基础模型")
    parser.add_argument("--resume_from", type=str,
                        default=None,
                        help="从检查点恢复（SFT 或 GRPO 目录）")

    # 训练配置
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="最大训练样本数")
    parser.add_argument("--sft_epochs", type=int, default=3,
                        help="SFT 训练轮数")
    parser.add_argument("--grpo_epochs", type=int, default=2,
                        help="GRPO 训练轮数")
    parser.add_argument("--num_samples_per_prompt", type=int, default=4,
                        help="GRPO 每 prompt 样本数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--kl_coef", type=float, default=0.04,
                        help="KL 惩罚系数")
    parser.add_argument("--eps_clip", type=float, default=0.2,
                        help="PPO clip 范围")

    # 目录配置
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                        help="数据目录")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")

    # 评测配置
    parser.add_argument("--eval_max_samples", type=int, default=100,
                        help="评测样本数")

    args = parser.parse_args()

    # 环境检查
    check_environment()

    # 准备数据
    if not args.skip_data_prep:
        prepare_data(
            raw_data_dir="./data/raw",
            processed_data_dir=args.data_dir,
            max_samples=args.max_samples,
        )

    train_data = os.path.join(args.data_dir, "train.jsonl")
    eval_data = os.path.join(args.data_dir, "eval.jsonl")

    sft_model_path = None
    grpo_output_dir = None

    # SFT 阶段
    if not args.skip_sft:
        if args.resume_from and os.path.exists(args.resume_from):
            # 如果 resume_from 是 SFT 目录
            sft_model_path = args.resume_from
        else:
            sft_model_path = run_sft(
                model_name=args.model_name,
                train_data=train_data,
                eval_data=eval_data if os.path.exists(eval_data) else None,
                output_dir=os.path.join(args.output_dir, "sft"),
                num_epochs=args.sft_epochs,
                max_samples=args.max_samples,
            )

        if sft_model_path is None:
            print("SFT 阶段失败！")
            return

    # GRPO 阶段
    if not args.skip_grpo:
        if sft_model_path is None and args.resume_from:
            sft_model_path = args.resume_from

        if sft_model_path is None:
            print("没有 SFT 模型，跳过 GRPO 阶段")
        else:
            grpo_output_dir = run_grpo(
                model_name=args.model_name,
                sft_model_path=sft_model_path,
                train_data=train_data,
                eval_data=eval_data if os.path.exists(eval_data) else None,
                output_dir=os.path.join(args.output_dir, "grpo"),
                num_epochs=args.grpo_epochs,
                max_samples=args.max_samples,
                num_samples_per_prompt=args.num_samples_per_prompt,
                learning_rate=args.learning_rate,
                kl_coef=args.kl_coef,
                eps_clip=args.eps_clip,
            )

    # 评测阶段
    if not args.skip_eval:
        model_to_eval = None

        if grpo_output_dir:
            model_to_eval = grpo_output_dir
        elif sft_model_path:
            model_to_eval = sft_model_path
        elif args.resume_from:
            model_to_eval = args.resume_from

        if model_to_eval and os.path.exists(model_to_eval):
            run_evaluation(
                model_path=model_to_eval,
                output_dir=os.path.join(args.output_dir, "eval"),
                max_samples=args.eval_max_samples,
            )
        else:
            print("没有可评测的模型")

    print("\n" + "=" * 60)
    print("全部流程完成！")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    if sft_model_path:
        print(f"SFT 模型: {sft_model_path}")
    if grpo_output_dir:
        print(f"GRPO 模型: {grpo_output_dir}")


if __name__ == "__main__":
    main()
