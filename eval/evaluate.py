"""
评测模块：评估训练后的模型在各个基准上的性能
支持: NL4OPT, OptMATH-Bench, IndustryOR
"""
import os
import re
import json
import torch
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grpo.reward import RewardFunction, RewardResult


@dataclass
class EvalResult:
    """单个样本的评测结果"""
    sample_id: str
    problem_text: str
    model_response: str
    ground_truth: Optional[float]
    reward_result: RewardResult
    generation_time: float
    is_correct: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """基准测试的总体结果"""
    benchmark_name: str
    num_samples: int
    num_correct: int
    num_executed: int
    num_format_valid: int
    accuracy: float
    execution_rate: float
    format_rate: float
    mean_reward: float
    std_reward: float
    avg_generation_time: float
    results: List[EvalResult]
    per_category_stats: Optional[Dict[str, Dict]] = None


def load_benchmark_data(
    benchmark_name: str,
    data_path: Optional[str] = None,
) -> List[Dict]:
    """加载基准测试数据集"""
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 内置数据加载（尝试从 HuggingFace）
    built_in_benchmarks = {
        "nl4opt": "AI4Math/NL4OPT",
        "optmath": "AI4Math/OptMATH",
    }

    if benchmark_name in built_in_benchmarks:
        try:
            from datasets import load_dataset
            ds = load_dataset(built_in_benchmarks[benchmark_name], trust_remote_code=True)
            if "test" in ds:
                return list(ds["test"])
            elif "validation" in ds:
                return list(ds["validation"])
            else:
                for split in ds:
                    return list(ds[split])
        except Exception as e:
            print(f"无法加载 {benchmark_name}: {e}")

    return []


def build_prompt(problem_text: str, include_thought: bool = True) -> str:
    """构建评测 prompt"""
    if include_thought:
        return (
            f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。"
            f"请先分析问题，再建立数学模型，最后给出完整的 Python/Pyomo 代码。"
            f"\n\n问题描述：\n{problem_text}\n\n"
            f"请按以下格式回答：\n"
            f"## 问题分析\n"
            f"## 数学模型\n"
            f"## Pyomo 实现\n"
            f"```python\n# 你的代码\n```"
        )
    else:
        return (
            f"请为以下运筹优化问题编写 Pyomo 代码求解。"
            f"\n\n问题描述：\n{problem_text}"
        )


class BenchmarkEvaluator:
    """基准测试评估器"""

    def __init__(
        self,
        model_path: str,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        reward_fn: Optional[RewardFunction] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_samples: int = 1,
    ):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.reward_fn = reward_fn or RewardFunction()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_samples = num_samples

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )

        # 如果是 PeftModel，不需要特殊处理
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        do_sample: bool = True,
    ) -> List[Tuple[str, List[float]]]:
        """
        为一个 prompt 生成多个样本

        Returns:
            List of (response, token_ids)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        results = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else 1.0,
                    top_p=self.top_p if do_sample else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                gen_ids = output[0][prompt_len:]
                response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                response = response.strip()
                results.append(response)

        return results

    def evaluate_single(
        self,
        problem_text: str,
        ground_truth: Optional[float] = None,
    ) -> EvalResult:
        """评估单个问题"""
        prompt = build_prompt(problem_text)

        start_time = time.time()
        responses = self.generate(prompt, do_sample=(self.num_samples > 1))
        generation_time = time.time() - start_time

        # 取最佳响应（根据奖励）
        best_response = responses[0]
        best_reward = float("-inf")
        best_reward_result = None

        for resp in responses:
            reward_result = self.reward_fn(resp, ground_truth=ground_truth, is_last=True)
            if reward_result.total_reward > best_reward:
                best_reward = reward_result.total_reward
                best_response = resp
                best_reward_result = reward_result

        # 判断是否正确
        is_correct = (
            best_reward_result is not None
            and best_reward_result.answer_reward > 0
        )

        return EvalResult(
            sample_id="",
            problem_text=problem_text,
            model_response=best_response,
            ground_truth=ground_truth,
            reward_result=best_reward_result or RewardResult(
                total_reward=best_reward,
                format_reward=0,
                execution_reward=0,
            ),
            generation_time=generation_time,
            is_correct=is_correct,
        )

    def evaluate_benchmark(
        self,
        benchmark_name: str,
        data_path: Optional[str] = None,
        num_samples: int = 0,
        max_eval_samples: int = 100,
        output_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        在基准数据集上评估模型

        Args:
            benchmark_name: 基准名称
            data_path: 数据文件路径
            num_samples: 每个问题的生成样本数（0=使用默认）
            max_eval_samples: 最大评估样本数
            output_path: 结果保存路径
        """
        print(f"\n{'=' * 60}")
        print(f"评估基准: {benchmark_name}")
        print(f"{'=' * 60}")

        # 加载数据
        data = load_benchmark_data(benchmark_name, data_path)
        if not data:
            print(f"警告: 无法加载 {benchmark_name} 数据")
            return None

        # 限制评估样本数
        if 0 < max_eval_samples < len(data):
            data = data[:max_eval_samples]

        print(f"评估样本数: {len(data)}")

        # 更新采样数
        if num_samples > 0:
            self.num_samples = num_samples

        results = []
        correct_count = 0
        executed_count = 0
        format_count = 0
        total_rewards = []

        for i, item in enumerate(data):
            problem_text = (
                item.get("problem", "") or
                item.get("text", "") or
                item.get("question", "") or
                item.get("source_sequence", "")
            )
            ground_truth_str = (
                item.get("answer", "") or
                item.get("optimal_value", "") or
                item.get("target_value", "")
            )
            ground_truth = None
            if ground_truth_str:
                try:
                    ground_truth = float(ground_truth_str)
                except (ValueError, TypeError):
                    pass

            sample_id = item.get("id", f"{benchmark_name}_{i}")

            print(f"[{i + 1}/{len(data)}] 评估: {sample_id[:50]}...", end=" ")

            try:
                eval_result = self.evaluate_single(problem_text, ground_truth)
                eval_result.sample_id = sample_id

                results.append(eval_result)
                total_rewards.append(eval_result.reward_result.total_reward)

                if eval_result.reward_result.is_valid:
                    executed_count += 1
                if eval_result.reward_result.format_reward > 0:
                    format_count += 1
                if eval_result.is_correct:
                    correct_count += 1

                status = "✓" if eval_result.is_correct else "✗"
                print(f"{status} Reward={eval_result.reward_result.total_reward:.2f} "
                      f"(F={eval_result.reward_result.format_reward:.1f}, "
                      f"E={eval_result.reward_result.execution_reward:.1f}, "
                      f"A={eval_result.reward_result.answer_reward:.1f})")

            except Exception as e:
                print(f"错误: {e}")
                results.append(EvalResult(
                    sample_id=sample_id,
                    problem_text=problem_text,
                    model_response="",
                    ground_truth=ground_truth,
                    reward_result=RewardResult(total_reward=0, format_reward=0, execution_reward=0),
                    generation_time=0,
                    is_correct=False,
                    error=str(e),
                ))

            # 每 10 个样本打印进度
            if (i + 1) % 10 == 0:
                current_acc = correct_count / (i + 1)
                current_exec = executed_count / (i + 1)
                print(f"\n  累计: Acc={current_acc:.2%}, Exec={current_exec:.2%}, "
                      f"MeanR={np.mean(total_rewards):.3f}")

        # 计算总体统计
        num_samples = len(results)
        benchmark_result = BenchmarkResult(
            benchmark_name=benchmark_name,
            num_samples=num_samples,
            num_correct=correct_count,
            num_executed=executed_count,
            num_format_valid=format_count,
            accuracy=correct_count / num_samples if num_samples > 0 else 0,
            execution_rate=executed_count / num_samples if num_samples > 0 else 0,
            format_rate=format_count / num_samples if num_samples > 0 else 0,
            mean_reward=np.mean(total_rewards) if total_rewards else 0,
            std_reward=np.std(total_rewards) if total_rewards else 0,
            avg_generation_time=np.mean([r.generation_time for r in results]) if results else 0,
            results=results,
        )

        # 打印总结
        print(f"\n{'=' * 60}")
        print(f"评估结果: {benchmark_name}")
        print(f"{'=' * 60}")
        print(f"  样本数:         {num_samples}")
        print(f"  正确数:         {correct_count}")
        print(f"  Pass@1 准确率:  {benchmark_result.accuracy:.2%}")
        print(f"  执行率:         {benchmark_result.execution_rate:.2%}")
        print(f"  格式正确率:     {benchmark_result.format_rate:.2%}")
        print(f"  平均奖励:       {benchmark_result.mean_reward:.3f} ± {benchmark_result.std_reward:.3f}")
        print(f"  平均生成时间:   {benchmark_result.avg_generation_time:.2f}s")
        print(f"{'=' * 60}")

        # 保存结果
        if output_path:
            self.save_results(benchmark_result, output_path)

        return benchmark_result

    def save_results(self, result: BenchmarkResult, output_path: str):
        """保存评测结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "benchmark": result.benchmark_name,
            "num_samples": result.num_samples,
            "accuracy": result.accuracy,
            "execution_rate": result.execution_rate,
            "format_rate": result.format_rate,
            "mean_reward": result.mean_reward,
            "std_reward": result.std_reward,
            "avg_generation_time": result.avg_generation_time,
        }

        with open(output_path / f"{result.benchmark_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 保存详细结果
        detailed = []
        for r in result.results:
            detailed.append({
                "id": r.sample_id,
                "problem_text": r.problem_text[:200],
                "model_response": r.model_response,
                "ground_truth": r.ground_truth,
                "is_correct": r.is_correct,
                "total_reward": r.reward_result.total_reward,
                "format_reward": r.reward_result.format_reward,
                "execution_reward": r.reward_result.execution_reward,
                "answer_reward": r.reward_result.answer_reward,
                "is_valid": r.reward_result.is_valid,
                "generation_time": r.generation_time,
                "error": r.error,
            })

        with open(output_path / f"{result.benchmark_name}_detailed.json", "w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_path}")


def evaluate_model(
    model_path: str,
    benchmarks: List[str],
    max_samples: int = 100,
    num_gen_samples: int = 1,
    output_dir: str = "./eval_results",
):
    """评估模型的综合函数"""
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建评估器
    evaluator = BenchmarkEvaluator(
        model_path=model_path,
        tokenizer=tokenizer,
        num_samples=num_gen_samples,
    )

    all_results = {}

    for benchmark in benchmarks:
        result = evaluator.evaluate_benchmark(
            benchmark_name=benchmark,
            max_eval_samples=max_samples,
            output_path=output_dir,
        )
        if result:
            all_results[benchmark] = result

    # 打印汇总
    print(f"\n{'=' * 60}")
    print("所有基准测试汇总")
    print(f"{'=' * 60}")
    print(f"{'Benchmark':<20} {'Accuracy':<12} {'Exec Rate':<12} {'Mean Reward':<12}")
    print("-" * 60)
    for name, result in all_results.items():
        print(f"{name:<20} {result.accuracy:>10.2%} {result.execution_rate:>10.2%} "
              f"{result.mean_reward:>10.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="模型评测脚本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径或 HuggingFace ID")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["nl4opt"],
                        choices=["nl4opt", "optmath", "industry_or"],
                        help="要评测的基准")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="最大评估样本数")
    parser.add_argument("--num_gen_samples", type=int, default=1,
                        help="每个问题生成的样本数（Pass@K 中的 K）")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="结果输出目录")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        num_gen_samples=args.num_gen_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
