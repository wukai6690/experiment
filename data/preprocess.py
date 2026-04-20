"""
数据预处理：将原始数据集转换为训练所需格式
支持: OptMATH, NL4OPT, IndustryOR, StepOPT
"""
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ORSample:
    """运筹优化样本的数据结构"""
    id: str
    problem_text: str          # 自然语言问题描述
    mathematical_model: str     # 数学模型（可选）
    code_solution: str          # Pyomo 代码解决方案
    answer: Optional[float]     # 最优值（用于验证）
    problem_type: str          # "LP", "MILP", "NLP"
    difficulty: str            # "easy", "medium", "hard"
    source: str                # 数据集来源


def extract_code_blocks(text: str) -> List[str]:
    """从文本中提取所有代码块"""
    pattern = r"```(?:python|pyomo)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)


def clean_code(code: str) -> str:
    """清理代码：移除空行、标准化缩进"""
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.rstrip()
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)


def validate_pyomo_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    验证 Pyomo 代码的基本格式
    返回: (是否有效, 错误信息)
    """
    if not code or len(code.strip()) < 10:
        return False, "代码太短或为空"

    required_patterns = [
        (r"import\s+pyomo", "缺少 'import pyomo'"),
        (r"from\s+pyomo\.(environ|core)", "缺少 Pyomo 环境导入"),
        (r"(Model|model)\s*=", "缺少模型定义"),
        (r"(\.solve|\.SolverFactory)", "缺少求解器调用"),
    ]

    for pattern, error_msg in required_patterns:
        if not re.search(pattern, code, re.IGNORECASE):
            return False, error_msg

    return True, None


def extract_mathematical_model(text: str) -> Optional[str]:
    """从文本中提取数学建模部分"""
    patterns = [
        r"##\s*(?:数学模型|Mathematical Model|优化模型)\s*\n(.*?)(?=```|$)",
        r"###\s*(?:数学建模|模型建立)\s*\n(.*?)(?=```|$)",
        r"\$数学模型\$\n(.*?)\$\$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 尝试提取 LaTeX 格式的数学模型
    latex_pattern = r"\$\$(.*?)\$\$"
    matches = re.findall(latex_pattern, text, re.DOTALL)
    if matches:
        return "\n".join(matches)

    return None


def extract_answer_from_code(code: str) -> Optional[float]:
    """从代码或输出中提取最优值"""
    # 匹配 "results['Objective']" 或 "model.objective()" 等模式
    patterns = [
        r"results\[.*?\]\s*[=:]\s*([-+]?\d*\.?\d+)",
        r"Value\(model\.\w+?\)\s*[=:]\s*([-+]?\d*\.?\d+)",
        r"Optimal\s+(?:value|objective)[:\s]+([-+]?\d*\.?\d+)",
        r"=\s*([-+]?\d*\.?\d+)\s*(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, code, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def detect_problem_type(text: str, code: str) -> str:
    """检测问题类型"""
    combined = text.lower() + " " + code.lower()

    # 检查整数变量特征
    if any(kw in combined for kw in ["integer", "binary", "Binary", "Integer",
                                       "within=Integers", "var.*Binary", "var.*Integer"]):
        return "MILP"

    # 检查非线性特征
    if any(kw in combined for kw in ["nonlinear", "quadratic", "sin(", "cos(",
                                       "exp(", "log(", "power(", "^", "**"]):
        return "NLP"

    return "LP"


def format_sft_example(sample: ORSample, add_thought: bool = True) -> Dict[str, str]:
    """
    将 ORSample 格式化为 SFT 训练样本

    Args:
        sample: ORSample 实例
        add_thought: 是否在响应中添加推理步骤

    Returns:
        {"prompt": ..., "response": ...}
    """
    if add_thought and sample.mathematical_model:
        response_parts = [
            "## 问题分析\n",
            _analyze_problem(sample.problem_text),
            "\n## 数学模型\n",
            sample.mathematical_model,
            "\n## Pyomo 实现\n",
            f"```python\n{sample.code_solution}\n```"
        ]
        response = "".join(response_parts)
    else:
        response = f"```python\n{sample.code_solution}\n```"

    prompt = f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\n\n问题描述：\n{sample.problem_text}"

    return {
        "prompt": prompt,
        "response": response,
        "id": sample.id,
        "problem_type": sample.problem_type,
        "difficulty": sample.difficulty,
        "ground_truth_answer": str(sample.answer) if sample.answer else "",
    }


def _analyze_problem(text: str) -> str:
    """生成问题分析（用于 Chain-of-Thought）"""
    # 简单的问题类型识别
    analysis_parts = []

    text_lower = text.lower()
    if "最大" in text or "max" in text_lower or "maximize" in text_lower:
        analysis_parts.append("这是一个**最大化**问题。")
    elif "最小" in text or "min" in text_lower or "minimize" in text_lower:
        analysis_parts.append("这是一个**最小化**问题。")

    if "运输" in text or "transport" in text_lower:
        analysis_parts.append("这是运输问题，适合用线性规划建模。")
    if "分配" in text or "assignment" in text_lower:
        analysis_parts.append("这是分配问题，可以用二分图匹配建模。")
    if "背包" in text or "knapsack" in text_lower:
        analysis_parts.append("这是背包问题，需要整数变量约束。")
    if "调度" in text or "scheduling" in text_lower:
        analysis_parts.append("这是调度问题，需要时间窗口约束。")

    if not analysis_parts:
        analysis_parts.append("这是一个标准的运筹优化问题，需要建立目标函数和约束条件。")

    return "".join(analysis_parts)


def preprocess_optmath(data: List[Dict]) -> List[ORSample]:
    """预处理 OptMATH 数据集"""
    samples = []
    for i, item in enumerate(data):
        problem_text = item.get("problem", item.get("text", item.get("question", "")))
        if not problem_text:
            continue

        code = item.get("code", item.get("solution", ""))
        math_model = item.get("model", item.get("mathematical_model", ""))
        answer_str = item.get("answer", item.get("optimal_value", ""))

        answer = None
        if answer_str:
            try:
                answer = float(answer_str)
            except (ValueError, TypeError):
                answer = extract_answer_from_code(code)

        sample = ORSample(
            id=f"optmath_{i}",
            problem_text=problem_text,
            mathematical_model=math_model,
            code_solution=code,
            answer=answer,
            problem_type=detect_problem_type(problem_text, code),
            difficulty=item.get("difficulty", "medium"),
            source="OptMATH",
        )
        samples.append(sample)

    return samples


def preprocess_nl4opt(data: List[Dict]) -> List[ORSample]:
    """预处理 NL4OPT 数据集"""
    samples = []
    for i, item in enumerate(data):
        nl_text = item.get("source_sequence", item.get("problem", ""))
        if not nl_text:
            continue

        code = item.get("target_sequence", item.get("code", ""))
        math_model = item.get("model", "")
        answer_str = item.get("answer", item.get("optimal_value", ""))

        answer = None
        if answer_str:
            try:
                answer = float(answer_str)
            except (ValueError, TypeError):
                answer = extract_answer_from_code(code)

        sample = ORSample(
            id=f"nl4opt_{i}",
            problem_text=nl_text,
            mathematical_model=math_model,
            code_solution=code,
            answer=answer,
            problem_type="MILP",
            difficulty=item.get("difficulty", "medium"),
            source="NL4OPT",
        )
        samples.append(sample)

    return samples


def preprocess_industry_or(data: List[Dict]) -> List[ORSample]:
    """预处理 IndustryOR / ORLM 数据集"""
    samples = []
    for i, item in enumerate(data):
        problem_text = (
            item.get("prompt", "") or
            item.get("question", "") or
            item.get("problem_text", "") or
            item.get("text", "")
        )
        if not problem_text:
            continue

        # IndustryOR 格式可能不同，尝试多个字段
        code = (
            item.get("response", "") or
            item.get("code", "") or
            item.get("solution", "") or
            item.get("answer", "")
        )
        math_model = item.get("math_model", item.get("mathematical_model", ""))
        answer_str = item.get("optimal_value", item.get("answer", ""))

        answer = None
        if answer_str:
            try:
                answer = float(answer_str)
            except (ValueError, TypeError):
                answer = extract_answer_from_code(code)

        sample = ORSample(
            id=f"industry_or_{i}",
            problem_text=problem_text,
            mathematical_model=math_model,
            code_solution=code,
            answer=answer,
            problem_type=detect_problem_type(problem_text, code),
            difficulty=item.get("difficulty", "medium"),
            source="IndustryOR",
        )
        samples.append(sample)

    return samples


def split_train_eval(
    samples: List[ORSample],
    eval_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[ORSample], List[ORSample]]:
    """将数据划分为训练集和验证集"""
    import random
    random.seed(seed)

    indices = list(range(len(samples)))
    random.shuffle(indices)

    eval_size = int(len(samples) * eval_ratio)
    eval_indices = set(indices[:eval_size])
    train_indices = set(indices[eval_size:])

    train_samples = [samples[i] for i in train_indices]
    eval_samples = [samples[i] for i in eval_indices]

    return train_samples, eval_samples


def save_processed_data(
    samples: List[ORSample],
    output_path: str,
    format: str = "jsonl"
):
    """保存处理后的数据"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                item = {
                    "id": sample.id,
                    "problem_text": sample.problem_text,
                    "mathematical_model": sample.mathematical_model,
                    "code_solution": sample.code_solution,
                    "answer": sample.answer,
                    "problem_type": sample.problem_type,
                    "difficulty": sample.difficulty,
                    "source": sample.source,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elif format == "chatml":
        # 保存为 ChatML 格式，适合直接用于 SFT
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                prompt = f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\n\n问题描述：\n{sample.problem_text}"
                response = f"## 数学模型\n{sample.mathematical_model}\n\n## Pyomo 实现\n```python\n{sample.code_solution}\n```"

                chat_item = {
                    "id": sample.id,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "answer": sample.answer,
                }
                f.write(json.dumps(chat_item, ensure_ascii=False) + "\n")

    print(f"已保存 {len(samples)} 条数据到: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="预处理运筹优化数据集")
    parser.add_argument("--input_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--format", type=str, default="chatml", choices=["jsonl", "chatml"])
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最大采样数量，0 表示使用全部数据")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # 扫描输入目录
    for jsonl_file in input_dir.rglob("*.jsonl"):
        print(f"处理文件: {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            data = json.load(f) if jsonl_file.stat().st_size < 1024 * 1024 * 100 else None
            if data is None:
                # 大文件，逐行读取
                data = []
                with open(jsonl_file, "r", encoding="utf-8") as f2:
                    for line in f2:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

        # 根据文件名判断数据集类型
        fname = jsonl_file.stem.lower()
        if "nl4opt" in fname:
            samples = preprocess_nl4opt(data)
        elif "optmath" in fname or "opt" in fname:
            samples = preprocess_optmath(data)
        elif "industry" in fname or "orlm" in fname:
            samples = preprocess_industry_or(data)
        else:
            samples = preprocess_industry_or(data)

        all_samples.extend(samples)

    # 采样
    if args.max_samples > 0 and len(all_samples) > args.max_samples:
        import random
        random.seed(42)
        all_samples = random.sample(all_samples, args.max_samples)

    print(f"共处理 {len(all_samples)} 条数据")

    # 划分数据集
    train_samples, eval_samples = split_train_eval(all_samples, args.eval_ratio)

    # 保存
    save_processed_data(train_samples, output_dir / "train.jsonl", args.format)
    save_processed_data(eval_samples, output_dir / "eval.jsonl", args.format)

    print(f"训练集: {len(train_samples)} 条")
    print(f"验证集: {len(eval_samples)} 条")


if __name__ == "__main__":
    main()
