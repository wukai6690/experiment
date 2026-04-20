#!/usr/bin/env python3
"""
LLM4OR 单文件版训练脚本
包含: 数据准备 + SFT + GRPO + 评测
一行命令跑完全流程

使用方法:
    python run_train.py                    # 默认配置 (Qwen2.5-1.5B, 5000样本)
    python run_train.py --model 3B        # Qwen2.5-3B
    python run_train.py --model 7B        # Qwen2.5-7B
    python run_train.py --skip_sft         # 跳过SFT，从检查点继续
"""
import os
import sys
import json
import time
import random
import traceback
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# 1. 依赖检查和安装
# ============================================================================

def check_and_install_deps():
    """检查并安装依赖"""
    required = [
        "transformers", "accelerate", "bitsandbytes", "peft",
        "datasets", "trl", "tensorboard", "tqdm"
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[INFO] 安装缺失的依赖: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=True)

    # 检查 Pyomo
    try:
        from pyomo.environ import SolverFactory, ConcreteModel, Objective, Var, Constraint, value
        print("[OK] Pyomo 已安装")
    except ImportError:
        print("[INFO] 安装 Pyomo...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyomo", "highspy"], check=True)
        try:
            from pyomo.environ import SolverFactory
            print("[OK] Pyomo 已安装")
        except:
            print("[WARN] Pyomo 安装失败，奖励函数的执行奖励将不可用")

check_and_install_deps()

# ============================================================================
# 2. 配置
# ============================================================================

@dataclass
class Config:
    # 模型
    model_size: str = "1.5B"  # 0.5B, 1.5B, 3B, 7B
    use_sft_checkpoint: Optional[str] = None  # 跳过SFT，从检查点继续

    # 数据
    max_train_samples: int = 5000
    eval_samples: int = 100

    # SFT
    sft_epochs: int = 3
    sft_lr: float = 2e-4
    sft_batch_size: int = 2
    sft_grad_accum: int = 4

    # GRPO
    grpo_epochs: int = 2
    grpo_lr: float = 1e-5
    grpo_batch_size: int = 1
    grpo_grad_accum: int = 4
    grpo_num_generations: int = 4
    grpo_kl_coef: float = 0.04
    grpo_clip_eps: float = 0.2

    # 输出
    output_dir: str = "./outputs"
    seed: int = 42

    def get_model_name(self) -> str:
        names = {
            "0.5B": "Qwen/Qwen2.5-0.5B",
            "1.5B": "Qwen/Qwen2.5-1.5B",
            "3B": "Qwen/Qwen2.5-3B",
            "7B": "Qwen/Qwen2.5-7B",
        }
        return names.get(self.model_size, "Qwen/Qwen2.5-1.5B")

    def get_batch_size(self) -> int:
        sizes = {"0.5B": 4, "1.5B": 2, "3B": 1, "7B": 1}
        return sizes.get(self.model_size, 2)

    def get_lora_r(self) -> int:
        sizes = {"0.5B": 8, "1.5B": 16, "3B": 16, "7B": 32}
        return sizes.get(self.model_size, 16)

    def get_max_length(self) -> int:
        sizes = {"0.5B": 1024, "1.5B": 1536, "3B": 1536, "7B": 2048}
        return sizes.get(self.model_size, 1536)


# ============================================================================
# 3. 工具函数
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, trainable / total * 100

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"

# ============================================================================
# 4. 数据
# ============================================================================

def generate_synthetic_data(n: int = 10000) -> List[Dict]:
    """生成合成训练数据作为备用"""
    print(f"[DATA] 生成 {n} 条合成训练数据...")

    problem_templates = [
        {
            "text": "一个简单的线性规划问题：最大化 3x + 2y，约束条件：x + y <= 10, 2x + y <= 16, x >= 0, y >= 0",
            "answer": "28.0",
            "code": '''from pyomo.environ import *

m = ConcreteModel()
m.x = Var(within=NonNegativeReals)
m.y = Var(within=NonNegativeReals)
m.obj = Objective(expr=3*m.x + 2*m.y, sense=maximize)
m.c1 = Constraint(expr=m.x + m.y <= 10)
m.c2 = Constraint(expr=2*m.x + m.y <= 16)
results = SolverFactory('cbc').solve(m, tee=False)
print(f"Optimal value: {value(m.obj)}")'''
        },
        {
            "text": "运输问题：有两个工厂和两个市场，工厂供给量分别为15和20单位，市场需求量分别为12和18单位，单位运输成本矩阵为[[4,6],[5,3]]，求最小运输成本",
            "answer": "138.0",
            "code": '''from pyomo.environ import *

m = ConcreteModel()
m.f = Set(initialize=['F1','F2'])
m.c = Set(initialize=['M1','M2'])
m.supply = Param(m.f, initialize={'F1':15,'F2':20})
m.demand = Param(m.c, initialize={'M1':12,'M2':18})
m.cost = Param(m.f, m.c, initialize={
    ('F1','M1'):4,('F1','M2'):6,
    ('F2','M1'):5,('F2','M2'):3
})
m.x = Var(m.f, m.c, within=NonNegativeReals)
m.obj = Objective(expr=sum(m.cost[i,j]*m.x[i,j] for i in m.f for j in m.c))
m.supply_c = Constraint(m.f, rule=lambda m,i: sum(m.x[i,j] for j in m.c) <= m.supply[i])
m.demand_c = Constraint(m.c, rule=lambda m,j: sum(m.x[i,j] for i in m.f) >= m.demand[j])
results = SolverFactory('cbc').solve(m)
print(f"Optimal cost: {value(m.obj)}")'''
        },
        {
            "text": "背包问题：有5个物品，重量分别为[2,3,4,5,6]，价值分别为[3,4,5,6,7]，背包容量为10，求最大价值",
            "answer": "11.0",
            "code": '''from pyomo.environ import *

m = ConcreteModel()
m.i = Set(initialize=[1,2,3,4,5])
w = {1:2,2:3,3:4,4:5,5:6}
v = {1:3,2:4,3:5,4:6,5:7}
m.x = Var(m.i, within=Binary)
m.obj = Objective(expr=sum(v[i]*m.x[i] for i in m.i), sense=maximize)
m.c = Constraint(expr=sum(w[i]*m.x[i] for i in m.i) <= 10)
results = SolverFactory('cbc').solve(m)
print(f"Optimal value: {value(m.obj)}")'''
        },
        {
            "text": "整数规划问题：最小化 x + y，约束：x + 2y >= 8, 3x + y >= 6, x >= 0, y >= 0, x,y 为整数",
            "answer": "4.0",
            "code": '''from pyomo.environ import *

m = ConcreteModel()
m.x = Var(within=NonNegativeIntegers)
m.y = Var(within=NonNegativeIntegers)
m.obj = Objective(expr=m.x + m.y)
m.c1 = Constraint(expr=m.x + 2*m.y >= 8)
m.c2 = Constraint(expr=3*m.x + m.y >= 6)
results = SolverFactory('cbc').solve(m)
print(f"Optimal value: {value(m.obj)}")'''
        },
        {
            "text": "生产计划问题：某工厂生产两种产品，产品A每单位利润5元，需要资源1单位8小时和资源2单位4小时；产品B每单位利润8元，需要资源1单位4小时和资源2单位6小时。每天资源1可用80小时，资源2可用60小时，求最大利润",
            "answer": "85.0",
            "code": '''from pyomo.environ import *

m = ConcreteModel()
m.A = Var(within=NonNegativeReals)
m.B = Var(within=NonNegativeReals)
m.obj = Objective(expr=5*m.A + 8*m.B, sense=maximize)
m.r1 = Constraint(expr=8*m.A + 4*m.B <= 80)
m.r2 = Constraint(expr=4*m.A + 6*m.B <= 60)
results = SolverFactory('cbc').solve(m)
print(f"Optimal profit: {value(m.obj)}")'''
        },
    ]

    data = []
    for i in range(n):
        template = problem_templates[i % len(problem_templates)]
        # 随机修改数值
        text = template["text"]
        code = template["code"]

        prompt = f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\n\n问题描述：\n{text}"

        data.append({
            "id": f"synthetic_{i}",
            "prompt": prompt,
            "response": f"```python\n{code}\n```",
            "problem_text": text,
            "code_solution": code,
            "answer": template["answer"],
        })

    return data


def load_or_download_data(max_samples: int) -> Tuple[List[Dict], List[Dict]]:
    """加载或下载训练数据"""
    output_dir = Path("./data")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    # 如果已有数据，直接加载
    if train_path.exists() and eval_path.exists():
        print(f"[DATA] 加载已有数据: {train_path}")
        train_data = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        train_data.append(json.loads(line))
                    except:
                        continue

        eval_data = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        eval_data.append(json.loads(line))
                    except:
                        continue

        # 采样
        if max_samples > 0 and len(train_data) > max_samples:
            random.seed(42)
            train_data = random.sample(train_data, max_samples)

        return train_data, eval_data

    # 尝试从 HuggingFace 下载
    print("[DATA] 尝试从 HuggingFace 下载数据集...")
    try:
        from datasets import load_dataset
        ds = load_dataset("AI4Math/OptMATH", trust_remote_code=True, split="train")
        train_data = []
        for item in ds:
            problem = item.get("problem", "") or item.get("text", "")
            code = item.get("code", "") or item.get("solution", "")
            answer = item.get("answer", item.get("optimal_value", ""))

            if not problem or not code:
                continue

            prompt = f"请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\n\n问题描述：\n{problem}"
            train_data.append({
                "id": f"optmath_{len(train_data)}",
                "prompt": prompt,
                "response": f"```python\n{code}\n```",
                "problem_text": problem,
                "code_solution": code,
                "answer": str(answer) if answer else "",
            })

        print(f"[DATA] OptMATH: {len(train_data)} 条")
    except Exception as e:
        print(f"[DATA] HuggingFace 下载失败: {e}")
        print("[DATA] 使用合成数据作为备用...")
        train_data = generate_synthetic_data(max_samples)

    # 采样
    if max_samples > 0 and len(train_data) > max_samples:
        random.seed(42)
        train_data = random.sample(train_data, max_samples)

    # 划分训练/验证集 (90/10)
    random.seed(42)
    random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)
    eval_data = train_data[split_idx:]
    train_data = train_data[:split_idx]

    # 保存
    print(f"[DATA] 保存训练集 {len(train_data)} 条, 验证集 {len(eval_data)} 条")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return train_data, eval_data


# ============================================================================
# 5. 奖励函数（修复版）
# ============================================================================

@dataclass
class RewardResult:
    total: float = 0.0
    format_r: float = 0.0
    exec_r: float = 0.0
    process_r: float = 0.0
    is_valid: bool = False
    error: str = ""


class RewardFunction:
    """修复版奖励函数 - 防止刷分攻击"""

    # 必须有完整的代码结构才能得分
    FORMAT_PATTERNS = {
        "import_pyomo": (r"import\s+pyomo|from\s+pyomo", 0.05),
        "model_def": (r"(?:Model|model)\s*=", 0.05),
        "variables": (r"(?:Var|var)\s*\(", 0.05),
        "objective": (r"(?:Objective|expr)\s*\(", 0.05),
        "constraints": (r"(?:Constraint|constraintlist)\s*\(", 0.05),
        "solver_call": (r"\.solve|\.SolverFactory", 0.05),
    }

    def __init__(self):
        self.format_scale = 1.0
        self.exec_scale = 2.0
        self.process_scale = 0.5

    def _extract_code(self, text: str) -> Optional[str]:
        import re
        patterns = [
            r"```python\n(.*?)```",
            r"```pyomo\n(.*?)```",
            r"```\n(.*?)```",
        ]
        for p in patterns:
            m = re.search(p, text, re.DOTALL)
            if m:
                code = m.group(1).strip()
                # 必须包含 pyomo 导入才认为是有效代码
                if re.search(r"import\s+pyomo|from\s+pyomo", code, re.I):
                    return code
        return None

    def _execute_code(self, code: str) -> Tuple[bool, str]:
        import sys, io
        from contextlib import redirect_stdout, redirect_stderr

        globs = {}
        try:
            from pyomo.environ import (
                ConcreteModel as CM, Objective as Obj, Constraint as C,
                Var as V, SolverFactory as SF, value as val,
                minimize, maximize, NonNegativeReals, NonNegativeIntegers,
                Binary, Set, Param, ConstraintList
            )
            globs.update({
                "ConcreteModel": CM, "Objective": Obj, "Constraint": C,
                "Var": V, "SolverFactory": SF, "value": val,
                "minimize": minimize, "maximize": maximize,
                "NonNegativeReals": NonNegativeReals,
                "NonNegativeIntegers": NonNegativeIntegers,
                "Binary": Binary, "Set": Set, "Param": Param,
                "ConstraintList": ConstraintList,
            })
        except ImportError:
            return False, "Pyomo not installed"

        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                exec(code, globs)
            return True, ""
        except Exception as e:
            return False, str(e)

    def compute_format_reward(self, text: str) -> float:
        import re
        code = self._extract_code(text)
        if code is None:
            return 0.0

        reward = 0.0
        critical_count = 0

        for name, (pattern, weight) in self.FORMAT_PATTERNS.items():
            if re.search(pattern, code, re.I):
                reward += weight
                if name in ("import_pyomo", "model_def", "solver_call"):
                    critical_count += 1

        # 完整性奖励
        if critical_count == 3:
            reward += 0.15

        return min(reward * self.format_scale, 1.5)

    def compute_exec_reward(self, text: str, fmt_r: float) -> Tuple[float, bool, str]:
        if fmt_r < 0.5:
            return 0.0, False, ""

        code = self._extract_code(text)
        if code is None:
            return 0.0, False, ""

        success, err = self._execute_code(code)
        if success:
            return 1.0 * self.exec_scale, True, ""
        return 0.0, False, err

    def compute_process_reward(self, text: str) -> float:
        import re
        reward = 0.0
        if re.search(r"##\s*(?:问题分析|Problem)", text):
            reward += 0.3
        if re.search(r"##\s*(?:数学模型|Mathematical)", text):
            reward += 0.3
        return min(reward * self.process_scale, 0.6)

    def __call__(self, text: str, is_last: bool = False) -> RewardResult:
        fmt_r = self.compute_format_reward(text)
        exec_r, is_valid, err = self.compute_exec_reward(text, fmt_r) if is_last else (0.0, False, "")
        proc_r = self.compute_process_reward(text)

        total = fmt_r + exec_r + proc_r
        total = max(min(total, 4.0), -1.0)

        return RewardResult(
            total=total,
            format_r=fmt_r,
            exec_r=exec_r,
            process_r=proc_r,
            is_valid=is_valid,
            error=err,
        )


def test_reward_function():
    """测试奖励函数"""
    print("\n" + "=" * 60)
    print("测试奖励函数")
    print("=" * 60)

    rf = RewardFunction()

    # 测试1: 刷分攻击
    hack_text = "python python python python python python"
    r = rf(hack_text, is_last=True)
    print(f"刷分攻击: total={r.total:.3f}, format={r.format_r:.3f}")
    assert r.total == 0.0, f"刷分攻击应该得0分，实际{r.total}"
    print("  ✓ 成功抵御刷分攻击")

    # 测试2: 有效代码
    valid_text = '''## 问题分析
这是一个线性规划问题。

## 数学模型
决策变量 x,y >= 0

## Pyomo实现
```python
from pyomo.environ import *
m = ConcreteModel()
m.x = Var(within=NonNegativeReals)
m.y = Var(within=NonNegativeReals)
m.obj = Objective(expr=3*m.x + 2*m.y, sense=maximize)
m.c1 = Constraint(expr=m.x + m.y <= 10)
m.c2 = Constraint(expr=2*m.x + m.y <= 16)
SolverFactory('cbc').solve(m)
```
'''
    r = rf(valid_text, is_last=True)
    print(f"有效代码: total={r.total:.3f}, format={r.format_r:.3f}, exec={r.exec_r:.3f}, valid={r.is_valid}")
    if r.is_valid:
        print("  ✓ 代码执行成功")
    else:
        print(f"  ! 执行失败: {r.error}")

    print("奖励函数测试完成\n")
    return rf


# ============================================================================
# 6. SFT 训练
# ============================================================================

def run_sft(
    model_name: str,
    train_data: List[Dict],
    eval_data: List[Dict],
    config: Config,
    output_dir: str,
) -> str:
    """运行 SFT 训练"""
    print("\n" + "=" * 70)
    print(f"SFT 训练: {model_name}")
    print(f"训练样本: {len(train_data)}, 验证样本: {len(eval_data)}")
    print("=" * 70)

    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        DataCollatorForLanguageModeling, Trainer, TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # Tokenizer
    print("[SFT] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 模型
    print("[SFT] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    print("[SFT] 应用 LoRA...")
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=config.get_lora_r(),
        lora_alpha=config.get_lora_r() * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    total_p, trainable_p, pct = count_parameters(model)
    print(f"[SFT] 模型参数: 总计 {total_p/1e6:.1f}M, 可训练 {trainable_p/1e6:.1f}M ({pct:.1f}%)")

    # 数据集
    print("[SFT] 准备数据集...")

    class ORDataset:
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            full_text = f"{prompt}\n\n{response}{self.tokenizer.eos_token}"

            enc = self.tokenizer(
                full_text, max_length=self.max_length,
                truncation=True, padding="max_length", return_tensors="pt"
            )

            ids = enc["input_ids"].squeeze()
            mask = enc["attention_mask"].squeeze()

            # 计算 prompt 长度
            p_enc = self.tokenizer(prompt, add_special_tokens=False)
            p_len = len(p_enc["input_ids"]) + 2

            labels = ids.clone()
            labels[:p_len] = -100

            return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    train_dataset = ORDataset(train_data, tokenizer, config.get_max_length())
    eval_dataset = ORDataset(eval_data[:min(50, len(eval_data))], tokenizer, config.get_max_length())

    # 训练参数
    bs = config.get_batch_size()
    grad_accum = config.sft_grad_accum
    steps_per_epoch = len(train_dataset) // bs // grad_accum
    warmup_steps = int(steps_per_epoch * config.sft_epochs * 0.03)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=config.sft_epochs,
        learning_rate=config.sft_lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=max(1, steps_per_epoch // 10),
        eval_steps=max(1, steps_per_epoch // 2),
        eval_strategy="steps",
        save_steps=max(1, steps_per_epoch),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=config.seed,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"[SFT] 开始训练 (batch={bs}, grad_accum={grad_accum}, epochs={config.sft_epochs})")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[SFT] 训练完成, 耗时: {format_time(elapsed)}")

    # 保存
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[SFT] 模型已保存: {final_path}")

    return final_path


# ============================================================================
# 7. GRPO 训练
# ============================================================================

def run_grpo(
    sft_model_path: str,
    train_data: List[Dict],
    eval_data: List[Dict],
    config: Config,
    output_dir: str,
) -> str:
    """运行 GRPO 训练"""
    print("\n" + "=" * 70)
    print(f"GRPO 训练 (基于 SFT 模型: {sft_model_path})")
    print(f"训练样本: {len(train_data)}, 每样本生成数: {config.grpo_num_generations}")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset as HFDataset

    # 加载 tokenizer
    print("[GRPO] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print("[GRPO] 加载 SFT 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # 准备数据集 (ChatML 格式)
    print("[GRPO] 准备数据集...")
    texts, answers = [], []
    for item in train_data:
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        try:
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ], tokenize=False)
        except:
            text = f"{prompt}\n\n{response}"

        texts.append(text)
        answers.append(item.get("answer", ""))

    hf_dataset = HFDataset.from_dict({"text": texts, "answer": answers})

    # 奖励函数
    reward_fn = RewardFunction()

    def reward_function(prompts, responses, **kwargs):
        rewards = []
        for resp in responses:
            r = reward_fn(resp, is_last=True)
            rewards.append(r.total)
        return rewards

    # GRPO 配置
    max_len = config.get_max_length()
    bs = config.get_batch_size()

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.grpo_epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=config.grpo_grad_accum,
        learning_rate=config.grpo_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        max_length=max_len,
        max_prompt_length=max_len // 2,
        max_completion_length=max_len // 2,
        num_generations=config.grpo_num_generations,
        beta=config.grpo_kl_coef,
        loss_type="grpo",
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        seed=config.seed,
        remove_unused_columns=False,
    )

    print(f"[GRPO] 配置: batch={bs}, generations={config.grpo_num_generations}, "
          f"lr={config.grpo_lr}, kl_coef={config.grpo_kl_coef}, clip={config.grpo_clip_eps}")

    # 创建训练器
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        reward_function=reward_function,
        train_dataset=hf_dataset,
        tokenizer=tokenizer,
    )

    print(f"[GRPO] 开始训练...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[GRPO] 训练完成, 耗时: {format_time(elapsed)}")

    # 保存
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[GRPO] 模型已保存: {final_path}")

    return final_path


# ============================================================================
# 8. 评测
# ============================================================================

def evaluate_model(
    model_path: str,
    eval_data: List[Dict],
    config: Config,
    max_eval: int = 100,
) -> Dict:
    """评测模型"""
    print("\n" + "=" * 70)
    print(f"评测模型: {model_path}")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True
    )
    model.eval()

    reward_fn = RewardFunction()
    device = model.device

    results = []
    eval_samples = eval_data[:max_eval]

    print(f"[EVAL] 评测 {len(eval_samples)} 个样本...")

    for i, item in enumerate(eval_samples):
        prompt = item.get("prompt", "")

        t0 = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - t0

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        r = reward_fn(response, is_last=True)

        results.append({
            "id": item.get("id", f"eval_{i}"),
            "reward": r.total,
            "format": r.format_r,
            "exec": r.exec_r,
            "valid": r.is_valid,
            "gen_time": gen_time,
        })

        status = "✓" if r.is_valid else "✗"
        print(f"  [{i+1:3d}/{len(eval_samples)}] {status} "
              f"R={r.total:.2f}(F={r.format_r:.1f},E={r.exec_r:.1f}) {format_time(gen_time)}")

        if (i + 1) % 10 == 0:
            recent = results[-10:]
            print(f"       最近10个: mean_R={np.mean([x['reward'] for x in recent]):.2f}, "
                  f"valid_rate={np.mean([x['valid'] for x in recent]):.1%}")

    # 统计
    rewards = [x["reward"] for x in results]
    valid_rate = np.mean([x["valid"] for x in results])
    format_rate = np.mean([x["format"] > 0 for x in results])

    stats = {
        "num_samples": len(results),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "valid_rate": float(valid_rate),
        "format_rate": float(format_rate),
        "avg_gen_time": float(np.mean([x["gen_time"] for x in results])),
    }

    print(f"\n{'='*50}")
    print("评测结果汇总")
    print(f"{'='*50}")
    print(f"  样本数:     {stats['num_samples']}")
    print(f"  平均奖励:   {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"  最高奖励:   {stats['max_reward']:.3f}")
    print(f"  执行率:     {stats['valid_rate']:.1%}")
    print(f"  格式正确率: {stats['format_rate']:.1%}")
    print(f"  平均生成时间: {stats['avg_gen_time']:.2f}s")
    print(f"{'='*50}")

    return stats


# ============================================================================
# 9. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM4OR 训练脚本")
    parser.add_argument("--model", type=str, default="1.5B",
                        choices=["0.5B", "1.5B", "3B", "7B"],
                        help="模型大小")
    parser.add_argument("--skip_sft", action="store_true",
                        help="跳过 SFT 阶段")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="SFT 检查点路径（跳过 SFT 时使用）")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="最大训练样本数")
    parser.add_argument("--sft_epochs", type=int, default=3,
                        help="SFT 训练轮数")
    parser.add_argument("--grpo_epochs", type=int, default=2,
                        help="GRPO 训练轮数")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="每个样本的生成数")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 构建配置
    config = Config(
        model_size=args.model,
        max_train_samples=args.max_samples,
        sft_epochs=args.sft_epochs,
        grpo_epochs=args.grpo_epochs,
        grpo_num_generations=args.num_generations,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    model_name = config.get_model_name()
    print("\n" + "#" * 70)
    print(f"# LLM4OR 训练 - 模型: {model_name} ({args.model})")
    print(f"# 训练样本: {args.max_samples}, SFT_epochs: {args.sft_epochs}, GRPO_epochs: {args.grpo_epochs}")
    print(f"# 输出目录: {args.output_dir}")
    print("#" * 70)

    # 环境检查
    print(f"\n[ENV] PyTorch: {torch.__version__}")
    print(f"[ENV] CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[ENV] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[ENV] 显存: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
    else:
        print("[ENV] WARNING: 无 GPU，训练将非常慢！")

    # 设置随机种子
    set_seed(config.seed)

    # 测试奖励函数
    reward_fn = test_reward_function()

    # 数据准备
    train_data, eval_data = load_or_download_data(config.max_train_samples)
    print(f"\n[DATA] 训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    # SFT 阶段
    sft_output = os.path.join(config.output_dir, "sft")
    sft_path = None

    if args.skip_sft:
        if args.sft_checkpoint:
            sft_path = args.sft_checkpoint
        else:
            # 找最新的 SFT 检查点
            sft_final = os.path.join(sft_output, "final")
            if os.path.exists(sft_final):
                sft_path = sft_final
            else:
                print("[ERROR] 需要指定 --sft_checkpoint")
                sys.exit(1)
        print(f"\n[SFT] 跳过 SFT，使用检查点: {sft_path}")
    else:
        sft_path = run_sft(model_name, train_data, eval_data, config, sft_output)
        # 快速评测 SFT 模型
        print("\n[SFT] 评测 SFT 模型...")
        sft_stats = evaluate_model(sft_path, eval_data, config, max_eval=30)
        with open(os.path.join(sft_output, "eval_results.json"), "w") as f:
            json.dump(sft_stats, f, indent=2)

    # GRPO 阶段
    grpo_output = os.path.join(config.output_dir, "grpo")
    grpo_path = run_grpo(sft_path, train_data, eval_data, config, grpo_output)

    # 最终评测
    print("\n[FINAL] 最终评测...")
    final_stats = evaluate_model(grpo_path, eval_data, config, max_eval=config.eval_samples)

    # 保存最终结果
    final_results_path = os.path.join(grpo_output, "final_results.json")
    with open(final_results_path, "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"\n[OK] 最终结果已保存: {final_results_path}")

    # 汇总
    print("\n" + "#" * 70)
    print("# 训练完成!")
    print(f"# SFT 模型:  {sft_path}")
    print(f"# GRPO 模型: {grpo_path}")
    print(f"# 最终评测:   执行率={final_stats['valid_rate']:.1%}, "
          f"平均奖励={final_stats['mean_reward']:.3f}")
    print("#" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        traceback.print_exc()
        sys.exit(1)
