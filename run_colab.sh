#!/bin/bash
# Google Colab 快速启动脚本
# 使用方法：在 Colab 中运行 !bash run_colab.sh

set -e

echo "=========================================="
echo "LLM4OR GRPO 训练 - Google Colab 启动"
echo "=========================================="

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
pip install \
    torch \
    transformers \
    accelerate \
    bitsandbytes \
    peft \
    datasets \
    pyyaml \
    tqdm \
    tensorboard \
    pyomo \
    highspy \
    scikit-learn

# 可选：安装 GLPK 求解器
echo "安装 GLPK 求解器..."
apt-get update -qq && apt-get install -qq -y glpk-utils

# 检查 GPU
echo "检查 GPU..."
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU 数量: {torch.cuda.device_count()}'); print(f'GPU 名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 下载数据集
echo "下载数据集..."
python -c "
from datasets import load_dataset
import json
import os

# 尝试下载 OptMATH 数据集
try:
    print('下载 OptMATH 数据集...')
    ds = load_dataset('AI4Math/OptMATH', trust_remote_code=True)
    os.makedirs('./data/processed', exist_ok=True)

    # 转换为训练格式
    train_data = []
    for split in ds:
        for item in ds[split]:
            prompt = '请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\n\n问题描述：\n' + (item.get('problem', '') or item.get('text', ''))
            code = item.get('code', '') or item.get('solution', '')
            answer = item.get('answer', item.get('optimal_value', ''))

            train_data.append({
                'id': item.get('id', ''),
                'prompt': prompt,
                'response': f'```python\n{code}\n```',
                'problem_text': item.get('problem', item.get('text', '')),
                'code_solution': code,
                'answer': str(answer) if answer else '',
            })

    with open('./data/processed/train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'成功保存 {len(train_data)} 条训练数据')
except Exception as e:
    print(f'下载数据集失败: {e}')
    print('将使用合成的测试数据...')
    os.makedirs('./data/processed', exist_ok=True)
    with open('./data/processed/train.jsonl', 'w', encoding='utf-8') as f:
        for i in range(100):
            item = {
                'id': f'train_{i}',
                'prompt': f'请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\n\n问题描述：\n一个简单的线性规划问题：最大化 3x + 2y，约束条件：x + y <= 10, 2x + y <= 16, x >= 0, y >= 0',
                'response': f'```python\n# 第{i}个示例代码\nfrom pyomo.environ import *\nm = ConcreteModel()\nm.x = Var(within=NonNegativeReals)\nm.y = Var(within=NonNegativeReals)\nm.obj = Objective(expr=3*m.x + 2*m.y, sense=maximize)\nm.c1 = Constraint(expr=m.x + m.y <= 10)\nm.c2 = Constraint(expr=2*m.x + m.y <= 16)\nSolverFactory(\"cbc\").solve(m).write()\n```',
                'problem_text': '一个简单的线性规划问题：最大化 3x + 2y，约束条件：x + y <= 10, 2x + y <= 16, x >= 0, y >= 0',
                'code_solution': f'from pyomo.environ import *\nm = ConcreteModel()\nm.x = Var(within=NonNegativeReals)\nm.y = Var(within=NonNegativeReals)\nm.obj = Objective(expr=3*m.x + 2*m.y, sense=maximize)\nm.c1 = Constraint(expr=m.x + m.y <= 10)\nm.c2 = Constraint(expr=2*m.x + m.y <= 16)\nSolverFactory("cbc").solve(m).write()',
                'answer': '28.0',
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print('已创建 100 条合成测试数据')
"

echo "=========================================="
echo "环境准备完成！"
echo "=========================================="
echo ""
echo "下一步，运行训练："
echo ""
echo "  # 方案A: 完整训练 (SFT + GRPO)"
echo "  !python train.py --model_name Qwen/Qwen2.5-1.5B --max_samples 5000"
echo ""
echo "  # 方案B: 仅 SFT"
echo "  !python sft/train_sft.py --model_name Qwen/Qwen2.5-1.5B --train_data ./data/processed/train.jsonl"
echo ""
echo "  # 方案C: 仅 GRPO (需要先有 SFT 模型)"
echo "  !python grpo/train_grpo.py --sft_model_path ./outputs/sft/final --train_data ./data/processed/train.jsonl"
echo ""
echo "  # 方案D: 小规模快速验证（推荐先运行这个）"
echo "  !python train.py --model_name Qwen/Qwen2.5-0.5B --max_samples 500 --grpo_epochs 1 --num_samples_per_prompt 2"
echo ""
echo "=========================================="
