#!/bin/bash
# ============================================================================
# 一键启动脚本 - 最简版本
# 直接运行，无需配置
# ============================================================================

# 步骤1: 安装依赖
echo "[1/3] 安装依赖..."
pip install -q transformers accelerate bitsandbytes peft datasets trl tensorboard tqdm pyomo highspy 2>/dev/null

# 步骤2: 创建数据
echo "[2/3] 创建训练数据..."
mkdir -p data/processed outputs/sft outputs/grpo
python -c "
import os, json, random

templates = [
    {
        'text': '最大化 3x + 2y，约束：x + y <= 10, 2x + y <= 16, x >= 0, y >= 0',
        'answer': '28.0',
        'code': 'from pyomo.environ import *\nm = ConcreteModel()\nm.x = Var(within=NonNegativeReals)\nm.y = Var(within=NonNegativeReals)\nm.obj = Objective(expr=3*m.x + 2*m.y, sense=maximize)\nm.c1 = Constraint(expr=m.x + m.y <= 10)\nm.c2 = Constraint(expr=2*m.x + m.y <= 16)\nSolverFactory(\"cbc\").solve(m)'
    },
    {
        'text': '运输问题：两工厂供给量15和20，两市场需求量12和18，成本矩阵[[4,6],[5,3]]，求最小成本',
        'answer': '138.0',
        'code': 'from pyomo.environ import *\nm = ConcreteModel()\nm.f = Set(initialize=[\"F1\",\"F2\"])\nm.c = Set(initialize=[\"M1\",\"M2\"])\nm.supply = Param(m.f, initialize={\"F1\":15,\"F2\":20})\nm.demand = Param(m.c, initialize={\"M1\":12,\"M2\":18})\nm.cost = Param(m.f, m.c, initialize={(\"F1\",\"M1\"):4,(\"F1\",\"M2\"):6,(\"F2\",\"M1\"):5,(\"F2\",\"M2\"):3})\nm.x = Var(m.f, m.c, within=NonNegativeReals)\nm.obj = Objective(expr=sum(m.cost[i,j]*m.x[i,j] for i in m.f for j in m.c))\nm.s = Constraint(m.f, rule=lambda m,i: sum(m.x[i,j] for j in m.c) <= m.supply[i])\nm.d = Constraint(m.c, rule=lambda m,j: sum(m.x[i,j] for i in m.f) >= m.demand[j])\nSolverFactory(\"cbc\").solve(m)'
    },
]

data = []
for i in range(5000):
    t = templates[i % len(templates)]
    data.append({
        'id': f'train_{i}',
        'prompt': f'请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\n\n问题描述：\n{t[\"text\"]}',
        'response': f'\`\`\`python\n{t[\"code\"]}\n\`\`\`',
        'answer': t['answer'],
    })

random.seed(42)
random.shuffle(data)
split = int(len(data) * 0.9)
os.makedirs('./data/processed', exist_ok=True)
for d, name in [(data[:split], 'train.jsonl'), (data[split:], 'eval.jsonl')]:
    with open(f'./data/processed/{name}', 'w') as f:
        for item in d:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'数据已创建: {len(data[:split])} 训练, {len(data[split:])} 验证')
"

# 步骤3: 启动训练
echo "[3/3] 启动训练..."
MODEL="${1:-1.5B}"
SAMPLES="${2:-5000}"

echo "模型: Qwen2.5-${MODEL}"
echo "样本: ${SAMPLES}"

python run_train.py \
    --model "${MODEL}" \
    --max_samples "${SAMPLES}" \
    --sft_epochs 3 \
    --grpo_epochs 2 \
    --num_generations 4 \
    --output_dir ./outputs

echo ""
echo "训练完成!"
echo "查看结果: cat outputs/grpo/final_results.json"
echo "查看日志: cat outputs/grpo/train.log"
