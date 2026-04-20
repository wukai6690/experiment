#!/bin/bash
# ============================================================================
# AutoDL 部署脚本 - 一键运行 LLM4OR 训练
# ============================================================================
#
# 使用方法:
#   1. 将本项目上传到 AutoDL
#   2. 修改下面的配置参数
#   3. 运行: bash run_autodl_full.sh
#
# ============================================================================

set -e

# ============================================================================
# 配置 (根据你的 AutoDL 机器修改)
# ============================================================================

# 模型选择: 0.5B, 1.5B, 3B, 7B
MODEL_SIZE="${MODEL_SIZE:-1.5B}"

# 训练样本数 (5000足够验证，建议不超过10000)
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

# SFT 轮数 (3轮足够)
SFT_EPOCHS="${SFT_EPOCHS:-3}"

# GRPO 轮数 (2-3轮)
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"

# 每个prompt的生成数 (GRPO的G值，越大优势估计越准但越慢)
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"

# 学习率
GRPO_LR="${GRPO_LR:-1e-5}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

# ============================================================================
# 环境准备
# ============================================================================

echo ""
echo "============================================================"
echo "LLM4OR 训练 - AutoDL 自动部署"
echo "============================================================"
echo "模型: Qwen2.5-${MODEL_SIZE}"
echo "样本数: ${MAX_SAMPLES}"
echo "SFT epochs: ${SFT_EPOCHS}"
echo "GRPO epochs: ${GRPO_EPOCHS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# 创建目录
mkdir -p data/processed
mkdir -p "${OUTPUT_DIR}/sft"
mkdir -p "${OUTPUT_DIR}/grpo"
mkdir -p "${OUTPUT_DIR}/eval"

# 检查 GPU
echo "[1/6] 检查 GPU 环境..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  错误: 需要 GPU 才能训练!')
    exit(1)
"

# 安装 Python 依赖
echo ""
echo "[2/6] 安装 Python 依赖..."
pip install -q \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.41.0 \
    peft>=0.10.0 \
    datasets>=2.18.0 \
    trl>=0.12.0 \
    tensorboard>=2.15.0 \
    tqdm>=4.66.0

# 安装 Pyomo 和求解器
echo ""
echo "[3/6] 安装 Pyomo 和求解器..."
pip install -q pyomo highspy

# 验证 Pyomo
python -c "
from pyomo.environ import SolverFactory, ConcreteModel, Var, Objective
print('  Pyomo: OK')
s = SolverFactory('cbc')
print('  CBC求解器: OK')
"

# ============================================================================
# 数据准备
# ============================================================================

echo ""
echo "[4/6] 准备训练数据..."

DATA_DIR="./data/processed"
TRAIN_PATH="${DATA_DIR}/train.jsonl"
EVAL_PATH="${DATA_DIR}/eval.jsonl"

if [ -f "${TRAIN_PATH}" ] && [ -f "${EVAL_PATH}" ]; then
    TRAIN_COUNT=$(wc -l < "${TRAIN_PATH}")
    EVAL_COUNT=$(wc -l < "${EVAL_PATH}")
    echo "  数据已存在: 训练集 ${TRAIN_COUNT} 条, 验证集 ${EVAL_COUNT} 条"
else
    echo "  正在下载/生成数据..."

    python -c "
import os
import json
import random

# 合成数据模板
templates = [
    {
        'text': '一个简单的线性规划问题：最大化 3x + 2y，约束条件：x + y <= 10, 2x + y <= 16, x >= 0, y >= 0',
        'answer': '28.0',
        'code': '''from pyomo.environ import *
m = ConcreteModel()
m.x = Var(within=NonNegativeReals)
m.y = Var(within=NonNegativeReals)
m.obj = Objective(expr=3*m.x + 2*m.y, sense=maximize)
m.c1 = Constraint(expr=m.x + m.y <= 10)
m.c2 = Constraint(expr=2*m.x + m.y <= 16)
results = SolverFactory('cbc').solve(m, tee=False)
print(f\"Optimal value: {value(m.obj)}\")'''
    },
    {
        'text': '运输问题：有两个工厂和两个市场，工厂供给量分别为15和20单位，市场需求量分别为12和18单位，单位运输成本矩阵为[[4,6],[5,3]]，求最小运输成本',
        'answer': '138.0',
        'code': '''from pyomo.environ import *
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
print(f\"Optimal cost: {value(m.obj)}\")'''
    },
    {
        'text': '背包问题：有5个物品，重量分别为[2,3,4,5,6]，价值分别为[3,4,5,6,7]，背包容量为10，求最大价值',
        'answer': '11.0',
        'code': '''from pyomo.environ import *
m = ConcreteModel()
m.i = Set(initialize=[1,2,3,4,5])
w = {1:2,2:3,3:4,4:5,5:6}
v = {1:3,2:4,3:5,4:6,5:7}
m.x = Var(m.i, within=Binary)
m.obj = Objective(expr=sum(v[i]*m.x[i] for i in m.i), sense=maximize)
m.c = Constraint(expr=sum(w[i]*m.x[i] for i in m.i) <= 10)
results = SolverFactory('cbc').solve(m)
print(f\"Optimal value: {value(m.obj)}\")'''
    },
    {
        'text': '整数规划问题：最小化 x + y，约束：x + 2y >= 8, 3x + y >= 6, x >= 0, y >= 0, x,y 为整数',
        'answer': '4.0',
        'code': '''from pyomo.environ import *
m = ConcreteModel()
m.x = Var(within=NonNegativeIntegers)
m.y = Var(within=NonNegativeIntegers)
m.obj = Objective(expr=m.x + m.y)
m.c1 = Constraint(expr=m.x + 2*m.y >= 8)
m.c2 = Constraint(expr=3*m.x + m.y >= 6)
results = SolverFactory('cbc').solve(m)
print(f\"Optimal value: {value(m.obj)}\")'''
    },
    {
        'text': '生产计划问题：某工厂生产两种产品，产品A每单位利润5元，需要资源1单位8小时和资源2单位4小时；产品B每单位利润8元，需要资源1单位4小时和资源2单位6小时。每天资源1可用80小时，资源2可用60小时，求最大利润',
        'answer': '85.0',
        'code': '''from pyomo.environ import *
m = ConcreteModel()
m.A = Var(within=NonNegativeReals)
m.B = Var(within=NonNegativeReals)
m.obj = Objective(expr=5*m.A + 8*m.B, sense=maximize)
m.r1 = Constraint(expr=8*m.A + 4*m.B <= 80)
m.r2 = Constraint(expr=4*m.A + 6*m.B <= 60)
results = SolverFactory('cbc').solve(m)
print(f\"Optimal profit: {value(m.obj)}\")'''
    },
]

os.makedirs('./data/processed', exist_ok=True)
n = ${MAX_SAMPLES}
data = []
for i in range(n):
    t = templates[i % len(templates)]
    prompt = f'请为以下运筹优化问题建立数学模型并编写 Pyomo 代码求解。\\n\\n问题描述：\\n{t[\"text\"]}'
    data.append({
        'id': f'train_{i}',
        'prompt': prompt,
        'response': f'\`\`\`python\\n{t[\"code\"]}\\n\`\`\`',
        'problem_text': t['text'],
        'code_solution': t['code'],
        'answer': t['answer'],
    })

random.seed(42)
random.shuffle(data)
split = int(len(data) * 0.9)
train, eval_ = data[:split], data[split:]

with open('./data/processed/train.jsonl', 'w') as f:
    for item in train:
        f.write(json.dumps(item, ensure_ascii=False) + '\\n')
with open('./data/processed/eval.jsonl', 'w') as f:
    for item in eval_:
        f.write(json.dumps(item, ensure_ascii=False) + '\\n')

print(f'  生成 {len(train)} 条训练数据, {len(eval_)} 条验证数据')
"

    echo "  数据准备完成"
fi

# ============================================================================
# 启动训练
# ============================================================================

echo ""
echo "[5/6] 启动训练..."
echo "============================================================"

# 计算实际 batch size
case ${MODEL_SIZE} in
    0.5B) BS=4 ;;
    1.5B) BS=2 ;;
    3B)   BS=1 ;;
    7B)   BS=1 ;;
    *)    BS=2 ;;
esac

echo "实际 batch size: ${BS}"

# 启动训练 (使用 nohup 后台运行，日志输出到 train.log)
nohup python run_train.py \
    --model "${MODEL_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --sft_epochs "${SFT_EPOCHS}" \
    --grpo_epochs "${GRPO_EPOCHS}" \
    --num_generations "${NUM_GENERATIONS}" \
    --output_dir "${OUTPUT_DIR}" \
    > train.log 2>&1 &

TRAIN_PID=$!
echo ""
echo "训练已启动 (PID: ${TRAIN_PID})"
echo "日志输出: train.log"
echo "实时查看: tail -f train.log"
echo ""
echo "开始监控训练进度..."
echo "============================================================"

# ============================================================================
# 监控训练进度
# ============================================================================

sleep 10

# 检查进程是否还在运行
check_alive() {
    kill -0 ${TRAIN_PID} 2>/dev/null
}

# 实时监控
while check_alive; do
    if [ -f "train.log" ]; then
        # 显示最新输出
        CLEAR
        echo ""
        echo "============================================================"
        echo "训练日志 (${TRAIN_PID}) - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        tail -n 30 train.log
        echo "============================================================"
    fi
    sleep 30
done

# 训练结束
echo ""
echo "============================================================"
echo "训练完成!"
echo "============================================================"

# 显示最终结果
if [ -f "train.log" ]; then
    echo ""
    echo "--- 最终日志 ---"
    tail -n 50 train.log
fi

# 检查评测结果
if [ -f "${OUTPUT_DIR}/grpo/final_results.json" ]; then
    echo ""
    echo "--- GRPO 最终评测结果 ---"
    cat "${OUTPUT_DIR}/grpo/final_results.json"
fi

echo ""
echo "============================================================"
echo "全部完成!"
echo "SFT 模型:  ${OUTPUT_DIR}/sft/final/"
echo "GRPO 模型: ${OUTPUT_DIR}/grpo/final/"
echo "TensorBoard: tensorboard --logdir=${OUTPUT_DIR}/"
echo "============================================================"
