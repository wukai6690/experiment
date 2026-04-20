#!/bin/bash
# AutoDL 启动脚本
# 使用方法：保存为 run_autodl.sh，上传到 AutoDL 上运行

set -e

echo "=========================================="
echo "LLM4OR GRPO 训练 - AutoDL 启动"
echo "=========================================="

# 创建工作目录
mkdir -p data/processed
mkdir -p outputs/sft
mkdir -p outputs/grpo
mkdir -p outputs/eval

# 安装依赖
echo "安装 Python 依赖..."
pip install -q \
    torch \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.41.0 \
    peft>=0.10.0 \
    datasets>=2.18.0 \
    pyyaml \
    tqdm \
    tensorboard \
    pyomo>=6.6.0

# 安装求解器 (HiGHS 和 CBC)
echo "安装求解器..."
pip install -q highspy glpk

# 验证安装
echo "验证安装..."
python -c "
import torch
import transformers
import peft
import pyomo
import highspy
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('Pyomo: OK')
print('HiGHS: OK')
"

# 检查 Pyomo 求解器
python -c "
from pyomo.environ import SolverFactory
solver = SolverFactory('cbc')
print('CBC 求解器: OK')
solver = SolverFactory('highs')
print('HiGHS 求解器: OK')
"

echo "=========================================="
echo "环境准备完成！"
echo "=========================================="

# 默认配置
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
KL_COEF="${KL_COEF:-0.04}"

echo "默认配置:"
echo "  模型: $MODEL_NAME"
echo "  样本数: $MAX_SAMPLES"
echo "  SFT epochs: $SFT_EPOCHS"
echo "  GRPO epochs: $GRPO_EPOCHS"
echo ""

# 运行完整训练流程
echo "开始训练..."
python train.py \
    --model_name "$MODEL_NAME" \
    --max_samples "$MAX_SAMPLES" \
    --sft_epochs "$SFT_EPOCHS" \
    --grpo_epochs "$GRPO_EPOCHS" \
    --num_samples_per_prompt "$NUM_SAMPLES" \
    --learning_rate "$LEARNING_RATE" \
    --kl_coef "$KL_COEF" \
    --output_dir ./outputs \
    --eval_max_samples 100

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="

# 查看输出
echo "查看训练输出..."
ls -la outputs/sft/
ls -la outputs/grpo/
ls -la outputs/eval/

echo ""
echo "查看 TensorBoard 日志..."
echo "tensorboard --logdir=outputs/"
