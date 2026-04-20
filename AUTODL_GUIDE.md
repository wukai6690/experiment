# AutoDL 部署指南

## 步骤 1: 上传项目到 AutoDL

将整个项目文件夹上传到 AutoDL。可以使用以下任一方式：

### 方式 A: 通过 AutoDL 网页界面上传
1. 打开 AutoDL 控制台
2. 上传项目 ZIP 文件
3. 解压到工作目录

### 方式 B: 使用 scp 上传（如果你有远程服务器访问）
```bash
scp -r ./d_ai_for_moudle_experiment user@autodl-server:/root/workspace/
```

---

## 步骤 2: 登录 AutoDL 并选择机器

1. 登录 AutoDL 控制台
2. 选择合适的 GPU 机器：
   - **Qwen2.5-1.5B**: 推荐 RTX 4090 / A5000 / A100 40G
   - **Qwen2.5-3B**: 推荐 A100 40G 或多卡
   - **Qwen2.5-7B**: 推荐 A100 80G 或多卡

---

## 步骤 3: 修改配置文件（可选）

编辑 `run_autodl_full.sh` 顶部的配置：

```bash
# 模型选择
MODEL_SIZE="1.5B"      # 可选: 0.5B, 1.5B, 3B, 7B

# 训练样本数
MAX_SAMPLES=5000       # 5000足够，显存不够可以降到2000

# 训练轮数
SFT_EPOCHS=3           # SFT轮数，3轮足够
GRPO_EPOCHS=2          # GRPO轮数

# 其他参数
NUM_GENERATIONS=4       # 每个prompt生成4个样本
GRPO_LR=1e-5           # GRPO学习率
```

---

## 步骤 4: 一键启动

在 AutoDL 终端中运行：

```bash
cd /root/workspace/d_ai_for_moudle_experiment
bash run_autodl_full.sh
```

---

## 步骤 5: 监控训练

训练启动后会自动在后台运行，可以通过以下方式监控：

### 查看实时日志
```bash
tail -f train.log
```

### 查看 TensorBoard
```bash
tensorboard --logdir=./outputs/
# 然后在浏览器中打开 AutoDL 提供的端口
```

### 查看进程状态
```bash
ps aux | grep run_train.py
```

---

## 显存不够时的解决方案

### 问题 1: CUDA out of memory

**解决方案：**
1. 减小 batch size（修改 `run_train.py` 中的 `sft_batch_size` 和 `grpo_batch_size`）
2. 减小 `max_samples`
3. 使用更小的模型

```bash
# 使用更小的batch size
python run_train.py --model 1.5B --sft_epochs 3 --grpo_epochs 2 --max_samples 3000
```

### 问题 2: NCCL 通信错误

**解决方案：**
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
python run_train.py ...
```

### 问题 3: PyTorch 版本不兼容

**解决方案：**
```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 中断后恢复训练

### 恢复 SFT
```bash
# SFT 不支持直接恢复，但有检查点
ls outputs/sft/
# 找到最新的检查点
python run_train.py --skip_sft --sft_checkpoint ./outputs/sft/checkpoint-XXX
```

### 恢复 GRPO
```bash
# GRPO 同样有检查点
python run_train.py --skip_sft --sft_checkpoint ./outputs/sft/final --resume_from ./outputs/grpo
```

---

## 查看最终结果

训练完成后：

```bash
# 查看 GRPO 最终评测结果
cat outputs/grpo/final_results.json

# 查看 SFT 评测结果
cat outputs/sft/eval_results.json
```

预期输出格式：
```json
{
  "num_samples": 100,
  "mean_reward": 2.345,
  "std_reward": 1.234,
  "max_reward": 3.500,
  "valid_rate": 0.75,
  "format_rate": 0.85,
  "avg_gen_time": 5.67
}
```

---

## 快速测试（不花钱）

如果你想先在本地测试代码是否正确：

### 在 AutoDL 上用 CPU 测试（慢，但不花钱验证代码）
```bash
python run_train.py --model 0.5B --max_samples 50 --sft_epochs 1 --grpo_epochs 1
```

### 只测试 SFT（最快验证）
```bash
python run_train.py --skip_sft --sft_checkpoint ./outputs/sft/final
# 或者
python sft/train_sft.py --model_name Qwen/Qwen2.5-0.5B --max_samples 100
```

---

## 常见错误处理

### 错误: `undefined symbol: ncclCommWindowDeregister`
```
# 解决方案：降级 PyTorch 或使用兼容版本
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### 错误: `AttributeError: module 'transformers' has no attribute 'LoraConfig'`
```
# 解决方案：更新 transformers
pip install transformers>=4.40.0
```

### 错误: `CUDA out of memory`
```
# 解决方案1: 使用更小的batch size
python run_train.py --model 1.5B --max_samples 2000

# 解决方案2: 减小最大序列长度
# 修改 run_train.py 中的 max_length 参数

# 解决方案3: 使用Qwen2.5-0.5B
python run_train.py --model 0.5B
```

### 错误: 奖励一直是负数
```
这是正常的！初始阶段模型输出大多无法执行。
应该随着训练进行逐渐上升。
如果一直不变，检查奖励函数实现。
```

---

## 推荐配置

| 显存 | 模型 | batch_size | max_samples | 预估时间 |
|-----|------|-----------|------------|---------|
| 24GB (RTX 4090) | 1.5B | 2 | 3000 | ~3小时 |
| 40GB (A100) | 3B | 1 | 5000 | ~4小时 |
| 80GB (A100) | 7B | 1 | 5000 | ~6小时 |
| 24GB (RTX 4090) | 3B | 1 | 2000 | ~3小时 |
