"""
SFT 训练脚本：使用 LoRA 对 LLM 进行指令微调
这是 GRPO 训练的前置阶段，目的是让模型学会 OR 建模的基本格式
"""
import os
import sys
import json
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ORPromptResponseDataset(Dataset):
    """
    运筹优化建模数据集
    格式: {"prompt": ..., "response": ..., "id": ..., "answer": ...}
    """
    data: List[Dict]
    tokenizer: AutoTokenizer
    max_length: int = 2048

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        # 构建完整序列: prompt + response + eos
        full_text = f"{prompt}\n\n{response}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # 计算 prompt 长度
        prompt_encoding = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        # response 起始位置（包含换行和特殊 token）
        # prompt + "\n\n" + response 的格式
        labels = input_ids.clone()
        # prompt 部分（以及中间的 special tokens）用 -100 掩盖
        labels[:prompt_len + 2] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_jsonl_dataset(
    file_path: str,
    max_samples: int = 0,
    shuffle: bool = False,
) -> List[Dict]:
    """加载 JSONL 格式的数据集"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if shuffle:
        import random
        random.seed(42)
        random.shuffle(data)

    if max_samples > 0 and len(data) > max_samples:
        data = data[:max_samples]

    logger.info(f"加载了 {len(data)} 条数据 from {file_path}")
    return data


def setup_lora_config(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """配置 LoRA"""
    # Qwen2 系列的目标模块
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def train_sft(
    model_name: str,
    train_data: List[Dict],
    eval_data: Optional[List[Dict]] = None,
    output_dir: str = "./outputs/sft",
    max_length: int = 2048,
    num_epochs: int = 3,
    per_device_batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    max_samples: int = 0,
    use_flash_attention: bool = True,
):
    """
    运行 SFT 训练

    Args:
        model_name: HuggingFace 模型名或本地路径
        train_data: 训练数据列表
        eval_data: 验证数据列表（可选）
        output_dir: 输出目录
        max_length: 最大序列长度
        num_epochs: 训练轮数
        per_device_batch_size: 每个 GPU 的 batch size
        learning_rate: 学习率
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        warmup_ratio: 预热比例
        logging_steps: 日志记录步数
        save_steps: 保存步数
        eval_steps: 评估步数
        max_samples: 最大训练样本数（0=全部）
        use_flash_attention: 是否使用 Flash Attention
    """
    model_name = model_name.strip('"')
    logger.info(f"加载模型: {model_name}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model_kwargs = {
        "torch_dtype": torch.float32,
        "device_map": None,
        "trust_remote_code": True,
    }

    if use_flash_attention and torch.cuda.is_available():
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("已启用 Flash Attention 2")
        except Exception as e:
            logger.warning(f"Flash Attention 不可用: {e}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # 冻结基础模型参数，只训练 LoRA
    for param in model.parameters():
        param.requires_grad = False

    # 应用 LoRA
    lora_config = setup_lora_config(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    model.print_trainable_parameters()

    # 准备数据集
    train_dataset = ORPromptResponseDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    eval_dataset = None
    if eval_data:
        eval_dataset = ORPromptResponseDataset(
            data=eval_data,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 计算训练步数
    total_steps = (len(train_dataset) // per_device_batch_size) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        dataloader_num_workers=2,
        seed=42,
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    logger.info("开始 SFT 训练...")
    trainer.train()

    # 保存模型
    logger.info(f"保存模型到: {output_dir}")
    trainer.save_model(str(Path(output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(output_dir) / "final"))

    logger.info("SFT 训练完成！")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT 训练脚本")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-1.5B",
                        help="模型名称或路径")
    parser.add_argument("--train_data", type=str,
                        default="./data/processed/train.jsonl",
                        help="训练数据路径")
    parser.add_argument("--eval_data", type=str,
                        default="./data/processed/eval.jsonl",
                        help="验证数据路径")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/sft",
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=1536,
                        help="最大序列长度")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--per_device_batch_size", type=int, default=2,
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最大样本数，0=全部")
    parser.add_argument("--no_flash_attention", action="store_true",
                        help="禁用 Flash Attention")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从 checkpoint 恢复训练")

    args = parser.parse_args()

    # 加载数据
    train_data = load_jsonl_dataset(args.train_data, max_samples=args.max_samples, shuffle=True)
    eval_data = None
    if os.path.exists(args.eval_data):
        eval_data = load_jsonl_dataset(args.eval_data)

    if not train_data:
        logger.error(f"训练数据为空: {args.train_data}")
        sys.exit(1)

    # 开始训练
    train_sft(
        model_name=args.model_name,
        train_data=train_data,
        eval_data=eval_data,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_samples=args.max_samples,
        use_flash_attention=not args.no_flash_attention,
    )


if __name__ == "__main__":
    main()
