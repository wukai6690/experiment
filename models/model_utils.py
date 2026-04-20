"""
模型加载与量化工具
支持: 8-bit 量化, LoRA, DeepSpeed ZeRO
"""
import os
import torch
import warnings
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


def get_model_dtype() -> torch.dtype:
    """获取模型数据类型"""
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_with_quantization(
    model_name_or_path: str,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    use_flash_attention: bool = True,
    trust_remote_code: bool = True,
    device_map: Optional[str] = "auto",
) -> Tuple[Any, Any]:
    """
    加载带量化的模型

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    assert not (load_in_8bit and load_in_4bit), "只能选择一种量化方式"

    bnb_config = None
    if load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    elif load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_model_dtype(),
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )

    # 确保有 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": get_model_dtype(),
    }

    # 可选：启用 Flash Attention 2
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # 处理双引号问题
    model_name_or_path = model_name_or_path.strip('"')

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    return model, tokenizer


def load_model_for_training(
    model_name_or_path: str,
    lora_config: Optional[Dict] = None,
    gradient_checkpointing: bool = True,
    use_flash_attention: bool = True,
) -> Tuple[Any, Any]:
    """
    加载用于训练的模型（不带量化，使用 LoRA）

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name_or_path = model_name_or_path.strip('"')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float32,  # 训练用 fp32
        "device_map": None,
        "trust_remote_code": True,
    }

    if use_flash_attention and torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    # 启用梯度检查点
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # 应用 LoRA
    if lora_config:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            target_modules=lora_config.get("target_modules", None),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    """保存模型和 tokenizer"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"模型已保存到: {output_dir}")


def merge_lora_weights(model, output_dir: str):
    """合并 LoRA 权重到基础模型"""
    from peft import PeftModel
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
        save_model_and_tokenizer(model, None, output_dir)
    return model


# ===== 推荐模型配置 =====
MODEL_CONFIGS = {
    "Qwen2.5-0.5B": {
        "repo": "Qwen/Qwen2.5-0.5B",
        "lora_r": 8,
        "lora_alpha": 16,
        "batch_size": 4,
        "max_seq_len": 1024,
        "min_gpu_mem_gb": 4,
    },
    "Qwen2.5-1.5B": {
        "repo": "Qwen/Qwen2.5-1.5B",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,
        "max_seq_len": 1536,
        "min_gpu_mem_gb": 8,
    },
    "Qwen2.5-3B": {
        "repo": "Qwen/Qwen2.5-3B",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 1,
        "max_seq_len": 1536,
        "min_gpu_mem_gb": 12,
    },
    "Qwen2.5-7B": {
        "repo": "Qwen/Qwen2.5-7B",
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 1,
        "max_seq_len": 2048,
        "min_gpu_mem_gb": 24,
    },
}


def get_model_config(model_name: str) -> Dict:
    """获取模型配置"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    # 尝试作为 HuggingFace repo 路径
    if "/" in model_name:
        return {
            "repo": model_name,
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 2,
            "max_seq_len": 1536,
            "min_gpu_mem_gb": 12,
        }
    raise ValueError(f"未知模型: {model_name}")


def check_gpu_memory() -> Dict[str, float]:
    """检查 GPU 显存"""
    if not torch.cuda.is_available():
        return {"total": 0, "available": 0, "used": 0}

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    available = total - allocated

    return {
        "total": round(total, 2),
        "allocated": round(allocated, 2),
        "available": round(available, 2),
    }


def estimate_required_memory(
    model_name: str,
    batch_size: int,
    seq_len: int,
    use_8bit: bool = False,
) -> float:
    """估算所需显存（GB）"""
    config = get_model_config(model_name)
    model_size_gb = {
        "Qwen2.5-0.5B": 1.0,
        "Qwen2.5-1.5B": 3.0,
        "Qwen.5-3B": 6.0,
        "Qwen2.5-7B": 14.0,
    }.get(model_name, 10.0)

    if use_8bit:
        model_size_gb *= 0.5
    else:
        model_size_gb *= 2  # fp16/fp32 overhead

    # 注意力激活值估算（约与参数量相当）
    activation_gb = model_size_gb * 0.5 * batch_size

    # KV cache
    kv_cache_gb = model_size_gb * 0.25 * batch_size * seq_len / 1024

    return model_size_gb + activation_gb + kv_cache_gb
