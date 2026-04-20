"""
分段 KL 惩罚实现
这是 SIRL 论文的核心创新：对推理链和建模代码施加不同程度的 KL 约束
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SegmentConfig:
    """分段配置"""
    name: str                    # 分段名称
    start_pattern: str           # 开始标记
    end_pattern: str             # 结束标记
    kl_coefficient: float        # KL 惩罚系数
    allow_exploration: bool      # 是否允许探索


# 默认配置：对推理链放松，对建模+代码严格
DEFAULT_SEGMENTS = [
    SegmentConfig(
        name="reasoning",
        start_pattern="",
        end_pattern="## 数学模型",
        kl_coefficient=0.1,      # 推理链：轻惩罚，允许探索
        allow_exploration=True,
    ),
    SegmentConfig(
        name="math_model",
        start_pattern="## 数学模型",
        end_pattern="## Pyomo",
        kl_coefficient=0.5,       # 数学建模：中等惩罚
        allow_exploration=True,
    ),
    SegmentConfig(
        name="code",
        start_pattern="## Pyomo",
        end_pattern="```",
        kl_coefficient=1.0,      # 代码段：严格惩罚，保持语法
        allow_exploration=False,
    ),
]


def find_segment_boundaries(
    text: str,
    segments: List[SegmentConfig],
) -> List[Tuple[int, int, SegmentConfig]]:
    """
    在文本中定位各个分段

    Returns:
        [(start_pos, end_pos, segment_config), ...]
    """
    import re
    boundaries = []

    # 将文本按分段配置进行切分
    # 首先找到所有关键标记的位置
    markers = {}
    for seg in segments:
        if seg.start_pattern:
            for match in re.finditer(re.escape(seg.start_pattern), text):
                markers[match.start()] = ("start", seg)
        if seg.end_pattern:
            for match in re.finditer(re.escape(seg.end_pattern), text):
                markers[match.start()] = ("end", seg)

    # 按位置排序
    sorted_markers = sorted(markers.items(), key=lambda x: x[0])

    # 构建分段
    current_pos = 0
    current_seg = None

    for pos, (marker_type, seg) in sorted_markers:
        if marker_type == "start" and pos >= current_pos:
            # 开始新分段
            if current_seg is not None and current_pos < pos:
                boundaries.append((current_pos, pos, current_seg))
            current_pos = pos
            current_seg = seg

    # 添加最后一个分段
    if current_seg is not None:
        boundaries.append((current_pos, len(text), current_seg))

    return boundaries


def compute_segment_kl_penalty(
    ref_logprobs: torch.Tensor,
    ref_logprobs_model: torch.Tensor,
    segment_boundaries: List[Tuple[int, int, SegmentConfig]],
    response_length: int,
) -> Tuple[float, Dict[str, float]]:
    """
    计算分段 KL 惩罚

    Args:
        ref_logprobs: 参考策略的 logprobs [seq_len]
        ref_logprobs_model: 当前策略的 logprobs [seq_len]
        segment_boundaries: 分段边界信息
        response_length: 响应序列长度

    Returns:
        (总KL惩罚, 各分段KL值)
    """
    import re

    total_kl = 0.0
    segment_kls = {}

    if len(segment_boundaries) == 0:
        # 没有分段信息，对整体应用中等惩罚
        kl = F.kl_div(
            ref_logprobs_model,
            ref_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        return kl * 0.3, {"all": kl}

    for start, end, seg_config in segment_boundaries:
        if end <= 0 or start >= response_length:
            continue

        # 截取到实际响应长度
        start = max(0, start)
        end = min(end, response_length)

        if end <= start:
            continue

        segment_logprobs_ref = ref_logprobs[start:end]
        segment_logprobs_curr = ref_logprobs_model[start:end]

        if len(segment_logprobs_ref) == 0:
            continue

        # 计算 KL 散度: D_KL(ref || curr) = sum(ref * (ref - curr))
        kl = F.kl_div(
            segment_logprobs_curr,
            segment_logprobs_ref,
            reduction="batchmean",
            log_target=True,
        ).item()

        # 应用分段系数
        weighted_kl = kl * seg_config.kl_coefficient
        total_kl += weighted_kl
        segment_kls[seg_config.name] = kl

    # 如果有些部分没被覆盖（没有匹配到分段），对剩余部分应用默认惩罚
    total_covered = sum(end - start for start, end, _ in segment_boundaries)
    if total_covered < response_length:
        remaining = response_length - total_covered
        default_kl_coef = 0.2  # 对未覆盖部分使用默认中等惩罚
        total_kl += remaining / response_length * default_kl_coef
        segment_kls["default"] = remaining / response_length * default_kl_coef

    return total_kl, segment_kls


class SegmentKLTracker:
    """分段 KL 追踪器：记录训练过程中各分段的 KL 值"""

    def __init__(self, segment_names: List[str]):
        self.segment_names = segment_names
        self.history: List[Dict[str, float]] = []
        self.step_count = 0

    def record(self, segment_kls: Dict[str, float]):
        """记录当前 step 的各分段 KL 值"""
        self.history.append(segment_kls)
        self.step_count += 1

    def get_average(self) -> Dict[str, float]:
        """获取历史平均 KL 值"""
        if not self.history:
            return {name: 0.0 for name in self.segment_names}

        avg = {name: 0.0 for name in self.segment_names}
        for record in self.history:
            for name in self.segment_names:
                avg[name] += record.get(name, 0.0)

        for name in self.segment_names:
            avg[name] /= len(self.history)

        return avg

    def get_recent(self, n: int = 10) -> Dict[str, float]:
        """获取最近 n 次的平均 KL 值"""
        if not self.history:
            return {name: 0.0 for name in self.segment_names}

        recent = self.history[-n:]
        avg = {name: 0.0 for name in self.segment_names}
        for record in recent:
            for name in self.segment_names:
                avg[name] += record.get(name, 0.0)

        for name in self.segment_names:
            avg[name] /= len(recent)

        return avg

    def summary(self) -> str:
        """生成摘要报告"""
        avg = self.get_average()
        recent = self.get_recent(10)

        lines = [
            f"Segment KL Tracker (Total Steps: {self.step_count})",
            "-" * 50,
            f"{'Segment':<20} {'Average KL':<15} {'Recent(10) KL':<15}",
            "-" * 50,
        ]

        for name in self.segment_names:
            lines.append(
                f"{name:<20} {avg.get(name, 0):<15.4f} {recent.get(name, 0):<15.4f}"
            )

        lines.append("-" * 50)
        return "\n".join(lines)
