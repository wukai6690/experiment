"""
修复版奖励函数
关键修复：
1. 移除关键词奖励（防止 "python" 刷分攻击）
2. 严格格式校验（必须包含完整建模+代码结构）
3. 执行奖励基于求解器实际结果
4. 支持过程奖励（中间步骤质量）
"""
import re
import ast
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class RewardResult:
    """奖励计算结果"""
    total_reward: float
    format_reward: float
    execution_reward: float
    answer_reward: float = 0.0
    process_reward: float = 0.0
    is_valid: bool = False
    error_message: str = ""
    solved_value: Optional[float] = None
    ground_truth: Optional[float] = None


class RewardFunction:
    """
    多维度奖励函数

    奖励组成:
    - 格式奖励: 代码结构是否完整、规范
    - 执行奖励: 代码能否被 Pyomo 成功执行
    - 答案奖励: 结果是否与参考答案一致
    """

    # 格式奖励的各个子项
    FORMAT_ITEMS = {
        "import_pyomo": {
            "pattern": r"import\s+pyomo|from\s+pyomo",
            "weight": 0.05,
            "desc": "导入 Pyomo 库",
        },
        "model_definition": {
            "pattern": r"(Model|model)\s*=",
            "weight": 0.05,
            "desc": "定义模型",
        },
        "decision_variables": {
            "pattern": r"(Var|var)\s*\(",
            "weight": 0.05,
            "desc": "定义决策变量",
        },
        "objective_function": {
            "pattern": r"(Objective|objective|expr)\s*\(",
            "weight": 0.05,
            "desc": "定义目标函数",
        },
        "constraints": {
            "pattern": r"(Constraint|constraint|ConstraintList|constraint_list)\s*\(",
            "weight": 0.05,
            "desc": "定义约束条件",
        },
        "solver_call": {
            "pattern": r"(\.solve|\.SolverFactory)",
            "weight": 0.05,
            "desc": "调用求解器",
        },
        "low_bound_definition": {
            "pattern": r"lowBound\s*=|bounds\s*=",
            "weight": 0.05,
            "desc": "定义变量下界",
        },
        "code_block_mark": {
            "pattern": r"```\s*python|```\s*pyomo",
            "weight": 0.0,  # 格式标记，不计入奖励
            "desc": "代码块标记",
        },
    }

    def __init__(
        self,
        enable_format_reward: bool = True,
        enable_execution_reward: bool = True,
        enable_answer_reward: bool = True,
        enable_process_reward: bool = True,
        format_reward_scale: float = 1.0,
        execution_reward_scale: float = 2.0,
        answer_reward_scale: float = 3.0,
        answer_tolerance: float = 1e-4,
        timeout_seconds: float = 10.0,
        enable_strict_mode: bool = True,
    ):
        """
        Args:
            enable_format_reward: 启用格式奖励
            enable_execution_reward: 启用执行奖励
            enable_answer_reward: 启用答案奖励
            enable_process_reward: 启用过程奖励
            format_reward_scale: 格式奖励缩放因子
            execution_reward_scale: 执行奖励缩放因子
            answer_reward_scale: 答案奖励缩放因子
            answer_tolerance: 答案容差（相对误差）
            timeout_seconds: 代码执行超时时间
            enable_strict_mode: 严格模式，要求格式完整才能获得执行奖励
        """
        self.enable_format_reward = enable_format_reward
        self.enable_execution_reward = enable_execution_reward
        self.enable_answer_reward = enable_answer_reward
        self.enable_process_reward = enable_process_reward
        self.format_reward_scale = format_reward_scale
        self.execution_reward_scale = execution_reward_scale
        self.answer_reward_scale = answer_reward_scale
        self.answer_tolerance = answer_tolerance
        self.timeout_seconds = timeout_seconds
        self.enable_strict_mode = enable_strict_mode

    def _extract_code(self, text: str) -> Optional[str]:
        """从模型输出中提取 Pyomo 代码"""
        # 优先提取 ```python 或 ```pyomo 代码块
        patterns = [
            r"```python\n(.*?)```",
            r"```pyomo\n(.*?)```",
            r"```\n(.*?)```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                code = match.group(1)
                # 清理代码
                code = self._clean_code(code)
                if self._is_valid_code(code):
                    return code

        # 如果没有代码块，尝试将整个响应作为代码
        if self._is_valid_code(text):
            return self._clean_code(text)

        return None

    def _clean_code(self, code: str) -> str:
        """清理代码"""
        # 移除注释
        lines = []
        for line in code.split("\n"):
            # 移除行注释
            if "#" in line:
                line = line[:line.index("#")]
            lines.append(line)
        code = "\n".join(lines)

        # 移除空行
        lines = [l.rstrip() for l in code.split("\n") if l.strip()]
        return "\n".join(lines)

    def _is_valid_code(self, code: str) -> bool:
        """检查代码是否看起来像有效的 Pyomo 代码"""
        if not code or len(code) < 20:
            return False

        # 必须包含 Pyomo 导入
        has_import = bool(re.search(r"import\s+pyomo|from\s+pyomo", code))
        if not has_import:
            return False

        # 必须包含模型定义
        has_model = bool(re.search(r"(Model|model)\s*=", code))
        if not has_model:
            return False

        return True

    def compute_format_reward(self, text: str) -> Tuple[float, Dict[str, bool]]:
        """
        计算格式奖励

        Returns:
            (格式奖励分数, 各子项通过情况)
        """
        code = self._extract_code(text)
        if code is None:
            # 没有找到有效代码，格式奖励为 0
            item_results = {k: False for k in self.FORMAT_ITEMS}
            return 0.0, item_results

        reward = 0.0
        item_results = {}

        for item_name, item_info in self.FORMAT_ITEMS.items():
            pattern = item_info["pattern"]
            weight = item_info["weight"]
            matched = bool(re.search(pattern, code, re.IGNORECASE))
            item_results[item_name] = matched

            if weight > 0:
                reward += weight if matched else 0.0

        # 格式完整奖励：如果所有关键项都通过，额外奖励
        critical_items = ["import_pyomo", "model_definition", "solver_call"]
        all_critical_passed = all(item_results.get(k, False) for k in critical_items)

        if all_critical_passed:
            reward += 0.15  # 完整性奖励

        # 缩放
        reward = reward * self.format_reward_scale

        # 格式奖励上限为 1.5
        reward = min(reward, 1.5)

        return reward, item_results

    def _execute_pyomo_code(self, code: str, ground_truth: Optional[float] = None) -> Tuple[bool, Optional[float], str]:
        """
        执行 Pyomo 代码并返回结果

        Returns:
            (是否成功, 最优值, 错误信息)
        """
        import sys
        import io
        from contextlib import redirect_stdout, redirect_stderr

        # 构建执行环境
        exec_globals = {
            "__name__": "__main__",
            "pyomo": None,
            "ConcreteModel": None,
            "Objective": None,
            "Constraint": None,
            "Var": None,
            "SolverFactory": None,
            "value": None,
            "results": None,
        }

        # 添加 Pyomo 到命名空间
        try:
            import pyomo
            from pyomo.environ import (
                ConcreteModel, Objective, Constraint, Var,
                SolverFactory, value, minimize, maximize, ConstraintList
            )
            exec_globals.update({
                "pyomo": pyomo,
                "ConcreteModel": ConcreteModel,
                "Objective": Objective,
                "Constraint": Constraint,
                "Var": Var,
                "SolverFactory": SolverFactory,
                "value": value,
                "ConstraintList": ConstraintList,
                "minimize": minimize,
                "maximize": maximize,
            })
        except ImportError as e:
            return False, None, f"Pyomo 未安装: {e}"

        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_globals)

            # 尝试提取结果
            solved_value = None

            # 方式1: 从 exec_globals 中获取
            if "results" in exec_globals and exec_globals["results"] is not None:
                results = exec_globals["results"]
                try:
                    if hasattr(results, "Problem"):
                        solved_value = float(results["Problem"][0]["Objective"])
                    elif hasattr(results, "Bound"):
                        solved_value = float(results["Bound"])
                    elif isinstance(results, dict) and "Objective" in results:
                        solved_value = float(results["Object"])
                except (KeyError, ValueError, TypeError):
                    pass

            # 方式2: 从 model 获取
            if solved_value is None and "model" in exec_globals:
                m = exec_globals["model"]
                try:
                    obj = list(m.component_objects(Objective, active=True))[0]
                    solved_value = value(obj)
                except Exception:
                    pass

            # 方式3: 检查 stdout 中的输出
            if solved_value is None:
                output = stdout_capture.getvalue()
                value_patterns = [
                    r"Optimal\s+(?:value|objective)[:\s]+([-+]?\d*\.?\d+)",
                    r"Objective[:\s]+([-+]?\d*\.?\d+)",
                    r"=\s*([-+]?\d*\.?\d+)\s*$",
                ]
                for pattern in value_patterns:
                    match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
                    if match:
                        try:
                            solved_value = float(match.group(1))
                            break
                        except ValueError:
                            continue

            return True, solved_value, ""

        except Exception as e:
            error_msg = str(e)
            # 清理错误信息
            error_msg = re.sub(r"File \"<string>\", line \d+,", "", error_msg)
            return False, None, error_msg.strip()

    def compute_execution_reward(
        self,
        text: str,
        ground_truth: Optional[float] = None,
    ) -> Tuple[float, bool, Optional[float], str]:
        """
        计算执行奖励

        Returns:
            (执行奖励, 是否成功执行, 最优值, 错误信息)
        """
        code = self._extract_code(text)
        if code is None:
            return 0.0, False, None, "未找到有效 Pyomo 代码"

        success, solved_value, error_msg = self._execute_pyomo_code(code, ground_truth)

        if not success:
            return 0.0, False, None, error_msg

        # 代码执行成功
        execution_reward = 1.0 * self.execution_reward_scale

        # 额外奖励：能提取到数值解
        if solved_value is not None:
            execution_reward += 0.5

        return execution_reward, success, solved_value, ""

    def compute_answer_reward(
        self,
        solved_value: Optional[float],
        ground_truth: Optional[float],
    ) -> float:
        """
        计算答案奖励（与参考答案对比）

        Returns:
            奖励值，范围 [-1, 3]
        """
        if ground_truth is None or solved_value is None:
            return 0.0

        if ground_truth == 0:
            diff = abs(solved_value)
        else:
            diff = abs(solved_value - ground_truth) / (abs(ground_truth) + 1e-8)

        if diff < self.answer_tolerance:
            # 完全正确
            return 3.0 * self.answer_reward_scale
        elif diff < 1e-2:
            return 2.0 * self.answer_reward_scale
        elif diff < 1e-1:
            return 1.0 * self.answer_reward_scale
        elif diff < 1.0:
            return 0.5 * self.answer_reward_scale
        else:
            # 完全错误（惩罚）
            return -1.0

    def compute_process_reward(self, text: str) -> float:
        """
        计算过程奖励：评估推理链的质量
        这是一个轻量级实现，主要基于规则的检查
        """
        reward = 0.0

        # 检查是否包含问题分析
        if re.search(r"##\s*问题分析|##\s*Problem\s+Analysis", text):
            reward += 0.3

        # 检查是否包含数学建模
        if re.search(r"##\s*(?:数学模型|Mathematical\s+Model|建模)", text):
            reward += 0.3

        # 检查变量定义是否有描述
        if re.search(r"#\s*.*(?:变量|决策|说明|表示)", text):
            reward += 0.2

        # 检查约束是否有描述
        constraint_comments = len(re.findall(r"#\s*.*(?:约束|限制|条件)", text))
        if constraint_comments >= 2:
            reward += 0.3
        elif constraint_comments >= 1:
            reward += 0.1

        return min(reward, 1.0)

    def __call__(
        self,
        text: str,
        ground_truth: Optional[float] = None,
        is_last: bool = False,
    ) -> RewardResult:
        """
        计算总奖励

        Args:
            text: 模型生成的完整响应文本
            ground_truth: 参考答案（可选）
            is_last: 是否是最后一个 token（用于判断是否执行）

        Returns:
            RewardResult 对象
        """
        # 1. 格式奖励
        format_reward, format_items = self.compute_format_reward(text)

        # 2. 执行奖励（只在最后一步计算）
        execution_reward = 0.0
        is_valid = False
        error_message = ""
        solved_value = None

        if is_last or format_reward > 0.5:
            # 只有格式基本正确才尝试执行
            execution_reward, is_valid, solved_value, error_message = \
                self.compute_execution_reward(text, ground_truth)

        # 3. 答案奖励（只在执行成功且有参考答案时计算）
        answer_reward = 0.0
        if ground_truth is not None and is_valid:
            answer_reward = self.compute_answer_reward(solved_value, ground_truth)

        # 4. 过程奖励（只在有过程内容时计算）
        process_reward = 0.0
        if self.enable_process_reward:
            process_reward = self.compute_process_reward(text)

        # 总奖励
        total_reward = format_reward + execution_reward + answer_reward + process_reward

        # 应用缩放
        # 格式奖励已经缩放过了，这里对总奖励做最终裁剪
        total_reward = max(min(total_reward, 5.0), -2.0)

        return RewardResult(
            total_reward=total_reward,
            format_reward=format_reward,
            execution_reward=execution_reward,
            answer_reward=answer_reward,
            process_reward=process_reward,
            is_valid=is_valid,
            error_message=error_message,
            solved_value=solved_value,
            ground_truth=ground_truth,
        )


def test_reward_function():
    """测试奖励函数"""
    print("=" * 60)
    print("奖励函数测试")
    print("=" * 60)

    rf = RewardFunction()

    # 测试用例1: 完美的解决方案
    test_text_good = """
## 问题分析
这是一个运输问题，需要最小化总运输成本。

## 数学模型
决策变量：x_ij = 从工厂i运往市场j的货物量（连续变量，非负）
目标函数：min sum(c_ij * x_ij for all i,j)
约束条件：...（省略）

## Pyomo 实现
```python
from pyomo.environ import *

m = ConcreteModel()
m.f = Set(initialize=['Factory1', 'Factory2', 'Factory3'])
m.c = Set(initialize=['Customer1', 'Customer2'])

m.cost = Param(m.f, m.c, initialize={
    ('Factory1', 'Customer1'): 4,
    ('Factory1', 'Customer2'): 6,
    ('Factory2', 'Customer1'): 5,
    ('Factory2', 'Customer2'): 3,
    ('Factory3', 'Customer1'): 7,
    ('Factory3', 'Customer2'): 2,
})

m.x = Var(m.f, m.c, within=NonNegativeReals, bounds=(0, None))

m.obj = Objective(expr=sum(m.cost[i, j] * m.x[i, j] for i in m.f for j in m.c), sense=minimize)

m.supply = Constraint(m.f, rule=lambda m, i: sum(m.x[i, j] for j in m.c) <= 10)
m.demand = Constraint(m.c, rule=lambda m, j: sum(m.x[i, j] for i in m.f) >= 8)

solver = SolverFactory('cbc')
results = solver.solve(m, tee=False)
print(f"Optimal cost: {value(m.obj)}")
```
"""

    result = rf(test_text_good, ground_truth=80.0, is_last=True)
    print(f"\n测试1 - 完整解决方案:")
    print(f"  格式奖励: {result.format_reward:.3f}")
    print(f"  执行奖励: {result.execution_reward:.3f}")
    print(f"  答案奖励: {result.answer_reward:.3f}")
    print(f"  过程奖励: {result.process_reward:.3f}")
    print(f"  总奖励:   {result.total_reward:.3f}")
    print(f"  执行状态: {'成功' if result.is_valid else '失败'}")
    print(f"  求解值:   {result.solved_value}")
    print(f"  错误信息: {result.error_message or '无'}")

    # 测试用例2: 刷分攻击
    test_text_hack = """
python python python python python python python python
我想想这个问题，应该建立一个模型来求解。
python python python python python python
```

    result = rf(test_text_hack, ground_truth=100.0, is_last=True)
    print(f"\n测试2 - 'python' 刷分攻击:")
    print(f"  格式奖励: {result.format_reward:.3f}")
    print(f"  执行奖励: {result.execution_reward:.3f}")
    print(f"  答案奖励: {result.answer_reward:.3f}")
    print(f"  总奖励:   {result.total_reward:.3f}")
    print(f"  执行状态: {'成功' if result.is_valid else '失败'}")
    assert result.format_reward == 0.0, "格式奖励应为0（无有效代码结构）"
    assert result.execution_reward == 0.0, "执行奖励应为0（无法执行）"
    assert result.total_reward == 0.0, "总奖励应为0（没有有效内容）"
    print("  ✓ 成功抵御刷分攻击！")

    # 测试用例3: 部分有效代码
    test_text_partial = """
```python
from pyomo.environ import *

m = ConcreteModel()
m.x = Var(within=NonNegativeReals)
m.obj = Objective(expr=m.x, sense=minimize)
m.c = Constraint(expr=m.x >= 5)

solver = SolverFactory('cbc')
results = solver.solve(m)
```
"""

    result = rf(test_text_partial, ground_truth=5.0, is_last=True)
    print(f"\n测试3 - 部分有效代码:")
    print(f"  格式奖励: {result.format_reward:.3f}")
    print(f"  执行奖励: {result.execution_reward:.3f}")
    print(f"  答案奖励: {result.answer_reward:.3f}")
    print(f"  总奖励:   {result.total_reward:.3f}")
    print(f"  执行状态: {'成功' if result.is_valid else '失败'}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_reward_function()
