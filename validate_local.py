#!/usr/bin/env python3
"""
本地验证脚本 - 在任何机器上快速验证代码正确性
不需要 GPU，不需要完整训练，只验证代码逻辑
"""
import sys
import json
import time
import traceback

def test_reward_function():
    """测试奖励函数"""
    print("\n" + "=" * 60)
    print("测试 1: 奖励函数")
    print("=" * 60)

    # 导入奖励函数
    sys.path.insert(0, str(__file__).replace("validate_local.py", ""))
    from run_train import RewardFunction, RewardResult

    rf = RewardFunction()

    # 测试1: 刷分攻击
    print("\n[1.1] 刷分攻击测试...", end=" ")
    hack = "python python python python python"
    r = rf(hack, is_last=True)
    assert r.total == 0.0, f"刷分攻击应得0分, 实际={r.total}"
    print(f"OK (reward={r.total:.2f})")

    # 测试2: 关键词干扰
    print("[1.2] 关键词干扰测试...", end=" ")
    noise = "这是一个优化问题。我来建立一个线性规划模型。让我用python写代码求解。"
    r = rf(noise, is_last=True)
    assert r.total == 0.0, f"无有效代码应得0分, 实际={r.total}"
    print(f"OK (reward={r.total:.2f})")

    # 测试3: 有效但无法执行的代码
    print("[1.3] 无效代码测试...", end=" ")
    invalid = "```python\nfrom pyomo.environ import *\nm = ConcreteModel()\n# 没有变量\n```"
    r = rf(invalid, is_last=True)
    fmt_r = r.format_r
    assert fmt_r > 0, "有导入和模型定义应该有格式分"
    assert r.exec_r == 0, "无法执行的代码不应有执行分"
    print(f"OK (format={r.format_r:.2f}, exec={r.exec_r:.2f})")

    # 测试4: 完整有效代码
    print("[1.4] 有效代码测试...", end=" ")
    valid = """```python
from pyomo.environ import *
m = ConcreteModel()
m.x = Var(within=NonNegativeReals)
m.y = Var(within=NonNegativeReals)
m.obj = Objective(expr=3*m.x + 2*m.y, sense=maximize)
m.c1 = Constraint(expr=m.x + m.y <= 10)
m.c2 = Constraint(expr=2*m.x + m.y <= 16)
results = SolverFactory('cbc').solve(m)
```"""
    r = rf(valid, is_last=True)
    assert r.format_r > 0, "有效代码应该有格式分"
    print(f"format={r.format_r:.2f}, exec={r.exec_r:.2f}, total={r.total:.2f}")

    if r.is_valid:
        print("  代码执行: 成功")
    else:
        print(f"  代码执行: 失败 ({r.error})")

    print("\n奖励函数测试通过!")


def test_data_generation():
    """测试数据生成"""
    print("\n" + "=" * 60)
    print("测试 2: 数据生成")
    print("=" * 60)

    sys.path.insert(0, str(__file__).replace("validate_local.py", ""))
    from run_train import generate_synthetic_data

    print("\n[2.1] 生成合成数据...", end=" ")
    data = generate_synthetic_data(100)
    assert len(data) == 100, f"应生成100条数据, 实际{len(data)}"
    print(f"OK ({len(data)} 条)")

    print("[2.2] 检查数据格式...", end=" ")
    for item in data[:5]:
        assert "prompt" in item, "缺少 prompt 字段"
        assert "response" in item, "缺少 response 字段"
        assert "answer" in item, "缺少 answer 字段"
        assert "```python" in item["response"], "response 应包含代码块"
    print("OK")

    print("[2.3] 检查 prompt 格式...", end=" ")
    for item in data[:5]:
        assert "运筹优化问题" in item["prompt"], "prompt 应包含引导语"
        assert len(item["problem_text"]) > 10, "problem_text 不应为空"
    print("OK")

    print("\n数据生成测试通过!")


def test_imports():
    """测试依赖导入"""
    print("\n" + "=" * 60)
    print("测试 3: 依赖导入")
    print("=" * 60)

    print("\n[3.1] PyTorch...", end=" ")
    import torch
    print(f"OK ({torch.__version__})")

    print("[3.2] Transformers...", end=" ")
    import transformers
    print(f"OK ({transformers.__version__})")

    print("[3.3] PEFT...", end=" ")
    import peft
    print(f"OK ({peft.__version__})")

    print("[3.4] TRL...", end=" ")
    import trl
    print(f"OK ({trl.__version__})")

    print("[3.5] Pyomo...", end=" ")
    try:
        from pyomo.environ import SolverFactory, ConcreteModel, Var, Objective, Constraint
        print("OK")
    except ImportError:
        print("未安装 (仅影响执行奖励，不影响训练)")

    print("\n依赖检查完成!")


def test_model_loading():
    """测试模型加载（CPU模式，不需要GPU）"""
    print("\n" + "=" * 60)
    print("测试 4: 模型加载 (CPU, 小模型)")
    print("=" * 60)

    print("\n[4.1] 加载 tokenizer...", end=" ")
    from transformers import AutoTokenizer
    # 使用一个肯定存在的模型
    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print("OK")
    except Exception as e:
        print(f"跳过 (需要网络): {e}")
        return

    print("[4.2] 加载模型 (CPU, float32)...", end=" ")
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("OK")
    except Exception as e:
        print(f"跳过 (需要网络或内存): {e}")
        return

    print("[4.3] 生成测试...", end=" ")
    prompt = "请为以下问题写Pyomo代码: 最大化 3x + 2y, 约束 x + y <= 10"
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tok.decode(outputs[0], skip_special_tokens=True)
    assert len(response) > len(prompt), "生成失败"
    print(f"OK (生成了 {len(response)} 字符)")

    print("\n模型加载测试通过!")


def main():
    print("\n" + "#" * 60)
    print("# LLM4OR 本地验证脚本")
    print("#" * 60)

    all_passed = True
    tests = [
        ("依赖导入", test_imports),
        ("数据生成", test_data_generation),
        ("奖励函数", test_reward_function),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n测试 '{name}' 失败: {e}")
            traceback.print_exc()
            all_passed = False

    # GPU 测试（可选）
    try:
        import torch
        if torch.cuda.is_available():
            try:
                test_model_loading()
            except Exception as e:
                print(f"\n模型加载测试失败: {e}")
                traceback.print_exc()
        else:
            print("\n[SKIP] GPU 测试 (无可用 GPU)")
    except:
        pass

    print("\n" + "=" * 60)
    if all_passed:
        print("全部测试通过! 代码可以正常运行.")
    else:
        print("部分测试失败, 请检查错误信息.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
