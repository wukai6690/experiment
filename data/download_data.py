"""
数据下载脚本
从 HuggingFace / GitHub 下载训练和评测数据
"""
import os
import requests
import zipfile
import json
import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk


HF_DATASETS = {
    # OptMATH: 优化建模数据集，格式良好
    "optmath": "AI4Math/OptMATH",
    # NL4OPT: 经典的 NL-to-MILP 翻译数据集
    "nl4opt": "AI4Math/NL4OPT",
    # StepOPT: 带推理步骤的优化建模数据
    "stepopt": "samwu-learn/StepOPT",
    # Resocratic: 包含推理过程的优化问题数据
    "resocratic": "TheBloke/Resocratic-27k",  # 可能需要替换为真实数据集名
}


def download_from_huggingface(dataset_name: str, output_dir: str, split: str = "train"):
    """从 HuggingFace 下载数据集"""
    print(f"正在下载数据集: {dataset_name}")
    try:
        if dataset_name in HF_DATASETS:
            ds = load_dataset(HF_DATASETS[dataset_name], trust_remote_code=True)
            output_path = Path(output_dir) / dataset_name
            output_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(output_path))
            print(f"数据集已保存到: {output_path}")
            return ds
        else:
            print(f"未知数据集: {dataset_name}")
            return None
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def download_nl4opt_from_url(output_dir: str):
    """从 GitHub 下载 NL4OPT 原始数据"""
    github_raw_url = "https://raw.githubusercontent.com/nl4opt/nl4opt/main/data/"
    output_path = Path(output_dir) / "nl4opt"
    output_path.mkdir(parents=True, exist_ok=True)

    files = [
        "train.json",
        "valid.json",
        "test.json",
    ]

    for fname in files:
        url = github_raw_url + fname
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(output_path / fname, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                print(f"下载完成: {fname}")
            else:
                print(f"文件不存在或下载失败: {url} (status: {resp.status_code})")
        except Exception as e:
            print(f"下载 {fname} 时出错: {e}")

    return output_path


def download_orlm_data(output_dir: str):
    """下载 ORLM / IndustryOR 数据集 (GitHub release)"""
    output_path = Path(output_dir) / "orlm"
    output_path.mkdir(parents=True, exist_ok=True)

    github_api = "https://api.github.com/repos/AI4Math/ORLM/releases/latest"
    try:
        resp = requests.get(github_api, timeout=30)
        if resp.status_code == 200:
            release_data = resp.json()
            print(f"ORLM 最新版本: {release_data['tag_name']}")
            for asset in release_data.get("assets", []):
                print(f"  - {asset['name']}")
        else:
            print(f"无法获取 ORLM release 信息")
    except Exception as e:
        print(f"获取 ORLM release 时出错: {e}")

    print(f"\n请手动从 GitHub 下载 ORLM 数据: https://github.com/AI4Math/ORLM")
    print(f"推荐下载 IndustryOR 数据集，保存到: {output_path}")
    return output_path


def load_local_json(filepath: str) -> dict:
    """加载本地 JSON 数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_datasets(output_path: str, dataset_names: list):
    """合并多个数据集为统一格式"""
    merged = []
    for name in dataset_names:
        path = Path(output_path) / name
        if path.exists():
            # 尝试加载 HuggingFace 格式
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(path))
                for item in ds["train"]:
                    merged.append(item)
            except:
                # 尝试加载 JSON
                json_file = path / "train.json"
                if json_file.exists():
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        merged.extend(data if isinstance(data, list) else [data])
    print(f"合并了 {len(merged)} 条数据")
    return merged


def main():
    parser = argparse.ArgumentParser(description="下载 LLM4OR 训练数据")
    parser.add_argument("--output_dir", type=str, default="./data/raw",
                        help="数据保存目录")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["optmath"],
                        choices=list(HF_DATASETS.keys()) + ["nl4opt", "orlm"],
                        help="要下载的数据集")
    parser.add_argument("--download_nl4opt", action="store_true",
                        help="下载 NL4OPT 数据")
    parser.add_argument("--download_orlm", action="store_true",
                        help="下载 ORLM IndustryOR 数据")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name in args.datasets:
        if ds_name == "nl4opt":
            if args.download_nl4opt:
                download_nl4opt_from_url(args.output_dir)
        elif ds_name == "orlm":
            if args.download_orlm:
                download_orlm_data(args.output_dir)
        else:
            download_from_huggingface(ds_name, args.output_dir)

    print("\n数据下载完成！")
    print(f"数据目录: {args.output_dir}")


if __name__ == "__main__":
    main()
