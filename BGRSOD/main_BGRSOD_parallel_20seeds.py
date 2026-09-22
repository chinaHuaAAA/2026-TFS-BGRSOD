"""BGRSOD grid experiment with seed-level parallelism.

The calculation and result-file structure are the same as in the original
runner.  For every (omega, alpha) pair, seeds 1--20 are evaluated in parallel
with independent worker processes.  Results are reordered by seed before the
mean, standard deviation, detailed table, and pickle file are written.

Run this file from the Outlier_Detection project root.  BGRSOD.py and its
dependencies are expected in ./OD, and only ./mydata6/*.mat is processed.
"""

from __future__ import annotations

import os

# Prevent every worker process from starting another full set of BLAS threads.
# These variables must be set before importing NumPy/SciPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import importlib
import itertools
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
F_NAME = "BGRSOD"
DATA_DIR = "Datasets"
RESULT_ROOT = "Result"
RANDOM_SEEDS = list(range(1, 21))

# Four processes is a conservative default.  Increase it only when RAM is
# sufficient; each process has its own BGRSOD intermediate arrays.
N_JOBS = min(4, os.cpu_count() or 1)

# Arrays larger than this are memory-mapped by joblib instead of being copied
# separately for every submitted seed task.
MAX_NBYTES = "10M"

CURRENT_DIR = os.getcwd()
OD_DIR = os.path.join(CURRENT_DIR, "OD")
if OD_DIR not in sys.path:
    sys.path.insert(0, OD_DIR)


def sort_files_by_size(folder_path: str):
    """Return .mat files sorted from small to large."""
    all_files = (
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(".mat")
        and os.path.isfile(os.path.join(folder_path, file))
    )
    files_with_size = [(path, os.path.getsize(path)) for path in all_files]
    return sorted(files_with_size, key=lambda item: item[1])


def check_if_processed(file_path: str, result_path: str, algorithm_name: str):
    """Skip only results that really contain the requested 20 seeds."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    stats_excel = os.path.join(
        result_path, algorithm_name, base_name + "_stats.xlsx"
    )
    pkl_file = os.path.join(result_path, algorithm_name, base_name + ".pkl")
    detailed_excel = os.path.join(
        result_path, algorithm_name, base_name + "_detailed.xlsx"
    )

    if not all(
        os.path.exists(path)
        for path in (stats_excel, detailed_excel, pkl_file)
    ):
        return False

    try:
        # The detailed workbook is small. Checking it avoids loading a possibly
        # very large pkl containing every saved outlier-score vector.
        details = pd.read_excel(
            detailed_excel,
            usecols=["Param1", "Param2", "Random_Seed"],
        )
        groups = details.groupby(["Param1", "Param2"], sort=False)
        complete = len(groups) > 0 and all(
            sorted(group["Random_Seed"].astype(int).tolist()) == RANDOM_SEEDS
            for _, group in groups
        )
        if complete:
            print(f"数据集 {base_name} 已有完整20次结果，跳过...")
            return True
        print(f"数据集 {base_name} 的旧结果不是完整20次，将重新计算...")
    except Exception as error:
        print(f"数据集 {base_name} 的旧结果无法验证，将重新计算: {error}")
    return False


def run_one_seed(
    trandata: np.ndarray,
    labels: np.ndarray,
    p1: int,
    p2: float,
    random_seed: int,
):
    """Worker task for one parameter pair and one random seed."""
    # Explicitly restore the local module path inside Windows worker processes.
    if OD_DIR not in sys.path:
        sys.path.insert(0, OD_DIR)
    module = importlib.import_module(F_NAME)
    detector = getattr(module, F_NAME)

    start = time.perf_counter()
    scores = detector(
        trandata,
        omega=p1,
        alpha=p2,
        random_state=random_seed,
    )
    elapsed = time.perf_counter() - start

    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size != labels.size:
        raise ValueError(
            f"seed={random_seed}: returned {scores.size} scores for "
            f"{labels.size} samples"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"seed={random_seed}: anomaly scores contain NaN or Inf")

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = float(auc(fpr, tpr))
    return random_seed, auc_score, scores, elapsed


def run_seeds_in_parallel(
    trandata: np.ndarray,
    labels: np.ndarray,
    p1: int,
    p2: float,
):
    """Run all configured seeds concurrently and return seed-ordered results."""
    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        batch_size=1,
        pre_dispatch=N_JOBS,
        max_nbytes=MAX_NBYTES,
        mmap_mode="r",
    )(
        delayed(run_one_seed)(trandata, labels, p1, p2, seed)
        for seed in RANDOM_SEEDS
    )
    results.sort(key=lambda item: item[0])
    return results


def main():
    os.makedirs(os.path.join(RESULT_ROOT, F_NAME), exist_ok=True)

    # Import once in the parent process to fail early if OD/BGRSOD.py or one of
    # its dependencies cannot be found. Workers import the same module locally.
    module = importlib.import_module(F_NAME)
    if not callable(getattr(module, F_NAME, None)):
        raise ImportError(f"OD/{F_NAME}.py does not define callable {F_NAME}")

    param1 = 2 ** np.arange(1, 7)
    param2 = np.arange(0, 1.1, 0.1)
    param_grid = list(itertools.product(param1, param2))

    print(f"参数网格大小: {len(param_grid)} 个参数组合")
    print(f"每个参数组合重复: {len(RANDOM_SEEDS)} 次")
    print(f"并行进程数: {N_JOBS}")
    print("如果内存占用过高，请把代码顶部的 N_JOBS 改为 2。\n")

    sorted_files = sort_files_by_size(DATA_DIR)
    if not sorted_files:
        raise FileNotFoundError(f"{DATA_DIR} 中没有找到 .mat 数据集")

    for file_path, _ in sorted_files:
        if check_if_processed(file_path, RESULT_ROOT, F_NAME):
            continue

        try:
            mat_data = sio.loadmat(file_path)
            if "trandata" not in mat_data:
                raise KeyError("MAT文件中不存在变量 trandata")

            data = np.asarray(mat_data["trandata"])
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError(
                    "trandata至少需要一列特征和一列标签，"
                    f"当前形状为 {data.shape}"
                )

            trandata = np.asarray(data[:, :-1], dtype=float)
            labels = np.asarray(data[:, -1]).reshape(-1)
            if not np.isfinite(trandata).all():
                raise ValueError("特征中包含NaN或Inf")
            if not np.isfinite(labels.astype(float)).all():
                raise ValueError("标签中包含NaN或Inf")

            trandata = MinMaxScaler().fit_transform(trandata)
            print(f"----------- {file_path} -----------\n")

            all_auc = []
            all_scores = []
            all_times = []
            param_combinations = []

            for p1, p2 in param_grid:
                wall_start = time.perf_counter()
                seed_results = run_seeds_in_parallel(
                    trandata,
                    labels,
                    int(p1),
                    float(p2),
                )
                wall_time = time.perf_counter() - wall_start

                auc_scores_for_param = [item[1] for item in seed_results]
                score_runs_for_param = [item[2] for item in seed_results]
                time_for_param = [item[3] for item in seed_results]

                all_auc.append(auc_scores_for_param)
                all_scores.append(score_runs_for_param)
                all_times.append(time_for_param)
                param_combinations.append((int(p1), float(p2)))

                print(
                    "p1=%.2f, p2=%.2f, 平均AUC=%.4f, 标准差=%.4f, "
                    "并行墙钟时间=%.2fs"
                    % (
                        p1,
                        p2,
                        np.mean(auc_scores_for_param),
                        np.std(auc_scores_for_param),
                        wall_time,
                    )
                )

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = os.path.join(RESULT_ROOT, F_NAME)

            stats_data = []
            for index, (p1, p2) in enumerate(param_combinations):
                stats_data.append(
                    {
                        "Param1": p1,
                        "Param2": p2,
                        "AUC_mean": np.mean(all_auc[index]),
                        "AUC_std": np.std(all_auc[index]),
                        "AUC_max": max(all_auc[index]),
                        "AUC_min": min(all_auc[index]),
                        "Time_mean": np.mean(all_times[index]),
                        "Time_std": np.std(all_times[index]),
                    }
                )

            pd.DataFrame(stats_data).to_excel(
                os.path.join(output_dir, base_name + "_stats.xlsx"),
                index=False,
                header=True,
            )

            # The original code still used range(1, 6) here.  This version
            # correctly writes all 20 seeds.
            detailed_data = []
            for index, (p1, p2) in enumerate(param_combinations):
                for seed_index, seed in enumerate(RANDOM_SEEDS):
                    detailed_data.append(
                        {
                            "Param1": p1,
                            "Param2": p2,
                            "Random_Seed": seed,
                            "AUC": all_auc[index][seed_index],
                            "Time": all_times[index][seed_index],
                        }
                    )

            pd.DataFrame(detailed_data).to_excel(
                os.path.join(output_dir, base_name + "_detailed.xlsx"),
                index=False,
                header=True,
            )

            data_dict = {
                "AUC": all_auc,
                "OF": all_scores,
                "Time": all_times,
                "param1": [pair[0] for pair in param_combinations],
                "param2": [pair[1] for pair in param_combinations],
                "param_combinations": param_combinations,
                "random_seeds": RANDOM_SEEDS,
            }
            with open(os.path.join(output_dir, base_name + ".pkl"), "wb") as stream:
                pickle.dump(data_dict, stream, protocol=pickle.HIGHEST_PROTOCOL)

            average_aucs = [np.mean(values) for values in all_auc]
            best_index = int(np.argmax(average_aucs))
            best_p1, best_p2 = param_combinations[best_index]
            print(
                "%s 最佳参数 p1=%.2f, p2=%.2f, 平均AUC=%.4f, 标准差=%.4f\n"
                % (
                    file_path,
                    best_p1,
                    best_p2,
                    average_aucs[best_index],
                    np.std(all_auc[best_index]),
                )
            )

        except Exception as error:
            print(f"处理文件 {file_path} 时出错: {error}")
            continue


if __name__ == "__main__":
    main()
