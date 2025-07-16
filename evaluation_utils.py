import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import pandas as pd
import glob


from metric import compute_ccq, compute_ccq_normal, f1_score
from metric import get_uncertainty_by_std, get_uncertainty_by_var, get_error_by_abs, get_error_by_mse
from metric import rAULC, corr
from metric import get_uncertainty_std
from plot import plot_corr_rAULC, plot_uncertainty_error_bar, save_as_heatmap


# === Process one sample ===
def process_single_segmentation(args) -> Tuple[float, float, float, float]:
    base_name, pred_dir, model_type, samples, relax = args

    if model_type == 'iterative': 
        postfix = "iter2"
    elif model_type == 'vanila': 
        postfix = "iter0"
    else: 
        raise ValueError(f"Error: The model type {model_type} is not supported yet!")

    pred_stack = []
    for sample_id in range(samples):
        path = os.path.join(pred_dir, f"{base_name}_sample{sample_id}_{postfix}.png")
        arr = np.array(Image.open(path)).astype(np.float32) / 255.0
        pred_stack.append(arr)

    pred_mean = np.mean(pred_stack, axis=0)

    mask_path = os.path.join(pred_dir, f"{base_name}_mask.png")
    mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0

    if relax:
        corr, comp, qual = compute_ccq(pred_mean, mask, threshold=0.5, slack=5)
    else:
        corr, comp, qual = compute_ccq_normal(pred_mean, mask, threshold=0.5)

    corr_f1, comp_f1, _ = compute_ccq_normal(pred_mean, mask, threshold=0.5)
    f1 = f1_score(corr_f1, comp_f1)

    return corr, comp, qual, f1

# === Main Evaluation Function ===
def segmentation_evaluation(pred_dir: str, model_type: str, samples: int, relax: bool, save_path: str):
    all_files = os.listdir(pred_dir)
    base_names = sorted(set(
        "_".join(f.split("_")[:-1])  # from <name>_mask.png
        for f in all_files if f.endswith("_mask.png")
    ))

    args_list = [(bn, pred_dir, model_type, samples, relax) for bn in base_names]
    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_segmentation, args) for args in args_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(future.result())

    # Tính trung bình cộng cho tất cả mẫu
    results_np = np.array(results)
    mean_result = np.mean(results_np, axis=0)

    df = pd.DataFrame([mean_result], columns=["correctness", "completeness", "quality", "f1_score"])
    csv_path = os.path.join(save_path, "segmentation_evaluation.csv")
    df.to_csv(csv_path, index=False)

def process_single_calibration(args):
    base_name, pred_dir, model_type, samples, n_rows, n_cols = args
    pred_list = []

    if model_type == 'iterative':
        n_iters = 3
        if samples == 1:
            for i in range(n_iters):
                path = os.path.join(pred_dir, f"{base_name}_sample0_iter{i}.png")
                arr = np.array(Image.open(path)).astype(np.float32) / 255.0
                pred_list.append(arr)
            pred_stack = np.array(pred_list)
        else:
            iter_means = []
            for i in range(n_iters):
                sample_stack = []
                for s in range(samples):
                    path = os.path.join(pred_dir, f"{base_name}_sample{s}_iter{i}.png")
                    arr = np.array(Image.open(path)).astype(np.float32) / 255.0
                    sample_stack.append(arr)
                iter_mean = np.mean(sample_stack, axis=0)
                iter_means.append(iter_mean)
            pred_stack = np.array(iter_means)
    else:
        for s in range(samples):
            path = os.path.join(pred_dir, f"{base_name}_sample{s}_iter0.png")
            arr = np.array(Image.open(path)).astype(np.float32) / 255.0
            pred_list.append(arr)
        pred_stack = np.array(pred_list)

    mask_path = os.path.join(pred_dir, f"{base_name}_mask.png")
    mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
    pred_mean = np.mean(pred_stack, axis=0)

    unc_std = get_uncertainty_std(pred_stack)
    unc_path = os.path.join(pred_dir, f"{base_name}_uncertainty.png")
    save_as_heatmap(unc_std, unc_path, cmap='plasma')

    # Calculate metrics
    result = []
    result.append(("var", "mse", get_uncertainty_by_var(pred_stack, 0, n_rows, n_cols),
                                   get_error_by_mse(pred_mean, mask, n_rows, n_cols), base_name))
    
    result.append(("var", "abs", get_uncertainty_by_var(pred_stack, 0, n_rows, n_cols),
                                  get_error_by_abs(pred_mean, mask, n_rows, n_cols), base_name))
    
    result.append(("std", "mse", get_uncertainty_by_std(pred_stack, 0, n_rows, n_cols),
                                  get_error_by_mse(pred_mean, mask, n_rows, n_cols), base_name))
    
    result.append(("std", "abs", get_uncertainty_by_std(pred_stack, 0, n_rows, n_cols),
                                  get_error_by_abs(pred_mean, mask, n_rows, n_cols), base_name))

    return result  # list of tuples


def calibration_evaluation(pred_dir: str, model_type: str, samples: int,
                           n_rows: int, n_cols: int, save_path: str, bad_sample_path: str):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(bad_sample_path, exist_ok=True)

    all_files = os.listdir(pred_dir)
    base_names = sorted(set(
        "_".join(f.split("_")[:-1])
        for f in all_files if f.endswith("_mask.png")
    ))

    args_list = [(bn, pred_dir, model_type, samples, n_rows, n_cols) for bn in base_names]

    result_dict = {
        ("var", "mse"): {"uncs": [], "errors": [], "names": []},
        ("var", "abs"): {"uncs": [], "errors": [], "names": []},
        ("std", "mse"): {"uncs": [], "errors": [], "names": []},
        ("std", "abs"): {"uncs": [], "errors": [], "names": []},
    }

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_calibration, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Calibration"):
            result_list = future.result()
            for unc_type, err_type, uncs, errors, name in result_list:
                result_dict[(unc_type, err_type)]["uncs"].extend(uncs)
                result_dict[(unc_type, err_type)]["errors"].extend(errors)
                result_dict[(unc_type, err_type)]["names"].extend([name] * len(uncs))

    results_summary = []

    for (unc_type, err_type), data in result_dict.items():
        uncs = data["uncs"]
        errors = data["errors"]
        names = data["names"]

        # Summary
        score_rAULC = rAULC(uncs, errors)
        score_corr = corr(uncs, errors)
        results_summary.append([unc_type, err_type, score_rAULC, score_corr])

        # Plot
        plot_corr_rAULC(uncs, errors,
            title=f"{unc_type.upper()} vs {err_type.upper()}",
            filename=f"scatter_{unc_type}_{err_type}.png",
            SAVE_PATH=save_path)

        plot_uncertainty_error_bar(
            uncs=uncs,
            errors=errors,
            save_path=save_path,
            filename=f"barplot_{unc_type}_{err_type}.png",
            max_samples=4000
        )

        # Bad sample detection only for std + abs
        if (unc_type, err_type) == ("std", "abs"):
            unc_thresh = np.percentile(uncs, 30)
            err_thresh = np.percentile(errors, 70)

            save_bad_samples(
                base_names=names,
                uncs=uncs,
                errors=errors,
                pred_dir=pred_dir,
                bad_sample_path=bad_sample_path,
                unc_thresh=unc_thresh,
                err_thresh=err_thresh
            )

    # Summary CSV
    df = pd.DataFrame(results_summary, columns=["unc_type", "error_type", "rAULC", "correlation"])
    df.to_csv(os.path.join(save_path, "calibration_evaluation.csv"), index=False)


def save_bad_samples(
    base_names, uncs, errors,
    pred_dir, bad_sample_path,
    unc_thresh, err_thresh
):
    os.makedirs(bad_sample_path, exist_ok=True)
    bad_entries = []

    for base_name, unc, err in zip(base_names, uncs, errors):
        if unc < unc_thresh and err > err_thresh:
            # Copy input & mask
            for suffix, newname in [
                (".png", "_input.png"),
                ("_mask.png", "_mask.png"),
                ("_uncertainty.png", "_unc.png")
            ]:
                src = os.path.join(pred_dir, base_name + suffix)
                dst = os.path.join(bad_sample_path, base_name + newname)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)

            # Copy all outputs matching _sampleX_iterY.png
            output_files = glob.glob(os.path.join(pred_dir, f"{base_name}_sample*_iter*.png"))
            for file_path in output_files:
                filename = os.path.basename(file_path)
                dst = os.path.join(bad_sample_path, filename)
                shutil.copyfile(file_path, dst)

            # Record bad entry
            bad_entries.append([base_name, unc, err])

    # Save bad.csv
    df = pd.DataFrame(bad_entries, columns=["base_name", "uncertainty", "error"])
    df.to_csv(os.path.join(bad_sample_path, "bad.csv"), index=False)
