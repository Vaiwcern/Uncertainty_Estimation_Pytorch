import os
import argparse
import yaml

from evaluation_utils import segmentation_evaluation, calibration_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Unet model.')

    parser.add_argument('--metric_type', type=str, required=True,
        help="Evaluation type. Options: 'segmentation', 'calibration', 'out-of-distribution'")

    parser.add_argument('--prediction_path', type=str, required=True,
        help="Directory that saved predictions.")

    parser.add_argument('--epoch', type=int, required=True,
        help="The epoch at which model is evaluated.")

    parser.add_argument('--model_type', type=str, required=True,
        help="The model type: 'iterative', 'vanila'.")

    parser.add_argument('--samples', type=int, required=True,
        help="Number of stochastic samples (e.g., for MC dropout).")

    parser.add_argument('--relaxed_ccq', type=lambda x: x.lower() == 'true', choices=[True, False], required=True,
        help="Use relaxed CCQ metric (with slack): 'true', 'false'.")

    parser.add_argument('--n_rows', type=int, required=False,
        help="Cut images into n_rows x n_cols for calibration evaluation.")

    parser.add_argument('--n_cols', type=int, required=False,
        help="Cut images into n_rows x n_cols for calibration evaluation.")

    parser.add_argument('--save_path', type=str, required=True,
        help="Directory to save evaluation results.")

    parser.add_argument('--bad_sample_path', type=str, required=True,
        help="Directory to save bad samples.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Constraint
    if args.metric_type == "calibration":
        if not args.n_rows or not args.n_cols:
            raise ValueError("Arguments '--n_rows' and '--n_cols' are required for calibration estimation.")

    # Path
    pred_path = os.path.join(args.prediction_path, f"epoch_{args.epoch}")
    save_path = os.path.join(args.save_path, f"epoch_{args.epoch}")
    bad_sample_path = os.path.join(args.bad_sample_path, f"epoch_{args.epoch}")

    # === Log setting ===
    os.makedirs(save_path, exist_ok=True)
    setting_log_path = os.path.join(save_path, f"setting_{args.metric_type}_evaluation.yaml")

    with open(setting_log_path, "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # === Segmentation Evaluation ===
    if args.metric_type == 'segmentation':
        segmentation_evaluation(
            pred_dir=pred_path,
            model_type=args.model_type,
            samples=args.samples,
            relax=args.relaxed_ccq,
            save_path=save_path,
        )
    elif args.metric_type == 'calibration':
        calibration_evaluation(
            pred_dir=pred_path,
            model_type=args.model_type,
            samples=args.samples,
            n_rows=args.n_rows, 
            n_cols=args.n_cols,
            save_path=save_path,
            bad_sample_path=bad_sample_path
        )
    elif args.metric_type == 'out-of-distribution':
        raise NotImplementedError("OOD evaluation is not yet implemented.")
    else:
        raise ValueError(f"Unsupported evaluation type: {args.eval_type}")
