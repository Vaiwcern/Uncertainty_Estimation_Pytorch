import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from datetime import datetime
import yaml

from predict_utils import predict_and_save_results, merge_prediction_results
from custom_dataset.dataset_factory import DatasetFactory

import argparse
import torch
import torch.multiprocessing as mp
import logging
from logging.handlers import RotatingFileHandler
from filelock import FileLock  # pip install filelock

from ddp_setup import ddp_setup, find_free_port, ddp_cleanup

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
🔍 Predict using trained iterative U-Net model (PyTorch, multi-GPU supported)

This script loads a checkpoint of the iterative U-Net model and performs prediction
on the test set with optional MC Dropout (via training mode).
Supports iterative refinement, stochastic sampling, and automatic saving of predicted masks.
"""
    )

    parser.add_argument('--dataset', type=str, required=True,
        help="Name of dataset: options = ['RT', 'Mass', 'Drive', 'Nuclei']")
    
    parser.add_argument('--dataset_path', type=str, required=True,
        help="Path to dataset directory.")
    
    parser.add_argument('--model_path', type=str, required=True,
        help="Path to the saved model directory containing 'setting.yaml' and weights.")
    
    parser.add_argument('--epoch', type=int, required=True,
        help="Epoch number of the checkpoint to load (e.g., 25 will load 'model_epoch_25.pth').")
    
    parser.add_argument('--save_path', type=str, required=True,
        help="Directory to save predicted masks and logs.")
    
    parser.add_argument('--training_mode', type=lambda x: x.lower() == 'true', choices=[True, False], required=True,
        help="Enable training mode during prediction (dropout & batchnorm active). "
             "Set to True to apply MC Dropout (samples > 1).")

    parser.add_argument('--batch_size', type=int, required=True,
        help="Batch size for prediction per GPU.")

    # parser.add_argument('--iterative', type=int, required=True,
    #     help="Number of iterative refinement steps per sample (e.g., 3).")

    parser.add_argument('--samples', type=int, required=True,
        help="Number of MC samples per input. Must be > 1 if training_mode=True; must be 1 if False.")
    
    parser.add_argument('--gpus', type=str, required=True,
        help="Comma-separated GPU IDs to use. Example: '0,1,2'")

    args = parser.parse_args()

    # 🚨 Check logic between training_mode and samples
    if args.training_mode:
        assert args.samples > 1, "🚨 If training_mode=True (MC Dropout), samples must be > 1"
    else:
        assert args.samples == 1, "🚨 If training_mode=False (deterministic), samples must be exactly 1"

    return args

def main_worker(rank, args, world_size):
    # === Logger for each rank ===
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    log_dir = os.path.join(args.save_path, "predict_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"epoch{args.epoch}_rank{rank}__{timestamp}.log")
    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout

    # Set up DDP
    ddp_setup(rank, world_size)
    print(f"[GPU {rank}] Spawned process started")
    print(f"[GPU {rank}] Local batch size: {args.batch_size} → Global batch size: {args.batch_size * world_size}")

    # === Load dataset ===
    data_loader = DatasetFactory.get_test_loader(
        name=args.dataset,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        add_channel=False,  
        num_workers=8,
        distributed=True
    )

    # === Predict and Save ===
    predict_and_save_results(
        model_path=args.model_path,
        epoch=args.epoch,
        data_loader=data_loader,
        save_path=args.save_path,
        training=args.training_mode,
        samples=args.samples,
        rank=rank,
        world_size=world_size, 
    )

    ddp_cleanup()

if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.makedirs(args.save_path, exist_ok=True)

    # === Save config ===
    setting_log_path = os.path.join(args.save_path, "setting.yaml")
    with open(setting_log_path, "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # === Redirect logs ===
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    log_dir = os.path.join(args.save_path, "predict_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"predict__epoch{args.epoch}__{timestamp}.log")
    log_file = open(log_file_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    # === Set up distributed training === 
    gpu_ids = list(map(int, args.gpus.split(",")))
    world_size = len(gpu_ids)
    
    port = find_free_port()
    os.environ["MASTER_PORT"] = str(port)

    # === Predicting === 
    mp.spawn(main_worker, args=(args, world_size), nprocs=world_size, join=True)
    
    
    # === Logger for merge ===
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    log_dir = os.path.join(args.save_path, "predict_logs")
    log_file_path = os.path.join(log_dir, f"epoch{args.epoch}_merge__{timestamp}.log")
    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout

    merge_prediction_results(args.save_path, args.epoch, world_size)
    