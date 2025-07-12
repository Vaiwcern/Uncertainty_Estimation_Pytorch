import os
import torch
import numpy as np
from tqdm import tqdm
import imageio.v3 as imageio
import yaml
from concurrent.futures import ProcessPoolExecutor
from torch.nn.parallel import DistributedDataParallel as DDP

from model.dru import druv2  # Replace if using different model
import shutil
from glob import glob
import cv2


def save_sample_wrapper(args):
    return save_sample(*args)

def load_yaml_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def load_model_from_folder(folder_path, epoch):
    config = load_yaml_config(os.path.join(folder_path, "setting.yaml"))

    if config['model'] == "iterative": 
        model = druv2(
            args=None,
            n_classes=1,
            steps=config.get("steps", 3),
            hidden_size=32,
            in_channels=3 + int(config.get("add_channel", False)),
            is_batchnorm=True,
            dropout_rate=config.get("dropout_rate", 0.0),
        )
    else: 
        raise NotImplementedError(f"Model type {config['model']} not implemented yet!")

    weight_path = os.path.join(folder_path, f"model_epoch_{epoch}.pth")
    state_dict = torch.load(weight_path, map_location="cpu")

    # 👉 Strip 'module.' prefix if exists
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model


def save_sample(save_path, filename, image_tensor, mask_tensor, preds):
    base = os.path.splitext(filename)[0]

    # Chuyển input từ tensor về numpy (RGB)
    input_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    input_np = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)  # cv2 dùng BGR

    # Chuyển mask
    mask_np = (mask_tensor.squeeze().numpy() * 255).astype(np.uint8)

    # Lưu input và mask
    cv2.imwrite(os.path.join(save_path, f"{base}_input.png"), input_np)
    cv2.imwrite(os.path.join(save_path, f"{base}_mask.png"), mask_np)

    # Lưu từng prediction
    for sample_idx, sample_preds in enumerate(preds):
        for iter_idx, pred in enumerate(sample_preds):
            pred_np = (np.clip(pred[0], 0, 1) * 255).astype(np.uint8)
            if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                pred_np = pred_np[0]  # (1, H, W) → (H, W)
            fname = f"{base}_sample{sample_idx}_iter{iter_idx}.png"
            cv2.imwrite(os.path.join(save_path, fname), pred_np)


def predict_and_save_results(model_path, epoch, data_loader, save_path, training, samples, rank, world_size):
    rank_save_path = os.path.join(save_path, f"epoch_{epoch}", f"rank_{rank}")
    os.makedirs(rank_save_path, exist_ok=True)

    model = load_model_from_folder(model_path, epoch)
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model.train() if training else model.eval()

    results_to_save = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"[GPU {rank}] 🔍 Predicting"):
            images, masks, filenames = batch
            images = images.to(device)              # [B, C, H, W]
            masks = masks.cpu()                     # keep masks on CPU for saving

            B, _, H, W = images.shape
            preds_per_batch = [[] for _ in range(B)]  # list of [samples][steps] for each sample

            for _ in range(samples):
                h = torch.zeros(B, 512, 32, 32).to(device)
                s = torch.zeros(B, 1, H, W).to(device)

                list_st, _ = model(images, h, s)  # list of [B, 1, H, W] per step

                # Apply sigmoid and move to CPU
                list_st_cpu = [torch.sigmoid(x).cpu().numpy() for x in list_st]

                # Append each step's output per sample
                for i in range(B):
                    steps_per_sample = [list_st_cpu[t][i] for t in range(len(list_st_cpu))]  # [steps]
                    preds_per_batch[i].append(steps_per_sample)  # append 1 sample → [samples][steps]

            # Save results for all samples in batch
            for i in range(B):
                results_to_save.append((
                    rank_save_path,
                    filenames[i],
                    images[i].cpu(),
                    masks[i],
                    preds_per_batch[i],  # shape: [samples][steps]
                ))

    # Save all predicted samples
    with ProcessPoolExecutor(max_workers=46//world_size) as executor:
        list(tqdm(executor.map(save_sample_wrapper, results_to_save), total=len(results_to_save), desc=f"[GPU {rank}] 💾 Saving"))

def merge_prediction_results(save_path, epoch, world_size):
    """
    Merge prediction results from rank_* folders into epoch_{epoch}/,
    then delete the rank_* folders.
    """
    epoch_path = os.path.join(save_path, f"epoch_{epoch}")
    merged_count = 0

    for rank_id in range(world_size):
        rank_folder = os.path.join(epoch_path, f"rank_{rank_id}")
        if not os.path.exists(rank_folder):
            continue

        for file_path in glob(os.path.join(rank_folder, "*.png")):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(epoch_path, filename)

            if os.path.exists(dest_path):
                print(f"⚠️ Skipping duplicate file: {filename}")
                continue

            shutil.move(file_path, dest_path)
            merged_count += 1

        shutil.rmtree(rank_folder)
        print(f"🧹 Removed: {rank_folder}")

    print(f"✅ Merged all rank folders into {epoch_path}")
    print(f"📦 Total merged predictions: {merged_count}")
