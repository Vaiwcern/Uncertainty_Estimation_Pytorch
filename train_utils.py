import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim
from model.dru import druv2
import time

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from loss_factory import LossFactory
from torch.amp import autocast, GradScaler

def train(
    model: str,
    train_loader,
    input_channels: int,
    num_epoch: int,
    save_path: str,
    loss_function: str,
    norm_type: str,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    save_per_epoch: int = 5,
    rank: int = 0, 
    world_size: int = 1,
) -> None:
    # Print dataset info
    if rank == 0:
        print(f"[GPU {rank}] ✅ Starting training loop...")
        print(f"[GPU {rank}] Total images: {len(train_loader.dataset)}")
        print(f"[GPU {rank}] Steps per epoch: {len(train_loader)}")

    if model == "iterative": 
        my_model = druv2(args=None,
                                n_classes=1,              
                                steps=3,
                                hidden_size=32,         
                                feature_scale=1,
                                in_channels=3,
                                is_deconv=True,
                                is_batchnorm=True,
                                dropout_rate=dropout_rate
                            )
    else: 
        raise NotImplementedError("Model type not implemented yet")
    
    # Move model to GPU
    device = torch.device(f'cuda:{rank}')
    my_model.to(device)

    # Wrap model with DDP
    my_model = DDP(my_model, device_ids=[rank])

    # Choose the optimizer
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    loss_fn = LossFactory.get_loss_fn(loss_function)

    # Mixed precision
    scaler = GradScaler(device="cuda")


    # Initialize metrics
    accuracy_metric = torchmetrics.Accuracy(task='binary').to(device)
    iou_metric = torchmetrics.JaccardIndex(task='binary', num_classes=2).to(device)
    f1_metric = torchmetrics.F1Score(task='binary', num_classes=2).to(device)
    # auc_metric = torchmetrics.AUROC(task='binary', num_classes=2).to(device)
    # prauc_metric = torchmetrics.AveragePrecision(task='binary').to(device)


    best_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    # min_delta = 0.001    

    # Start training
    start = time.time()
    
    for epoch in range(num_epoch):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        my_model.train()  # Set model to training mode
        running_loss = 0.0

        # Reset metrics at the start of the epoch
        accuracy_metric.reset()
        iou_metric.reset()
        f1_metric.reset()
        # auc_metric.reset()
        # prauc_metric.reset()

        for i, (images, masks, _) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            s = torch.zeros_like(masks).to(device)
            h = torch.zeros(images.size(0), 512, 32, 32).to(device)

            optimizer.zero_grad()  # Clear gradients

            with autocast(device_type="cuda"):
                outputs, _ = my_model(images, h, s)  # Forward pass
                loss = 0 
                for pred in outputs: 
                    loss += loss_fn(pred, masks)  # Compute loss

            # Backward pass and optimization

            scaler.scale(loss).backward()       
            scaler.step(optimizer)              
            scaler.update()                    

            # Update the metrics
            running_loss += loss.item()

            outputs_sigmoid = torch.sigmoid(pred)  # Apply sigmoid to logits
            outputs_thresholded = outputs_sigmoid > 0.5  # Threshold to get binary predictions

            # Metrics
            accuracy_metric.update(outputs_thresholded.int(), masks.int())
            iou_metric.update(outputs_thresholded.int(), masks.int())
            f1_metric.update(outputs_thresholded.int(), masks.int())
            # auc_metric.update(outputs_sigmoid, masks.int())
            # prauc_metric.update(outputs_sigmoid, masks.int())

            # Print every `save_per_epoch` steps
            if rank == 0: 
                if i % save_per_epoch == 0:
                    print(f"[GPU {rank}] Epoch [{epoch+1}/{num_epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.10f}", flush=True)

        # Print summary of metrics for the epoch
        print(f"[GPU {rank}] Epoch [{epoch+1}/{num_epoch}] | Loss: {running_loss / len(train_loader):.10f} "
              f"| Acc: {accuracy_metric.compute():.4f} "
              f"| IoU: {iou_metric.compute():.4f} "
              f"| F1: {f1_metric.compute():.4f} " )
            #   f"| AUC: {auc_metric.compute():.4f} "
            #   f"| PRAUC: {prauc_metric.compute():.4f}")

        # Save the model every `save_per_epoch` epochs
        if rank == 0:
            if (epoch + 1) % save_per_epoch == 0:
                torch.save(my_model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))

        avg_loss = running_loss / len(train_loader)
        if rank == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                torch.save(my_model.state_dict(), os.path.join(save_path, f"best_model.pth"))
                print(f"[GPU {rank}] ✅ New best loss: {best_loss:.10f} (model saved)")
            else:
                epochs_no_improve += 1
                print(f"[GPU {rank}] 🔁 No improvement in loss for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= patience:
                print(f"[GPU {rank}] 🛑 Early stopping triggered at epoch {epoch+1}")
                break
        
        scheduler.step(avg_loss)  

        torch.cuda.empty_cache()

    end = time.time()
    print(f"⏱️ Total time training: {(end - start):.2f} seconds")

    dist.barrier() 
