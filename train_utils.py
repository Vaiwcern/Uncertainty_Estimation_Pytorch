import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from model.unet import Unet
from evaluation.metric import IoUMetric, F1ScoreMetric, AUCMetric, PRAUCMetric  
import time

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
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> None:
    # Print dataset info
    print("Total images:", len(train_loader.dataset))
    print("Steps per epoch:", len(train_loader))
 
    my_model = Unet(model_type=model, input_channels=input_channels, dropout_rate=dropout_rate, norm_type=norm_type)

    # Move model to GPU
    my_model.to(device)
    
    # Choose the optimizer
    optim = optim.Adam(my_model.parameters(), lr=learning_rate)
    
    if loss_function == 'focal': 
        loss_fn = CustomLosses.focal_loss(gamma=2.0, from_logits=True)
    elif loss_function == 'bce': 
        loss_fn = CustomLosses.binary_crossentropy_loss(from_logits=True)
    elif loss_function == 'dice': 
        loss_fn = CustomLosses.dice_loss(from_logits=True)
    elif loss_function == 'dice_bce': 
        loss_fn = CustomLosses.dice_bce_loss(from_logits=True)
    elif loss_function == 'dice_focal': 
        loss_fn = CustomLosses.dice_focal_loss(from_logits=True)
    elif loss_function == 'iou': 
        loss_fn = CustomLosses.iou_loss(from_logits=True)
    else: 
        raise ValueError("Loss function not supported.")

    # Start training
    start = time.time()
    
    for epoch in range(num_epoch):
        my_model.train()  # Set model to training mode
        running_loss = 0.0
        running_metrics = {
            'acc': 0,
            'iou': 0,
            'f1': 0,
            'auc': 0,
            'prauc': 0
        }

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optim.zero_grad()  # Clear gradients
            outputs = my_model(images)  # Forward pass
            loss = loss_fn(outputs, masks)  # Compute loss

            # Backward pass and optimization
            loss.backward()
            optim.step()

            # Update the metrics
            running_loss += loss.item()

            # Calculate metrics
            # You need to implement these metrics as PyTorch functions (IoUMetric, F1ScoreMetric, etc.)
            acc = IoUMetric(outputs, masks)
            iou = IoUMetric(outputs, masks)
            f1 = F1ScoreMetric(outputs, masks)
            auc = AUCMetric(outputs, masks)
            prauc = PRAUCMetric(outputs, masks)

            running_metrics['acc'] += acc.item()
            running_metrics['iou'] += iou.item()
            running_metrics['f1'] += f1.item()
            running_metrics['auc'] += auc.item()
            running_metrics['prauc'] += prauc.item()

            # Log every `save_per_epoch` steps
            if i % save_per_epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Print summary of metrics for the epoch
        print(f"Epoch [{epoch+1}/{num_epoch}] | Loss: {running_loss / len(train_loader):.4f} "
              f"| Acc: {running_metrics['acc'] / len(train_loader):.4f} "
              f"| IoU: {running_metrics['iou'] / len(train_loader):.4f} "
              f"| F1: {running_metrics['f1'] / len(train_loader):.4f} "
              f"| AUC: {running_metrics['auc'] / len(train_loader):.4f} "
              f"| PRAUC: {running_metrics['prauc'] / len(train_loader):.4f}")

        # Save the model every `save_per_epoch` epochs
        if (epoch + 1) % save_per_epoch == 0:
            torch.save(my_model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))

    end = time.time()
    print(f"⏱️ Total time training: {(end - start):.2f} seconds")

class CustomLosses:
    @staticmethod
    def binary_crossentropy_loss(from_logits=True):
        """
        Binary Cross-Entropy Loss with optional sigmoid application.
        If from_logits=True, the loss applies sigmoid internally.
        """
        if from_logits:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.BCELoss()

    @staticmethod
    def dice_loss(pred, target, from_logits=True, smooth=1e-6):
        """
        Dice Loss with optional sigmoid application.
        If from_logits=True, sigmoid will be applied internally.
        """
        if from_logits:
            pred = torch.sigmoid(pred)

        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        return 1 - (2 * intersection + smooth) / (union + smooth)

    @staticmethod
    def dice_bce_loss(pred, target, from_logits=True, smooth=1e-6):
        """
        Dice Loss + Binary Cross-Entropy Loss with optional sigmoid application.
        """
        bce_loss = CustomLosses.binary_crossentropy_loss(from_logits)(pred, target)
        dice_loss = CustomLosses.dice_loss(pred, target, from_logits, smooth)
        return bce_loss + dice_loss

    @staticmethod
    def dice_focal_loss(pred, target, gamma=2.0, from_logits=True, smooth=1e-6):
        """
        Dice Loss + Focal Loss with optional sigmoid application.
        """
        if from_logits:
            # Không cần áp dụng sigmoid ở đây, vì BCEWithLogitsLoss đã xử lý.
            bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
        else:
            bce_loss = nn.BCELoss(reduction='none')(torch.sigmoid(pred), target)
        
        focal_loss = torch.pow(1 - torch.sigmoid(pred), gamma) * bce_loss
        focal_loss = torch.mean(focal_loss)

        # Dice Loss
        dice_loss = CustomLosses.dice_loss(pred, target, from_logits, smooth)

        return focal_loss + dice_loss


    @staticmethod
    def iou_loss(pred, target, from_logits=True, smooth=1e-6):
        """
        Intersection over Union (IoU) Loss with optional sigmoid application.
        """
        if from_logits:
            pred = torch.sigmoid(pred)

        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        return 1 - (intersection + smooth) / (union + smooth)

    @staticmethod
    def focal_loss(gamma=2.0, from_logits=True):
        """
        Binary Focal Loss with optional sigmoid application.
        """
        def focal_loss_fn(pred, target):
            if from_logits:
                bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
            else:
                bce_loss = nn.BCELoss(reduction='none')(torch.sigmoid(pred), target)
            
            focal_loss = torch.pow(1 - torch.sigmoid(pred), gamma) * bce_loss
            return torch.mean(focal_loss)
        return focal_loss_fn
