# custom_loss.py
import torch
import torch.nn as nn

class CustomLosses:
    @staticmethod
    def binary_crossentropy_loss(from_logits=True):
        return nn.BCEWithLogitsLoss() if from_logits else nn.BCELoss()

    @staticmethod
    def dice_loss(pred, target, from_logits=True, smooth=1e-6):
        if from_logits:
            pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        return 1 - (2 * intersection + smooth) / (union + smooth)

    @staticmethod
    def dice_bce_loss(pred, target, from_logits=True, smooth=1e-6):
        bce = CustomLosses.binary_crossentropy_loss(from_logits)(pred, target)
        dice = CustomLosses.dice_loss(pred, target, from_logits, smooth)
        return bce + dice

    @staticmethod
    def dice_focal_loss(pred, target, gamma=2.0, from_logits=True, smooth=1e-6):
        bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target) if from_logits else nn.BCELoss(reduction='none')(torch.sigmoid(pred), target)
        focal = torch.pow(1 - torch.sigmoid(pred), gamma) * bce
        focal = torch.mean(focal)
        dice = CustomLosses.dice_loss(pred, target, from_logits, smooth)
        return focal + dice

    @staticmethod
    def iou_loss(pred, target, from_logits=True, smooth=1e-6):
        if from_logits:
            pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        return 1 - (intersection + smooth) / (union + smooth)

    @staticmethod
    def focal_loss(gamma=2.0, from_logits=True):
        def fn(pred, target):
            bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target) if from_logits else nn.BCELoss(reduction='none')(torch.sigmoid(pred), target)
            focal = torch.pow(1 - torch.sigmoid(pred), gamma) * bce
            return torch.mean(focal)
        return fn
