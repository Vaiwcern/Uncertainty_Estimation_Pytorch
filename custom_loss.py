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
    def focal_loss(gamma=2.0, alpha=0.85, from_logits=True):
        """
        Binary Focal Loss implementation.

        Args:
            gamma (float): focusing parameter for modulating factor (1 - pt)
            alpha (float or None): balancing factor, e.g., 0.25
            from_logits (bool): whether pred is raw logits or sigmoid probabilities

        Returns:
            Callable: loss function
        """
        def fn(pred, target):
            if from_logits:
                pred_prob = torch.sigmoid(pred)
            else:
                pred_prob = pred

            # Clamp để tránh log(0)
            pred_prob = torch.clamp(pred_prob, min=1e-6, max=1.0 - 1e-6)

            # BCE loss thủ công để dễ kết hợp focal
            bce_loss = - (target * torch.log(pred_prob) + (1 - target) * torch.log(1 - pred_prob))

            # pt: xác suất đúng
            pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
            focal_weight = (1 - pt) ** gamma

            if alpha is not None:
                alpha_t = torch.where(target == 1, alpha, 1 - alpha)
                loss = alpha_t * focal_weight * bce_loss
            else:
                loss = focal_weight * bce_loss

            return loss.mean()
        return fn

