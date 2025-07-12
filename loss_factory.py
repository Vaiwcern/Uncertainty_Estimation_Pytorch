# loss_factory.py
from custom_loss import CustomLosses

class LossFactory:
    @staticmethod
    def get_loss_fn(name: str):
        name = name.lower()
        if name == 'focal':
            return CustomLosses.focal_loss()
        elif name == 'bce':
            return CustomLosses.binary_crossentropy_loss()
        elif name == 'dice':
            return CustomLosses.dice_loss
        elif name == 'dice_bce':
            return CustomLosses.dice_bce_loss
        elif name == 'dice_focal':
            return CustomLosses.dice_focal_loss
        elif name == 'iou':
            return CustomLosses.iou_loss
        else:
            raise ValueError(f"Unsupported loss function: {name}")
