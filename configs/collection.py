import torch
from interf_seg.src.model import UNET, SegNet, MSUnet
from interf_seg.src.loss_fn import IoULoss, ScaledIoULoss, IoUWithBCELoss


module_collection = {"UNET": UNET,
                     "SegNet": SegNet,
                     "MSUnet": MSUnet,
                     "ScaledIoULoss": ScaledIoULoss,
                     "IoULoss": IoULoss,
                     "IoUWithBCELoss": IoUWithBCELoss,
                     "Adam": torch.optim.Adam,
                     "StepLR": torch.optim.lr_scheduler.StepLR,
                     }
