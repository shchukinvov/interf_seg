import torch
from interf_seg.src.model import UNET, SegNet, MSUnet, AttUnet
from interf_seg.src.loss_fn import IoULoss, ScaledIoULoss, IoUWithBCELoss


module_collection = {"UNET": UNET,
                     "SegNet": SegNet,
                     "MSUnet": MSUnet,
                     "AttUnet": AttUnet,
                     "ScaledIoULoss": ScaledIoULoss,
                     "IoULoss": IoULoss,
                     "IoUWithBCELoss": IoUWithBCELoss,
                     "BCELoss": torch.nn.BCELoss,
                     "Adam": torch.optim.Adam,
                     "StepLR": torch.optim.lr_scheduler.StepLR,
                     }
