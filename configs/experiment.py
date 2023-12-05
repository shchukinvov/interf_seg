import os

TRAIN_IMAGE_DIR = "../data/train/interfs"
TRAIN_MASK_DIR = "../data/train/masks"

VAL_IMAGE_INDICES = ["9001", "9002", "9003", "9004", "9005", "9006", "9007",
                     "9008", "9009", "9010", "9011", "9012"]
VAL_IMAGE_DIR = '../data/test/interfs'
VAL_MASK_DIR = '../data/test/masks'


TRAINING = True
PRETRAIN = False
LOAD_CP = "pretrain_model_27.pth"
SAVE_MODEL = True

CP_PATH = '../experiments_reg/'
EXPERIMENT_LIST = '../experiments_reg/experiment_list.json'

EXPERIMENT_PARAMS = {
    "model": {
        "name": "AttUnet",
        "params": {
            "in_channels": 1,
            "features": (16, 32, 64, 128)
        }
    },

    "criterion": {
        "name": "IoULoss",
        "params": {
            # "alpha_bce": 1
            # "gamma_fp": 1.5,
            # "gamma_fn": 1.5,
        }
    },

    "optimizer": {
        "name": "Adam",
        "params": {
            "lr": 4e-4,
            "betas": (0.9, 0.999),
            # "weight_decay": 0.01
        }
    },

    "scheduler": {
        "name": "StepLR",
        "params": {
            "step_size": 3,
            "gamma": 0.98
        }
    },

    "seed": 4,
    "num_epochs": 350,
    "batch_size": 10
}
