import os
import torch
from torchvision.transforms.v2 import Resize
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import cv2
from PIL import  Image
import matplotlib.pyplot as plt
import json
import random
from math import ceil, sqrt
from dataset import init_test_loader
from interf_seg.configs.transforms import *
from interf_seg.configs.experiment import *


class ExperimentLogger:
    """
    Logging experiments
    """
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(filename):
            with open(filename, 'x') as file:
                file.write('{"counter": 0, "experiments": []}')

        if not isinstance(self.read(), dict):
            raise TypeError("Corrupted data")

    def __len__(self):
        return self.read()["counter"]

    def read(self):
        with open(self.filename, 'r', encoding='utf-8') as file:
            return json.load(file)

    def write(self, data):
        data = json.dumps(data)
        data = json.loads(str(data))
        with open(self.filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)

    def parse_and_log(self, val_scores, model_path,  **kwargs):
        exp_data = self.read()
        counter, exp_list = exp_data["counter"], exp_data["experiments"]
        exp_info = {
            "id": counter + 1,
            "model_path": model_path,
            "metrics": {
                "IoU_mean": round(val_scores[0], 4),
                "IoU_min": round(val_scores[1], 4),
            },
            "experiment_params": kwargs
        }
        counter += 1
        exp_list.append(exp_info)
        updated_exp_data = {"counter": counter, "experiments": exp_list}
        self.write(updated_exp_data)


def seed_all(seed: int) -> None:
    """
    Seed everything for deterministic experiment
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def show_losses(train_loss: list, val_loss: list) -> None:
    """
    :param train_loss: list of train losses during training
    :param val_loss: list of validation losses during training
    :return: loss/epoch plot
    """
    k = ceil(len(train_loss) / len(val_loss))
    plt.plot(train_loss, label='train_loss')
    plt.plot(np.arange(0, len(val_loss) * k, k), val_loss, label='val_loss')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def show_model_preds(model, image: np.ndarray) -> None:

    height, width = image.shape

    t_image = VAL_TRANSFORMS(image=image)['image']
    image = ToTensorV2()(image=image)['image']

    device = model.device()
    model.eval()
    with torch.no_grad():
        t_image = t_image.unsqueeze(0).to(device)
        mask = (model(t_image) > 0.5).squeeze(0).float()

    mask = Resize((height, width), antialias=True)(mask).cpu()

    masked_image = image * mask
    figure = plt.figure(figsize=(12, 6))

    figure.add_subplot(1, 2, 1).set_title("Raw image")
    plt.imshow(image.permute(1, 2, 0).numpy(), cmap="gray")

    figure.add_subplot(1, 2, 2).set_title("Masked image")
    plt.imshow(masked_image.permute(1, 2, 0).numpy(), cmap="gray")

    plt.show()


def train_fn(model, dataloaders, criterion, optimizer, num_epochs, device, scheduler=None, validation=True):
    if validation:
        train_dataloader, val_dataloader = dataloaders
        train_loss = []
        val_loss = []
    else:
        train_dataloader = dataloaders
        train_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch+1, num_epochs), flush=True)

        with tqdm(total=len(train_dataloader)) as progress:
            losses = []
            model.train()

            for inputs, masks in train_dataloader:
                inputs = inputs.to(device).clamp(0, 1)
                masks = masks.to(device).clamp(0, 1)

                optimizer.zero_grad()
                preds = model(inputs)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                progress.update()

            mean_loss = sum(losses) / len(losses)
            train_loss.append(mean_loss)
            progress.set_postfix({f'Train loss': mean_loss})

        if validation:
            if epoch % 4 == 0:
                with tqdm(total=len(val_dataloader)) as progress:
                    losses = []
                    model.eval()
                    for inputs, masks in val_dataloader:
                        inputs = inputs.to(device).clamp(0, 1)
                        masks = masks.to(device).clamp(0, 1)

                        with torch.set_grad_enabled(False):
                            preds = model(inputs)
                            loss = criterion(preds, masks)

                        losses.append(loss.item())
                        progress.update()

                    mean_loss = sum(losses) / len(losses)
                    val_loss.append(mean_loss)
                    progress.set_postfix({f'Val loss': mean_loss})

        # Early Stopping
        # if val_loss.index(min(val_loss), -12) + 11 < len(val_loss):
        #     print(f'Early Stopping on epoch={epoch+1} with validation loss={val_loss[-1]}')
        #     return train_loss, val_loss

        if scheduler:
            scheduler.step()

    return train_loss, val_loss if validation else train_loss


def calc_iou_score(prediction: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """
    :param prediction: tensor containing prediction mask
    :param target: tensor containing target mask
    :param eps: add smoothing
    :return: IoU score for given prediction and target
    """
    assert prediction.device == target.device, "Different devices"

    prediction = prediction.view(-1)
    target = target.view(-1)

    intersection = (prediction * target).sum()
    total = (prediction + target).sum()
    union = total - intersection

    iou = intersection / (union + eps)

    return iou.item()


def calc_dice_score(prediction: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """
    :param prediction: tensor containing prediction mask
    :param target: tensor containing target mask
    :param eps: add smoothing
    :return: dice score for given prediction and target
    """
    assert prediction.device == target.device, "Different devices"

    prediction = prediction.view(-1)
    target = target.view(-1)

    intersection = (prediction * target).sum()

    dice = (2.*intersection + eps)/(prediction.sum() + target.sum() + eps)

    return dice.item()


def validate_model(model, save_predictions=False, postprocessing=False, show_results=False):
    """
    Validating model
    """
    iou_scores = []
    dice_scores = []
    masked_images = []
    device = model.device()
    test_loader = init_test_loader(image_indices=VAL_IMAGE_INDICES,
                                   image_dir=VAL_IMAGE_DIR,
                                   mask_dir=VAL_MASK_DIR,
                                   test_transforms=VAL_TRANSFORMS,
                                   batch_size=1)

    with tqdm(total=len(test_loader)) as progress:
        model.eval()

        for image, mask, shape in test_loader:
            image = image.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                if not postprocessing:
                    predicted_mask = (model(image) > 0.5).float()
                else:
                    predicted_mask = morph_transforms(model(image), 0.5, mode="MORPH_CLOSE", kernel_size=(9, 9))

                iou_score = calc_iou_score(predicted_mask, mask)
                dice_score = calc_dice_score(predicted_mask, mask)

                masked_image = (image*predicted_mask).squeeze(0)
                masked_image = Resize(shape, antialias=True)(masked_image)

            iou_scores.append(iou_score)
            dice_scores.append(dice_score)
            masked_images.append(masked_image)

            progress.update()

    mean_iou, min_iou = (sum(iou_scores)/len(iou_scores)), min(iou_scores)
    mean_dice, min_dice = (sum(dice_scores) / len(dice_scores)), min(dice_scores)

    print('Mean IoU score is {:.3f} | Min IoU score is {:.3f}'.format(mean_iou, min_iou))
    print('Mean Dice score is {:.3f} | Min Dice score is {:.3f}'.format(mean_dice, min_dice))

    if save_predictions:
        for ind, image in enumerate(masked_images):
            save_image(image, f"saved_images/{ind+1}.bmp")

    if show_results:
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(hspace=0.33, wspace=0.15)
        length = len(masked_images)
        size = ceil(sqrt(length))
        for i in range(length):
            fig.add_subplot(size, size, i + 1).set_title('IoU score={:.3f} | Dice score={:.3f}'.
                                                         format(iou_scores[i], dice_scores[i]),
                                                         fontdict={'fontsize': 8})
            plt.imshow(masked_images[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")

        plt.show()

    return mean_iou, min_iou


def show_model_featuremap(model, image: np.ndarray) -> None:
    if not hasattr(model, "show_featuremap"):
        raise AttributeError("show_featuremap is not implemented in this model")

    model.eval()
    image = VAL_TRANSFORMS(image)
    device = model.device()
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        feature_map = model.show_featuremap(x)
        for key, value in feature_map.items():
            side = ceil(sqrt(value.shape[1]))
            fig = plt.figure(figsize=(10, 10))
            fig.suptitle(key)
            for idx, channel in enumerate(value.squeeze(0)):
                fig.add_subplot(side, side, idx+1)
                plt.imshow(channel.cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.show()


def morph_transforms(x: torch.Tensor, threshold: float, mode: str, kernel_size: tuple[int, int]) -> torch.Tensor:
    """
    Apply morphological transforms to the model's output
    :param x: Predicted mask with values from 0 to 1
    :param threshold: Binarize mask with given threshold
    :param mode: MORPH_ERODE | MORPH_DILATE | MORPH_OPEN | MORPH_CLOSE
    :param kernel_size: (int, int)
    :return: Processed mask
    """
    device = x.device
    if x.is_cuda:
        x = x.cpu()
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input should be torch.Tensor")
    if not x.shape[-3] == 1:
        raise RuntimeError("Image should have only 1 channel")
    if (len(kernel_size) != 2
            and not isinstance(kernel_size, tuple)
            and not isinstance((kernel_size[0], kernel_size[1]), int)):
        raise RuntimeError("Kernel should be (int, int)")

    modes = {"MORPH_ERODE": cv2.MORPH_ERODE,
             "MORPH_DILATE": cv2.MORPH_DILATE,
             "MORPH_OPEN": cv2.MORPH_OPEN,
             "MORPH_CLOSE": cv2.MORPH_CLOSE
             }
    if mode not in modes:
        raise RuntimeError(f'{mode} is not supported')

    if len(x.shape) > 3:
        masks = []
        for mask in x:
            mask = (mask > threshold).squeeze(0).float().numpy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            masks.append(torch.tensor(cv2.morphologyEx(mask, modes[mode], kernel=kernel)).unsqueeze(0).to(device))
        return torch.stack(masks, dim=0)

    else:
        mask = (x > threshold).squeeze(0).float().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        return torch.tensor(cv2.morphologyEx(mask, modes[mode], kernel=kernel)).unsqueeze(0).to(device)
