import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Normalize
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data, image_dir, mask_dir, transforms=None):
        self.data = data
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.data[index])
        mask_path = os.path.join(self.mask_dir, self.data[index])
        image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask


class TestDataset(Dataset):
    def __init__(self, image_indices, image_dir, mask_dir, transforms=None):
        self.image_indices = image_indices
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        path = self.image_indices[index] + ".bmp"
        image_path = os.path.join(self.image_dir, path)
        image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
        im_shape = image.shape
        mask_path = os.path.join(self.mask_dir, path)
        mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask, im_shape


def init_dataloaders(image_dir, mask_dir, train_transforms, val_transforms, test_size, batch_size=16, random_state=0, num_workers=2):
    full_data = os.listdir(image_dir)
    train_data, val_data = train_test_split(full_data,
                                            test_size=test_size,
                                            shuffle=True,
                                            random_state=random_state)

    train_dataset = CustomDataset(data=train_data,
                                  image_dir=image_dir,
                                  mask_dir=mask_dir,
                                  transforms=train_transforms)

    val_dataset = CustomDataset(data=val_data,
                                image_dir=image_dir,
                                mask_dir=mask_dir,
                                transforms=val_transforms)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader


def init_test_loader(image_indices, image_dir, mask_dir, test_transforms, batch_size=1, num_workers=2):
    test_dataset = TestDataset(image_indices=image_indices,
                               image_dir=image_dir,
                               mask_dir=mask_dir,
                               transforms=test_transforms)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return test_loader


""" TEST """
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from interf_seg.configs.transforms import *
    from interf_seg.configs.experiment import *
    test_data = os.listdir(TRAIN_IMAGE_DIR)
    test_set = iter(CustomDataset(test_data, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, transforms=TRAIN_TRANSFORMS))
    # test_set = iter(TestDataset(VAL_IMAGE_INDICES, VAL_IMAGE_DIR, VAL_MASK_DIR, transforms=TRAIN_TRANSFORMS))
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        im, _ = next(test_set)
        fig.add_subplot(5, 5, i+1)
        plt.imshow(im.permute(1, 2, 0).numpy(), cmap="gray")
    plt.show()
