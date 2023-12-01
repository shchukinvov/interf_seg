import cv2
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from time import time


class InterfGenerator(Dataset):
    def __init__(self, size=100, shape=400):
        super(InterfGenerator, self).__init__()
        self.size = size
        self.shape = (shape, shape)
        self.center = shape // 2
        self.max_r = np.sqrt(2) * (shape / 2)
        self.dist_angle_mat = self._dist_angle_matrix()
        self.mask_modes = ["rectangle", "circle", "ellipse"]
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32),
            v2.Resize((256, 256), antialias=True),
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        mask = self._generate_random_mask(mode=np.random.choice(self.mask_modes),
                                          range_=np.random.uniform(0.1, 0.9),
                                          seed=idx)

        background = self._generate_background(noise_mean=np.random.uniform(20, 80),
                                               noise_std=np.random.uniform(30, 110),
                                               seed=idx)

        interf = self._generate_interf(range_=np.random.uniform(0.1, 1.),
                                       X=np.random.uniform(0, 40),
                                       Y=np.random.uniform(0, 40),
                                       A=np.random.uniform(0, 10),
                                       FIA=np.random.uniform(0, 90),
                                       D=np.random.uniform(0, 10),
                                       C_3=np.random.uniform(0, 10),
                                       FIC=np.random.uniform(0, 90),
                                       S=np.random.uniform(0, 10))

        noised_cropped_interf = np.clip(interf * mask + background, 0, 255)

        # Transforms
        image, mask = np.expand_dims(noised_cropped_interf, axis=2), np.expand_dims(mask, axis=2)

        image, mask = self.transforms(image, mask)

        m, s = image.min(), image.max() - image.min() + 1e-6
        image = v2.Normalize(mean=[m], std=[s])(image)

        return image, mask

    def _dist_angle_matrix(self):
        grid = np.indices(self.shape)
        return np.stack((self._distance_mat(grid[0], grid[1]), self._angle_mat(grid[0], grid[1])), axis=0)

    def _distance_mat(self, x, y):
        return np.sqrt((x - self.center)**2 + (y - self.center)**2) / self.max_r

    def _angle_mat(self, x, y):
        return np.arctan2((y - self.center), (x - self.center))

    def _tilt(self, tilt_x, tilt_y):
        return (tilt_x * self.dist_angle_mat[0] * np.sin(self.dist_angle_mat[1]) +
                tilt_y * self.dist_angle_mat[0] * np.cos(self.dist_angle_mat[1]))

    def _astigmatism(self, a, fia):
        phi_mat = self.dist_angle_mat[1] + np.pi * fia / 180
        return a * (self.dist_angle_mat[0]**2) * np.cos(2 * phi_mat)

    def _defocus(self, d):
        return d * (2 * self.dist_angle_mat[0]**2 - 1)

    def _coma(self, c, fic):
        phi_mat = self.dist_angle_mat[1] + np.pi * fic / 180
        return c * (3 * self.dist_angle_mat[0]**3 - 2 * self.dist_angle_mat[0]) * np.cos(phi_mat)

    def _spherical(self, s):
        return s * (6 * self.dist_angle_mat[0]**4 - 6 * self.dist_angle_mat[0]**2 + 1)

    def _generate_background(self, noise_mean, noise_std, seed=0):
        np.random.seed(seed)
        return np.random.normal(loc=noise_mean, scale=noise_std, size=self.shape)

    def _generate_random_mask(self, mode, range_, seed=0):
        np.random.seed(seed)
        mask = np.zeros(self.shape, dtype=np.uint8)
        if mode == "rectangle":
            size = int(self.shape[0] * range_)
            start_x, start_y = (np.random.randint(0, int(self.shape[0] * (1 - range_))),
                                np.random.randint(0, int(self.shape[0] * (1 - range_))))
            mask = cv2.rectangle(mask, (start_x, start_y), (start_x + size, start_y + size), color=1, thickness=-1)

        elif mode == "circle":
            radius = int(self.shape[0] / 2 * range_) - 1
            center = (np.random.randint(int(self.shape[0] / 2 * range_), int(self.shape[0] * (1 - range_ / 2))+1),
                      np.random.randint(int(self.shape[0] / 2 * range_), int(self.shape[0] * (1 - range_ / 2))+1))
            mask = cv2.circle(mask, center, radius, color=1, thickness=-1)

        elif mode == "ellipse":
            angle = np.random.randint(0, 180)
            angle_end = np.random.randint(45, 360)
            ratio = np.random.uniform(0.2, 0.9)
            axes_x = int(self.shape[0] / 2 * range_) - 1
            center_x = np.random.randint(axes_x + 1, int(self.shape[0] - axes_x + 1))
            axes_y = int(axes_x * ratio)
            center_y = np.random.randint(axes_y + 1, int(self.shape[0] - axes_y + 1))
            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), angle, 0, angle_end, color=1, thickness=-1)
        return mask

    def _generate_interf(self, range_=1., **kwargs):
        X = kwargs.get("X", 0)
        Y = kwargs.get("Y", 0)
        A = kwargs.get("A", 0)
        FIA = kwargs.get("FIA", 0)
        D = kwargs.get("D", 0)
        C_3 = kwargs.get("C_3", 0)
        FIC = kwargs.get("FIC", 0)
        S = kwargs.get("S", 0)
        phase_mat = np.pi * (self._tilt(X, Y) +
                             self._astigmatism(A, FIA) +
                             self._defocus(D) +
                             self._coma(C_3, FIC) +
                             self._spherical(S))

        return 127.5 * range_ * np.sin(phase_mat) + 127.5 * range_


def init_interf_generator(size, shape, batch_size=10, num_workers=2):
    interf_dataset = InterfGenerator(size=size, shape=shape)

    interf_loader = DataLoader(dataset=interf_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)

    return interf_loader

""" TEST """
if __name__ == "__main__":
    test = InterfGenerator(100, 400)
    fig = plt.figure(figsize=(8, 6))
    for im, msk in test:
        fig.add_subplot(1, 2, 1)
        plt.imshow(im.permute(1, 2, 0).numpy(), cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(msk.permute(1, 2, 0).numpy(), cmap="gray")
        plt.show()
        break
