import torch
from torchvision.transforms import v2

RANDOM_TRANSFORMS = v2.RandomApply([
    v2.RandomRotation((-90, 90)),
    v2.RandomCrop((256, 256), pad_if_needed=True, padding_mode="constant", fill=0.),
],
    p=0.5)

TYPE_TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32),
])

BASIC_TRANSFORMS = v2.Compose([
    v2.Resize((256, 256), antialias=True),
])

TRAIN_TRANSFORMS = v2.Compose([
    TYPE_TRANSFORMS,
    RANDOM_TRANSFORMS,
    BASIC_TRANSFORMS,
])

VAL_TRANSFORMS = v2.Compose([
    TYPE_TRANSFORMS,
    BASIC_TRANSFORMS,
])


""" TEST """
if __name__ == "__main__":
    from PIL import Image
    im_path1 = 'interfs/0001.bmp'
    im_path2 = 'masks/0001.bmp'
    im1 = Image.open(im_path1).convert("L")
    im2 = Image.open(im_path2).convert("1")
    t_im1, t_im2 = TRAIN_TRANSFORMS(im1, im2)
    mean, std = t_im1.min(), t_im1.max()-t_im1.min()
    t_im1 = v2.Normalize(mean=[mean], std=[std])(t_im1)
    print(f' Max t_im1={t_im1.max():.3f} | Min t_im1={t_im1.min():.3f}')
    print(f' Max t_im2={t_im2.max():.3f} | Min t_im2={t_im2.min():.3f}')