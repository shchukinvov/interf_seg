import albumentations as A
from albumentations.pytorch import ToTensorV2


TRAIN_TRANSFORMS = A.Compose([
    A.PadIfNeeded(256, 256, value=0, mask_value=0),
    A.Rotate(p=0.6),
    A.RandomCrop(256, 256, p=0.2),
    A.Resize(256, 256, always_apply=True),
    A.ToFloat(max_value=255),
    ToTensorV2(),
])

VAL_TRANSFORMS = A.Compose([
    A.PadIfNeeded(256, 256, value=0, mask_value=0),
    A.Resize(256, 256, always_apply=True),
    A.ToFloat(max_value=255),
    ToTensorV2(),
])



""" TEST """
if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    im_path1 = "D:/ML_Projects/interf_seg/data/test/interfs/9009.bmp"
    im_path2 = "D:/ML_Projects/interf_seg/data/test/masks/9009.bmp"
    im1 = np.array(Image.open(im_path1).convert("L"), dtype=np.uint8)
    im2 = np.array(Image.open(im_path2).convert("1"), dtype=np.uint8)
    t = TRAIN_TRANSFORMS(image=im1, mask=im2)
    t_im = t["image"]
    t_msk = t["mask"]
    print(t_im.dtype)
    print(f' Max t_im1={t_im.max():.3f} | Min t_im1={t_im.min():.3f}')
    print(f' Max t_im2={t_msk.max():.3f} | Min t_im2={t_msk.min():.3f}')
    fig = plt.figure()
    fig.add_subplot(121)
    plt.imshow(t_im.permute(1, 2, 0).numpy(), cmap="gray")
    fig.add_subplot(122)
    plt.imshow(t_msk.numpy(), cmap="gray")
    plt.show()
