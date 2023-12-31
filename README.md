# interf_seg
Interferograms snapshots segmentation.

## Description
Basic [information](https://en.wikipedia.org/wiki/Interferometry) about interferometry and interferograms.

Example of interferogram is given below.
![1](https://upload.wikimedia.org/wikipedia/commons/e/e8/Optical_flat_interference_fringes.jpg)


## Main task
Build CV ML model to removing background from interferogram snapshot.

## Minor task
Build classic CV algorithm using the obtained features of the model.

## Validate metrics
- **IoU score**;
- **Dice score**;

## Progress
1. Collected and annotated 212 grayscale images;
2. `U-Net` as the baseline model;
3. Tested following cases:
    - `U-Net`, `SegNet`, `MSU-Net`, `AttU-Net` models;
    - `BCELoss`, `IoULoss`, `ScaledIoULoss`, `IoUWithBCELoss` losses;
    - Pretraining model on large amount of generated data;
    - Postprocessing based on morphological transforms;
4.  Best results at the moment:
    - `IoUmean = 0.939`
    - `IoUmin = 0.776`

    ##### Received by following params:
     - `Attention U-Net` with `features = (8, 16, 32, 64)`
     - `Adam` optimizer
     - Learning rate: `4e-4` 
     - Batch size: `10`
     - Num epochs: `450`

Some images

![1](https://github.com/shchukinvov/interf_seg/blob/main/figure/results_1.png)
![2](https://github.com/shchukinvov/interf_seg/blob/main/figure/results_2.png)

## TODO list
- Implement augmentations based on diffraction artifacts caused by dirt in optical system;
