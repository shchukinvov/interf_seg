import torch
import torch.nn as nn


class ScaledIoULoss(nn.Module):
    def __init__(self, gamma_fp=1, gamma_fn=1):
        super(ScaledIoULoss, self).__init__()
        self.eps = 1e-6
        self.g_fp = gamma_fp
        self.g_fn = gamma_fn

    def forward(self, predictions, targets):

        # predictions = nn.functional.sigmoid(predictions)
        predictions = predictions.view(-1)                              #    _______________________
        targets = targets.view(-1)                                      #   |  | target  |          |
                                                                        #   |  | FN______|_____     |
        tp = (predictions * targets).sum()                              #   |  |   |  TP |     |    |
        fp = predictions.sum() - tp                                     #   |  |___|_____|     |    |
        fn = targets.sum() - tp                                         #   |      |    FP     |    |
        iou = tp / (tp + self.g_fp * fp + self.g_fn * fn + self.eps)    #   |______|_predicted_|____|

        return 1 - iou


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.eps = 1e-6

    def forward(self, predictions, targets):

        # predictions = nn.functional.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        total = (predictions + targets).sum()
        union = total - intersection

        iou = intersection / (union + self.eps)

        return 1 - iou


class IoUWithBCELoss(nn.Module):
    def __init__(self, alpha_bce):
        super(IoUWithBCELoss, self).__init__()
        self.eps = 1e-6
        self.alpha_bce = alpha_bce

    def forward(self, predictions, targets):

        # predictions = nn.functional.sigmoid(predictions)
        predictions = predictions.clamp(0, 1)
        targets = targets.clamp(0, 1)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        bce = nn.functional.binary_cross_entropy(predictions, targets) / 100

        intersection = (predictions * targets).sum()
        total = (predictions + targets).sum()
        union = total - intersection

        iou = intersection / (union + self.eps)

        return (1 - iou) + self.alpha_bce * bce


""" TEST """
if __name__ == "__main__":
    pred = torch.rand(1, 1, 224, 224)
    target = torch.rand(1, 1, 224, 224)
    loss = IoUWithBCELoss(alpha_bce=0.5)
    l = loss(pred, target)
    print(l.item())
