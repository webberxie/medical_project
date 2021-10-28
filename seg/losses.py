import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

def binary_cross_entropyloss(prob, target, weight=None):
    loss = - weight * (target * torch.log(prob) + (1 - target) * (torch.log(1 - prob)))
    loss = torch.sum(loss) / torch.numel(target)
    if torch.numel(target) == 0:
        print('cuowu')
    print(loss, torch.sum(loss))
    return loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        #weight = 2 * target.clone()
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        #bce = binary_cross_entropyloss(input, target, weight)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
        #return 0.2 * bce + torch.log((torch.exp(dice) + torch.exp(-dice)) / 2.0)


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
