import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss', 'BoundarywithMaskLoss', 'BoundarywithMaskLoss2']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class BoundarywithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.segLoss = BCEDiceLoss()
        self.bdrLoss = nn.MSELoss()
    
    def boundary_loss(self, pred, gt):
        bce = F.binary_cross_entropy_with_logits(pred, gt)
        return self.bdrLoss(pred, gt) + bce
    
    def forward(self, inputs, target, target_ditance, task_ids, t, N, Taskloss_flg=False):
        #* 1. boundary loss
        loss_boundary = self.boundary_loss((inputs['dis'].to(torch.float64)), target_ditance.to(dtype = torch.float64, device=inputs['dis'].device))
        #* 2. mask loss
        loss_mask = self.segLoss(inputs['mask'], target)
        
        return  (1 - t/N) * loss_boundary + (t/N) * loss_mask
    
class BoundarywithMaskLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.segLoss = BCEDiceLoss()
        self.bdrLoss = nn.MSELoss()
    
    def boundary_loss(self, pred, gt):
        return self.bdrLoss(pred, gt) + F.binary_cross_entropy_with_logits(pred, gt)     # mse:bce = 1:1 비율

    def forward(self, inputs, target, target_ditance, task_ids, t, N, Taskloss_flg=False):
        #print('loss2')
        #* 1. boundary loss
        loss_boundary = self.boundary_loss(inputs['dis'].to(torch.float64), target_ditance.to(dtype = torch.float64, device=inputs['dis'].device))
        #loss_boundary = self.bdrLoss(inputs['dis'].to(torch.float64), target_ditance.to(dtype = torch.float64, device=inputs['dis'].device))
        #* 2. mask loss
        loss_mask = self.segLoss(inputs['mask'], target, task_ids, Taskloss_flg)
        
        return  (1 - t/N) * loss_boundary + (t/N) * loss_mask
