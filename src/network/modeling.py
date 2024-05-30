import torch
import torch.nn as nn
from src.network.CMUNet import CMUNet

class Model_R2C(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, l=7, k=7):
        super(Model_R2C, self).__init__()

        #* 1. Regression model
        self.RegModel = CMUNet(img_ch, output_ch, l, k)
        #* 2. Segmentation model
        self.SegModel = CMUNet(img_ch, output_ch, l, k)

    def forward(self, x):
        #* 1. regression model forward      : img -> distance map([b,1,h,w])
        distancemap = self.RegModel(x)  
        #* 2. segmentation model forward    : distance map([b,3,h,w]) -> mask
        mask = self.SegModel(torch.concat([distancemap, distancemap, distancemap], dim=1))
        
        return {"dis" : distancemap, "mask" : mask}