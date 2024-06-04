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
    

class CMUNet_DoubleHead(CMUNet):
    def __init__(self, img_ch=3, output_ch=1, l=7, k=7):
        """
        Args:
            img_ch : input channel.
            output_ch: output channel.
            l: number of convMixer layers
            k: kernal size of convMixer

        """
        super(CMUNet_DoubleHead, self).__init__()
        #* Boundary head 추가
        self.bdry_head = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.ConvMixer(x5)

        x4 = self.msag4(x4)
        x3 = self.msag3(x3)
        x2 = self.msag2(x2)
        x1 = self.msag1(x1)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)  # mask head
        d1_boundary = self.bdry_head(d2)
        return {'mask': d1, 'boundary': d1_boundary}