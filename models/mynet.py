# encoding: utf-8
"""
2D UNet for medical images.
Author: Jason.Fang
Update time: 28/12/2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Type, Any, Callable, Union, List, Optional
# from convolution.SFConv import SpecConv2d as SpecConv
# from convolution.FConv_2d import TensorTrain as FConv
# from convolution.FConvSN_2d import SpecConv2d as FConvSN
# from convolution.DPConv import SpecConv2d as DPConv


#https://github.com/milesial/Pytorch-UNet
class MyNet(nn.Module):
    def __init__(self,  n_classes, bilinear=True, **kwargs: Any):
        super(MyNet, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        numFit = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(numFit, n_classes)
        # self.encoder.conv1 = nn.Conv2d(2, 512, kernel_size=7, stride=2, padding=3, bias=False)
        self.sigmoid = nn.Softmax(dim=-1)
            

    def forward(self, x):
        logits = self.encoder(x)
        logits = self.sigmoid(logits)
        # return logits, features
        return logits

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
    
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss
    

if __name__ == "__main__":
    #for debug   
    img = torch.rand(2, 1, 304, 304).cuda() #.to(torch.device('cuda:%d'%4))
    unet = UNet(n_channels=1, n_classes=1).cuda()
    out = unet(img)
    print(out.size())
    print(unet)