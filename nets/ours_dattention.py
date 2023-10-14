import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from nets.vgg import VGG16
from nets.Vitmodeling import Embeddings,Block
from torch.nn.modules.utils import _pair
import numpy as np
import math
import graph.ConvGRU2 as ConvGRU
from torch_geometric.nn import GCNConv,GATConv,DenseGCNConv,AGNNConv,GatedGraphConv,GraphConv
torch.cuda.empty_cache()

class unetUp_G(nn.Module):
    def __init__(self, in_size, out_size,F_g, F_l, F_int):
        super(unetUp_G, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Att = Attention_SG(F_g, F_l, F_int)
        # self.spa=SpatialAttention(kernel_size=7)
        # self.conv3=nn.Conv2d(F_int*2,F_int, kernel_size=3, padding=1)

    def forward(self, inputs1, inputs2):
        inputs2=self.up(inputs2)
        inputs1_2,inputs1_1=self.Att(inputs2,inputs1)
        # inputs1_2=self.spa(inputs1)
        # inputs1_=(inputs1_1+inputs1_2)*inputs1
        outputs = torch.cat((inputs1_1,inputs1_2, inputs2), 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class da_network(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(da_network, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp_G(in_filters[3], out_filters[3],F_g=512, F_l=512, F_int=256)
        # 128,128,256
        self.up_concat3 = unetUp_G(in_filters[2], out_filters[2],F_g=512, F_l=256, F_int=128)
        # 256,256,128
        self.up_concat2 = unetUp_G(in_filters[1], out_filters[1],F_g=256, F_l=128, F_int=64)
        # 512,512,64
        self.up_concat1 = unetUp_G(in_filters[0], out_filters[0],F_g=128, F_l=64, F_int=32)

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        final = self.final(up1)

        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


class Attention_SG(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_SG, self).__init__()
        self.spa=SpatialAttention_new(kernel_size=1,F_int=F_int)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv=nn.Conv2d( F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        gx=self.conv(g1+x1)
        psi = self.relu(gx)
        psi1 = self.psi(psi)
        psi2=self.spa(g1+x1)
        # spa
        out1=psi1*(g1+x1)
        out2 = psi2 * (g1 + x1)
        out=self.sigmoid(out1+out2)


        return g1*out,x1*out

class SpatialAttention_new(nn.Module):
    def __init__(self, kernel_size=1,F_int=256):
        super(SpatialAttention_new, self).__init__()
        # assert kernel_size in (3,7), "kernel size must be 3 or 7"
        # padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(F_int*2,F_int,kernel_size, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=2, keepdim=True)
        avgout = torch.mean(avgout, dim=3, keepdim=True)
        maxout, _ = torch.max(x, dim=2, keepdim=True)
        maxout, _ = torch.max(maxout, dim=3, keepdim=True)
        x1 = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x1)
        return self.sigmoid(x1)








class Attention_G(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_G, self).__init__()
        self.spa=SpatialAttention(kernel_size=7)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi1 = self.psi(psi)
        psi2=self.spa(g1+x1)
        psi_=psi1+psi2
        return g*(psi1+psi2),x*(psi1+psi2)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1)   #
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x1)
        return self.sigmoid(x1)