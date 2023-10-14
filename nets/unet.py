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

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):

        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained,in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[  :4 ](inputs)
        feat2 = self.vgg.features[4 :9 ](feat1)
        feat3 = self.vgg.features[9 :16](feat2)
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



#---------------------------------------------------------------------------------#
# New model transfomer
#---------------------------------------------------------------------------------#

class unetUp_A(nn.Module):
    def __init__(self, in_size, out_size,F_g, F_l, F_int):
        super(unetUp_A, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Att = Attention_block(F_g, F_l, F_int)

    def forward(self, inputs1, inputs2):
        inputs2=self.up(inputs2)
        inputs1=self.Att(inputs2,inputs1)
        outputs = torch.cat((inputs1, inputs2), 1)
        del inputs1,inputs2
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

class vgg_Unet_Attention(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(vgg_Unet_Attention, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        #F_g是input2的通道数，F_1是input1的通道数，F_g的通道数需要调整
        self.up_concat4 = unetUp_A(in_filters[3], out_filters[3],F_g=512, F_l=512, F_int=256)
        # 128,128,256
        self.up_concat3 = unetUp_A(in_filters[2], out_filters[2],F_g=512, F_l=256, F_int=128)
        # 256,256,128
        self.up_concat2 = unetUp_A(in_filters[1], out_filters[1],F_g=256, F_l=128, F_int=64)
        # 512,512,64
        self.up_concat1 = unetUp_A(in_filters[0], out_filters[0],F_g=128, F_l=64, F_int=32)

        # final conv (without any concat)
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


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
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

        return x * psi1








#--------------------------------------------------------------------------------#
#                      Graph_test
#--------------------------------------------------------------------------------#

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


class Graph_network(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(Graph_network, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        #F_g is the number of channels for input2，F_1 is the number of channels for input1，the number of channels for F_g needs to be adjusted
        self.up_concat4 = unetUp_G(in_filters[3], out_filters[3],F_g=512, F_l=512, F_int=256)
        # 128,128,256
        self.up_concat3 = unetUp_G(in_filters[2], out_filters[2],F_g=512, F_l=256, F_int=128)
        # 256,256,128
        self.up_concat2 = unetUp_G(in_filters[1], out_filters[1],F_g=256, F_l=128, F_int=64)
        # 512,512,64
        self.up_concat1 = unetUp_G(in_filters[0], out_filters[0],F_g=128, F_l=64, F_int=32)

        # self.GICN=GraphInterConnection(out_filters[2],patch_num=16)

        self.GECN = GraphExterConnection(all_channel=32,output_channel=512, patch_num=3)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.conv5 = nn.Conv2d(544, out_filters[3], kernel_size=3, padding=1)#out_filters[3]+32=544

        self.conv3 = nn.Conv2d(out_filters[2] * 2, out_filters[2], kernel_size=3, padding=1)

    def forward(self, inputs):
        feat1 = self.vgg.features[:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)



        temp5=self.GECN(feat5,3)
        # # join the input network and see the results
        feat5_=self.conv5(torch.cat([feat5,temp5],1))

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

class GraphInterConnection(nn.Module):
    def __init__(self, all_channel=256,patch_num=16):
        super(GraphInterConnection, self).__init__()
        self.coa1=coattention(all_channel)
        # self.conv_fusion = nn.Conv2d(all_channel*patch_num, all_channel, kernel_size=3, padding=1)
        self.conv_fusion = nn.Conv2d(all_channel, all_channel, kernel_size=3, padding=1)
    def forward(self,temp,patch_num=16):
        temp1=[]
        for ii in range(len(temp)):
            self_att=self.coa1(temp[ii],temp[ii])
            for jj in range(len(temp)):
                # self_att=torch.add(self_att,torch.div(self.coa1(temp[ii],temp[jj]),float(patch_num+1)))  # 1为通道所在位置 拼接换为平均
               if jj!=ii:
                    A=self.coa1(temp[ii],temp[jj])
                    self_att=torch.add(self_att,A)
                    # self_att=torch.cat([self_att,A],1)
                    # del A
                # temp2.append(self.coa1(temp[ii],temp[jj]))
            temp1.append(torch.div(self_att,patch_num))

        return temp1

class GraphExterConnection(nn.Module):
    def __init__(self, all_channel=32,output_channel=512,patch_num=4):
        super(GraphExterConnection, self).__init__()
        self.conv1=nn.Conv2d(output_channel,all_channel, kernel_size=3,padding=1)
        self.coa1=coattention(all_channel)
        self.conv_fusion = nn.Conv2d(all_channel * patch_num, all_channel, kernel_size=3, padding=1, bias=True)


        self.ConvGRU = ConvGRU.ConvGRUCell(all_channel, all_channel, kernel_size=1)  # 可替换为gnn
        # self.GCN=GatedGraphConv(all_channel, num_layers=1)
    def forward(self, temp, patch_num=3):
        temp=self.conv1(temp)
        temp1=torch.chunk(temp, patch_num, dim=0)
        temp2=temp.clone()
        del temp
        attention1 = self.conv_fusion(torch.cat([self.coa1(temp1[0].clone(), temp1[0].clone()),
                                                 self.coa1(temp1[0].clone(), temp1[1].clone()),
                                                 self.coa1(temp1[0].clone(), temp1[2].clone())],
                                                1))  # message passing with concat operation
        attention2 = self.conv_fusion(torch.cat([self.coa1(temp1[1].clone(), temp1[1].clone()),self.coa1(temp1[1].clone(), temp1[0].clone()),
                                                 self.coa1(temp1[1].clone(), temp1[2].clone())], 1))
        attention3 = self.conv_fusion(torch.cat([self.coa1(temp1[2].clone(), temp1[2].clone()),self.coa1(temp1[2].clone(), temp1[0].clone()),
                                                 self.coa1(temp1[2].clone(), temp1[1].clone())], 1))
        h_v1 = self.ConvGRU(attention1, temp1[0].clone())

        h_v2 = self.ConvGRU(attention2, temp1[1].clone())

        h_v3 = self.ConvGRU(attention3, temp1[2].clone())
        temp2[0] = h_v1
        temp2[1] = h_v2
        temp2[2] = h_v3

        #batchsize=4
        # attention1 = self.conv_fusion(torch.cat([self.coa1(temp1[0].clone(), temp1[0].clone()),
        #                                          self.coa1(temp1[0].clone(), temp1[1].clone()),
        #                                          self.coa1(temp1[0].clone(), temp1[2].clone()),
        #                                          self.coa1(temp1[0].clone(), temp1[3].clone())],
        #                                         1))  # message passing with concat operation
        # attention2 = self.conv_fusion(
        #     torch.cat([self.coa1(temp1[1].clone(), temp1[1].clone()), self.coa1(temp1[1].clone(), temp1[0].clone()),
        #                self.coa1(temp1[1].clone(), temp1[2].clone()), self.coa1(temp1[1].clone(), temp1[3].clone())], 1))
        # attention3 = self.conv_fusion(
        #     torch.cat([self.coa1(temp1[2].clone(), temp1[2].clone()), self.coa1(temp1[2].clone(), temp1[0].clone()),
        #                self.coa1(temp1[2].clone(), temp1[1].clone()), self.coa1(temp1[2].clone(), temp1[3].clone())], 1))
        # attention4 = self.conv_fusion(
        #     torch.cat([self.coa1(temp1[3].clone(), temp1[2].clone()), self.coa1(temp1[3].clone(), temp1[0].clone()),
        #                self.coa1(temp1[3].clone(), temp1[1].clone()), self.coa1(temp1[3].clone(), temp1[3].clone())],
        #               1))
        # h_v1 = self.ConvGRU(attention1, temp1[0].clone())
        #
        # h_v2 = self.ConvGRU(attention2, temp1[1].clone())
        #
        # h_v3 = self.ConvGRU(attention3, temp1[2].clone())
        # h_v4 = self.ConvGRU(attention4, temp1[3].clone())
        # temp2[0] = h_v1
        # temp2[1] = h_v2
        # temp2[2] = h_v3
        # temp2[3] = h_v4
        return temp2

class coattention(nn.Module):  # Calculate the correlation between two vectors
    def  __init__(self, all_channel=256):	#473./8=60
        super(coattention, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel,bias = False)
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.channel=all_channel

    def forward(self,exemplar, query):
        fea_size = query.size()[2:]
           #		 #all_dim = exemplar.shape[1]*exemplar.shape[2]
        # exemplar_flat = exemplar.contiguous().view(-1, self.channel, (fea_size[0] * fea_size[1]))  # N,C,H*W
        #         query_flat = query.contiguous().view(-1, self.channel, fea_size[0] * fea_size[1])
        exemplar_flat = exemplar.contiguous().view(-1, self.channel, (fea_size[0] * fea_size[0]))  # N,C,H*W
        query_flat = query.contiguous().view(-1, self.channel, fea_size[0] * fea_size[0])
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        # C=torch.transpose(A, 1, 2).contiguous()
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        # query_att = torch.bmm(exemplar_flat, A).contiguous() #Pay attention to whether to use interaction and the structure of Residuals in this place
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[0])
        # input2_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        # input2_mask = self.gate(input2_att)
        input1_mask1 = self.gate_s(input1_mask)
        # input2_mask = self.gate_s(input2_mask)
        input1_att1 = input1_att * input1_mask1
        # input2_att = input2_att * input2_mask

        return input1_att1



def getpatch(input,patch_size=64):
    B, C, H, W = input.shape
    output=[]
    for i in range(int(H/patch_size)):
        for j in range(int(H/patch_size)):
            output.append(input[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size])
    return output

def catpatch(temp,output_size=64):
    B, C, H, W = temp[0].shape
    output=torch.zeros(B, C, output_size, W)
    for i in range(int(output_size/H)):
        temp1=temp[i*int(output_size/H)]
        for j in range(int(output_size/H)):
            if j !=0:
                temp1=torch.cat([temp1,temp[i*int(output_size/H)+j]],3)
        if i==0:
            output = temp1
        else:
            output=torch.cat([output,temp1],2)
    return output

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
        # spa
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

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi1 = self.psi(psi)
        psi2=self.spa(g1+x1)
        # spa
        out1=psi1*(g1+x1)
        out2 = psi2 * (g1 + x1)
        out=self.sigmoid(out1+out2)
        #对psi1、psi2进行计算

        # W0=[]
        # a=psi1[0]
        # w0=np.linalg.norm(psi1[0].cpu()-psi1[0].cpu())
        # w0=np.linalg.norm(psi1[0]-psi1[0])+np.linalg.norm(psi1[0]-psi1[1])+np.linalg.norm(psi1[0]-psi1[2])
        # W0.append(np.linalg.norm((psi1[0]-psi1[0]))/w0)
        # W0.append(np.linalg.norm((psi1[0] - psi1[1]))/w0)
        # W0.append(np.linalg.norm((psi1[0] - psi1[2])) / w0)
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



#---------------------------------------------------------------------------------#
# new model             feature fusion
#---------------------------------------------------------------------------------#

class unetUp_AF(nn.Module):
    def __init__(self, in_size, out_size,F_g, F_l, F_int):
        super(unetUp_AF, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.iaff=iAFF(F_g, F_l, F_int)
        self.iaff = AFF(F_g, F_l, F_int)

    def forward(self, inputs1, inputs2):
        inputs2=self.up(inputs2)
        inputs1_=self.iaff(inputs2,inputs1)
        # inputs1=self.Att(inputs2,inputs1)
        outputs = torch.cat((inputs1_, inputs2), 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

class vgg_Unet_AF(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(vgg_Unet_AF, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        #F_g is the number of channels for input2，F_1 is the number of channels for input1，the number of channels for F_g needs to be adjusted
        self.up_concat4 = unetUp_AF(in_filters[3], out_filters[3],F_g=512, F_l=512, F_int=512)
        # 128,128,256
        self.up_concat3 = unetUp_AF(in_filters[2], out_filters[2],F_g=512, F_l=256, F_int=256)
        # 256,256,128
        self.up_concat2 = unetUp_AF(in_filters[1], out_filters[1],F_g=256, F_l=128, F_int=128)
        # 512,512,64
        self.up_concat1 = unetUp_AF(in_filters[0], out_filters[0],F_g=128, F_l=64, F_int=64)

        # final conv (without any concat)
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


class AF_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AF_block, self).__init__()
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
        psi = self.psi(psi)

        return x * psi


class DAF(nn.Module):
    '''
     DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual
class iAFF(nn.Module):
    '''
    multi-feature fusion iAFF
    '''
    def __init__(self,F_g, F_l, F_int):
        super(iAFF, self).__init__()
        channels=F_int
        r=4
        inter_channels = int(channels // r)


        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # global  attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Second local attention
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # Second global  attention
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        x_ = self.W_g(x)
        residual_ = self.W_x(residual)
        xa = x_ + residual_
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x_ * wei + residual_ * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x_ * wei2 + residual_ * (1 - wei2)
        return xo



class AFF(nn.Module):
    '''
    multi-feature fusion AFF
    '''

    def __init__(self, F_g, F_l, F_int):
        super(AFF, self).__init__()
        channels = F_int
        r = 4
        inter_channels = int(channels // r)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        x_ = self.W_g(x)
        residual_ = self.W_x(residual)
        xa = x_ + residual_
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x_ * wei + 2 * residual_ * (1 - wei)
        return xo


class MS_CAM(nn.Module):
    '''
    Single feature channel weighting, similar to SE module
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei