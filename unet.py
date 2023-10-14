import colorsys
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Variable
from nets.ours_dattention import da_network
from nets.unet import Unet as unet,vgg_Unet_Attention,vgg_Unet_AF,Graph_network
from utils.metrics import f_score
import csv
from torchvision import transforms as T
from nets.unet_contrast import U_Net,AttU_Net,R2AttU_Net,R2U_Net
from nets.IAM_NOIHAM import da_network_noIHAM


#--------------------------------------------#
#   two parameters need to be modified to predict using a self trained model
#   model_path and num_classes need to be modified!
#   If there is a shape mismatch
#   be sure to pay attention to the model during training_ Path and num_ Modification of classes number
#--------------------------------------------#
class Unet(object):
    _defaults = {
        "model_path"        : './logs/log_f/iiam.pth',
        "model_image_size"  : (512,512, 3),
        "num_classes"       : 2,
        "cuda"              : True,
        #--------------------------------#
        #   The blend parameter is used to control whether to mix the recognition results with the original image
        #--------------------------------#
        "blend"             : False
    }

    #---------------------------------------------------#
    #   Initialize UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    #---------------------------------------------------#
    #   Obtain all categories
    #---------------------------------------------------#
    def generate(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        # self.net = unet(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()
        # self.net = vgg_Unet_Attention(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()
        # self.net = AttU_Net(img_ch=3,output_ch=2).eval()
        # self.net = U_Net(img_ch=3, output_ch=2).eval()  # unet
        # self.net = vgg_Unet_AF(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()
        # self.net = Graph_network(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()
        # self.net = da_network(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()
        self.net = da_network_noIHAM(num_classes=self.num_classes, in_channels=self.model_image_size[-1]).eval()
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('{} model loaded.'.format(self.model_path))

        if self.num_classes == 2:
            self.colors = [(255, 255, 255),  (0, 0, 0)]
        elif self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            # Set different colors for the picture frame
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                        for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self ,image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image,nw,nh
    #---------------------------------------------------#
    #   detect images
    #---------------------------------------------------#
    def detect_image(self, image):
        # make a backup of the original image
        old_img = copy.deepcopy(image)

        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))

        images = [np.array(image)/255]
        images = np.transpose(images,(0,3,1,2))

        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images =images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        if self.blend:
            image = Image.blend(old_img,image,0.7)
        
        return image




