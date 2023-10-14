from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import matplotlib.pyplot as plt

def letterbox_image(image, label , size):
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))

    return new_image, new_label

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class DeeplabDataset(Dataset):
    def __init__(self,train_lines,image_size,num_classes,random_data):
        super(DeeplabDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes
        self.random_data = random_data

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.1, hue=.0, sat=1.1, val=1.1):
        image = image.convert("RGB")
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.5,1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label = label.convert("L")
        
        # flip image or not
        flip = rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data,label


    def __getitem__(self, index):
        # if index == 0:
        #     shuffle(self.train_lines)
            
        annotation_line = self.train_lines[index]
        name = annotation_line.split()[0]
        # reading images from files
        # jpg = Image.open(r"./Medical_Datasets/Images" + '/' + name + ".bmp")
        # png = Image.open(r"./Medical_Datasets/Labels" + '/' + name + ".png")
        jpg = Image.open(r"./Medical_Datasets/Images" + '/' + name + ".jpg")
        png = Image.open(r"./Medical_Datasets/Labels" + '/' + name + ".jpg")
        if self.random_data:
            jpg, png = self.get_random_data(jpg,png,(int(self.image_size[1]),int(self.image_size[0])))
        else:
            jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]),int(self.image_size[0])))

        # reading images from files
        png = np.array(png)

        modify_png = np.zeros_like(png)
        modify_png[png <= 127.5] = 1
        
        seg_labels = modify_png
        seg_labels = np.eye(self.num_classes+1)[seg_labels.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes+1))

        jpg = np.transpose(np.array(jpg),[2,0,1])/255
        return jpg, modify_png, seg_labels, name
        # return jpg, modify_png, seg_labels,name,x,y,xw,yh

# def ROI(img):
#     imgray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # img1 = cv2.drawContours(imgray, contours, -1, (255, 255, 255), 3)
#     x, y, w, h = cv2.boundingRect(contours[0])
#     xw = x + w
#     yh = y + h
#     for i in range(len(contours)):
#         if i != 0:
#             x1, y1, w1, h1 = cv2.boundingRect(contours[i])
#             if x1 < x: x = x1
#             if y1 < y: y = y1
#             if (x1 + w1) > xw: xw = (x1 + w1)
#             if (y1 + h1) > yh: yh = (y1 + h1)
#     # img2 = cv2.rectangle(imgray, (x, y), (xw,yh), (255, 255, 0), 2)
#     # cv2.imshow('img',img2)
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()
#     return x, y, xw, yh

# def get_realROI(x,y,xw,yh):
#
#     return r_x,r_y,r_xw,r_yh


# use of collate_fn in DataLoader
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    names=[]

    for img, png, labels,name in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
        names.append(name)

    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)

    return images, pngs, seg_labels,names