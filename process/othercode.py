import cv2
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
def ROI(img):
    imgray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img1 = cv2.drawContours(imgray, contours, -1, (255, 255, 255), 3)

    x, y, w, h = cv2.boundingRect(contours[0])
    xw=x+w
    yh=y+h
    for i in range(len(contours)):
        if i!=0:
            x1,y1,w1,h1=cv2.boundingRect(contours[i])
            if x1<x: x=x1
            if y1<y: y=y1
            if (x1+w1)>xw: xw=(x1+w1)
            if (y1+h1)>yh: yh=(y1+h1)
    # img2 = cv2.rectangle(imgray, (x, y), (xw,yh), (255, 255, 0), 2)
    img2=np.array(imgray)
    img2_ = img2[114:146,293:325]
    # img2_=img2[y:yh,x:xw]  #h w
    image=Image.fromarray(img2_)

    plt.figure("img")
    plt.imshow(image)
    plt.show()
    # cv2.imshow('img',image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return x,y,xw,yh



if __name__=='__main__':
    img = Image.open(r"./Medical_Datasets/Images/xxxx.jpg").convert('RGB')
    ROI(img)