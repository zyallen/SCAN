import os
import cv2
import numpy as np

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def save_box(img,save_path):
    cv2.imwrite(save_path,img)

def getbox(img_path,save_path):
    img = cv2.imread(img_path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for index in range(len(contours)):
        cnt = contours[index]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(img, [box], -1, (0, 0, 255), 2)

    # img = cv2.drawContours(img, contours, -1, (0,255,0), 3)  #-1 draw all contours
    # cv_show(img, 'img')

if __name__=='__main__':
    img_path = 'Medical_Datasets/test_GT/xxxx.jpg'