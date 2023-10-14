import numpy as np
import os
import nibabel as nib
import imageio
import cv2

# class ImageRename():
#     def __init__(self):
#         self.path = path


def rename(path):
    image_paths = os.listdir(path)
    # image_paths = list(map(lambda x: os.path.join(self.path, x), os.listdir(self.path)))
    # image_paths.sort(cmp=None, key=None, reverse=False)
    total_num = len(image_paths)
    i = 0
    for item in image_paths:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path), item)
            filename = int(item.split('.')[0])
            dst = os.path.join(os.path.abspath(path), format(str(filename)) + '.jpg')
            os.rename(src, dst)
            print(src, dst)
        i = i + 1
    print(total_num, i)

def renamelabel(path1,path2):
    image_paths = os.listdir(path1)
    total_num = len(image_paths)
    i = 0
    for item in image_paths:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path1), item)
            filename = (total_num-int(item.split('.')[0]))
            dst = os.path.join(os.path.abspath(path2), format(str(filename)) + '.jpg')
            img=cv2.imread(src)
            cv2.imwrite(dst,img)
            # os.rename(src, dst)
            print(int(item.split('.')[0]),filename)
        i = i + 1
    print(total_num, i)


if __name__ == '__main__':
    path='./dataset/train/labels/'
    path2='./dataset/train/label/'
    path_dir=os.listdir(path)
    total_dir=len(path_dir)
    i = 0
    for item in path_dir:
        path1 = path+path_dir[i]+'/'
        path_2=path2+path_dir[i]+'/'
        # rename(path1) # rename image
        renamelabel(path1,path_2)
        i=i+1
    print("end")