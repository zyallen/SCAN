import numpy as np
import os
import cv2

def reflip(path):
    image_paths = os.listdir(path)
    # image_paths = list(map(lambda x: os.path.join(self.path, x), os.listdir(self.path)))
    total_num = len(image_paths)
    i = 0
    for item in image_paths:
        if item.endswith('.jpg'):
            img = cv2.imread(path + image_paths[i])
            img = np.fliplr(img)
            img = np.rot90(img, 1)
            cv2.imwrite(path+image_paths[i], img)
        i = i + 1
    print(total_num, i)

if __name__=='__main__':
    path = './Image_Segmentation-master/dataset/train/label/'
    path_dir = os.listdir(path)
    total_dir = len(path_dir)
    i = 0
    for item in path_dir:
        path1 = path + path_dir[i] + '/'
        reflip(path1)
        i = i + 1
        print("end")
