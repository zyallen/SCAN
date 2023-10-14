import  os
from PIL import Image
from PIL import ImageChops
import numpy as np


def getpatch(root):
    image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
    image_len = len(image_paths)

    for index in range(image_len):
        image_path = image_paths[index]
        filename = image_path.split('/')[-1][:-len(".jpg")]
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)
        w,h=image.size
        item_width = int(w/ 4)
        item_height = int(h/ 4)
        box_list = []
        # (left, upper, right, lower)
        for j in range(0, 4):
            for i in range(0, 4):
                box = (i * item_width, j*item_height, (i + 1) * item_width, (j+1)*item_height)
                box_list.append(box)
        image_list = [image.crop(box) for box in box_list]

        index = 0
        for image in image_list:
            image.save('./Medical_Datasets/Labels_GetPatch/'+filename+'_'+str(index) + '.jpg')
            index += 1


if __name__=='__main__':
    image_path = './Medical_Datasets/Images/'
    GT_path = './Medical_Datasets/Labels/'
    getpatch(GT_path)
