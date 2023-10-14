#  '''
# There are several points to note about predict.py
# 1. Batch prediction is not possible. If you want to make a batch prediction, you can use os. listdir() to traverse the folder and use Image.open to open the image file for prediction.
# 2. If you want to save, use r_ Image. save ("img. jpg") can be saved.
# 3. If you want the original and split images to not mix, you can set the blend parameter to False.
# 4. If you want to obtain the corresponding area based on the mask, you can refer to detect_image, the part that uses the predicted results to plot.
# seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
# for c in range(self.num_classes):
#     seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
#     seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
#     seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
# '''
import os
from PIL import Image
import torch
from unet import Unet
from nets.unet_contrast import U_Net,AttU_Net,R2AttU_Net,R2U_Net
from torch.autograd import Variable
from utils.metrics import f_score
from torch.utils.data import DataLoader
from utils.dataloader_medical1 import DeeplabDataset, deeplab_dataset_collate
import time
from process.niigz2img import jpgtoniigz
unet = Unet()
# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = unet.detect_image(image)
#         r_image.show()
# inputs_size = [512,512,3]
# NUM_CLASSES = 2
# with open(r"./Medical_Datasets/ImageSets/Segmentation/test.txt", "r") as f:
#     test_lines = f.readlines()
# test_dataset = DeeplabDataset(test_lines, inputs_size, NUM_CLASSES, True)
# gentest = DataLoader(test_dataset, num_workers=0, pin_memory=True,
#                             drop_last=True, collate_fn=deeplab_dataset_collate)
start_time = time.time()
test_path='./Medical_Datasets/new_test/test_img/'
# test_path='./Medical_Datasets/new_test/GUOAIYU/GUOAIYU/'
image_paths = list(map(lambda x: os.path.join(test_path, x), os.listdir(test_path)))
image_len = len(image_paths)
for i in image_paths:
    file=i.split('/')[-1]
    test_paths = list(map(lambda x: os.path.join(i, x), os.listdir(i)))
    print(test_paths)
    for index in range(len(test_paths)):
        image_path = test_paths[index]
        filename = image_path.split('\\')[-1][:-len(".jpg")]
        # label_path=GT_path+ filename +'.jpg'
        test_image = Image.open(image_path)
        # label_image=Image.open(label_path)

        Result = unet.detect_image(test_image)
        # temp='Medical_Datasets/new_test/result/'+file+ '/' + filename + '.jpg'
        Result.save('Medical_Datasets/new_test/result/'+file+ '/' + filename + '.jpg')

        # Result.save(file + '/' + filename + '.jpg')
        # Result.save('Medical_Datasets/new_test/our/pred/' + filename + '.jpg')
        # -----------------------------------------#
        #   Calculate the accuracy function of the test set and save the training results
        # -----------------------------------------#
        # (_f_score, precision, recall) = unet.get_score(test_image, label_image)

waste_time=time.time()-start_time
print(waste_time)
jpgpath = './Medical_Datasets/new_test/result/'
niipath = './Medical_Datasets/new_test/result/'
jpgtoniigz(jpgpath, niipath)
