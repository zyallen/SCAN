import  os
from PIL import Image
from PIL import ImageChops
import numpy as np
import nibabel as nib  # nii format usually uses this package
import imageio

import SimpleITK as sitk
import glob
import cv2

def toRGB(root):
    # root = './Image_Segmentation/Image_Segmentation-master/dataset/images' #single channel image directory
    image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
    image_len = len(image_paths)
    for index in range(image_len):
        image_path = image_paths[index]
        filename = image_path.split('\\')[-1][:-len(".jpg")]
        # filename = image_path.split('\\')[-1][:-len(".jpg")] # test dataset conversion
        image = Image.open(image_path).convert('RGB')
        image.save(root +'/'+ filename + '.jpg')

def nii_to_jpg(filepath,imgfile):
    # filepath = './Image_Segmentation/Image_Segmentation-master/dataset/nii'
    # imgfile = './Image_Segmentation/Image_Segmentation-master/dataset/labels'
    filenames = os.listdir(filepath)  # read nii folder
    slice_trans = []

    for f in filenames:
        # Start reading nii file
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  # reading nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii.gz', '')
        img_f_path = os.path.join(imgfile, fname)


        # Start converting to image
        (x, y, z) = img.shape
        print(img.shape)
        # image = np.expand_dims(a, axis=2)
        # img_f_path=img_f_path+fname+'_'
        for i in range(z):  # z is a sequence of images
            silce = img_fdata[:, :, i]
            # slice1=np.expand_dims(slice,axis=2)
            # slice2 = np.concatenate((slice1, slice1, slice1), axis=-1)
            imageio.imwrite(os.path.join(img_f_path + '_{}.jpg'.format(i)), silce)
            # 保存图像

def jpgtoniigz(imgpath,savepath):
    image_paths = list(map(lambda x: os.path.join(imgpath, x), os.listdir(imgpath)))
    image_len = len(image_paths)
    for index in range(image_len):
        image_path=list(map(lambda x: os.path.join(image_paths[index], x), os.listdir(image_paths[index])))
        n=len(image_path)
        filename = image_paths[index].split('/')[-1]
        allImg = []
        savepath1=savepath+filename+'.nii.gz'
        for ii in range(n):
            imagepath=image_paths[index]+'/'+filename+'_'+str(ii)+'.jpg'
            image=Image.open(imagepath).convert('L')
            img1=np.array(image)
            img1[img1 <= 127.5] = 0
            img1[img1 > 127.5] = 1
            img = np.fliplr(img1)
            img = np.rot90(img, 1)
            allImg.append(img)

        allimg=np.array(allImg)

        nii_file = sitk.GetImageFromArray(allimg)
        sitk.WriteImage(nii_file, savepath1)

def nii_removezero(filepath,savepath):
    # filepath = './Image_Segmentation/Image_Segmentation-master/dataset/nii'
    # imgfile = './Image_Segmentation/Image_Segmentation-master/dataset/labels'
    filenames = os.listdir(filepath + 'seg')
    slice_trans = []

    for f in filenames:

        # liver_paths=filepath+'seg'
        # liver_path = os.path.join(liver_paths, f)
        # liver = nib.load(liver_path)  # 读取nii
        # liver_fdata = liver.get_fdata()

        liver_paths = filepath + 'seg'
        liver_path = os.path.join(liver_paths, f)
        liver = sitk.ReadImage(liver_path)
        # liver = sitk.Cast(sitk.RescaleIntensity(liver), sitk.sitkUInt8)
        origin = liver.GetOrigin()
        direction = liver.GetDirection()
        xyz_thickness = liver.GetSpacing()
        liver_fdata = sitk.GetArrayFromImage(liver)

        # Starting to convert to image SimpleITK loading data is channel_first
        (z, y, x) = liver_fdata.shape
        print(liver_fdata.shape)
        min_x=0
        max_x=x
        min_y=0
        max_y=y
        min_z=0
        max_z=z
        # for i in range(x):
        #     if np.amax(liver_fdata[i, :, :]) > 0.0:
        #         min_x=i
        #         break
        # for i in reversed(range(x)):
        #     if np.amax(liver_fdata[i, :, :]) > 0.0:
        #         max_x = i
        #         break
        # for i in range(y):  # y
        #     if np.amax(liver_fdata[:, i, :]) > 0.0:
        #         min_y=i
        #         break
        # for i in reversed(range(y)):  # y
        #     if np.amax(liver_fdata[:, i, :]) > 0.0:
        #         max_y = i
        #         break
        for i in range(z):
            if np.amax(liver_fdata[i,:, :]) > 0.0:
                min_z=i
                break
        for i in reversed(range(z)):
            if np.amax(liver_fdata[i,:, :]) > 0.0:
                max_z = i
                break
        temp_liver=liver_fdata[min_z:max_z,min_y:max_y,min_x:max_x]
        nii_liver = sitk.GetImageFromArray(temp_liver)
        print(nii_liver)
        nii_liver.SetOrigin(origin)
        nii_liver.SetDirection(direction)
        nii_liver.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
        sitk.WriteImage(nii_liver, savepath+'seg/'+f)

        # save image、label、source
        image_paths = filepath + 'img'
        image_path = os.path.join(image_paths, f)
        image = sitk.ReadImage(image_path)
        image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
        image_fdata = sitk.GetArrayFromImage(image)

        temp_image = image_fdata[min_z:max_z,min_y:max_y,min_x:max_x]
        nii_image = sitk.GetImageFromArray(temp_image)
        nii_image.SetOrigin(origin)
        nii_image.SetDirection(direction)
        nii_image.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
        sitk.WriteImage(nii_image, savepath + 'img/' + f)

        # label_paths = filepath + 'label'
        # label_path = os.path.join(label_paths, f)
        # label = nib.load(label_path)
        # label_fdata = label.get_fdata()
        #
        # temp_label = label_fdata[min_x:max_x,min_y:max_y,min_z:max_z]
        # nii_label = sitk.GetImageFromArray(np.swapaxes(temp_label, 2, 0))
        # sitk.WriteImage(nii_label, savepath + 'label/' + f)
        #
        # source_paths = filepath + 'source'
        # source_path = os.path.join(source_paths, f)
        # source = nib.load(source_path)
        # source_fdata = source.get_fdata()
        #
        # temp_source = source_fdata[min_x:max_x,min_y:max_y,min_z:max_z]
        # nii_source = sitk.GetImageFromArray(np.swapaxes(temp_source, 2, 0))
        # sitk.WriteImage(nii_source, savepath + 'source/' + f)


if __name__ == '__main__':
    # filepath='./Medical_Datasets/new_test/new_seg'
    # imgfile = './Medical_Datasets/new_test/new_seg_img'
    # nii_to_jpg(filepath,imgfile)
    # toRGB(imgfile)


    # imgpath='./Medical_Datasets/new_test/result/'
    # savepath='./Medical_Datasets/new_test/result/'
    # jpgtoniigz(imgpath,savepath)

    imgpath='./DataSet/bileduck/raw/'
    savepath='./DataSet/bileduck/raw/'
    nii_removezero(imgpath,savepath)