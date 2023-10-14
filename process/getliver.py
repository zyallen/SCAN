import  os
from PIL import Image
from PIL import ImageChops
import numpy as np
import SimpleITK as sitk

def getliver_jpg(root,GT_paths):
    image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
    image_len = len(image_paths)
    for index in range(image_len):
        image_path = image_paths[index]
        filename = image_path.split('/')[-1][:-len(".jpg")]
        GT_path = GT_paths + filename + '.jpg'
        liver_path = './Medical_Datasets/liverlabel/' + filename + '.jpg'
        image = Image.open(image_path).convert('RGB')
        liver_labels1 = Image.open(liver_path).convert('RGB')
        image = ImageChops.multiply(image, liver_labels1)
        GT = Image.open(GT_path)
        GT_zero = np.array(GT)
        image_zero = np.array(image)

        if np.max(image_zero) != 0.:
            liver_labels2 = Image.open(liver_path)
            # GT = Image.open(GT_path)
            vessel_labels = ImageChops.multiply(GT, liver_labels2)
            image.save(
                './Medical_Datasets/Images/' + filename + '.jpg')
            vessel_labels.save(
                './Medical_Datasets/Labels/' + filename + '.jpg')

def getliver_nii2(image_path,liverseg_path,vessel1_path,vessel2_path):
    vessel1_paths = list(map(lambda x: os.path.join(vessel1_path, x), os.listdir(vessel1_path)))
    image_len = len(vessel1_paths)
    for name in vessel1_paths:
        filename = name.split('\\')[-1][:-len(".nii.gz")]
        vessel2_paths = vessel2_path + '/' + filename + '.nii.gz'
        image_paths = image_path + '/' + filename + '.nii.gz'
        liverseg_paths = liverseg_path + '/' + filename + '.nii.gz'
        ct = sitk.ReadImage(os.path.join(image_paths))
        liver_seg = sitk.ReadImage(os.path.join(liverseg_paths))
        vessel1 = sitk.ReadImage(os.path.join(name))
        vessel2 = sitk.ReadImage(os.path.join(vessel2_paths))

        origin =  ct.GetOrigin()
        direction =  ct.GetDirection()
        xyz_thickness =  ct.GetSpacing()

        vessel1_array = sitk.GetArrayFromImage(vessel1)
        vessel2_array = sitk.GetArrayFromImage(vessel2)


        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(liver_seg)

        vessel_array = (vessel1_array + vessel2_array) * seg_array
        vessel_array[vessel_array>=0.5] = 1
        vessel_array[vessel_array <0.5] = 0

        image_array=ct_array*seg_array
        image_save='D:/MedicalDataset/OpenData/3Diradb/imagenii/'+filename+'.nii.gz'
        vessel_save = 'D:/MedicalDataset/OpenData/3Diradb/vessel/' + filename + '.nii.gz'
        newImg = sitk.GetImageFromArray(image_array)
        newImg.SetOrigin(origin)
        newImg.SetDirection(direction)
        newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
        sitk.WriteImage(newImg, image_save)

        newImg1 = sitk.GetImageFromArray(vessel_array)
        newImg1.SetOrigin(origin)
        newImg1.SetDirection(direction)
        newImg1.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
        sitk.WriteImage(newImg, vessel_save)


def getliver_nii(image_path,liverseg_path,vessel1_path):
    vessel1_paths = list(map(lambda x: os.path.join(vessel1_path, x), os.listdir(vessel1_path)))
    image_len = len(vessel1_paths)
    for name in vessel1_paths:
        filename = name.split('\\')[-1][:-len(".nii.gz")]
        # vessel2_paths = vessel2_path + '/' + filename + '.nii.gz'
        image_paths = image_path + '/' + filename + '.nii.gz'
        liverseg_paths = liverseg_path + '/' + filename + '.nii.gz'
        ct = sitk.ReadImage(os.path.join(image_paths))
        liver_seg = sitk.ReadImage(os.path.join(liverseg_paths))
        vessel1 = sitk.ReadImage(os.path.join(name))
        # vessel2 = sitk.ReadImage(os.path.join(vessel2_paths))
        # int32类型保存会出错，需要转换成uint16类型
        ct = sitk.Cast(sitk.RescaleIntensity(ct), sitk.sitkUInt16)
        liver_seg = sitk.Cast(sitk.RescaleIntensity(liver_seg), sitk.sitkUInt16)
        vessel1 = sitk.Cast(sitk.RescaleIntensity(vessel1), sitk.sitkUInt16)

        origin =  ct.GetOrigin()
        direction =  ct.GetDirection()
        xyz_thickness =  ct.GetSpacing()

        vessel1_array = sitk.GetArrayFromImage(vessel1)
        # vessel2_array = sitk.GetArrayFromImage(vessel2)


        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(liver_seg)

        seg_array[seg_array >= 1] = 1
        seg_array[seg_array < 1] = 0
        vessel_array = vessel1_array * seg_array
        vessel_array[vessel_array>=1]=1
        vessel_array[vessel_array <1] = 0

        image_array=ct_array * seg_array
        image_save='./project/CBIM-Medical-Image-Segmentation/DataSet/bileduck/raw/img/'+filename+'.nii.gz'

        newImg = sitk.GetImageFromArray(image_array)
        newImg.SetOrigin(origin)
        newImg.SetDirection(direction)
        newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
        sitk.WriteImage(newImg, image_save)

        vessel_save = './project/CBIM-Medical-Image-Segmentation/DataSet/bileduck/raw/seg/'+filename+'.nii.gz'
        newImg1 = sitk.GetImageFromArray(vessel_array)
        newImg1.SetOrigin(origin)
        newImg1.SetDirection(direction)
        newImg1.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
        sitk.WriteImage(newImg1, vessel_save)


def getliver_jpg1(root,GT1_paths,GT2_paths):
    image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
    image_len = len(image_paths)
    for index in range(image_len):
        image_path = image_paths[index]
        filename = image_path.split('/')[-1][:-len(".jpg")]
        GT1_path = GT1_paths + filename + '.jpg'
        GT2_path=GT2_paths + filename + '.jpg'
        liver_path = 'D:/MedicalDataset/OpenData/3Diradb/Liver_seg/image/' + filename + '.jpg'
        image = Image.open(image_path).convert('RGB')
        liver_labels1 = Image.open(liver_path).convert('RGB')
        image = ImageChops.multiply(image, liver_labels1)
        GT1 = Image.open(GT1_path)
        GT2 = Image.open(GT2_path)
        GT=ImageChops.add(GT1,GT2)
        GT_zero = np.array(GT1)
        image_zero = np.array(image)

        if np.max(image_zero) != 0.:
            liver_labels2 = Image.open(liver_path)
            # GT = Image.open(GT_path)
            vessel_labels = ImageChops.multiply(GT, liver_labels2)
            vessel_labels=vessel_labels.convert('L')
            image.save(
                'D:/MedicalDataset/OpenData/3Diradb/dataset/images/' + filename + '.jpg')
            vessel_labels.save(
                'D:/MedicalDataset/OpenData/3Diradb/dataset/labels/' + filename + '.jpg')

if __name__=='__main__':
    # root = 'D:/MedicalDataset/OpenData/3Diradb/Patient/image/'
    # GT_paths = 'D:/MedicalDataset/OpenData/3Diradb/Patient/image/'
    # getliver_jpg(root, GT_paths)

    # root = 'D:/MedicalDataset/OpenData/3Diradb/Patient/image/'
    # GT1_paths = 'D:/MedicalDataset/OpenData/3Diradb/vessel_1/image/'
    # GT2_paths = 'D:/MedicalDataset/OpenData/3Diradb/vessel_2/image/'
    # getliver_jpg1(root, GT1_paths,GT2_paths)

    image_path='./project/CBIM-Medical-Image-Segmentation/DataSet/bileduck/raw_origin/img'
    liverseg_path='./20220420/data/liverlabel_nii'
    vessel1_path='./project/CBIM-Medical-Image-Segmentation/DataSet/bileduck/raw_origin/seg'
    getliver_nii(image_path, liverseg_path, vessel1_path)
