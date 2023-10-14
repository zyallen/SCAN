#coding=utf-8
import SimpleITK as sitk
import os


def dcm2nii(dcms_path, nii_path):
    # 1. Build a dicom sequence file reader and execute (i.e. "packaging and integration" of dicom sequence files)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    # 2. Convert the integrated data into an array and obtain the basic information of the dicom file
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3. Convert the array to img and save it as. nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)



if __name__ == '__main__':
    dcms_path = r'D:/MedicalDataset/OpenData/3Dircadb1'  # path where the dicom sequence file is located

    dcms_paths = list(map(lambda x: os.path.join(dcms_path, x), os.listdir(dcms_path)))
    for index in range(len(dcms_paths)):
        # patient_nii
        dcms_path1=dcms_paths[index]+'/PATIENT_DICOM/PATIENT_DICOM'
        nii_path = r'D:/MedicalDataset/OpenData/3Diradb/Patient/'+str(index+1)+'.nii.gz'  # required. nii.gz file save path
        dcm2nii(dcms_path1, nii_path)
        # liver_nii
        dcms_path1 = dcms_paths[index] + '/MASKS_DICOM/MASKS_DICOM/liver'
        nii_path = r'D:/MedicalDataset/OpenData/3Diradb/liver_seg/'+str(index+1)+'.nii.gz'   # required .nii.gz file save path
        dcm2nii(dcms_path1, nii_path)
        # vessel1_nii
        dcms_path1 = dcms_paths[index] + '/MASKS_DICOM/MASKS_DICOM/venoussystem' #hepatic vessel
        nii_path = r'D:/MedicalDataset/OpenData/3Diradb/vessel_1/' +str(index+1)+'.nii.gz'  # required .nii.gz file save path
        if(os.path.exists(dcms_path1)):
            dcm2nii(dcms_path1, nii_path)
        # vessel2_nii
        dcms_path1 = dcms_paths[index] + '/MASKS_DICOM/MASKS_DICOM/portalvein' #hepatic vessel
        nii_path = r'D:/MedicalDataset/OpenData/3Diradb/vessel_2/'+str(index+1)+'.nii.gz'   # required .nii.gz file save path
        if (os.path.exists(dcms_path1)):
            dcm2nii(dcms_path1, nii_path)


