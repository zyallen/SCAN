import SimpleITK as sitk
import numpy as np
import os

def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_name)


def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


image_path = './temp'
# label_path='./liver_dataset/nii_labels'
saved_path = './temp'


image_paths = list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path)))
# name_list = ['volume-0.nii']
for name in image_paths:
    filename = name.split('\\')[-1][:-len(".nii.gz")]
    # label_paths = label_path+'/'+ filename + '_seg.nii.gz'
    # label_paths = label_path + '/' + filename + '.nii.gz'
    ct = sitk.ReadImage(os.path.join(image_path, filename))
    # seg= sitk.ReadImage(os.path.join(label_paths))
    origin = ct.GetOrigin()
    direction = ct.GetDirection()
    xyz_thickness = ct.GetSpacing()
    ct_array = sitk.GetArrayFromImage(ct)
    # seg_array = sitk.GetArrayFromImage(seg)
    # seg_bg = seg_array < 1
    # seg_vessel = seg_array >= 1
    # ct_bg = ct_array * seg_bg
    # ct_vessel = ct_array * seg_vessel

    # vessel_min = ct_vessel.min()
    # vessel_max = ct_vessel.max()

    # by liver
    # vessel_wide = vessel_max - vessel_min
    # vessel_center = (vessel_max + vessel_min) / 2
    vessel_wide = 400
    vessel_center = 30
    vessel_wl = window_transform(ct_array, vessel_wide, vessel_center, normal=True)
    saved_name = os.path.join(saved_path, filename+'.nii.gz')
    saved_preprocessed(vessel_wl, origin, direction, xyz_thickness, saved_name)

