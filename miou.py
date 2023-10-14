import argparse
import json
from os.path import join

import numpy as np
from PIL import Image


# Set label width W and length H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):  
    print('Num classes', num_classes)  

    #-----------------------------------------#
    #   Create a matrix with all zeros, which is a confusion matrix
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   Obtain a list of validation set label paths for direct reading
    #   Obtain a path list of validation set image segmentation results for easy direct reading
    #------------------------------------------------#
    gt_imgs = [join(gt_dir, x + ".jpg") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".jpg") for x in png_name_list]

    #------------------------------------------------#
    #   Read each (image label)
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   Read an image segmentation result and convert it into a numpy array
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   Read a corresponding label and convert it into a numpy array
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # If the image segmentation result is different from the size of the label, this image will not be calculated
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
        # Output the average mIoU value of all categories in the currently calculated images for every 10 calculated images
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                    100 * np.nanmean(per_class_iu(hist)),
                                                    100 * np.nanmean(per_class_PA(hist))))
    #------------------------------------------------#
    #   Calculate the mIoU values per category for all validation set images
    #------------------------------------------------#
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    #------------------------------------------------#
    #   Output mIoU values by category
    #------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(round(mPA[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   Calculate the average mIoU value of all categories on all validation set images
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))  
    return mIoUs


if __name__ == "__main__":
    gt_dir = r".\Medical_Datasets\new_test\our\label"
    pred_dir = r".\Medical_Datasets\new_test\our\pred"
    png_name_list = open("./Medical_Datasets/ImageSets/Segmentation/test.txt",'r').read().splitlines()
    #------------------------------#
    #   classification number+1
    #   2+1
    #------------------------------#
    num_classes = 2
    #--------------------------------------------#
    #   Distinguished types
    #--------------------------------------------#
    name_classes = ["background","vessel"]
    compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes)
