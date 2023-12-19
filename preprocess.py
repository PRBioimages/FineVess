# performing preprocessing of new raw data
import cv2
import numpy as np
from matplotlib import pyplot as plt
from dvn.utils import get_itk_array, get_itk_image, make_itk_image, write_itk_image
import os as os
import argparse
from cv2_rolling_ball import subtract_background_rolling_ball
import tensorflow as tf
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk


source_folder = "/home/xdyang/FineVess/data/raw_data"#"/public/yangxiaodu/nnunet/save_data/nnUNet_raw_data_base/nnUNet_raw_data"
output_fold="/home/xdyang/FineVess/data/pre_data/"#"/public/yangxiaodu/nnunet/save_data/nnUNet_raw_data_base/nnUNet_raw_data/all_data_preprocessed/"
test_cases = subfiles(source_folder, suffix=".nii.gz", join=False)
for t in test_cases:
    img_file = join(source_folder, t)
    img = get_itk_array(img_file)
    imgCLAHE = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    roll_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    background = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    img = img.astype(np.uint8)

    constrast = {}
    brightness = {}
    prefix = os.path.basename(img_file).split('.')[0]
    print(prefix)
    for i in range(img.shape[0]):
        constrast[i] = img[i, :, :].std()
        brightness[i] = img[i, :, :].mean()
        # print(prefix, constrast, brightness)
    constrast = list(constrast.values())
    brightness = list(brightness.values())
    if max(constrast) <= 12 and max(brightness) <= 10:
        print(max(constrast), max(brightness))
        print("This volume need perform CLAHE")
        Flag=True
    else:
        Flag=False

    for i in range(img.shape[0]):
        if Flag:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            imgCLAHE[i, :, :] = clahe.apply(img[i, :, :])
        else:
            imgCLAHE[i,:,:] = img[i,:,:]

        roll_img[i,:,:], background[i,:,:] = subtract_background_rolling_ball(imgCLAHE[i,:,:], 50, light_background=False, use_paraboloid=False,
                                                       do_presmooth=True)
    roll_img = roll_img.astype(np.float32)

    image = make_itk_image(roll_img)

    nii_filename= output_fold+prefix+ '.nii.gz'

    write_itk_image(image, nii_filename)




