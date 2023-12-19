import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import *
import scipy.misc
from skimage.morphology import skeletonize_3d
from scipy import ndimage as ndi
from dvn.utils import get_itk_array, make_itk_image, write_itk_image, get_itk_image
# from batchgenerators.utilities.file_and_folder_operations import *
from skimage.io import imread
import SimpleITK as sitk
from evaluation import metric_dice, f1_socre, accuracy_bin, sensitivity, specificity, precision
from skimage.measure import label, regionprops
import nibabel as nib

#用于血管分析部分
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用编号为1，2号的GPU


def parse_args():
    parser = argparse.ArgumentParser(description='analyse binary vessel segmentation')
    parser.add_argument('--filenames1', dest='filenames1', type=str,
                        default='/public/yangxiaodu/vessap2/data_self/raw_seg_vessels_result.txt')
    # parser.add_argument('--test_labelFns', dest='test_labelFns', type=str, default='testing_labels_self.txt')
    parser.add_argument('--filenames2', dest='filenames2', type=str,
                        default='/public/yangxiaodu/vessap2/data_self/preprocessed_seg_vessels_result.txt')
    parser.add_argument('--maskFilename', dest='maskFn', type=str,
                        default=None,  #对应的稀疏标注，用来测试指标
                        help='a mask file to be applied to the predictions')
    parser.add_argument('--output', dest='output', type=str,
                        default='/public/yangxiaodu/nnunet/save_data/nnUNet_trained_models/nnUNet/3d_fullres/Task508_other_vessels_raw/nnUNetTrainerV2_loss_ignore_label2_cew_sb_CBAM_decoder_btc_moreDA_BN__nnUNetPlansv2.1/new_all_pre_add_out',
                        help='output folder for storing predictions (default: current working directory)')
    parser.add_argument('--f', dest='format', type=str, default='.nii.gz',
                        help='NIFTI file format for saving outputs (default: .nii.gz)')
    parser.add_argument('--txt', dest='txt', type=bool, default=False,
                        help='choose iuput way. If using txt files, choose txt=True (default: .nii.gz)')
    args = parser.parse_args()

    return args


def save_data(data, img, filename):
    out_img = make_itk_image(data, img)
    write_itk_image(out_img, filename)

def run():
    args = parse_args()
    outputFn = args.output
    txt=args.txt
    fmt = args.format
    filenames1 = args.filenames1
    filenames2 = args.filenames2
    masks = args.maskFn



    print('----------------------------------------')
    print(' Postprocessing Parameters ')
    print('----------------------------------------')
    print('txt',txt)
    print('Input files:', filenames1)
    print('Input files:', filenames2)
    print('Mask file:', masks)
    print('Output folder:', outputFn)

    print('Output format:', fmt)


    with open(os.path.abspath(args.filenames1)) as f:
        iFn1 = f.readlines()
    iFn1 = [x.strip() for x in iFn1]

    with open(os.path.abspath(args.filenames2)) as f:
        iFn2 = f.readlines()
    iFn2 = [x.strip() for x in iFn2]
    if masks is not None:
        with open(os.path.abspath(args.maskFn)) as f:
            mFn = f.readlines()
        mFn = [x.strip() for x in mFn]
    else:
        mFn=[]

    i = 0
    # dice_acc = {}
    # dice_acc1 = {}
    dice1={}
    f1score1 = {}
    f1score2={}
    acc_voxel1 = {}
    acc_voxel2 = {}
    sensitivity2={}
    specificity2={}
    precision2={}
    name = {}


    #
    #对于新数据，这种txt输入方法不太适用。
    source_folder = '/public/yangxiaodu/clearmap/data/postprocess_all1code/analysis'
    train_cases = subfiles(join(source_folder, 'patch'), suffix=".nii.gz", join=False)
    if txt==False:
        iFn1=[]
        iFn2=[]
        for i, t in enumerate(train_cases):
            iFn1.append(join(source_folder, 'patch',t))
            iFn2.append(join(source_folder, 'add_new_all_pre_add',t.split('.')[0][0:-22])+'.nii.gz')

        for ifn1, ifn2 in zip(iFn1, iFn2):
            prefix = os.path.basename(ifn1).split('.')[0]
            refined_patch = get_itk_array(ifn1)
            preprocessed_result = get_itk_array(ifn2)  # 检测每一个连通域

            sum_patch=np.sum(refined_patch)
            image=np.sum(preprocessed_result)
            ratio=sum_patch/image

            print(prefix,ratio)



if __name__ == '__main__':
    run()