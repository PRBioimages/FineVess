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


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing-based segmentation refinement')
    parser.add_argument('--filenames1', dest='filenames1', type=str,
                        default='/public/yangxiaodu/vessap2/data_self/raw_seg_vessels_result.txt')
    parser.add_argument('--filenames2', dest='filenames2', type=str,
                        default='/public/yangxiaodu/vessap2/data_self/preprocessed_seg_vessels_result.txt')
    parser.add_argument('--maskFilename', dest='maskFn', type=str,
                        default=None,
                        help='a mask file to be applied to the predictions')
    parser.add_argument('--output', dest='output', type=str,
                        default='/home/xdyang/FineVess/results/pre_refine',
                        help='output folder for storing refinement results (default: current working directory)')
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
    dice1={}
    acc_voxel1 = {}
    sensitivity2={}
    specificity2={}
    precision2={}
    name = {}

    source_folder = '/home/xdyang/FineVess/results/nnunet'
    train_cases = subfiles(join(source_folder, 'raw_data_seg_results'), suffix=".nii.gz", join=False)
    if txt==False:
        iFn1=[]
        iFn2=[]
        mFn=[]
        for i, t in enumerate(train_cases):
            iFn1.append(join(source_folder, 'raw_data_seg_results',t))
            iFn2.append(join(source_folder, 'pre_data_seg_results',t))
    if mFn!=[]:
        for ifn1, ifn2,mfn in zip(iFn1, iFn2, mFn):
            prefix = os.path.basename(ifn1).split('.')[0]
            raw_result = get_itk_array(ifn1)
            preprocessed_result = get_itk_array(ifn2)
            # dice_test = round(metric_dice(raw_result, preprocessed_result), 4)
            # print(prefix, dice_test)
            dice = []
            for j in range(raw_result.shape[0]):
                dice.append(round(metric_dice(raw_result[j, :, :], preprocessed_result[j, :, :]), 4))
            # print(prefix, dice)

            fig = plt.figure()

            plt.plot(dice)

            plt.title(prefix)
            # plt.title('acc')
            plt.ylabel('dice')
            # plt.ylabel('acc')
            plt.xlabel('slice')
            # plt.legend(loc='lower right')

            y1_min = np.argmin(dice)
            y1_max = np.argmax(dice)
            show_min = '[' + str(y1_min) + ' ' + str(dice[y1_min]) + ']'
            show_max = '[' + str(y1_max) + ' ' + str(dice[y1_max]) + ']'
            plt.plot(y1_min, dice[y1_min], 'ko')
            plt.plot(y1_max, dice[y1_max], 'ko')
            plt.annotate(show_min, xy=(y1_min, dice[y1_min]), xytext=(y1_min, dice[y1_min]))
            plt.annotate(show_max, xy=(y1_max, dice[y1_max]), xytext=(y1_max, dice[y1_max]))

            plt.show()
            # predict
            fig.savefig(outputFn+"/dice_plot/"+ prefix + ".jpg")


            prefix = os.path.basename(ifn1).split('.')[0]
            raw_result = get_itk_array(ifn1)
            preprocessed_result=get_itk_array(ifn2)
            print(dice[y1_min])
            if dice[y1_min] >= 0.4:
                print(prefix)
                preprocessed_result_int = preprocessed_result.astype(np.int)
                label_img, num = label(preprocessed_result_int, connectivity=preprocessed_result.ndim, return_num=True)
                preprocessed_region = regionprops(label_img)

                pre_region_all_coord = []
                for o in range(len(preprocessed_region)):
                    coord_list = preprocessed_region[o].coords
                    for v in range(len(coord_list)):
                        pre_region_all_coord.append(tuple([coord_list[v][0], coord_list[v][1], coord_list[v][2]]))

                # record the coordinates
                raw_result_int = raw_result.astype(np.int)
                label_img, num = label(raw_result_int, connectivity=raw_result_int.ndim, return_num=True)
                raw_region = regionprops(label_img)


                raw_region_all_coord = []
                for o in range(len(raw_region)):
                    coord_list = raw_region[o].coords
                    for v in range(len(coord_list)):
                        raw_region_all_coord.append(tuple([coord_list[v][0], coord_list[v][1], coord_list[v][2]]))


                for o in range(len(preprocessed_region)):
                    coord_list2 = preprocessed_region[o].coords
                    preprocessed_region_list2 = []
                    for v in range(len(coord_list2)):
                        preprocessed_region_list2.append(tuple([coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]]))
                    if len(list(set(preprocessed_region_list2).intersection(set(raw_region_all_coord)))) == 0:
                        for v in range(len(coord_list2)):
                            preprocessed_result[coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]] = 0

                for o in range(len(raw_region)):
                    coord_list2 = raw_region[o].coords
                    raw_region_list2 = []
                    for v in range(len(coord_list2)):
                        raw_region_list2.append(tuple([coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]]))
                    if len(list(set(raw_region_list2).intersection(set(pre_region_all_coord)))) == 0:
                        for v in range(len(coord_list2)):
                            preprocessed_result[coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]] = 1

            else:
                preprocessed_result =raw_result

            preprocessed_result = preprocessed_result.astype(np.float32)
            ofn2 = os.path.join(outputFn + '/' + prefix + '.nii.gz')
            save_data(data=preprocessed_result, img=get_itk_image(ifn1), filename=ofn2)


            mask = get_itk_array(mfn)
            index = []
            for ind in range(mask.shape[0]):
                if np.any(mask[ind, :, :] != 2) == True:
                    # if np.any(mask[ind, :, :]) == True:
                    index.append(ind)
            # print(index)
            data1 = preprocessed_result[index, :, :]
            mask1 = mask[index, :, :]
            dice1[i] = round(metric_dice(mask1, data1), 4)
            acc_voxel1[i] = round(accuracy_bin(mask1, data1), 4)
            sensitivity2[i] = round(sensitivity(mask1, data1), 4)
            specificity2[i] = round(specificity(mask1, data1), 4)
            precision2[i] = round(precision(mask1, data1), 4)
            name[i] = prefix
            i = i + 1
        Len = {}
        Sum = {}
        Avg = {}
        for k, p in enumerate([dice1, acc_voxel1,sensitivity2, specificity2, precision2]):
            Len[k] = len(p)
            Sum[k] = sum(p.values())
            Avg[k] = Sum[k] / Len[k]
        path = args.output + '/metric_other.txt'
        f = open(path, 'w')
        f.writelines([path, "\n", "dice1:", str(dice1), str(Avg[0]),
                      # "\n", "f1_score1:",str(f1score1),str(Avg[1]),"\n","f1score2:",str(f1score2),str(Avg[2]),
                      "\n", "acc_voxel1:", str(acc_voxel1), str(Avg[1]),  # "\n","acc_voxel2:",str(acc_voxel2),str(Avg[4]),
                      "\n", "sensitivity2:", str(sensitivity2), str(Avg[2]), "\n", "specificity2:", str(specificity2),
                      str(Avg[3]),
                      "\n", "precision2:", str(precision2), str(Avg[4]), "\n", "name:", str(name)])
        f.close()
        # print("dice_probs", dice_acc)
        print("dice1", dice1)
        print("acc_voxel1", acc_voxel1)
        print("sensitivity2", sensitivity2)
        print("specificity2", specificity2)
        print("precision2", precision2)
        print('finished!')
    else:
        for ifn1, ifn2 in zip(iFn1, iFn2):
            prefix = os.path.basename(ifn1).split('.')[0]
            raw_result = get_itk_array(ifn1)
            preprocessed_result = get_itk_array(ifn2)
            # dice_test = round(metric_dice(raw_result, preprocessed_result), 4)
            # print(prefix, dice_test)
            dice = []
            for j in range(raw_result.shape[0]):
                dice.append(round(metric_dice(raw_result[j, :, :], preprocessed_result[j, :, :]), 4))
            # print(prefix, dice)

            fig = plt.figure()

            plt.plot(dice)

            plt.title(prefix)
            # plt.title('acc')
            plt.ylabel('dice')
            # plt.ylabel('acc')
            plt.xlabel('slice')
            # plt.legend(loc='lower right')

            y1_min = np.argmin(dice)
            y1_max = np.argmax(dice)
            show_min = '[' + str(y1_min) + ' ' + str(dice[y1_min]) + ']'
            show_max = '[' + str(y1_max) + ' ' + str(dice[y1_max]) + ']'
            plt.plot(y1_min, dice[y1_min], 'ko')
            plt.plot(y1_max, dice[y1_max], 'ko')
            plt.annotate(show_min, xy=(y1_min, dice[y1_min]), xytext=(y1_min, dice[y1_min]))
            plt.annotate(show_max, xy=(y1_max, dice[y1_max]), xytext=(y1_max, dice[y1_max]))

            plt.show()
            # predict
            # save_path = '/public/yangxiaodu/nnunet/save_data/nnUNet_trained_models/nnUNet/3d_fullres/Task508_other_vessels_raw/nnUNetTrainerV2_loss_ignore_label2_cew_sb_CBAM_decoder_btc_moreDA_BN__nnUNetPlansv2.1/new_all_pre_add_out/dice_plot/'
            fig.savefig(outputFn+"/dice_plot/"+ prefix + ".jpg")

            prefix = os.path.basename(ifn1).split('.')[0]
            raw_result = get_itk_array(ifn1)
            preprocessed_result = get_itk_array(ifn2)
            print(dice[y1_min])
            if dice[y1_min] >= 0.4:
                print(prefix)
                preprocessed_result_int = preprocessed_result.astype(np.int)
                label_img, num = label(preprocessed_result_int, connectivity=preprocessed_result.ndim, return_num=True)
                preprocessed_region = regionprops(label_img)

                pre_region_all_coord = []
                for o in range(len(preprocessed_region)):
                    coord_list = preprocessed_region[o].coords
                    for v in range(len(coord_list)):
                        pre_region_all_coord.append(tuple([coord_list[v][0], coord_list[v][1], coord_list[v][2]]))

                raw_result_int = raw_result.astype(np.int)
                label_img, num = label(raw_result_int, connectivity=raw_result_int.ndim, return_num=True)
                raw_region = regionprops(label_img)

                raw_region_all_coord = []
                for o in range(len(raw_region)):
                    coord_list = raw_region[o].coords
                    for v in range(len(coord_list)):
                        raw_region_all_coord.append(tuple([coord_list[v][0], coord_list[v][1], coord_list[v][2]]))

                for o in range(len(preprocessed_region)):
                    coord_list2 = preprocessed_region[o].coords
                    preprocessed_region_list2 = []
                    for v in range(len(coord_list2)):
                        preprocessed_region_list2.append(
                            tuple([coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]]))
                    if len(list(set(preprocessed_region_list2).intersection(set(raw_region_all_coord)))) == 0:
                        for v in range(len(coord_list2)):
                            preprocessed_result[coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]] = 0


                for o in range(len(raw_region)):
                    coord_list2 = raw_region[o].coords
                    raw_region_list2 = []
                    for v in range(len(coord_list2)):
                        raw_region_list2.append(tuple([coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]]))
                    if len(list(set(raw_region_list2).intersection(set(pre_region_all_coord)))) == 0:
                        for v in range(len(coord_list2)):
                            preprocessed_result[coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]] = 1

            else:
                preprocessed_result = raw_result

            preprocessed_result = preprocessed_result.astype(np.float32)
            ofn2 = os.path.join(outputFn + '/' + prefix + '.nii.gz')
            save_data(data=preprocessed_result, img=get_itk_image(ifn1), filename=ofn2)

if __name__ == '__main__':
    run()