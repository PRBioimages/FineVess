import os
import argparse
from dvn.utils import get_itk_array, get_itk_image, make_itk_image, write_itk_image
import numpy as np
from dvn.metrics import dice_score
from keras import backend as K
import tensorflow as tf

def NumIn(s):
    for char in s:
        if char.isdigit():
            return True
    return False

# def save_path():
#     import os
#     rootdir1 = os.path.join('/public/yangxiaodu/nnunet/save_data/nnUNet_trained_models/nnUNet/3d_fullres/Task508_other_vessels_raw/nnUNetTrainerV2_loss_ignore_label2_cew_sb_CBAM_decoder_btc_moreDA_BN__nnUNetPlansv2.1/all/other')
#     # read
#
#     write_path1 = open('/public/yangxiaodu/vessap2/code_self/nnunet_other_data.txt', 'w')
#     file_list=os.listdir(rootdir1)
#     file_list.sort(key=lambda x:x[3:5])
#     file_name_list=[]
#     for i in range(len(file_list)):
#         if NumIn(file_list[i])==True:
#             write_path1.write(os.path.join(rootdir1, file_list[i]) + '\n')
#     write_path1.close()
#
#
#     rootdir2 = os.path.join('/public/yangxiaodu/nnunet/save_data/nnUNet_raw_data_base/nnUNet_raw_data/Task515_other_real_vessels/labelsTr')
#     # read
#     write_path2 = open('/public/yangxiaodu/vessap2/code_self/nnunet_other_label.txt', 'w')
#     file_list=os.listdir(rootdir2)
#     file_list.sort(key=lambda x:x[3:5])
#     for i in range(len(file_list)):
#         if NumIn(file_list[i])==True:
#             write_path2.write(os.path.join(rootdir2, file_list[i]) + '\n')
#     write_path2.close()

def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--filenames', dest='filenames', type=str, default='/home/xdyang/FineVess/result/evaluation_data.txt')
    parser.add_argument('--maskFilename', dest='maskFn', type=str, default='/home/xdyang/FineVess/result/evaluation_label.txt',
                   help='a mask file to be applied to the predictions')
    parser.add_argument('--output', dest='output', type=str,
                        default="/home/xdyang/FineVess/results/nnunet/raw_data_seg_results",
                        help='output folder for storing predictions (default: current working directory)')
    args = parser.parse_args()
    return args

def generate_data(filenames, maskFn=None):
    data = get_itk_array(filenames)
    mask = get_itk_array(maskFn)
    # return np.asarray(data, dtype='float32'), np.asarray(mask, dtype='float32')
    return data, mask

def metric_dice(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def get_tp_fp_tn_fn(y_true, y_pred):
    y_pred_positive=y_pred
    y_pred_negative = 1 - y_pred_positive
    y_positive =y_true
    y_negative = 1 - y_positive
    # TP = tf.cast(K.sum(y_positive * y_pred_positive),tf.float32)
    # TN = tf.cast(K.sum(y_negative * y_pred_negative),tf.float32)
    # FP =  tf.cast(K.sum(y_negative * y_pred_positive),tf.float32)
    # FN = tf.cast(K.sum(y_positive * y_pred_negative),tf.float32)
    TP = np.sum(y_positive * y_pred_positive)
    TN = np.sum(y_negative * y_pred_negative)
    FP = np.sum(y_negative * y_pred_positive)
    FN = np.sum(y_positive * y_pred_negative)
    # print(TP,TN,FP,FN)
    # image = tf.cast(image, tf.float32)
    return TP, TN, FP, FN

def acc(y_true, y_pred):
    TP, TN, FP, FN = get_tp_fp_tn_fn(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + K.epsilon())
    return ACC

def sensitivity(y_true, y_pred):
    """ recall """
    TP, TN, FP, FN = get_tp_fp_tn_fn(y_true, y_pred)
    SE = TP/(TP + FN + K.epsilon())
    return SE


def specificity(y_true, y_pred):
    TP, TN, FP, FN = get_tp_fp_tn_fn(y_true, y_pred)
    SP = TN / (TN + FP + K.epsilon())
    return SP

def precision(y_true, y_pred):
    TP, TN, FP, FN = get_tp_fp_tn_fn(y_true, y_pred)
    PC = TP/(TP + FP + K.epsilon())
    return PC

def f1_socre(y_true, y_pred):
    """ dice """
    SE = sensitivity(y_true, y_pred)
    PC = precision(y_true, y_pred)
    F1 = 2 * SE * PC / (SE + PC + K.epsilon())
    return F1


def accuracy_bin(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    A=np.equal(y_true_f, y_pred_f)
    count= np.sum(A!= 0)
    return float(count)/float(len(y_true_f))




def run():
    args = parse_args()
    print('----------------------------------------')
    print(' Testing Parameters ')
    print('----------------------------------------')
    print('test files:', args.filenames)
    print('Mask file:', args.maskFn)
    with open(os.path.abspath(args.filenames)) as f:
        iFn = f.readlines()
    iFn = [x.strip() for x in iFn]

    with open(os.path.abspath(args.maskFn)) as f:
        mFn = f.readlines()
    mFn = [x.strip() for x in mFn]

    i = 0
    dice1={}
    f1score1 = {}
    f1score2={}
    acc_voxel1 = {}
    acc_voxel2 = {}
    sensitivity2={}
    specificity2={}
    precision2={}
    name = {}
    for ifn, mfn in zip(iFn, mFn):
        data, mask = generate_data(ifn, mfn)
        print('Volume size: ', data.shape)
        index = []
        for ind in range(mask.shape[0]):
            if np.any(mask[ind, :, :] != 2) == True:
            # if np.any(mask[ind, :, :]) == True:
                index.append(ind)
        print(index)
        data1 = data[index, :, :]
        mask1 = mask[index, :, :]
        asd=0
        num=0
        mask1_b = mask1.astype(bool)
        data1_b = data1.astype(bool)
        for o in range(mask1.shape[0]):
            if np.sum(data1_b[o])!=0:
                surface_distances = surfdist.compute_surface_distances(
                    mask1_b[o], data1_b[o], spacing_mm=(1.0, 1.0))
                avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
                assd=(avg_surf_dist[0]+avg_surf_dist[1])/2
                asd=asd+assd
                num=num+1
        asd1[i]=round(asd/num,4)
        dice1[i] = round(metric_dice(mask1, data1), 4)
        f1score1[i] = round(dice_score(mask1, data1), 4)
        f1score2[i]= round(f1_socre(mask1, data1), 4)
        acc_voxel1[i] = round(accuracy_bin(mask1, data1), 4)
        acc_voxel2[i]=round(acc(mask1, data1), 4)
        sensitivity2[i] = round(sensitivity(mask1, data1), 4)
        specificity2[i] = round(specificity(mask1, data1), 4)
        precision2[i] = round(precision(mask1, data1), 4)
        prefix = os.path.basename(ifn).split('.')[0]
        name[i] = prefix
        i = i + 1
    Len={}
    Sum={}
    Avg={}
    for k, p in enumerate([dice1,f1score1,f1score2,acc_voxel1,acc_voxel2,sensitivity2,specificity2,precision2]):
        Len[k] = len(p)
        Sum[k] = sum(p.values())
        Avg[k] = Sum[k] / Len[k]
    path=args.output+'/metric_other.txt'
    f = open(path,'w')
    f.writelines([path,"\n","dice1:",str(dice1),str(Avg[0]),
                  #"\n", "f1_score1:",str(f1score1),str(Avg[1]),"\n","f1score2:",str(f1score2),str(Avg[2]),
                  "\n","acc_voxel1:",str(acc_voxel1),str(Avg[3]),#"\n","acc_voxel2:",str(acc_voxel2),str(Avg[4]),
                  "\n","sensitivity2:",str(sensitivity2),str(Avg[5]),"\n","specificity2:",str(specificity2),str(Avg[6]),
                  "\n","precision2:",str(precision2),str(Avg[7]),"\n","name:",str(name)])
    f.close()
    # print("dice_probs", dice_acc)
    print("dice1", dice1)
    print("f1_score1", f1score1)
    print("f1_score2", f1score2)
    print("acc_voxel1", acc_voxel1)
    print("acc_voxel2", acc_voxel2)
    print("sensitivity2", sensitivity2)
    print("specificity2", specificity2)
    print("precision2", precision2)
    print('finished!')
if __name__ == "__main__":
    # save_path()
    args=run()
