#this code was designed first for the subvolumes with the size of 512*512*#. We appliled it to the volumes(1024*1024*#) by cropping them.
#it needs ClearMap environments.
#In this code, "subvolume" refers to the "part" in the article as well as in the supplementary material.
import argparse
import os
# print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import scipy.misc
from skimage import morphology
from skimage.morphology import skeletonize_3d
from scipy import ndimage as ndi
from dvn.utils import get_itk_array, make_itk_image, write_itk_image, get_itk_image
# from batchgenerators.utilities.file_and_folder_operations import *
from skimage.io import imread
import SimpleITK as sitk
from evaluation import metric_dice, f1_socre, accuracy_bin, sensitivity, specificity, precision
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from skimage.morphology import disk, rectangle, binary_dilation, binary_erosion, binary_closing, binary_opening, \
    rectangle, remove_small_objects
import networkx as nx
from scipy.spatial import distance

import sys

sys.path.append('/public/yangxiaodu/clearmap')
import os

import nibabel as nib
import SimpleITK as sitk


import ClearMap.IO.Workspace as wsp
import ClearMap.IO.IO as io
# import ClearMap.ImageProcessing.Experts.Vasculature as vasc
import ClearMap.ImageProcessing.MachineLearning.VesselFilling.VesselFilling as vf
from ClearMap.convert import convert_to_tiff, make_itk_image, write_itk_image, get_itk_array, convert_to_nii_gz, \
    norm_data
import numpy as np
from keras import backend as K


def parse_args():
    parser = argparse.ArgumentParser(description='Postprocessing binary vessel segmentation')
    parser.add_argument('--filenames', dest='filenames', type=str,
                        default='/public/yangxiaodu/clearmap/data/pre_refine_data.txt')  #you need create a new txt file.
    parser.add_argument('--maskFilename', dest='maskFn', type=str,
                        default=None,
                        help='a mask file to be applied to the predictions')
    parser.add_argument('--win', dest='win', type=int,
                        default=12,  # A parameter used in large vessel filling
                        help='win')
    parser.add_argument('--output1', dest='output1', type=str,
                        default='/public/yangxiaodu/clearmap/data/post_all/postprocess_output_1',  #you can place the results in .../FineVess/results/post_refine
                        help='output folder for storing predictions (default: current working directory)')
    parser.add_argument('--output2', dest='output2', type=str,
                        default='/public/yangxiaodu/clearmap/data/post_all/postprocess_output_2',
                        help='output folder for storing predictions (default: current working directory)')
    parser.add_argument('--output3', dest='output3', type=str,
                        default='/public/yangxiaodu/clearmap/data/post_all/postprocess_output_3',
                        help='output folder for storing predictions (default: current working directory)')
    parser.add_argument('--f', dest='format', type=str, default='.nii.gz',
                        help='NIFTI file format for saving outputs (default: .nii.gz)')
    args = parser.parse_args()

    return args

def norm_data_255(data):
    data = data - np.min(data)
    data = data * 255.0 / (np.max(data)-np.min(data))
    # print 'normed data:', np.min(data), np.max(data)
    return data


def fill_holes(data):
    data = np.asarray(ndi.binary_fill_holes(data), dtype='uint8')
    return data


def save_data(data, img, filename):
    out_img = make_itk_image(data, img)
    write_itk_image(out_img, filename)


def get_spacing(fn):
    # img = imread(fn)
    img = nib.load(os.path.abspath(fn))
    # img_affine = image.affine
    img = img.get_data()
    img_itk = sitk.GetImageFromArray(img.astype(np.float32))
    spacing = np.array(img_itk.GetSpacing())
    return spacing


def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    # return mask_sel,max_area
    return mask_sel, max_area


def find_real_region(mask_sel,outputFn,prefix,sub_maxvImg_list1,sub_maxvImg_list2,valid_subvolume,j):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    temp_mask={}
    temp_mask_t={}
    temp_dice={}
    for c in range(len(contours)):
        temp_mask[c]=mask_sel.copy()
        for k in range(len(contours)):
            if k!=c:
                cv2.fillPoly(temp_mask[c], [contours[k]], 0)
        temp_mask_t[c],temp_pengzhang=dilation_and_fill(temp_mask[c],outputFn,prefix,j=j)
        temp_dice[c]=metric_dice(temp_mask_t[c],sub_maxvImg_list1)
        if j + 1 in valid_subvolume:
            temp_dice[c] =temp_dice[c]+ metric_dice(temp_mask_t[c], sub_maxvImg_list2)


    # temp_dice_list=list(temp_dice)
    temp_dice_list=list(temp_dice.values())
    max_idx = np.argmax(temp_dice_list)
    max_area = cv2.contourArea(contours[max_idx])

    return temp_mask[max_idx], max_area


def calc_length(img):
    image = img
    image.astype(dtype='uint8', copy=False)
    sum_v = np.sum(image)
    return sum_v

def get_vessel_length(img,outputFn=None,prefix=None,j=100):
    thr=0
    binary = img > thr
    skeleton = morphology.skeletonize(binary)
    skeleton.astype(dtype='uint8', copy=False)


    if outputFn!=None:
        skeleton_show=skeleton*255
        ofn2 = os.path.join(outputFn + '/' + prefix + '_'+str(j)+'skeleton.png')
        cv2.imwrite(ofn2, skeleton_show)

    length=calc_length(skeleton)

    return length,skeleton

def extract_radius(segmentation, centerlines):
    image = segmentation
    skeleton = centerlines
    transf = ndi.distance_transform_edt(image, return_indices=False)
    radius_matrix = transf * skeleton
    av_rad = np.true_divide(radius_matrix.sum(), (radius_matrix != 0).sum())
    return av_rad

def dilation_and_fill(maxvImg,outputFn=None,prefix=None,maxi=100,j=0,selem_num=5):
    selem = disk(selem_num)
    pengzhang = binary_dilation(maxvImg, selem)
    pengzhang = pengzhang.astype(np.uint8)
    pengzhang_show = pengzhang * 255

    if outputFn!=None:
        if j==0:
            ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'whole' + str(maxi) + 'maxvImg_pengzhang.png')
            cv2.imwrite(ofn2, pengzhang_show)
        else:
            ofn2 = os.path.join(outputFn + '/' + prefix + '_' + str(j) + 'sub' + str(maxi) + 'maxvImg_pengzhang.png')
            cv2.imwrite(ofn2, pengzhang_show)

    tianchong = ndi.binary_fill_holes(pengzhang)
    tianchong = tianchong.astype(np.uint8)
    tianchong_show = tianchong * 255

    if outputFn!=None:
        if j==0:
            ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'whole' + str(maxi) + 'maxvImg_tianchong.png')
            cv2.imwrite(ofn2, tianchong_show)
        else:
            ofn2 = os.path.join(outputFn + '/' + prefix + '_' + str(j) + 'sub' + str(maxi) + 'maxvImg_tianchong.png')
            cv2.imwrite(ofn2, tianchong_show)

    return tianchong,pengzhang



def find_max_region_in_volume(subvolume,j,outputFn,prefix,sub_maxvImg_list=None,valid_subvolume=None,refine=False):  #J is the serial number of the subvolume
    win = 12
    # win=10
    step = 4
    thr = 0
    maxrawdata = []
    maxvImg = []
    maxi = 0
    max_area = 0
    center_slice_list={}
    for i, ilayer in enumerate(range(win // 2, subvolume.shape[0] - win // 2 + 1, step)):
        tmpdata = subvolume[ilayer - win // 2:ilayer + win // 2 + 1, :, :]
        data_1 = np.sum(tmpdata, axis=0)

        data_1 = data_1 > thr
        data_1 = data_1.astype(np.uint8)
        data_2=data_1.copy()

        show_picture_data_1 = data_1 * 255
        ofn2 = os.path.join(outputFn + '/' + prefix + '_' + str(j)+str(i) + 'data_1.png')
        cv2.imwrite(ofn2, show_picture_data_1)

        if refine==False:
            lsd_data, area = find_max_region(data_1)
        else:
            lsd_data,area= find_real_region(data_1,outputFn,prefix,sub_maxvImg_list[j-1],sub_maxvImg_list[j+1],valid_subvolume,j)


        lsd_data = lsd_data > thr
        lsd_data = lsd_data.astype(np.uint8)

        show_picture_lsd_data = lsd_data * 255
        ofn2 = os.path.join(outputFn + '/' + prefix + '_' + str(j)+str(i) + 'lsd_data.png')
        # scipy.misc.toimage(lsd_data).save(ofn2)
        cv2.imwrite(ofn2, show_picture_lsd_data)

        #Store the four center slices first
        center_slice_list[i]=lsd_data

        if area > max_area:
            max_area = area
            maxrawdata = data_2
            maxvImg = lsd_data
            maxi = i
    if maxrawdata!=[]:
        print('The image is %dth for %dth' % (maxi,j))
        show_picture = maxrawdata * 255
        ofn2 = os.path.join(outputFn + '/' + prefix + '_' + str(j)+str(maxi) + 'maxrawdata.png')
        cv2.imwrite(ofn2, show_picture)
        show_picture1 = maxvImg * 255
        ofn2 = os.path.join(outputFn + '/' + prefix + '_'+str(j)+ str(maxi) + 'maxvImg.png')
        cv2.imwrite(ofn2, show_picture1)
    return maxrawdata,maxvImg,maxi,center_slice_list

def find_each_slice_projection(volume,outputFn,prefix):
    win = 12
    thr=0
    projection=np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=volume.dtype)
    for z in range(volume.shape[0]):
        if np.sum(volume[z])!=0:  #Don't treat the slices that don't have blood vessels
            tmpdata =volume[z - min(win // 2,z-0):z + min(win // 2,volume.shape[0]-z) + 1, :, :]  #For each slice, you can traverse the slices around it.
            data_1 = np.sum(tmpdata, axis=0)
            data_1 = data_1 > thr
            data_1 = data_1.astype(np.uint8)
            projection[z,:,:]=data_1
    tem_tianchong=np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=volume.dtype)
    tem_pengzhang=np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]), dtype=volume.dtype)
    for z in range(projection.shape[0]):
        tem_tianchong[z,:,:],tem_pengzhang[z,:,:]=dilation_and_fill(projection[z,:,:],outputFn=outputFn,prefix=prefix,selem_num=10)
    return projection,tem_pengzhang,tem_tianchong


# To detect edge pixels, add a zero border
def connect_skeleton(skeleton_volume,ifother=True):
    skeleton_volume_pad = np.pad(skeleton_volume, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
    r_other=50
    r_self=60
    if ifother==True:
        r_limit=r_other
    else:
        r_limit=r_self
    for z in range(skeleton_volume_pad.shape[0]):
        label_img, num = label(skeleton_volume_pad[z, :, :], connectivity=2, return_num=True)
        region_skeleton_slice = regionprops(label_img)
        endpoints_list_list = {}
        for c in range(len(region_skeleton_slice)):
            endpoints_list_list[c] = []
            for v in range(len(region_skeleton_slice[c].coords)):  # Iterate through each pixel point of this connectivity skeleton to find the endpoint pixels
                coord_list = region_skeleton_slice[c].coords
                x0 = int(coord_list[v][0])
                y0 = int(coord_list[v][1])
                sum = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        sum = sum + skeleton_volume_pad[z, x0 + i, y0 + j]
                if sum == 2:
                    endpoints_list_list[c].append(tuple([x0, y0]))

        for c in range(len(region_skeleton_slice)):
            other_region_points_list = []
            for c1 in range(len(region_skeleton_slice)):
                if c1 != c:
                    other_region_points_list.extend(endpoints_list_list[c1])
            # the endpoints of the other connectivity domains have all been brought together in one list.

            for v in range(len(endpoints_list_list[c])):
                other_self_points = []
                for e in range(len(endpoints_list_list[c])):
                    if e!=v:
                        other_self_points.append(tuple([endpoints_list_list[c][e][0],endpoints_list_list[c][e][1]]))

                x1 = int(endpoints_list_list[c][v][0])
                y1 = int(endpoints_list_list[c][v][1])
                min_r = 500
                if ifother==True:
                    for v1 in range(len(other_region_points_list)):  # Find the point nearest to (x1,y1).
                        x2 = int(other_region_points_list[v1][0])
                        y2 = int(other_region_points_list[v1][1])
                        r = pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)
                        if r < min_r:
                            min_r = r
                            min_x = x2
                            min_y = y2
                else:
                    for v1 in range(len(other_self_points)):
                        x2 = int(other_self_points[v1][0])
                        y2 = int(other_self_points[v1][1])
                        r = pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)
                        if r < min_r:
                            min_r = r
                            min_x = x2
                            min_y = y2
                if min_r < r_limit:
                    # plt.plot([x1, min_x], [y1, min_y])

                    if x1 > min_x:
                        x1, min_x = min_x, x1
                        y1, min_y = min_y, y1
                    # for x in [x1,min_x]:
                    if x1 != min_x:
                        x_y_list = []
                        for x in range(x1, min_x + 1):
                            x = x
                            y = (y1 - min_y) / (x1 - min_x) * (x - x1) + y1
                            skeleton_volume_pad[z, x, int(y)] = 1
                            # Completes missing pixels on the y-axis
                            x_y_list.append(tuple([x, int(y)]))
                        for k in range(len(x_y_list) - 1):
                            if abs(x_y_list[k + 1][1] - x_y_list[k][1]) != 1:
                                a = abs(x_y_list[k + 1][1] - x_y_list[k][1])
                                for i in range(a - 1):
                                    if x_y_list[k + 1][1] - x_y_list[k][1] < 0:
                                        skeleton_volume_pad[z, x_y_list[k][0], (x_y_list[k][1] - (i + 1))] = 1
                                    else:
                                        skeleton_volume_pad[z, x_y_list[k][0], (x_y_list[k][1] + i + 1)] = 1
                    else:
                        if y1 > min_y:
                            y1, min_y = min_y, y1
                            x1, min_x = min_x, x1
                        for y in range(y1, min_y + 1):
                            x = x1
                            y = y
                            skeleton_volume_pad[z, x, int(y)] = 1

        # lianjie_show = skeleton_volume_pad[z,:,:] * 255
        # ofn2 = os.path.join(outputFn + '/' + prefix + '_'+str(z)+ 'lianjie.png')
        # cv2.imwrite(ofn2, lianjie_show)
    skeleton_volume_lianjie = skeleton_volume_pad[:, 1:-1, 1:-1]
    return  skeleton_volume_lianjie


def fill_skeleton(skeleton_volume_lianjie):
   skeleton_volume_lianjie_fill = np.zeros((skeleton_volume_lianjie.shape[0],skeleton_volume_lianjie.shape[1],skeleton_volume_lianjie.shape[2]), dtype=skeleton_volume_lianjie.dtype)
   skeleton_volume_lianjie = skeleton_volume_lianjie.astype(np.uint8)

   for z in range(skeleton_volume_lianjie.shape[0]):
       skeleton_volume_lianjie_fill[z, :, :] = ndi.binary_fill_holes(skeleton_volume_lianjie[z, :, :])
   skeleton_volume_lianjie_fill = skeleton_volume_lianjie_fill.astype(np.float32)

   return skeleton_volume_lianjie_fill

def get_skeleton(volume_bv):
    skeleton_volume = np.zeros((volume_bv.shape[0], volume_bv.shape[1], volume_bv.shape[2]), dtype=volume_bv.dtype)
    # for z in range(volume_bv.shape[0]):
    #     length_notneed,skeleton_volume[z,:,:]=get_vessel_length(pengzhang_volume_3d[z,:,:])
    # ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'volume_bv_pengzhang_3d_2i_skeleton.nii.gz')
    # save_data(data=skeleton_volume, img=get_itk_image(ifn), filename=ofn2)
    for z in range(volume_bv.shape[0]):
        length_notneed, skeleton_volume[z, :, :] = get_vessel_length(volume_bv[z, :, :])

    return skeleton_volume

def get_bv(data,valid_subvolume,sub_center_slice_list_list,sub_maxvImg_list,slice_num):
    volume_bv=np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype)
    for idx in range(len(valid_subvolume)):
        sub_center_slice_list=sub_center_slice_list_list[valid_subvolume[idx]-1]
        sub_maxvImg_next=sub_maxvImg_list[valid_subvolume[idx]]
        subvolume=data[(valid_subvolume[idx]-1) * slice_num:(valid_subvolume[idx]-1)* slice_num + slice_num, :, :]
        subvolume_bv=np.zeros((subvolume.shape[0], subvolume.shape[1], subvolume.shape[2]), dtype=data.dtype)

        label_img, num = label(sub_maxvImg_next, connectivity=2, return_num=True)
        region_sub_maxvImg_next= regionprops(label_img)
        for o in range(len(region_sub_maxvImg_next)):
            object_list= []
            object_list_coords = region_sub_maxvImg_next[o].coords
            for v in range(len(object_list_coords)):
                object_list.append(tuple([object_list_coords[v][0], object_list_coords[v][1]]))
        b_all={}
        for z in range(subvolume.shape[0]):

            #subvolume_bv[z,:,:]=subvolume[z,:,:]*sub_maxvImg_next

            label_img, num = label(subvolume[z,:,:], connectivity=2, return_num=True)
            region_subvolume_slice = regionprops(label_img)
            dice_eo={}
            b={}
            for o in range(len(region_subvolume_slice)):
                each_object=np.zeros((subvolume.shape[1], subvolume.shape[2]), dtype=data.dtype)
                object_list_slice = []
                object_list_coords =region_subvolume_slice[o].coords
                for v in range(len(object_list_coords)):
                    object_list_slice.append(tuple([object_list_coords[v][0], object_list_coords[v][1]]))
                    # each_object[object_list_coords[v][0], object_list_coords[v][1]]=1
                    # each_object_show=each_object*255
                if len(list(set(object_list_slice).intersection(set(object_list)))) / len(object_list_slice)>=0.4:
                # if len(list(set(object_list_slice).intersection(set(object_list)))) != 0:
                    # dice_eo[o]=metric_dice(each_object,sub_maxvImg_next)
                    # b[o]=len(list(set(object_list_slice).intersection(set(object_list)))) / len(object_list_slice)
                    # ofn2 = os.path.join(output_test + '/' + prefix + 'each_object.png')
                    # cv2.imwrite(ofn2, each_object_show)
            # b_all[z]=b
                    for v in range(len(object_list_slice)):
                        subvolume_bv[z,object_list_slice[v][0], object_list_slice[v][1]] = 1

        volume_bv[(valid_subvolume[idx]-1) * slice_num:(valid_subvolume[idx]-1)* slice_num + slice_num, :, :]=subvolume_bv
    return volume_bv

def fill_edge_hole(data):
    data_pad1 = np.pad(data, ((0, 0), (1, 1), (0, 0)), 'constant', constant_values=1)
    data_pad1_filled = np.zeros((data_pad1.shape[0], data_pad1.shape[1], data_pad1.shape[2]), dtype=data_pad1.dtype)
    for z in range(data_pad1.shape[0]):
        data_pad1_filled[z, :, :] = ndi.binary_fill_holes(data_pad1[z, :, :])
    data_filled1 = data_pad1_filled[:, 1:-1, :]

    data_pad2 = np.pad(data_filled1, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=1)
    data_pad2_filled = np.zeros((data_pad2.shape[0], data_pad2.shape[1], data_pad2.shape[2]), dtype=data_pad2.dtype)
    for z in range(data_pad2.shape[0]):
        data_pad2_filled[z, :, :] = ndi.binary_fill_holes(data_pad2[z, :, :])
    data_filled2 = data_pad2_filled[:, :, 1:-1]
    data_filled2 = data_filled2.astype(np.float32)

    return data_filled2


def skeleton_connect_fill_add(volume,ifn,outputFn,prefix,volume_bv,iter=3):
    iter=iter+1
    for i in range(iter):
        skeleton_volume = get_skeleton(volume)
        ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'volume_bv_pz_skeleton'+str(i)+'.nii.gz')
        save_data(data=skeleton_volume, img=get_itk_image(ifn), filename=ofn2)

        if i==iter-1:
            skeleton_volume_lianjie = connect_skeleton(skeleton_volume, ifother=False)
        else:
            skeleton_volume_lianjie = connect_skeleton(skeleton_volume, ifother=True)
        ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'volume_bv_p_lianjie_0409'+str(i)+'.nii.gz')
        save_data(data=skeleton_volume_lianjie, img=get_itk_image(ifn), filename=ofn2)

        skeleton_volume_lianjie_fill = fill_skeleton(skeleton_volume_lianjie)
        ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'volume_bv_p_lianjie_0409_fill'+str(i)+'.nii.gz')
        save_data(data=skeleton_volume_lianjie_fill, img=get_itk_image(ifn), filename=ofn2)

        skeleton_volume_lianjie_fill_add = np.logical_or(skeleton_volume_lianjie_fill, volume_bv)
        skeleton_volume_lianjie_fill_add = skeleton_volume_lianjie_fill_add.astype(np.float32)
        ofn3 = os.path.join(outputFn + '/' + prefix + '_' + 'volume_bv_p_lianjie_0409_fill_add'+str(i)+'.nii.gz')
        save_data(data=skeleton_volume_lianjie_fill_add, img=get_itk_image(ifn), filename=ofn3)

        volume=skeleton_volume_lianjie_fill_add
        volume_bv=skeleton_volume_lianjie_fill_add

    return ofn3

def clearmap_filling(ifn,directory):
    convert_to_tiff([ifn], directory)
    space = tuple(get_spacing(ifn))
    ws = wsp.Workspace('TubeMap', directory=directory);
    prefix = os.path.basename(ifn).split('.')[0]
    io.convert_files(prefix + '.tif', extension='npy', path=directory,processes=12, verbose=True);
    expression_mypicture = prefix + '.npy'
    ws.update(mypicture=expression_mypicture)
    ws.info()
    # mask = get_itk_array(mfn)
    source = ws.filename('mypicture');
    sink = ws.filename('binary', postfix='filled');
    io.delete_file(sink)
    source = io.as_source(source);
    source.dtype = bool
    processing_parameter = vf.default_fill_vessels_processing_parameter.copy();
    processing_parameter.update(size_max=200,
                                size_min='fixed',
                                axes=all,
                                overlap=50);

    vf.fill_vessels(source, sink, resample=1, threshold=0.5, cuda=None, processing_parameter=processing_parameter,
                    verbose=True)

    io.convert_files('binary_filled' + '.npy', extension='tif', path=directory,
                     processes=12, verbose=True);
    result = directory + '/' + 'binary_filled' + '.tif'
    bin = get_itk_array(result)  # bin.shape:(127,512,512,4);dtype:uint8
    bin = ~bin
    bin = norm_data(bin)
    bin = np.array(bin, dtype='uint8')
    bin_result = bin[:, :, :, 0]
    bin_result1 = make_itk_image(bin_result)
    write_itk_image(bin_result1,directory + '/' + prefix + 'filled' + 'max_size_200_resample=1_o50' + '.nii.gz')
    print(directory + '/' + prefix + 'filled'+ '.nii.gz')

    return  bin_result






def filling_small_holes_and_conneting(ifn,data_tubemap,outputFn,mfn=[]):
    prefix = os.path.basename(ifn).split('.')[0]
    data = get_itk_array(ifn)

    tubemap_hole_patch = data_tubemap.copy()
    tubemap_hole_patch[data == 1] = 0
    ofn = os.path.join(outputFn + '/' + prefix + '_' + 'tubemap_patch.nii.gz')
    save_data(data=tubemap_hole_patch, img=get_itk_image(ifn), filename=ofn)

    data_pad1_filled_no_edge = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype)
    for z in range(data_pad1_filled_no_edge.shape[0]):
        data_pad1_filled_no_edge[z, :, :] = ndi.binary_fill_holes(data[z, :, :])

    hole_patch2 = data_pad1_filled_no_edge.copy()
    hole_patch2[data == 1] = 0

    ofn = os.path.join(outputFn + '/' + prefix + '_' + 'hole_patch.nii.gz')
    save_data(data=hole_patch2, img=get_itk_image(ifn), filename=ofn)

    data_new_patch = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype)

    hole_patch_int = hole_patch2.astype(np.int)
    tubemap_hole_patch_int = tubemap_hole_patch.astype(np.int)

    voxel_excluded = []

    for z in range(hole_patch2.shape[0]):
        label_img, num = label(hole_patch_int[z, :, :], connectivity=2, return_num=True)
        region_hole_patch2 = regionprops(label_img)
        label_img1, num1 = label(tubemap_hole_patch_int[z, :, :], connectivity=2, return_num=True)
        region_tubemap_hole_patch = regionprops(label_img1)

        for o in range(len(region_hole_patch2)):

            cood = region_hole_patch2[o].centroid

            object_list_c = []
            object_list_c_coords = region_hole_patch2[o].coords
            for v in range(len(region_hole_patch2[o].coords)):
                object_list_c.append(tuple([object_list_c_coords[v][0], object_list_c_coords[v][1]]))

            if tubemap_hole_patch[z, :, :][int(cood[0]), int(cood[1])] == 1:  # Make sure it's a real hole.

                for o1 in range(len(region_tubemap_hole_patch)):
                    object_list = region_tubemap_hole_patch[o1].coords
                    object_list1 = []
                    for v1 in range(len(region_tubemap_hole_patch[o1].coords)):
                        object_list1.append(tuple([object_list[v1][0], object_list[v1][1]]))

                    if (int(cood[0]), int(cood[1])) in object_list1:

                        if region_tubemap_hole_patch[o1].area <= 50 * region_hole_patch2[o].area and region_tubemap_hole_patch[o1].area > region_hole_patch2[o].area:  # use the region of tubemap patch
                            for c in range(len(object_list1)):
                                data_new_patch[z, :, :][object_list1[c][0], object_list1[c][1]] = 1
                            break
                        else:  # use the region of hole_patch
                            for c in range(len(object_list_c)):
                                data_new_patch[z, :, :][object_list_c[c][0], object_list_c[c][1]] = 1
                            if region_tubemap_hole_patch[o1].area > 50 * region_hole_patch2[o].area:
                                voxel_excluded.append(tuple([z, object_list_c[c][0], object_list_c[c][1]]))
                            break
    data_new_patch = data_new_patch.astype(np.float32)
    ofn2 = os.path.join(outputFn + '/' + prefix + '_'+ 'data_new_patch.nii.gz')
    save_data(data=data_new_patch, img=get_itk_image(ifn), filename=ofn2)

    # refine patch
    data_new_patch1 = data_new_patch.astype(np.int)
    label_img, num = label(data_new_patch1, connectivity=data.ndim, return_num=True)
    region_new_patch = regionprops(label_img)

    for o in range(len(region_new_patch)):
        boxing = region_new_patch[o].bbox
        # object_list_grow=[]
        coord_list = region_new_patch[o].coords

        region_new_patch_list = []

        for v in range(len(coord_list)):
            region_new_patch_list.append(
                tuple([coord_list[v][0], coord_list[v][1], coord_list[v][2]]))

        # if region_new_patch_list.isdisjoint(voxel_excluded):

        if len(list(set(region_new_patch_list).intersection(set(voxel_excluded)))) == 0 and region_new_patch[o].area > 6:  #It's easy to create square artifacts for small patches
            for v in range(len(region_new_patch[o].coords)):
                coord_list = region_new_patch[o].coords
                z0 = int(coord_list[v][0])
                x0 = int(coord_list[v][1])
                y0 = int(coord_list[v][2])
                zhan = np.zeros((data.shape[0] * data.shape[1] * data.shape[2], 3), dtype=int)
                pzhan = 1
                zhan[pzhan][1] = x0
                zhan[pzhan][2] = y0
                zhan[pzhan][0] = z0
                while pzhan > 0:
                    z1 = zhan[pzhan][0]
                    x1 = zhan[pzhan][1]
                    y1 = zhan[pzhan][2]
                    pzhan = pzhan - 1
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            for k in range(-1, 2):
                                if (z1 + i >= 0) & (z1 + i < data.shape[0]) & (x1 + j >= 0) & (
                                        x1 + j < data.shape[1]) & (y1 + k >= 0) & (y1 + k < data.shape[2]) \
                                        & (data_new_patch[z1 + i, x1 + j, y1 + k] != 1) & (
                                        tubemap_hole_patch[z1 + i, x1 + j, y1 + k] == 1) \
                                        & ((z1 + i) >= max(0, boxing[0] - 3)) & (
                                        (z1 + i) <= min(data.shape[0], boxing[3] + 3)) \
                                        & ((x1 + j) >= max(0, boxing[1] - 3)) & (
                                        (x1 + j) <= min(data.shape[1], boxing[4] + 3)) \
                                        & ((y1 + k) >= max(0, boxing[2] - 3)) & (
                                        (y1 + k) <= min(data.shape[2], boxing[5] + 3)):
                                    # & ((z1 + i, x1 + j, y1 + k) in object_list_grow):
                                    data_new_patch[z1 + i, x1 + j, y1 + k] = 1
                                    pzhan = pzhan + 1
                                    zhan[pzhan][1] = x1 + j
                                    zhan[pzhan][2] = y1 + k
                                    zhan[pzhan][0] = z1 + i
    data_new_patch = data_new_patch.astype(np.float32)
    ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'refine_data_new_patch.nii.gz')
    save_data(data=data_new_patch, img=get_itk_image(ifn), filename=ofn2)

    data_new = np.logical_or(data_new_patch, data)

    data_new = data_new.astype(np.float32)
    ofn3 = os.path.join(outputFn + '/' + prefix + '_'+ 'data_new1.nii.gz')
    save_data(data=data_new, img=get_itk_image(ifn), filename=ofn3)
    '''''''''''Hold off on the vascular filler network based on nnunet
    nnunet_filled_data1 = get_itk_array(nffn1)
    nnunet_filled_patch1 = nnunet_filled_data1.copy()
    nnunet_filled_patch1[data == 1] = 0

    # nnunet_filled_data2 = get_itk_array(nffn2)
    # nnunet_filled_patch2 = nnunet_filled_data2.copy()
    # nnunet_filled_patch2[data == 1] = 0

    # nnunet_filled_patch_add = np.logical_or(nnunet_filled_patch1, nnunet_filled_patch2)

    data_new_patch2= data_new_patch.astype(np.int)
    label_img, num = label(data_new_patch2, connectivity=data.ndim, return_num=True)
    region_new_patch2 = regionprops(label_img)

    # First record the coordinates of all nnunet patch
    nnunet_patch = nnunet_filled_patch1.astype(np.int)
    label_img, num = label(nnunet_patch, connectivity=data.ndim, return_num=True)
    region_nnunet_patch = regionprops(label_img)

    region_nnunet_patch_all_coord = []
    for o in range(len(region_nnunet_patch)):
        coord_list = region_nnunet_patch[o].coords
        for v in range(len(coord_list)):
            region_nnunet_patch_all_coord.append(
                tuple([coord_list[v][0], coord_list[v][1], coord_list[v][2]]))

    for o in range(len(region_new_patch2)):
        coord_list2 = region_new_patch2[o].coords
        region_new_patch_list2 = []
        for v in range(len(coord_list2)):
            region_new_patch_list2.append(
                tuple([coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]]))
        if len(list(set(region_new_patch_list2).intersection(set(region_nnunet_patch_all_coord)))) != 0:
            for v in range(len(coord_list2)):
                data_new_patch[coord_list2[v][0], coord_list2[v][1], coord_list2[v][2]] = 0

    data_new_patch = data_new_patch.astype(np.float32)
    ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'refine_data_new_patch2.nii.gz')
    save_data(data=data_new_patch, img=get_itk_image(ifn), filename=ofn2)

    #  Combining nnunet and tubemap-based patches
    patch_tube_nnunet = np.logical_or(data_new_patch, nnunet_filled_patch1)
    patch_tube_nnunet = patch_tube_nnunet.astype(np.float32)
    ofn3 = os.path.join(outputFn + '/' + prefix + '_' + 'patch_tube_nnunet.nii.gz')
    save_data(data=patch_tube_nnunet, img=get_itk_image(ifn), filename=ofn3)

    # Combining the final patch with the original image
    data_new2 = np.logical_or(patch_tube_nnunet, data)

    data_new2 = data_new2.astype(np.float32)
    ofn3 = os.path.join(outputFn + '/' + prefix + '_'+ 'data_new2.nii.gz')
    save_data(data=data_new2, img=get_itk_image(ifn), filename=ofn3)
    '''''''''''
    # data_new3 = fill_holes(data_new2)
    data_new3 = fill_holes(data_new)
    data_new3 = data_new3.astype(np.float32)
    ofn2 = os.path.join(outputFn + '/' + prefix + '_' + 'fill_data_new3.nii.gz')
    save_data(data=data_new3, img=get_itk_image(ifn), filename=ofn2)

    if mfn!=[]:
        mask = get_itk_array(mfn)
        index = []
        for ind in range(mask.shape[0]):
            if np.any(mask[ind, :, :] != 2) == True:
                # if np.any(mask[ind, :, :]) == True:
                index.append(ind)
        # print(index)
        data1 = data_new3[index, :, :]
        mask1 = mask[index, :, :]

        dice1 = round(metric_dice(mask1, data1), 4)
        acc_voxel1 = round(accuracy_bin(mask1, data1), 4)
        sensitivity2 = round(sensitivity(mask1, data1), 4)
        specificity2= round(specificity(mask1, data1), 4)
        precision2 = round(precision(mask1, data1), 4)
        name = prefix
    else:
        dice1=acc_voxel1=sensitivity2=specificity2=precision2=0
        name = prefix

    return data_new3,dice1,acc_voxel1,sensitivity2,specificity2,precision2,name



def big_vessel_filling(post1_result,ifn,outputFn2,directory,mfn=[]):
    prefix = os.path.basename(ifn).split('.')[0]
    data = post1_result
    spacing = get_spacing(ifn)
    whole_maxrawdata, whole_maxvImg, maxi, whole_center_slice_list = find_max_region_in_volume(data, 0,outputFn2,prefix)
    tianchong, pengzhang = dilation_and_fill(whole_maxvImg, outputFn2, prefix, maxi, j=0)

    dice = {}
    length = {}
    radius = {}
    dice[0] = 1
    length[0], skeleton = get_vessel_length(tianchong, outputFn2, prefix, 0)
    radius[0] = extract_radius(tianchong, skeleton)

    if radius[0] < 14:
        print("No large vessel in the volume")
        result=data
    else:
        subvolume_num = 5
        slice_num = data.shape[0] // subvolume_num
        slice_num_rest = data.shape[0] % subvolume_num

        valid_subvolume = []
        sub_maxvImg_list = {}
        average_dice = {}
        average_inter={}
        average_radius = {}
        sub_center_slice_list_list = {}

        for i in range(subvolume_num):
            sub_maxrawdata, sub_maxvImg, submaxi, sub_center_slice_list_list[i] = find_max_region_in_volume(data[i * slice_num:i * slice_num + slice_num, :, :], i + 1, outputFn2, prefix, refine=False)
            subtianchong, subpengzhang = dilation_and_fill(sub_maxvImg, outputFn2, prefix, submaxi, j=i + 1)

            dice_center_slice = {}
            dice_center_slice_inter = {}
            dice_add = 0
            dice_add_inter=0
            length_center_slice = {}
            radius_center_slice = {}
            radius_center_slice_add = 0

            for t in range(len(sub_center_slice_list_list[i])):
                subtianchong_center_slice, subpengzhang_center_slice = dilation_and_fill(
                    sub_center_slice_list_list[i][t], None, prefix, submaxi, j=i + 1)
                show_picture = subtianchong_center_slice * 255
                ofn2 = os.path.join(outputFn2 + '/' + prefix + '_' + str(i + 1) + str(t) + 'center_maxdata.png')
                cv2.imwrite(ofn2, show_picture)
                dice_center_slice[t] = metric_dice(subtianchong_center_slice, subtianchong)
                dice_add = dice_add + dice_center_slice[t]

                other_f = subtianchong_center_slice.flatten()
                max_f = subtianchong.flatten()
                intersection = np.sum(other_f * max_f)
                dice_center_slice_inter[t] = intersection / (np.sum(other_f) + 1e-8)
                dice_add_inter = dice_add_inter + dice_center_slice_inter[t]

                length_center_slice[t], subskeleton_center_slice = get_vessel_length(subtianchong_center_slice,
                                                                                     outputFn2, prefix, i + 1)
                radius_center_slice[t] = extract_radius(subtianchong_center_slice, subskeleton_center_slice)
                radius_center_slice_add = radius_center_slice_add + radius_center_slice[t]

            average_dice[i] = dice_add / len(sub_center_slice_list_list[i])
            average_inter[i] = dice_add_inter / len(sub_center_slice_list_list[i])
            average_radius[i] = radius_center_slice_add / len(sub_center_slice_list_list[i])

            dice[i + 1] = metric_dice(subtianchong, tianchong)
            length[i + 1], subskeleton = get_vessel_length(subtianchong, outputFn2, prefix, i + 1)
            radius[i + 1] = extract_radius(subtianchong, subskeleton)
            sub_maxvImg_list[i + 1] = subtianchong

            if dice[i + 1] >= 0.2 and radius[i + 1] >= max(radius[0] -10,14) and radius[i + 1] <= radius[0] + 10 \
                and average_inter[i]>=0.6 and average_radius[i] >=14:
                print('The %dth subvolume is included.' % (i + 1))
                valid_subvolume.append(i + 1)
            else:
                print("dice", dice[i + 1])
                print('radius', radius[i + 1])
                print('average_inter', average_inter[i])
                print('average_radius', average_radius[i])

        if slice_num_rest != 0:
            sub_maxrawdata_rest, sub_maxvImg_rest, submaxi, sub_center_slice_list_list[i + 1] = find_max_region_in_volume(data[(subvolume_num - 1) * slice_num + slice_num:data.shape[0], :, :], i + 1 + 1, outputFn2, prefix,refine=False)
            if sub_maxrawdata_rest != []:
                subtianchong, pengzhang = dilation_and_fill(sub_maxvImg_rest, outputFn2, prefix, submaxi, j=i + 1 + 1)

                dice_center_slice = {}
                dice_center_slice_inter={}
                dice_add = 0
                dice_add_inter=0
                length_center_slice = {}
                radius_center_slice = {}
                radius_center_slice_add = 0
                for t in range(len(sub_center_slice_list_list[i + 1])):
                    subtianchong_center_slice, subpengzhang_center_slice = dilation_and_fill(
                        sub_center_slice_list_list[i + 1][t], None, prefix, submaxi, j=i + 1 + 1)
                    show_picture = subtianchong_center_slice * 255
                    ofn2 = os.path.join(outputFn2 + '/' + prefix + '_' + str(i + 1 + 1) + str(t) + 'center_maxdata.png')
                    cv2.imwrite(ofn2, show_picture)

                    dice_center_slice[t] = metric_dice(subtianchong_center_slice, subtianchong)
                    dice_add = dice_add + dice_center_slice[t]

                    other_f = subtianchong_center_slice.flatten()
                    max_f = subtianchong.flatten()
                    intersection = np.sum(other_f * max_f)
                    dice_center_slice_inter[t]= intersection /(np.sum(other_f)+1e-8)
                    dice_add_inter=dice_add_inter+dice_center_slice_inter[t]

                    length_center_slice[t], subskeleton_center_slice = get_vessel_length(subtianchong_center_slice,outputFn2, prefix, i + 1 + 1)
                    radius_center_slice[t] = extract_radius(subtianchong_center_slice, subskeleton_center_slice)
                    radius_center_slice_add = radius_center_slice_add + radius_center_slice[t]

                average_dice[i + 1] = dice_add / len(sub_center_slice_list_list[i + 1])
                average_inter[i+1]= dice_add_inter / len(sub_center_slice_list_list[i + 1])
                average_radius[i + 1] = radius_center_slice_add / len(sub_center_slice_list_list[i + 1])

                dice[i + 1 + 1] = metric_dice(subtianchong, tianchong)
                length[i + 1 + 1], subskeleton = get_vessel_length(subtianchong, outputFn2, prefix, i + 1 + 1)
                radius[i + 1 + 1] = extract_radius(subtianchong, subskeleton)
                sub_maxvImg_list[i + 1 + 1] = subtianchong

                if dice[i + 1 + 1] >= 0.2 and radius[i + 1 + 1] >= max(radius[0] - 10,14) and radius[i + 1 + 1] <= radius[0] + 10 and average_inter[i+1]>=0.6 and average_radius[i+1] >=14:
                    print('The %dth subvolume is included' % (i + 1 + 1))
                    valid_subvolume.append(i + 1 + 1)
                else:
                    print("dice",dice[i + 1 + 1])
                    print('radius',radius[i + 1 + 1])
                    print('average_inter', average_inter[i+1])
                    print('average_radius',average_radius[i+1])


        # Find the subvolume that needs to be redetermined.:
        lack_slice = []
        not_do1 = False
        not_do2 = False
        not_do3 = False

        dice_list = list(dice.values())
        dice_list1=dice_list[1:]
        dice_max_idx = np.argmax(dice_list1)

        for idx in range(len(valid_subvolume) - 1):
            if valid_subvolume[idx + 1] - valid_subvolume[idx] != 1:
                # a = valid_subvolume[idx + 1] - valid_subvolume[idx]
                print('The list appears discontinuous')
                for idx in range(len(valid_subvolume)):
                    if valid_subvolume[idx]!=dice_max_idx+1:
                        if abs(valid_subvolume[idx]-(dice_max_idx+1))>2:
                            valid_subvolume.remove(valid_subvolume[idx])  #Remove some unnecessary subvolumes first.
                break

        for idx in range(len(valid_subvolume) - 1):
            if valid_subvolume[idx + 1] - valid_subvolume[idx] != 1:
                a = valid_subvolume[idx + 1] - valid_subvolume[idx]
                print('%d slices missing' % (a - 1))
                for i in range(a - 1):
                    lack_slice.append(valid_subvolume[idx] + i + 1)


        if not_do1 == False:
            # lack_sub_center_slice_list={}
            for lack_ids in range(len(lack_slice)):  # lack_slice=[2,4]
                lack_sub_maxrawdata, lack_sub_maxvImg, lack_submaxi, lack_sub_center_slice_list = find_max_region_in_volume(data[(lack_slice[lack_ids] - 1) * slice_num:(lack_slice[lack_ids] - 1) * slice_num + slice_num,:, :], lack_slice[lack_ids], outputFn2, prefix, sub_maxvImg_list, valid_subvolume,refine=True)
                lack_subtianchong, lack_subpengzhang = dilation_and_fill(lack_sub_maxvImg, outputFn2, prefix,lack_submaxi, j=lack_slice[lack_ids])

                dice_center_slice = {}
                dice_center_slice_inter={}
                dice_add = 0
                dice_add_inter=0
                length_center_slice = {}
                radius_center_slice = {}
                lack_sub_center_slice_list_tianchong={}
                radius_center_slice_add = 0
                for t in range(len(lack_sub_center_slice_list)):
                    subtianchong_center_slice, subpengzhang_center_slice = dilation_and_fill(
                        lack_sub_center_slice_list[t], None, prefix, lack_submaxi, j=lack_slice[lack_ids])
                    lack_sub_center_slice_list_tianchong[t]=subtianchong_center_slice
                    show_picture = subtianchong_center_slice * 255
                    ofn2 = os.path.join(outputFn2 + '/' + prefix + '_' + str(lack_slice[lack_ids]) + str(t) + 'refine_center_maxdata.png')
                    cv2.imwrite(ofn2, show_picture)
                    dice_center_slice[t] = metric_dice(subtianchong_center_slice, lack_subtianchong)
                    dice_add = dice_add + dice_center_slice[t]

                    other_f = subtianchong_center_slice.flatten()
                    max_f = lack_subtianchong.flatten()
                    intersection = np.sum(other_f * max_f)
                    dice_center_slice_inter[t] = intersection / (np.sum(other_f) + 1e-8)
                    dice_add_inter = dice_add_inter + dice_center_slice_inter[t]


                    length_center_slice[t], subskeleton_center_slice = get_vessel_length(subtianchong_center_slice,
                                                                                         outputFn2, prefix, lack_slice[lack_ids])
                    radius_center_slice[t] = extract_radius(subtianchong_center_slice, subskeleton_center_slice)
                    radius_center_slice_add = radius_center_slice_add + radius_center_slice[t]

                average_dice[lack_slice[lack_ids]-1] = dice_add / len(lack_sub_center_slice_list)
                average_inter[lack_slice[lack_ids] - 1] = dice_add_inter / len(lack_sub_center_slice_list)
                average_radius[lack_slice[lack_ids]-1] = radius_center_slice_add / len(lack_sub_center_slice_list)



                dice[lack_slice[lack_ids]] = metric_dice(lack_subtianchong, tianchong)
                length[lack_slice[lack_ids]], subskeleton = get_vessel_length(lack_subtianchong, outputFn2,prefix, lack_slice[lack_ids])
                radius[lack_slice[lack_ids]] = extract_radius(lack_subtianchong, subskeleton)



                if dice[lack_slice[lack_ids]] >= 0.2 and radius[lack_slice[lack_ids]] >= max(14,radius[0] - 10) and radius[lack_slice[lack_ids]] <= radius[0] + 10 and average_inter[lack_slice[lack_ids]-1]>=0.6 and average_radius[lack_slice[lack_ids]-1] >=14:
                    print('The %dth subvolume is included.' % (lack_slice[lack_ids]))
                    # valid_subvolume.append(i + 1)
                    # idx=valid_subvolume.index(str(lack_slice[lack_ids]-1))
                    sub_maxvImg_list[lack_slice[lack_ids]] = lack_subtianchong  # Store the maximum connectivity domain of each subvolume # Do an update

                    sub_center_slice_list_list[lack_slice[lack_ids] - 1] = lack_sub_center_slice_list  #Do an update

                    idx = valid_subvolume.index(lack_slice[lack_ids] - 1)
                    valid_subvolume.insert(idx + 1, lack_slice[lack_ids])
                    if average_dice[lack_slice[lack_ids]-1]<0.5:
                        new_lack_subtianchong = np.ones((data.shape[1], data.shape[2]),dtype=data.dtype)
                        for i in range(len(lack_sub_center_slice_list_tianchong)):
                            new_lack_subtianchong=lack_sub_center_slice_list_tianchong[i] * new_lack_subtianchong

                        sub_maxvImg_list[lack_slice[lack_ids]] = new_lack_subtianchong
                        a=sub_maxvImg_list[lack_slice[lack_ids]]
                        show_picture = a * 255
                        ofn2 = os.path.join(outputFn2 + '/' + prefix + '_' + str(lack_slice[lack_ids]) + 'refine_maxvimg.png')
                        cv2.imwrite(ofn2, show_picture)

                else:
                    print("dice", dice[lack_slice[lack_ids]])
                    print('radius', radius[lack_slice[lack_ids]])
                    print('average_inter', average_inter[lack_slice[lack_ids]-1])
                    print('average_radius', average_radius[lack_slice[lack_ids]-1])

            dice_list = list(dice.values())
            dice_list1 = dice_list[1:]
            dice_max_idx = np.argmax(dice_list1)
            if dice_max_idx+1 not in valid_subvolume:
                print("The subvolume in which the largest vessel is located does not meet the conditions for the presence of a large vessel, and there is a high probability that no large vessel exists in that volume")
                not_do2 = True
            if len(valid_subvolume) <= 1:
                print("Requires more than one valid subvolume!")
                not_do3= True

            if not_do2==False and not_do3==False:
                volume_bv = get_bv(data, valid_subvolume, sub_center_slice_list_list, sub_maxvImg_list,slice_num)  # Extracting the large vessel according to the largest central slice.
                s_no = 0
                for z in range(volume_bv.shape[0]):
                    if np.sum(volume_bv[z, :, :]) != 0:
                        s_no = s_no + 1

                ofn2 = os.path.join(outputFn2 + '/' + prefix + '_' + 'volume_bv_0409.nii.gz')
                save_data(data=volume_bv, img=get_itk_image(ifn), filename=ofn2)

                # Obtain grayscale projection
                volume_bv_sum_z = np.sum(volume_bv, axis=0)

                volume_bv_sum_z = volume_bv_sum_z.astype(np.uint8)
                volume_bv_sum_z = norm_data_255(volume_bv_sum_z)

                ofn2 = os.path.join(outputFn2 + '/' + prefix + 'bv_sum_z_g.png')
                cv2.imwrite(ofn2, volume_bv_sum_z)

                volume_bv_sum_z_binary = volume_bv_sum_z > 0
                volume_bv_sum_z_binary = volume_bv_sum_z_binary.astype(np.uint8)

                show_volume_bv_sum_z_binary = volume_bv_sum_z_binary * 255
                ofn2 = os.path.join(outputFn2 + '/' + prefix + 'bv_sum_z_b.png')
                cv2.imwrite(ofn2, show_volume_bv_sum_z_binary)

                # fill binary images
                volume_bv_sum_z_binary_filled = np.asarray(ndi.binary_fill_holes(volume_bv_sum_z_binary), dtype='uint8')
                show_volume_bv_sum_z_binary_filled = volume_bv_sum_z_binary_filled * 255
                ofn2 = os.path.join(outputFn2 + '/' + prefix + 'bv_sum_z_b_f.png')
                cv2.imwrite(ofn2, show_volume_bv_sum_z_binary_filled)

                #Skeleton connection and filling
                pengzhang_volume_3d = ndi.binary_dilation(volume_bv, iterations=2).astype(np.float32)
                ofn2 = os.path.join(outputFn2 + '/' + prefix + '_' + 'volume_bv_pengzhang_3d_2i.nii.gz')
                save_data(data=pengzhang_volume_3d, img=get_itk_image(ifn), filename=ofn2)

                skeleton_volume_lianjie_fill_add_fn=skeleton_connect_fill_add(pengzhang_volume_3d, ifn, outputFn2, prefix, volume_bv,iter=3)


                connect_fill_result = clearmap_filling(skeleton_volume_lianjie_fill_add_fn,directory)
                # Correct the result of TubeMap first: the tubemap will give two more blank lines in both the x and y directions
                connect_fill_result_pad = np.zeros((data.shape[0], data.shape[1] - 2, data.shape[2] - 2), dtype=data.dtype)
                connect_fill_result_pad[:, :, :] = connect_fill_result[:, :-2, :-2]
                connect_fill_result_pad = np.pad(connect_fill_result_pad, ((0, 0), (0, 2), (0, 2)), 'edge')

                # Vessel fill for each slice: # Use function to fill holes: holes at the edges are filled as well
                connect_fill_result_pad_fill = fill_edge_hole(connect_fill_result_pad)

                # Correct the contour of large vessels
                connect_fill_result_refine = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype)
                # connect_fill_result_refine[:s_no, :, :] = connect_fill_result_pad_fill[:s_no, :, :]
                connect_fill_result_refine[:, :, :] = connect_fill_result_pad_fill[:, :, :]
                bv_new = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype)
                # bv_new[:s_no,:,:]=connect_fill_result
                for z in range(data.shape[0]):
                    bv_new[z, :, :] = connect_fill_result_refine[z, :, :] * volume_bv_sum_z_binary_filled

                show_bv_new = bv_new * 255
                ofn2 = os.path.join(outputFn2 + '/' + prefix + 'bv_new.nii.gz')
                save_data(data=show_bv_new, img=get_itk_image(ifn), filename=ofn2)

                # Last step: corrode away the excess skeleton
                k = np.ones((5, 5), np.uint8)
                fushi_volume_2d = np.zeros((bv_new.shape[0], bv_new.shape[1], bv_new.shape[2]), dtype=data.dtype)
                for z in range(bv_new.shape[0]):
                    fushi_volume_2d[z, :, :] = cv2.erode(bv_new[z, :, :], k, iterations=1)
                ofn2 = os.path.join(outputFn2 + '/' + prefix + '_' + 'volume_bv_fushi_2d.nii.gz')
                save_data(data=fushi_volume_2d, img=get_itk_image(ifn), filename=ofn2)

                # Some of the vessels were corroded off during the erosion, hence combinate it with the original large vessel image
                bv_last = np.logical_or(fushi_volume_2d, volume_bv)

                # Combinate it with the original image
                whole_volume = np.logical_or(bv_last, data)
                whole_volume = whole_volume.astype(np.float32)
                ofn2 = os.path.join(outputFn2 + '/' + prefix + 'whole_volume_result.nii.gz')
                save_data(data=whole_volume, img=get_itk_image(ifn), filename=ofn2)


                result = whole_volume
            else:
                    result = data

        else:
            result = data

    if mfn != []:
        mask = get_itk_array(mfn)
        index = []
        for ind in range(mask.shape[0]):
            if np.any(mask[ind, :, :] != 2) == True:
                index.append(ind)
        # print(index)
        data1 = result[index, :, :]
        mask1 = mask[index, :, :]

        dice1 = round(metric_dice(mask1, data1), 4)
        acc_voxel1 = round(accuracy_bin(mask1, data1), 4)
        sensitivity2 = round(sensitivity(mask1, data1), 4)
        specificity2 = round(specificity(mask1, data1), 4)
        precision2 = round(precision(mask1, data1), 4)
        name = prefix
    else:
        dice1=acc_voxel1=sensitivity2=specificity2=precision2=0
        name = prefix

    return result, dice1, acc_voxel1, sensitivity2, specificity2, precision2, name

def denoise(data,outputFn3,ifn,n,mfn=[]):
    # for n in range(100,500,10):
    spacing = get_spacing(ifn)
    label_img, num = label(data, connectivity=data.ndim, return_num=True)
    region = regionprops(label_img)
    noise_patch = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=data.dtype)
    for o in range(len(region)):
        object_list_coords = region[o].coords
        pixel_num = region[o].area
        object_volume = spacing[0] * spacing[1] * spacing[2] * pixel_num
        if object_volume <= n:
            for v in range(len(object_list_coords)):
                noise_patch[object_list_coords[v][0], object_list_coords[v][1], object_list_coords[v][2]] = 1
                data[object_list_coords[v][0], object_list_coords[v][1], object_list_coords[v][2]] = 0

    prefix = os.path.basename(ifn).split('.')[0]
    ofn2 = os.path.join(outputFn3 + '/' + prefix + ' noise_patch.nii.gz')
    save_data(data=noise_patch, img=get_itk_image(ifn), filename=ofn2)

    ofn2 = os.path.join(outputFn3 + '/' + prefix + ' data_disnoise.nii.gz')
    save_data(data=data, img=get_itk_image(ifn), filename=ofn2)

    if mfn != []:
        mask = get_itk_array(mfn)
        index = []
        for ind in range(mask.shape[0]):
            if np.any(mask[ind, :, :] != 2) == True:
                # if np.any(mask[ind, :, :]) == True:
                index.append(ind)
        # print(index)
        data1 = data[index, :, :]
        mask1 = mask[index, :, :]

        dice1 = round(metric_dice(mask1, data1), 4)
        acc_voxel1 = round(accuracy_bin(mask1, data1), 4)
        sensitivity2 = round(sensitivity(mask1, data1), 4)
        specificity2 = round(specificity(mask1, data1), 4)
        precision2 = round(precision(mask1, data1), 4)
        name = prefix
    else:
        dice1=acc_voxel1=sensitivity2=specificity2=precision2=0
        name=prefix

    return data, dice1, acc_voxel1, sensitivity2, specificity2, precision2, name

def get_patch_data(volume4d, divs = (1,4,4), offset=(0,128,128)):
    patches = []
    shape = volume4d.shape  #（1，640，320，384）#(120,1024,1024）
    widths = [ int(s/d) for s,d in zip(shape, divs)]   #（1，64，64，64）#(120，256，256）
    patch_shape = [w+o*2 for w, o in zip(widths,offset)]    #(120,512,512)
    for x in np.arange(0, shape[1], widths[1]):
        for y in np.arange(0, shape[2], widths[2]):
            # for z in np.arange(0, shape[2], widths[2]):
            #     for t in np.arange(0, shape[3], widths[3]):
                    patch = np.zeros(patch_shape, dtype=volume4d.dtype)
                    x_s = x - offset[1] if x - offset[1] >= 0 else 0
                    x_e = x + widths[1] + offset[1] if x + widths[1] + offset[1] <= shape[1] else shape[1]
                    y_s = y - offset[2] if y - offset[2] >= 0 else 0
                    y_e = y + widths[2] + offset[2] if y + widths[2] + offset[2] <= shape[2] else shape[2]
                    # z_s = z - offset[2] if z - offset[2] >= 0 else 0
                    # z_e = z + widths[2] + offset[2] if z + widths[2] + offset[2] <= shape[2] else shape[2]
                    # t_s = t - offset[3] if t - offset[3] >= 0 else 0
                    # t_e = t + widths[3] +  offset[3] if t + widths[3] + offset[3] <= shape[3] else shape[3]
                    # vp = volume4d[x_s:x_e,y_s:y_e,z_s:z_e,t_s:t_e]
                    vp = volume4d[:,x_s:x_e, y_s:y_e]
                    px_s = offset[1] - (x - x_s)
                    px_e = px_s + (x_e - x_s)
                    py_s = offset[2] - (y - y_s)
                    py_e = py_s + (y_e - y_s)
                    # pz_s = offset[2] - (z - z_s)
                    # pz_e = pz_s + (z_e - z_s)
                    # pt_s = offset[3] - (t - t_s)
                    # pt_e = pt_s + (t_e - t_s)
                    # patch[px_s:px_e,py_s:py_e,pz_s:pz_e,pt_s:pt_e] = vp
                    patch[:,px_s:px_e, py_s:py_e] = vp
                    patches.append(patch)
    # return np.array(patches, dtype=volume4d.dtype)
    return patches

def get_volume_from_patches(patches, divs = (1,4,4), offset=(0,128,128)):
    new_shape = [(ps-of*2)*d for ps,of,d in zip(patches.shape[-3:],offset,divs)]
    volume3d = np.zeros(new_shape, dtype=(patches.dtype))
    shape = volume3d.shape  #(120,1024,1024)
    widths = [ int(s/d) for s,d in zip(shape,divs)]  #w=[120,256,256]
    index = 0
    for x in np.arange(0, shape[1], widths[1]):
        for y in np.arange(0, shape[2], widths[2]):
            # for z in np.arange(0, shape[2], widths[2]):
            #     for t in np.arange(0, shape[3], widths[3]):
            patch = patches[index]
            index = index + 1
            volume3d[:,x:x+widths[1],y:y+widths[2]] = patch[:,offset[1]:offset[1]+widths[1],offset[2]:offset[2]+widths[2]]
    return volume3d


import os


def run():
    args = parse_args()
    outputFn1 = args.output1
    outputFn2 = args.output2
    outputFn3 = args.output3
    fmt = args.format
    filenames = args.filenames
    masks = args.maskFn
    # fillmasks = args.filledFn2
    # fillmasks1 = args.nnunet_filledFn1
    # fillmasks2 = args.nnunet_filledFn2

    print('----------------------------------------')
    print(' Postprocessing Parameters ')
    print('----------------------------------------')
    print('Input files:', filenames)
    print('Mask file:', masks)
    print('Output folder:', outputFn1)
    print('Output folder:', outputFn2)
    print('Output format:', fmt)
    # print('ref_mask1:', fillmasks1)
    # print('ref_mask2:', fillmasks2)

    with open(os.path.abspath(args.filenames)) as f:
        iFn = f.readlines()
    iFn = [x.strip() for x in iFn]

    if masks is not None:
        with open(os.path.abspath(args.maskFn)) as f:
            mFn = f.readlines()
        mFn = [x.strip() for x in mFn]
    else:
        mFn=[]


    i = 0
    dice_list= {}
    acc_voxel_list = {}
    sensitivity_list = {}
    specificity_list = {}
    precision_list = {}
    name_list ={}

    for t in range(3):
        dice_list[t]= {}
        acc_voxel_list[t]= {}
        sensitivity_list[t] = {}
        specificity_list[t] = {}
        precision_list[t]= {}
        name_list[t] = {}


    directory = '/public/yangxiaodu/clearmap/data/post_all/big_vessels_filling_output1_tubemapoutput'  #you can place tubemap results in .../FineVess/results/post_refine
    # convert_to_tiff(iFn, directory)
    test_v={}
    v=0
    if mFn!=[]:
        # for ifn, mfn, nffn1 in zip(iFn, mFn,nfFn1):
        for ifn, mfn in zip(iFn, mFn):

            prefix = os.path.basename(ifn).split('.')[0]
            print('predicting features for :', ifn)

            data_tubemap=clearmap_filling(ifn, directory)

            print('predicting features for :', ifn)

            
            post1_result,dice_list[0][i],acc_voxel_list[0][i],sensitivity_list[0][i],specificity_list[0][i],precision_list[0][i],name_list[0][i]=filling_small_holes_and_conneting(ifn,data_tubemap,outputFn1,mfn)
            # post1_result, dice_list[0][i], acc_voxel_list[0][i], sensitivity_list[0][i], specificity_list[0][i],precision_list[0][i], name_list[0][i] = filling_small_holes_and_conneting(ifn, data_tubemap, nffn1,
            #                                                                           outputFn1, mfn)

            post2_result,dice_list[1][i],acc_voxel_list[1][i],sensitivity_list[1][i],specificity_list[1][i],precision_list[1][i],name_list[1][i]=big_vessel_filling(post1_result,ifn, outputFn2, directory, mfn)

            post3_result, dice_list[2][i], acc_voxel_list[2][i], sensitivity_list[2][i], specificity_list[2][i], precision_list[2][i], name_list[2][i] = denoise(post2_result, outputFn3, ifn, 150, mfn)
            i = i + 1


    for t in range(3):
        if t == 0:
            f = open(outputFn1 + '/' + 'metric.txt', 'a')
        if t == 1:
            f = open(outputFn2 + '/' + 'metric.txt', 'a')
        if t == 2:
            f = open(outputFn3 + '/' + 'metric.txt', 'a')
        f.write('name:' + str(name_list[t])  +'\n' + "dice_bins:" + str(dice_list[t]) +str(sum(dice_list[t].values())/len(dice_list[t])) +'\n' + "acc_voxel1:" + str(
            acc_voxel_list[t]) +str(sum(acc_voxel_list[t].values())/len(acc_voxel_list[t])) + "\n" + "sensitivity2:" + str(sensitivity_list[t]) +str(sum(sensitivity_list[t].values())/len(sensitivity_list[t])) +"\n" + "specificity2:" + str(
            specificity_list[t]) +str(sum(specificity_list[t].values())/len(specificity_list[t]))+ "\n" + "precision2:" + str(precision_list[t]) +str(sum(precision_list[t].values())/len(precision_list[t]))+ '\n')
        f.close()


if __name__ == '__main__':
    run()