import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from dvn.utils import get_itk_array,write_itk_image,get_itk_image,make_itk_image
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data#, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti
from collections import OrderedDict
from skimage.io import imread
import SimpleITK as sitk
#correct voxel space and create input files

def load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing,add_label=False):
    # img = imread(img_file)
    img = get_itk_image(img_file)
    # img_itk = sitk.GetImageFromArray(img.astype(np.float32))
    img.SetSpacing(np.array(spacing)[::-1])
    imageArray = sitk.GetArrayFromImage(img)
    imageArray = imageArray.astype(np.float32)
    image1 = make_itk_image(imageArray, img)
    write_itk_image(image1, img_out_base)
    # sitk.WriteImage(img_itk, img_out_base)
    if lab_file is not None:
        l = imread(lab_file)
        l[l > 0] = 1
        index = []
        if add_label is True:
            for ind in range(l.shape[0]):
                if np.all(l[ind, :, :] == 0) == True:
                    l[ind, :, :][:] = 2
                    index.append(ind)
            print(index)
        l_itk = sitk.GetImageFromArray(l.astype(np.uint8))
        l_itk.SetSpacing(np.array(spacing)[::-1])
        sitk.WriteImage(l_itk, anno_out)

if __name__ == '__main__':
    # '''''''''''''''
    """
    nnU-Net was originally built for 3D images. It is also strongest when applied to 3D segmentation problems because a 
    large proportion of its design choices were built with 3D in mind. Also note that many 2D segmentation problems, 
    especially in the non-biomedical domain, may benefit from pretrained network architectures which nnU-Net does not
    support.
    Still, there is certainly a need for an out of the box segmentation solution for 2D segmentation problems. And 
    also on 2D segmentation tasks nnU-Net cam perform extremely well! We have, for example, won a 2D task in the cell 
    tracking challenge with nnU-Net (see our Nature Methods paper) and we have also successfully applied nnU-Net to 
    histopathological segmentation problems. 
    Working with 2D data in nnU-Net requires a small workaround in the creation of the dataset. Essentially, all images 
    must be converted to pseudo 3D images (so an image with shape (X, Y) needs to be converted to an image with shape 
    (1, X, Y). The resulting image must be saved in nifti format. Hereby it is important to set the spacing of the 
    first axis (the one with shape 1) to a value larger than the others. If you are working with niftis anyways, then 
    doing this should be easy for you. This example here is intended for demonstrating how nnU-Net can be used with 
    'regular' 2D images. We selected the massachusetts road segmentation dataset for this because it can be obtained 
    easily, it comes with a good amount of training cases but is still not too large to be difficult to handle.
    """
    # now start the conversion to nnU-Net:
    spacing1 = (4.99, 2.83, 2.83)#(2,4,6,8,10,12,14,17,19,22,23)
    spacing2 = (5, 0.994, 0.994)#(1,5,7,9,13,20,21)
    spacing3 = (5, 1.59, 1.59)#(3)
    spacing4 = (5, 2.485, 2.485)#(15)
    spacing5 = (5, 0.621, 0.621)#(11,16,18)
    spacing6=(0.799,0.568,0.568) #(24-36)
    spacing7=(4,0.446,0.446) #(37)
    spacing8=(1.0,1.0,1.0)#(38-47)

    # task_name = 'Task508_glioma_vessels'
    task_name = 'other_data_raw'
    # task_name = 'other_data_preprocessed'
    print(task_name)
    target_base = join("/home/xdyang/FineVess/nnUNet-master/save_data/nnUNet_raw_data_base",'nnUNet_raw_data', task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")


    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)


    source_folder = '/home/xdyang/FineVess/data'
    train_cases = subfiles(join(source_folder, 'raw_data'), suffix=".nii.gz", join=False) #<class 'list'>: ['vessel_001_0000.tif', 'vessel_002_0000.tif', 'vessel_003_0000.tif', 'vessel_004_0000.tif', 'vessel_005_0000.tif', 'vessel_006_0000.tif', 'vessel_007_0000.tif', 'vessel_008_0000.tif', 'vessel_009_0000.tif', 'vessel_010_0000.tif', 'vessel_011_0000.tif', 'vessel_012_0000.tif', 'vessel_013_0000.tif', 'vessel_014_0000.tif', 'vessel_015_0000.tif', 'vessel_016_0000.tif', 'vessel_017_0000.tif', 'vessel_018_0000.tif', 'vessel_019_0000.tif']
    for i,t in enumerate(train_cases):
        # img_file = join(source_folder, 'jzl', t)
        # lab_file = join(source_folder, 'jzl_GT/' + t[0:10]+t[-4:])
        img_file = join(source_folder, 'raw_data',t)
        # lab_file = join(source_folder, 'mv_nii_GT/' + t[0:4] + '_y' + t[-7:])
        # if not isfile(lab_file):
        #     # not all cases are annotated in the manual dataset
        #     continue
        img_out_base = join(target_imagesTr, os.path.basename(img_file)[:-7] + '_0000'+".nii.gz")
        # anno_out = join(target_labelsTr, os.path.basename(lab_file)[:-4] + ".nii.gz")
        # i=i+23
        # if i+1<=37:
        #     if i + 1 in [2, 4, 6, 8, 10, 12, 14, 17, 19, 22, 23]:
        #         load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing1,False)
        #     elif i + 1 in [1, 5, 7, 9, 13, 20, 21]:
        #         load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing2,False)
        #     elif i + 1 in [3]:
        #         load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing3,False)
        #     elif i + 1 in [15]:
        #         load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing4,False)
        #     elif i + 1 in [11, 16, 18]:
        #         load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing5,False)
        #     elif i + 1 in [37]:
        #         load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing7,True)
        #     else:
        #         load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing6,True)
        #     print('0')
        # else:
        #     load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing8, False)
        ifn = os.path.basename(img_file)[:-7]
        print(ifn)
        if "#5-CD31-6" in ifn:
            space=(0.799,1.14,1.14)
            print("#5-CD31-6")
        elif "-10x" in ifn:
            space = (0.799, 1.14, 1.14)
            print("x10")
        elif "539" in ifn:
            space=(4, 0.446, 0.446)
            print("539")
        elif "-40x" in ifn:
            space=(0.799,0.285,0.285)
            print("-40x")
        else:
            space = (0.799, 0.568, 0.568)
            print("other")
        load_tiff_convert_to_nifti(img_file, None, img_out_base, None, space, False)
        # train_patient_names.append(casename)

    # '''''''''
    # """""""""""
    # task_name = 'other_data_raw'
    # print(task_name)
    # target_base = join("/data/run01/scz5927/nnUNet/save_data/nnUNet_raw_data_base", 'nnUNet_raw_data', task_name)
    # target_labelsTr = join(target_base, "labelsTr")
    # labelsTr = subfiles(target_labelsTr)
    # # finally we can call the utility for generating a dataset.json
    # # generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ("CD31"),
    # #                       labels={0: 'background', 1: 'vessels'}, dataset_name=task_name, license='hands off!')
    # # create a json file that will be needed by nnUNet to initiate the preprocessing process
    # json_dict = OrderedDict()
    # # json_dict['name'] = "other_vessels"
    # # json_dict['description'] = "other vessels segmentation"
    # json_dict['name'] = "other_data_raw"
    # json_dict['description'] = "new_jzl_vessels segmentation"
    # json_dict['tensorImageSize'] = "4D"
    # json_dict['reference'] = "none"
    # json_dict['licence'] = "see reference"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "CD31",
    # }
    # # labels differ for ACDC challenge
    # # json_dict['labels'] = {0: 'background', 1: 'vessels',2:'unlabeled'}
    # json_dict['labels'] = {0: 'background', 1: 'vessels'}
    # json_dict['numTraining'] = len(labelsTr)
    # json_dict['numTest'] = 0
    # json_dict['training'] = [{'image': "./imagesTr/%s" % i.split("/")[-1],
    #                           "label": "./labelsTr/%s" % i.split("/")[-1]} for i in labelsTr]
    # json_dict['test'] = []
    #
    # save_json(json_dict, os.path.join(target_base, "dataset.json"))

    '''''''''

    labelsTr = subfiles(target_labelsTr)
    # finally we can call the utility for generating a dataset.json
    # generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ("CD31"),
    #                       labels={0: 'background', 1: 'vessels'}, dataset_name=task_name, license='hands off!')
    # create a json file that will be needed by nnUNet to initiate the preprocessing process
    json_dict = OrderedDict()
    # json_dict['name'] = "other_vessels"
    # json_dict['description'] = "other vessels segmentation"
    json_dict['name'] = "preprocessed glioma_vessels"
    json_dict['description'] = "preprocessed glioma vessels segmentation with 2 label"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "none"
    json_dict['licence'] = "see reference"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CD31",
    }
    # labels differ for ACDC challenge
    # json_dict['labels'] = {0: 'background', 1: 'vessels',2:'unlabeled'}
    json_dict['labels'] = {0: 'background', 1: 'vessels'}
    json_dict['numTraining'] = len(labelsTr)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s" % i.split("/")[-1],
                              "label": "./labelsTr/%s" % i.split("/")[-1]} for i in labelsTr]
    json_dict['test'] = []

    save_json(json_dict, os.path.join(target_base, "dataset.json"))
    '''''''''
    # """""""
    # once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    # dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the
    # `nnUNet_plan_and_preprocess` command like this:
    #
    # > nnUNet_plan_and_preprocess -t 120 -pl3d None
    #
    # once that is completed, you can run the trainings as follows:
    # > nnUNet_train 2d nnUNetTrainerV2 120 FOLD
    #
    # (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)
    #
    # there is no need to run nnUNet_find_best_configuration because there is only one model to choose from.
    # Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    # for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    # a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    # `nnUNet_determine_postprocessing` command
    # """""""
