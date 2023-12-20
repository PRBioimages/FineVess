from nnunet.training.network_training.nnUNetTrainerV2_loss_ignore_label2_ce_w_osm import nnUNetTrainerV2_loss_ignore_label2_ce_w_osm
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from torch import nn
import torch
from nnunet.utilities.nd_softmax import softmax_helper

class nnUNetTrainerV2_loss_ignore_label2_cew_osm_CBAM_decoder_btc_moreDA_BN(nnUNetTrainerV2_loss_ignore_label2_ce_w_osm):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["p_rot"] = 0.3

        self.data_aug_params["scale_range"] = (0.65, 1.6)
        self.data_aug_params["p_scale"] = 0.3

        self.data_aug_params["do_elastic"] = True
        self.data_aug_params["p_eldef"] = 0.3

        self.data_aug_params["do_additive_brightness"] = True
        self.data_aug_params["additive_brightness_mu"] = 0
        self.data_aug_params["additive_brightness_sigma"] = 0.2
        self.data_aug_params["additive_brightness_p_per_sample"] = 0.3
        self.data_aug_params["additive_brightness_p_per_channel"] = 1

        self.data_aug_params['gamma_range'] = (0.5, 1.6)

        self.data_aug_params['num_cached_per_thread'] = 4
    def initialize_network(self):
        """inference_apply_nonlin to sigmoid + larger unet"""
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            # norm_op = nn.InstanceNorm3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            # norm_op = nn.InstanceNorm3d
            norm_op = nn.BatchNorm3d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, 320,
                                    ifCBAM_decoder_before_tc=True)
        # print(self.network)
        # if torch.cuda.is_available():
        #     self.network.cuda()
        # self.network.inference_apply_nonlin = nn.Sigmoid()
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper