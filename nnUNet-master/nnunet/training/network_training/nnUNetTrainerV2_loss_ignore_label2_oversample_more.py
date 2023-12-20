#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset


class nnUNetTrainerV2_loss_ignore_label2_oversample_more(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {},ignore_label=2)
        # self.loss = DC_and_ICE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, ignore_label=2)
        # self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.save_every = 50
        self.skip_blank=True
        self.oversample_foreground_percent=0.8
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,skip_blank=self.skip_blank,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,skip_blank=self.skip_blank,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val