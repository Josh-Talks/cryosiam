import os
import torch
import pickle
import random
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR

from monai.utils import set_determinism
from monai.data.utils import worker_init_fn
from monai.data import CacheDataset, list_data_collate, Dataset, SmartCacheDataset, ThreadDataLoader, set_track_meta
from monai.transforms import (
    OneOf,
    Compose,
    Identityd,
    LoadImaged,
    SpatialPadd,
    EnsureTyped,
    RandCoarseDropoutd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRanged
)

from .data import MrcReader, TiffReader
from .utils import basic_train_val_split
from .networks.nets import DenseSimSiam
from .transforms import RandomMaskedViews, RandomLowPassBlurd, RandomGaussianNoised, RandomHighPassSharpend


class DenseSimSiamModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._model = DenseSimSiam(block_type=self.config['parameters']['network']['block_type'],
                                   spatial_dims=self.config['parameters']['network']['spatial_dims'],
                                   n_input_channels=self.config['parameters']['network']['in_channels'],
                                   num_layers=self.config['parameters']['network']['num_layers'],
                                   num_filters=self.config['parameters']['network']['num_filters'],
                                   fpn_channels=self.config['parameters']['network']['fpn_channels'],
                                   no_max_pool=self.config['parameters']['network']['no_max_pool'],
                                   dim=self.config['parameters']['network']['dim'],
                                   pred_dim=self.config['parameters']['network']['pred_dim'],
                                   dense_dim=self.config['parameters']['network']['dense_dim'],
                                   dense_pred_dim=self.config['parameters']['network']['dense_pred_dim'],
                                   include_levels=self.config['parameters']['network']['include_levels_loss']
                                   if 'include_levels_loss' in self.config['parameters']['network'] else False,
                                   add_later_conv=self.config['parameters']['network']['add_fpn_later_conv']
                                   if 'add_fpn_later_conv' in self.config['parameters']['network'] else False,
                                   decoder_type=self.config['parameters']['network']['decoder_type']
                                   if 'decoder_type' in self.config['parameters']['network'] else 'fpn',
                                   decoder_layers=self.config['parameters']['network']['fpn_layers']
                                   if 'fpn_layers' in self.config['parameters']['network'] else 2)

        self.batch_size = self.config['hyper_parameters']['batch_size']
        self.spatial_dims = self.config['parameters']['network']['spatial_dims']
        self.lr = self.config['hyper_parameters']['lr']
        self.momentum = self.config['hyper_parameters']['momentum']
        self.weight_decay = self.config['hyper_parameters']['weight_decay']
        self.fix_pred_lr = self.config['hyper_parameters']['fix_pred_lr']
        self.max_epochs = self.config['hyper_parameters']['max_epochs']
        self.lr_warmup_epochs = self.config['hyper_parameters']['lr_warmup_epochs']
        self.include_levels_loss = self.config['parameters']['network']['include_levels_loss'] \
            if 'include_levels_loss' in config['parameters']['network'] else False
        self.include_global_loss = self.config['parameters']['network']['include_global_loss'] \
            if 'include_global_loss' in config['parameters']['network'] else True
        self.weight_dense_loss = self.config['parameters']['network']['weight_dense_loss'] \
            if 'weight_dense_loss' in config['parameters']['network'] else None

        self.use_noisy_input = self.config['parameters']['transforms']['use_noisy_input'] \
            if 'use_noisy_input' in self.config['parameters']['transforms'] else False

        self.save_hyperparameters()
        if int(self.config['parameters']['gpu_devices']) * int(self.config['parameters']['nodes']) > 1:
            self.sync_dist = True
            self.prepare_data_per_node = False
        else:
            self.sync_dist = False

    def get_image_transforms(self):
        image_1_transforms = [
            RandomLowPassBlurd(keys=['image_1'], prob=1.0,
                               sigma=self.config['parameters']['transforms']['low_pass_sigma_range']) if
            self.config['parameters']['transforms']['low_pass_sigma_range'] else None,
            RandomHighPassSharpend(keys=['image_1'], prob=1.0,
                                   sigma=self.config['parameters']['transforms']['high_pass_sigma_range'],
                                   sigma2=self.config['parameters']['transforms']['high_pass_sigma2_range']) if
            self.config['parameters']['transforms']['high_pass_sigma_range'] else None,
            RandomGaussianNoised(keys=['image_1'], prob=1.0,
                                 sigma=self.config['parameters']['transforms']['noise_sigma_range']) if
            self.config['parameters']['transforms']['noise_sigma_range'] else None,
            Compose([
                RandomGaussianNoised(keys=['image_1'], prob=1.0,
                                     sigma=self.config['parameters']['transforms']['noise_sigma_range']),
                RandomLowPassBlurd(keys=['image_1'], prob=1.0,
                                   sigma=self.config['parameters']['transforms']['low_pass_sigma_range']),
                RandomHighPassSharpend(keys=['image_1'], prob=1.0,
                                       sigma=self.config['parameters']['transforms']['high_pass_sigma_range'],
                                       sigma2=self.config['parameters']['transforms']['high_pass_sigma2_range'])
            ]) if self.config['parameters']['transforms']['combine_transforms'] else None,
            Identityd(keys=['image_1'])]
        image_1_transforms = [x for x in image_1_transforms if x is not None]
        image_2_transforms = [
            RandomLowPassBlurd(keys=['image_2'], prob=1.0,
                               sigma=self.config['parameters']['transforms']['low_pass_sigma_range']) if
            self.config['parameters']['transforms']['low_pass_sigma_range'] else None,
            RandomHighPassSharpend(keys=['image_2'], prob=1.0,
                                   sigma=self.config['parameters']['transforms']['high_pass_sigma_range'],
                                   sigma2=self.config['parameters']['transforms']['high_pass_sigma2_range']) if
            self.config['parameters']['transforms']['high_pass_sigma_range'] else None,
            RandomGaussianNoised(keys=['image_2'], prob=1.0,
                                 sigma=self.config['parameters']['transforms']['noise_sigma_range']) if
            self.config['parameters']['transforms']['noise_sigma_range'] else None,
            Compose([
                RandomGaussianNoised(keys=['image_2'], prob=1.0,
                                     sigma=self.config['parameters']['transforms']['noise_sigma_range']),
                RandomLowPassBlurd(keys=['image_2'], prob=1.0,
                                   sigma=self.config['parameters']['transforms']['low_pass_sigma_range']),
                RandomHighPassSharpend(keys=['image_2'], prob=1.0,
                                       sigma=self.config['parameters']['transforms']['high_pass_sigma_range'],
                                       sigma2=self.config['parameters']['transforms']['high_pass_sigma2_range'])
            ]) if self.config['parameters']['transforms']['combine_transforms'] else None,
            Identityd(keys=['image_2'])]
        image_2_transforms = [x for x in image_2_transforms if x is not None]
        return image_1_transforms, image_2_transforms

    def setup_transformations(self):
        reader = MrcReader if self.config['file_extension'] in ['.mrc', '.rec'] else TiffReader
        print('Local rank:', f'cuda:{self.trainer.local_rank}')
        device = f'cuda:{self.trainer.local_rank}' if self.trainer.local_rank is not None else 'cuda:0'

        if self.use_noisy_input:
            keys = ['image', 'noisy_image']
            keys1 = ['image_1', 'noisy_image_1']
            keys2 = ['image_2', 'noisy_image_2']
        else:
            keys = ['image']
            keys1 = ['image_1']
            keys2 = ['image_2']

        image_1_transforms, image_2_transforms = self.get_image_transforms()
        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=keys, reader=reader()),
                EnsureChannelFirstd(keys=keys, channel_dim='no_channel'),
                ScaleIntensityRanged(keys=keys, a_min=self.config['parameters']['data']['min'],
                                     a_max=self.config['parameters']['data']['max'], b_min=0, b_max=1),
                SpatialPadd(keys=keys, spatial_size=self.config['parameters']['data']['patch_size']),
                RandomMaskedViews(keys=keys,
                                  input_image_size=self.config['parameters']['data']['patch_size'],
                                  view_size=self.config['parameters']['data']['view_size'],
                                  overlap=self.config['parameters']['data']['view_overlap']),
                # transformations on the first view
                OneOf(transforms=image_1_transforms),
                # transformations on the second view
                OneOf(transforms=image_2_transforms),
                RandCoarseDropoutd(keys=keys1, prob=self.config['parameters']['transforms']['drop_out_prob'],
                                   holes=int(np.prod(self.config['parameters']['data']['view_size']) /
                                             np.prod([4] * self.spatial_dims) * 0.1),
                                   max_holes=int(np.prod(self.config['parameters']['data']['view_size']) /
                                                 np.prod([4] * self.spatial_dims) *
                                                 self.config['parameters']['transforms']['drop_out']),
                                   spatial_size=[1] * self.spatial_dims, max_spatial_size=[4] * self.spatial_dims,
                                   fill_value=0) if self.config['parameters']['transforms']['drop_out'] else
                Identityd(keys=keys1),
                RandCoarseDropoutd(keys=keys2, prob=self.config['parameters']['transforms']['drop_out_prob'],
                                   holes=int(np.prod(self.config['parameters']['data']['view_size']) /
                                             np.prod([4] * self.spatial_dims) * 0.1),
                                   max_holes=int(np.prod(self.config['parameters']['data']['view_size']) /
                                                 np.prod([4] * self.spatial_dims) *
                                                 self.config['parameters']['transforms']['drop_out']),
                                   spatial_size=[1] * self.spatial_dims, max_spatial_size=[4] * self.spatial_dims,
                                   fill_value=0) if self.config['parameters']['transforms']['drop_out'] else
                Identityd(keys=keys2),
                NormalizeIntensityd(keys=keys1 + keys2, subtrahend=self.config['parameters']['data']['mean'],
                                    divisor=self.config['parameters']['data']['std']),
                EnsureTyped(keys=keys1 + keys2, data_type='tensor', dtype=torch.float),
                EnsureTyped(keys=['mask_1', 'mask_2'], data_type='tensor', dtype=torch.long),
            ]
        )
        set_track_meta(False)
        return train_transforms

    def prepare_data(self):
        data_root = os.path.normpath(self.config['data_folder'])
        train_val_path = os.path.join(self.config['log_dir'], 'train_val_split.pkl')
        if not os.path.isfile(train_val_path):
            train_files, val_files = basic_train_val_split(data_root,
                                                           noisy_data_root=self.config['noisy_data_folder']
                                                           if 'noisy_data_folder' in self.config else None,
                                                           files=self.config['train_files'],
                                                           ratio=self.config['validation_ratio'],
                                                           patches_folder=self.config['patches_folder'],
                                                           noisy_patches_folder=self.config['noisy_patches_folder']
                                                           if 'noisy_patches_folder' in self.config else None,
                                                           file_ext=self.config['file_extension'])
            with open(train_val_path, 'wb') as f:
                pickle.dump({'train_files': train_files, 'val_files': val_files}, f)

    def initialization(self):
        # set data
        train_val_path = os.path.join(self.config['log_dir'], 'train_val_split.pkl')
        with open(train_val_path, 'rb') as f:
            data = pickle.load(f)
            train_files, val_files = data['train_files'], data['val_files']
        return train_files, val_files

    def setup(self, stage=None):
        train_files, val_files = self.initialization()
        if self.trainer.world_size > 1 and self.config['hyper_parameters']['cache_rate'] > 0:
            partition_len = len(train_files) // self.trainer.world_size
            global_rank = self.trainer.global_rank
            train_files = train_files[(partition_len * global_rank):(partition_len * global_rank + partition_len)]
            print(f'World size: {self.trainer.world_size}, global_rank: {global_rank}, '
                  f'start_index: {partition_len * global_rank}')
            partition_len = len(val_files) // self.trainer.world_size
            global_rank = self.trainer.global_rank
            val_files = val_files[(partition_len * global_rank):(partition_len * global_rank + partition_len)]
            print(f'World size: {self.trainer.world_size}, global_rank: {global_rank}, '
                  f'start_index: {partition_len * global_rank}')

        # set deterministic training for reproducibility
        set_determinism(seed=0)
        train_transforms = self.setup_transformations()

        if self.config['hyper_parameters']['cache_rate'] > 0:
            # cached datasets - 10x faster than regular datasets
            rate = self.config['hyper_parameters']['cache_rate']
            if 'replace_rate' in self.config['hyper_parameters']:
                replace_rate = self.config['hyper_parameters']['replace_rate']
                self.train_ds = SmartCacheDataset(data=train_files, transform=train_transforms,
                                                  replace_rate=replace_rate,
                                                  num_init_workers=1, num_replace_workers=1, shuffle=True,
                                                  seed=self.trainer.global_rank, cache_rate=rate)
                self.val_ds = SmartCacheDataset(data=val_files, transform=train_transforms, replace_rate=replace_rate,
                                                num_init_workers=1, num_replace_workers=1, shuffle=True,
                                                seed=self.trainer.global_rank, cache_rate=rate)
            else:
                self.train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=10,
                                             cache_num=len(train_files), cache_rate=rate, copy_cache=False)
                self.val_ds = CacheDataset(data=val_files, transform=train_transforms, num_workers=4,
                                           cache_num=len(val_files), cache_rate=rate, copy_cache=False)
        else:
            self.train_ds = Dataset(data=train_files, transform=train_transforms)
            self.val_ds = Dataset(data=val_files, transform=train_transforms)

    def train_dataloader(self):
        if self.config['hyper_parameters']['cache_rate'] > 0:
            train_loader = ThreadDataLoader(self.train_ds, num_workers=0, batch_size=self.batch_size, shuffle=True,
                                            buffer_size=80, collate_fn=list_data_collate, pin_memory=True,
                                            drop_last=True)
        else:
            train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=list_data_collate,
                                      shuffle=False if self.sync_dist else True, num_workers=10,
                                      persistent_workers=True, worker_init_fn=worker_init_fn,
                                      pin_memory=False, drop_last=True)
        return train_loader

    def val_dataloader(self):
        if self.config['hyper_parameters']['cache_rate'] > 0:
            val_loader = ThreadDataLoader(self.val_ds, num_workers=0, batch_size=self.batch_size,
                                          collate_fn=list_data_collate, pin_memory=True)
        else:
            val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=list_data_collate,
                                    num_workers=4, persistent_workers=True, worker_init_fn=worker_init_fn,
                                    pin_memory=False)
        return val_loader

    def forward(self, x1, x2):
        return self._model(x1, x2)

    def configure_optimizers(self):
        if self.fix_pred_lr:
            optim_params = [{'params': self._model.encoder.parameters(), 'fix_lr': False},
                            {'params': self._model.decoder.parameters(), 'fix_lr': False},
                            {'params': self._model.avg_pool.parameters(), 'fix_lr': False},
                            {'params': self._model.projector.parameters(), 'fix_lr': False},
                            {'params': self._model.global_projector.parameters(), 'fix_lr': False},
                            {'params': self._model.predictor.parameters(), 'fix_lr': True},
                            {'params': self._model.global_predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = self._model.parameters()
        lr = self.lr * self.batch_size / 256
        if self.weight_decay == 0:
            optimizer = torch.optim.SGD(optim_params, lr, momentum=self.momentum)
        else:
            optimizer = torch.optim.SGD(optim_params, lr, momentum=self.momentum, weight_decay=self.weight_decay)
        if self.lr_warmup_epochs is not None:
            scheduler_one = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / self.lr_warmup_epochs)
            scheduler_two = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler_one, scheduler_two],
                                     milestones=[self.lr_warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        lr_schedulers = {"scheduler": scheduler,
                         "monitor": "train_loss"}
        return [optimizer], [lr_schedulers]

    def loss(self, p1, p2, z1, z2, p1_global, p2_global, z1_global, z2_global, mask1, mask2,
             levels_p1=None, levels_p2=None, levels_z1=None, levels_z2=None):
        loss_dense = -(self._model.calculate_dense_loss(p1, z2, mask1, mask2).mean() +
                       self._model.calculate_dense_loss(p2, z1, mask2, mask1).mean()) * 0.5
        loss_global = -(self._model.calculate_global_loss(p1_global, z2_global).mean() +
                        self._model.calculate_global_loss(p2_global, z1_global).mean()) * 0.5
        if self.include_levels_loss:
            flag = self.include_levels_loss if isinstance(self.include_levels_loss, int) else None
            loss_levels1 = self._model.calculate_levels_loss(levels_p1, levels_z2, mask1, mask2, flag)
            loss_levels2 = self._model.calculate_levels_loss(levels_p2, levels_z1, mask2, mask1, flag)
            loss_levels = [-(loss1.mean() + loss2.mean()) * 0.5 for loss1, loss2 in zip(loss_levels1, loss_levels2)]
            return loss_dense, loss_global, loss_levels
        return loss_dense, loss_global

    def training_step(self, batch, batch_idx):
        inputs1, inputs2, mask1, mask2 = batch['image_1'], batch['image_2'], \
            batch['mask_1'], batch['mask_2']
        if self.use_noisy_input:
            inputs1 = batch[random.choices(['image_1', 'noisy_image_1'], weights=[4, 1], k=1)[0]]
        if self.include_levels_loss:
            p1, p2, z1, z2, p1_global, p2_global, z1_global, z2_global, \
                levels_p1_2, levels_p1_4, levels_p1_8, levels_p2_2, levels_p2_4, levels_p2_8, \
                levels_z1_2, levels_z1_4, levels_z1_8, levels_z2_2, levels_z2_4, levels_z2_8 = self.forward(inputs1,
                                                                                                            inputs2)
            loss_dense, loss_global, loss_levels = self.loss(p1, p2, z1, z2, p1_global, p2_global,
                                                             z1_global, z2_global, mask1, mask2,
                                                             [levels_p1_2, levels_p1_4, levels_p1_8],
                                                             [levels_p2_2, levels_p2_4, levels_p2_8],
                                                             [levels_z1_2, levels_z1_4, levels_z1_8],
                                                             [levels_z2_2, levels_z2_4, levels_z2_8])
            final_loss = loss_dense + sum(loss_levels) + loss_global
            self.log("train_loss", final_loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
            self.log("train_dense_loss", loss_dense, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
            for i in range(len(loss_levels)):
                self.log(f"train_levels_loss_{(i + 1)}", loss_levels[i], on_step=False, on_epoch=True,
                         sync_dist=self.sync_dist)
        else:
            p1, p2, z1, z2, p1_global, p2_global, z1_global, z2_global = self.forward(inputs1, inputs2)
            loss_dense, loss_global = self.loss(p1, p2, z1, z2, p1_global, p2_global,
                                                z1_global, z2_global, mask1, mask2)

            if self.include_global_loss:
                if self.weight_dense_loss is not None:
                    final_loss = self.weight_dense_loss * loss_dense + (1 - self.weight_dense_loss) * loss_global
                else:
                    final_loss = loss_dense + loss_global
                self.log("train_loss", final_loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
                self.log("train_dense_loss", loss_dense, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
                self.log("train_global_loss", loss_global, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
            else:
                final_loss = loss_dense
                self.log("train_loss", final_loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
                self.log("train_dense_loss", loss_dense, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        return final_loss

    def validation_step(self, batch, batch_idx):
        inputs1, inputs2, mask1, mask2 = batch['image_1'], batch['image_2'], \
            batch['mask_1'], batch['mask_2']
        if self.use_noisy_input:
            inputs1 = batch[random.choices(['image_1', 'noisy_image_1'], weights=[4, 1], k=1)[0]]
        if self.include_levels_loss:
            p1, p2, z1, z2, p1_global, p2_global, z1_global, z2_global, \
                levels_p1_2, levels_p1_4, levels_p1_8, levels_p2_2, levels_p2_4, levels_p2_8, \
                levels_z1_2, levels_z1_4, levels_z1_8, levels_z2_2, levels_z2_4, levels_z2_8 = self.forward(inputs1,
                                                                                                            inputs2)
            loss_dense, loss_global, loss_levels = self.loss(p1, p2, z1, z2, p1_global, p2_global,
                                                             z1_global, z2_global, mask1, mask2,
                                                             [levels_p1_2, levels_p1_4, levels_p1_8],
                                                             [levels_p2_2, levels_p2_4, levels_p2_8],
                                                             [levels_z1_2, levels_z1_4, levels_z1_8],
                                                             [levels_z2_2, levels_z2_4, levels_z2_8])
            final_loss = loss_dense + sum(loss_levels) + loss_global
            self.log("val_loss", final_loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
            self.log("val_dense_loss", loss_dense, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
            for i in range(len(loss_levels)):
                self.log(f"val_levels_loss_{(i + 1)}", loss_levels[i], on_step=False, on_epoch=True,
                         sync_dist=self.sync_dist)
            self.log("val_global_loss", loss_global, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        else:
            p1, p2, z1, z2, p1_global, p2_global, z1_global, z2_global = self.forward(inputs1, inputs2)
            loss_dense, loss_global = self.loss(p1, p2, z1, z2, p1_global, p2_global,
                                                z1_global, z2_global, mask1, mask2)
            if self.include_global_loss:
                if self.weight_dense_loss is not None:
                    final_loss = self.weight_dense_loss * loss_dense + (1 - self.weight_dense_loss) * loss_global
                else:
                    final_loss = loss_dense + loss_global
                self.log("val_loss", final_loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
                self.log("val_dense_loss", loss_dense, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
                self.log("val_global_loss", loss_global, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
            else:
                final_loss = loss_dense
                self.log("val_loss", final_loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
                self.log("val_dense_loss", loss_dense, on_step=False, on_epoch=True, sync_dist=self.sync_dist)

    def on_train_epoch_start(self):
        if self.config['hyper_parameters']['cache_rate'] > 0 and 'replace_rate' in self.config['hyper_parameters']:
            if self.current_epoch == 0:
                print('Start smart cache')
                self.trainer.train_dataloader.dataset.start()

    def on_train_epoch_end(self):
        if self.config['hyper_parameters']['cache_rate'] > 0 and 'replace_rate' in self.config['hyper_parameters']:
            if self.current_epoch == self.max_epochs:
                print('Shut down smart cache')
                self.trainer.train_dataloader.dataset.shutdown()
            else:
                print('New epoch update smart cache')
                self.trainer.train_dataloader.dataset.update_cache()

    def on_validation_epoch_start(self):
        if self.config['hyper_parameters']['cache_rate'] > 0 and 'replace_rate' in self.config['hyper_parameters']:
            if self.current_epoch == 0:
                print('Start val smart cache')
                self.trainer.val_dataloaders.dataset.start()

    def on_validation_epoch_end(self):
        if self.config['hyper_parameters']['cache_rate'] > 0 and 'replace_rate' in self.config['hyper_parameters']:
            if self.current_epoch == self.max_epochs:
                print('Shut down val smart cache')
                self.trainer.val_dataloaders.dataset.shutdown()
            else:
                print('New epoch update val smart cache')
                self.trainer.val_dataloaders.dataset.update_cache()
