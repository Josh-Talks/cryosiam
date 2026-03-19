import os
import torch
from typing import assert_never
import yaml

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from cryosiam.data_structure.dense_simsiam_config import load_dense_simsiam_config
from cryosiam.module import DenseSimSiamModule


def main(config_file_path):
    cfg = load_dense_simsiam_config(config_file_path)

    is_distributed = cfg.parameters.gpu_devices > 1 or cfg.parameters.nodes > 1
    os.makedirs(cfg.logging.log_dir, exist_ok=True)

    # initialise the LightningModule

    # check for checkpoint to continue trianing
    # checkpoint = torch.load(cfg.pretrained_model, weights_only=False)
    net = DenseSimSiamModule(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, "model"),
        filename="model_best",
        every_n_epochs=cfg.hyper_parameters.val_interval,
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)

    if cfg.logging.logger == "wandb":
        logger = WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.name,
            save_dir=os.path.join(cfg.logging.log_dir),
        )
    elif cfg.logging.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=os.path.join(cfg.logging.log_dir))

    else:
        assert_never(cfg.logging.logger)
