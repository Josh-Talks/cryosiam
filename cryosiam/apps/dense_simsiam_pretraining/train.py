import os
import typer
from typing import Annotated, assert_never

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from cryosiam.data_structure.dense_simsiam_config import load_dense_simsiam_config
from cryosiam.apps.dense_simsiam_pretraining.module import DenseSimSiamModule


def main(
    config_file_path: Annotated[
        str, typer.Option(help="Path to config file", exists=True)
    ],
):
    cfg = load_dense_simsiam_config(config_file_path)

    is_distributed = cfg.parameters.gpu_devices > 1 or cfg.parameters.nodes > 1
    os.makedirs(cfg.logging.log_dir, exist_ok=True)

    # initialise the LightningModule

    # check for checkpoint to continue trianing
    # checkpoint = torch.load(cfg.pretrained_model, weights_only=False)
    net = DenseSimSiamModule(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging.log_dir, cfg.logging.name),
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

    if is_distributed:
        if cfg.hyper_parameters.cache_rate > 0:
            use_distributed_sampler = False
        else:
            use_distributed_sampler = True
        trainer = Trainer(
            accelerator="gpu",
            devices=cfg.parameters.gpu_devices,
            num_nodes=cfg.parameters.nodes,
            strategy=DDPStrategy(find_unused_parameters=True),
            max_epochs=cfg.hyper_parameters.max_epochs,
            use_distributed_sampler=use_distributed_sampler,
            logger=logger,
            check_val_every_n_epoch=cfg.hyper_parameters.val_interval,
            callbacks=[checkpoint_callback, lr_monitor],
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=cfg.hyper_parameters.max_epochs,
            logger=logger,
            check_val_every_n_epoch=cfg.hyper_parameters.val_interval,
            callbacks=[checkpoint_callback, lr_monitor],
        )

    if cfg.continue_training:
        trainer.fit(
            net,
            ckpt_path=os.path.join(cfg.logging.log_dir, cfg.logging.name, "last.ckpt"),
        )
    else:
        trainer.fit(net)


if __name__ == "__main__":
    typer.run(main)
