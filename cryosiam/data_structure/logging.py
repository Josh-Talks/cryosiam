from pydantic import Discriminator
from typing import Annotated, Literal, Optional, Union


from .utils import _StrictModel


class LoggerConfig(_StrictModel):
    log_dir: str


class WandbConfig(LoggerConfig):
    logger: Literal["wandb"] = "wandb"
    project: str
    name: str


class TensorBoardConfig(LoggerConfig):
    logger: Literal["tensorboard"] = "tensorboard"
    name: str


logging_type = Annotated[Union[WandbConfig, TensorBoardConfig], Discriminator("logger")]
