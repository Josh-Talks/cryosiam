from __future__ import annotations

from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from pydantic import Field
import yaml

from .utils import _StrictModel
from .logging import logging_type


class DenseSimSiamDataConfig(_StrictModel):
    patch_size: List[int] = Field(..., min_items=1)
    view_size: List[int] = Field(..., min_items=1)
    view_overlap: float = Field(..., ge=0.0, le=1.0)
    min: float
    max: float
    mean: float
    std: float = Field(..., gt=0)


class DenseSimSiamTransformsConfig(_StrictModel):
    low_pass_sigma_range: Optional[Tuple[float, float]] = None
    high_pass_sigma_range: Optional[Tuple[float, float]] = None
    high_pass_sigma2_range: Optional[Tuple[float, float]] = None
    noise_sigma_range: Optional[Tuple[float, float]] = None
    combine_transforms: bool = False
    drop_out_prob: float = Field(0.0, ge=0.0, le=1.0)
    drop_out: float = Field(0.0, ge=0.0)
    use_noisy_input: bool = False


class DenseSimSiamNetworkConfig(_StrictModel):
    block_type: Literal["basic", "bottleneck"] = "bottleneck"
    spatial_dims: int = Field(..., ge=2)
    in_channels: int = Field(..., ge=1)
    num_layers: Union[int, Sequence[int]]
    num_filters: Sequence[int] = Field(..., min_items=1)
    fpn_channels: int = Field(..., ge=1)
    no_max_pool: bool
    dim: int = Field(..., ge=1)
    pred_dim: int = Field(..., ge=1)
    dense_dim: int = Field(..., ge=1)
    dense_pred_dim: int = Field(..., ge=1)

    include_levels_loss: Union[bool, int] = False
    add_fpn_later_conv: bool = False
    decoder_type: Literal["fpn", "bifpn"] = "fpn"
    fpn_layers: int = Field(2, ge=1)
    include_global_loss: bool = True
    weight_dense_loss: Optional[float] = Field(None, ge=0.0, le=1.0)


class DenseSimSiamParametersConfig(_StrictModel):
    nodes: int = Field(..., ge=1)
    gpu_devices: int = Field(..., ge=1)
    data: DenseSimSiamDataConfig
    transforms: DenseSimSiamTransformsConfig
    network: DenseSimSiamNetworkConfig


class DenseSimSiamHyperParametersConfig(_StrictModel):
    batch_size: int = Field(..., ge=1)
    lr: float = Field(..., gt=0)
    momentum: float = Field(..., ge=0.0, le=1.0)
    weight_decay: float = Field(0.0, ge=0.0)
    fix_pred_lr: bool
    max_epochs: int = Field(..., ge=1)
    lr_warmup_epochs: Optional[int] = Field(None, ge=1)
    val_interval: int = Field(..., ge=1)
    cache_rate: float = Field(0.0, ge=0.0, le=1.0)
    replace_rate: Optional[float] = Field(None, ge=0.0, le=1.0)


class DenseSimSiamConfig(_StrictModel):
    data_folder: str
    logging: logging_type
    file_extension: Literal[".mrc", ".rec", ".tif", ".tiff"]
    train_files: Optional[List[str]]
    validation_ratio: float = Field(..., ge=0.0, le=1.0)
    patches_folder: Optional[str] = None
    noisy_data_folder: Optional[str] = None
    noisy_patches_folder: Optional[str] = None
    continue_training: bool = False
    parameters: DenseSimSiamParametersConfig
    hyper_parameters: DenseSimSiamHyperParametersConfig

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> "DenseSimSiamConfig":
        if hasattr(cls, "model_validate"):
            return cls.model_validate(config_dict)
        return cls.parse_obj(config_dict)

    @classmethod
    def from_yaml(cls, config_path: str) -> "DenseSimSiamConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(
                "DenseSimSiam config YAML must contain a dictionary at the top level."
            )
        return cls.from_dict(loaded)

    def to_dict(self) -> dict:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


def normalize_dense_simsiam_config(
    config: Union[DenseSimSiamConfig, Mapping[str, Any]],
) -> DenseSimSiamConfig:
    if isinstance(config, DenseSimSiamConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be a DenseSimSiamConfig model or a mapping/dict.")
    return DenseSimSiamConfig.from_dict(config)


def load_dense_simsiam_config(config_path: str) -> DenseSimSiamConfig:
    return DenseSimSiamConfig.from_yaml(config_path)
