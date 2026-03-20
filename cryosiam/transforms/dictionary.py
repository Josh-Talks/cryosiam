import torch
import numpy as np
from monai.config import KeysCollection
from monai.utils import convert_to_tensor
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import convert_to_dst_type
from typing import Optional, Dict, Hashable, Tuple, Sequence, Union
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.compose import MapTransform, RandomizableTransform

from .array import (
    NumpyToTensor,
    RandomSharpen,
    ClipIntensity,
    ScaleIntensity,
    InvertIntensity,
    RandomLowPassBlur,
    RandomGaussianNoise,
    RandomHighPassSharpen,
    RandomMaskedViews,
)


class ClipIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.ClipIntensity`.
    """

    backend = ClipIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min=None,
        a_max=None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.clip = ClipIntensity(a_min=a_min, a_max=a_max)

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.clip(d[key])
        return d


class NumpyToTensord(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.ClipIntensity`.
    """

    backend = NumpyToTensor.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.convert = NumpyToTensor()

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.convert(d[key])
        return d


class ScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.ScaleIntensity`.
    """

    backend = ClipIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        lower_percentage=0.1,
        upper_percentage=99.9,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.scale = ScaleIntensity(
            lower_percentage=lower_percentage, upper_percentage=upper_percentage
        )

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scale(d[key])
        return d


class InvertIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.InvertIntensity`.
    """

    backend = InvertIntensity.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.inverter = InvertIntensity()

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.inverter(d[key])
        return d


class RandomSharpend(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomSharpen`.
    """

    backend = RandomSharpen.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma: Tuple[float, float],
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigma: range of sigma value for bluring Gaussian filter.
            prob: probability to apply the shapening.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.high_pass = RandomSharpen(sigma=sigma, prob=prob)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomSharpend":
        super().set_random_state(seed, state)
        self.high_pass.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma1, sigma2, etc.
        self.high_pass.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.high_pass(d[key], randomize=False)
        return d


class RandomLowPassBlurd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomLowPassBlur`.
    """

    backend = RandomLowPassBlur.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma: Tuple[float, float],
        ignore_zeros: bool = False,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigma: range of sigma value for Gaussian filter.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the blur.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.low_pass = RandomLowPassBlur(
            sigma=sigma, ignore_zeros=ignore_zeros, prob=prob
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomLowPassBlurd":
        super().set_random_state(seed, state)
        self.low_pass.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma, etc.
        self.low_pass.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.low_pass(d[key], randomize=False)
        return d


class RandomGaussianNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomGaussianNoise`.
    """

    backend = RandomGaussianNoise.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma: Tuple[float, float],
        ignore_zeros: bool = False,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigma: std of the added Gaussian noise.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the noise.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.noised = RandomGaussianNoise(
            sigma=sigma, ignore_zeros=ignore_zeros, prob=prob
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomGaussianNoise":
        super().set_random_state(seed, state)
        self.noised.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma, etc.
        self.noised.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.noised(d[key], randomize=False)
        return d


class RandomHighPassSharpend(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryoet_torch.transforms.RandomHighPassSharpen`.
    """

    backend = RandomHighPassSharpen.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma: Tuple[float, float],
        sigma2: Tuple[float, float],
        ignore_zeros: bool = False,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigma: range of sigma value for first Gaussian filter.
            sigma2: range of sigma value for second Gaussian filter.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the shapening.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.high_pass = RandomHighPassSharpen(
            sigma=sigma, sigma2=sigma2, ignore_zeros=ignore_zeros, prob=prob
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomHighPassSharpend":
        super().set_random_state(seed, state)
        self.high_pass.set_random_state(seed, state)
        return self

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma1, sigma2, etc.
        self.high_pass.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.high_pass(d[key], randomize=False)
        return d


class RandomMaskedViewsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`cryosiam.transforms.RandomMaskedViews`.

    Extracts two random overlapping views from input images and creates overlap-coordinate
    masks for each view.
    Handles multiple input keys (e.g., ["image"] or ["image", "noisy_image"]) and generates
    corresponding output with keys like ["image_1", "image_2", "mask_1", "mask_2"], or with
    noisy variants if present.
    """

    backend = RandomMaskedViews.backend

    def __init__(
        self,
        keys: KeysCollection,
        input_image_size: Union[Sequence[int], int],
        view_size: Union[Sequence[int], int],
        overlap: float = 0.5,
        overlap_mode: str = "diagonal",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed (e.g., ["image"] or ["image", "noisy_image"]).
                See also: :py:class:`monai.transforms.compose.MapTransform`
            input_image_size: size of the input patch (e.g., [64, 64] for 2D or [64, 64, 64] for 3D)
            view_size: size of each view to extract (e.g., [32, 32] for 2D or [32, 32, 32] for 3D)
            overlap: exact overlap fraction between the two views relative to the view
                area/volume (0.0 to 1.0). Default is 0.5.
            overlap_mode: how to distribute overlap across axes: 'side_by_side' (offset along last axis),
                'vertical' (offset along first axis), or 'diagonal' (symmetric offset on all axes).
                Default is 'diagonal'.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.masker = RandomMaskedViews(
            input_image_size=input_image_size,
            view_size=view_size,
            overlap=overlap,
            overlap_mode=overlap_mode,
        )

    def __call__(self, data: Dict) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Extract two views and masks from input images.

        Args:
            data: Dictionary containing input images with keys specified in self.keys.
                  Example: {"image": torch.Tensor, "noisy_image": torch.Tensor}

        Returns:
            Dictionary with extracted views and masks:
            Example: {
                "image_1": torch.Tensor,
                "image_2": torch.Tensor,
                "mask_1": torch.Tensor,
                "mask_2": torch.Tensor,
                "noisy_image_1": torch.Tensor,  # if present
                "noisy_image_2": torch.Tensor,  # if present
            }
            Note: masks are shared across all input images and generated only once.
            Each mask has shape [spatial_dims, 2] and stores inclusive overlap coordinates
            [start, end] in that view's local frame.
        """
        d = dict(data)

        # Generate shared random view positions once for all keys.
        # We do NOT call self.masker(img) because RandomMaskedViews.__call__
        # internally calls self.randomize() again, which would overwrite these
        # positions with new ones for every key.  Instead we use the internal
        # helpers directly so every key is cropped at the same locations.
        self.masker.randomize(None)

        output = {}
        mask1 = None
        mask2 = None

        for key in self.key_iterator(d):
            img = d[key]
            img_tensor = convert_to_tensor(img, track_meta=get_track_meta())

            # Extract views at the pre-generated (shared) positions
            view1 = self.masker._extract_view(img_tensor, self.masker.view1_start)
            view2 = self.masker._extract_view(img_tensor, self.masker.view2_start)

            view1_out, *_ = convert_to_dst_type(view1, dst=img, dtype=view1.dtype)
            view2_out, *_ = convert_to_dst_type(view2, dst=img, dtype=view2.dtype)

            output[f"{key}_1"] = view1_out
            output[f"{key}_2"] = view2_out

            # Masks depend only on the view positions, so create them once
            if mask1 is None:
                mask1 = torch.from_numpy(
                    self.masker._create_mask(
                        self.masker.view1_start, self.masker.view2_start
                    )
                ).long()
                mask2 = torch.from_numpy(
                    self.masker._create_mask(
                        self.masker.view2_start, self.masker.view1_start
                    )
                ).long()

        # Add shared masks (not prefixed with key name)
        output["mask_1"] = mask1
        output["mask_2"] = mask2

        return output
