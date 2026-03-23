import math
import torch
import numpy as np
from scipy import ndimage
from typing import Any, Optional, Tuple, Sequence, Union, Literal
from monai.data.meta_obj import get_track_meta
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform, RandomizableTransform
from monai.utils.type_conversion import (
    convert_data_type,
    convert_to_dst_type,
    convert_to_tensor,
)


class ClipIntensity(Transform):
    """
    Clip the intensity for the entire image with given minimum and maximum intensity values.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, a_min=None, a_max=None) -> None:
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        out = torch.clip(img, min=self.a_min, max=self.a_max)
        out, *_ = convert_data_type(data=out, dtype=img.dtype)
        return out


class NumpyToTensor(Transform):
    """
    Scale intensity for the entire image by removing lower and upper percentage
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t, *_ = convert_data_type(img, np.ndarray, dtype=np.float32)
        out, *_ = convert_data_type(data=img_t, dtype=img.dtype)
        return out


class ScaleIntensity(Transform):
    """
    Scale intensity for the entire image by removing lower and upper percentage
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, lower_percentage=0.1, upper_percentage=99.9) -> None:
        self.lower_percentage = lower_percentage
        self.upper_percentage = upper_percentage

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """

        img = convert_to_tensor(img, track_meta=get_track_meta())

        img_t, *_ = convert_data_type(img, np.ndarray, dtype=np.float32)
        min_val = np.percentile(img_t, self.lower_percentage)
        max_val = np.percentile(img_t, self.upper_percentage)
        img_t = (img_t - min_val) / (max_val - min_val)
        out = np.clip(img_t, 0, 1)

        out, *_ = convert_data_type(data=out, dtype=img.dtype)
        return out


class InvertIntensity(Transform):
    """
    Invert intensity for the entire image by multiplying with -1.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """

        img = convert_to_tensor(img, track_meta=get_track_meta())
        out = img * -1
        out, *_ = convert_data_type(data=out, dtype=img.dtype)

        return out


class RandomSharpen(RandomizableTransform):
    """
    Implement high pass filtering on an image.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, sigma: Tuple[float, float], prob: float = 0.1) -> None:
        """
        Args:
            sigma: range of sigma value for bluring Gaussian filter.
            prob: probability to apply the sharpening.
        """
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma
        if len(self.sigma_range) != 2:
            raise AssertionError(
                f"Sigma range should be a sequence of two elements (start, end), "
                f"but got values {self.sigma_range}."
            )
        if self.sigma_range[0] > self.sigma_range[1]:
            raise AssertionError(
                f"First element of sigma range should be smaller than the second element of the "
                f"range, but got values {self.sigma_range}."
            )
        self.prob = prob
        self.sigma1 = self.sigma_range[0]

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.sigma1 = self.R.uniform(low=self.sigma_range[0], high=self.sigma_range[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        img_t = torch.squeeze(img_t, 0)
        low_pass = torch.from_numpy(
            ndimage.gaussian_filter(img_t.numpy(), self.sigma1)
        ).float()
        sharpen = img_t + (img_t - low_pass)
        out_t = sharpen.unsqueeze(0)
        out, *_ = convert_to_dst_type(out_t, dst=img, dtype=out_t.dtype)
        return out


class RandomLowPassBlur(RandomizableTransform):
    """
    Implement low pass filtering on an image.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, sigma: Tuple[float, float], ignore_zeros: bool = False, prob: float = 0.1
    ) -> None:
        """
        Args:
            sigma: range of sigma value for Gaussian filter.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the blur.
        """
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma
        if len(self.sigma_range) != 2:
            raise AssertionError(
                f"Sigma range should be a sequence of two elements (start, end), "
                f"but got values {self.sigma_range}."
            )
        if self.sigma_range[0] > self.sigma_range[1]:
            raise AssertionError(
                f"First element of sigma range should be smaller than the second element of the "
                f"range, but got values {self.sigma_range}."
            )
        self.sigma = self.sigma_range[0]
        self.ignore_zeros = ignore_zeros

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.sigma = self.R.uniform(low=self.sigma_range[0], high=self.sigma_range[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        img_t = torch.squeeze(img_t, 0)
        mask = img_t == 0
        low_pass = torch.from_numpy(
            ndimage.gaussian_filter(img_t.numpy(), self.sigma)
        ).float()
        if self.ignore_zeros:
            low_pass[mask] = 0
        out_t = low_pass.unsqueeze(0)
        out, *_ = convert_to_dst_type(out_t, dst=img, dtype=out_t.dtype)
        return out


class RandomGaussianNoise(RandomizableTransform):
    """
    Implement random gaussian noise on an image.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, sigma: Tuple[float, float], ignore_zeros: bool = False, prob: float = 0.1
    ) -> None:
        """
        Args:
            sigma: std of the added Gaussian noise.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the noise.
        """
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma
        if len(self.sigma_range) != 2:
            raise AssertionError(
                f"Sigma range should be a sequence of two elements (start, end), "
                f"but got values {self.sigma_range}."
            )
        if self.sigma_range[0] > self.sigma_range[1]:
            raise AssertionError(
                f"First element of sigma range should be smaller than the second element of the "
                f"range, but got values {self.sigma_range}."
            )
        self.sigma = self.sigma_range[0]
        self.ignore_zeros = ignore_zeros

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.sigma = self.R.uniform(low=self.sigma_range[0], high=self.sigma_range[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        img_t = torch.squeeze(img_t, 0)
        mask = img_t == 0
        noise = np.random.normal(0, self.sigma, size=img_t.shape).astype(np.float32)
        noised = img_t + torch.from_numpy(noise)
        min_val, max_val = torch.min(noised), torch.max(noised)
        noised = (noised - min_val) / (max_val - min_val)
        if self.ignore_zeros:
            noised[mask] = 0
        out_t = noised.unsqueeze(0)
        out, *_ = convert_to_dst_type(out_t, dst=img, dtype=out_t.dtype)
        return out


class RandomHighPassSharpen(RandomizableTransform):
    """
    Implement high pass filtering on an image.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        sigma: Tuple[float, float],
        sigma2: Tuple[float, float],
        ignore_zeros: bool = False,
        prob: float = 0.1,
    ) -> None:
        """
        Args:
            sigma: range of sigma value for first Gaussian filter.
            sigma2: range of sigma value for second Gaussian filter.
            ignore_zeros: avoid applying the transformation of the values of zeros in the image
            prob: probability to apply the sharpening.
        """
        RandomizableTransform.__init__(self, prob)
        self.sigma_range = sigma
        self.sigma2_range = sigma2
        if len(self.sigma_range) != 2:
            raise AssertionError(
                f"Sigma range should be a sequence of two elements (start, end), "
                f"but got values {self.sigma_range}."
            )
        if len(self.sigma2_range) != 2:
            raise AssertionError(
                f"Sigma2 range should be a sequence of two elements (start, end), "
                f"but got values {self.sigma2_range}."
            )
        if self.sigma_range[0] > self.sigma_range[1]:
            raise AssertionError(
                f"First element of sigma range should be smaller than the second element of the "
                f"range, but got values {self.sigma_range}."
            )
        if self.sigma2_range[0] > self.sigma2_range[1]:
            raise AssertionError(
                f"First element of sigma2 range should be smaller than the second element of the "
                f"range, but got values {self.sigma2_range}."
            )
        self.prob = prob
        self.sigma1 = self.sigma_range[0]
        self.sigma2 = self.sigma2_range[0]
        self.ignore_zeros = ignore_zeros

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.sigma1 = self.R.uniform(low=self.sigma_range[0], high=self.sigma_range[1])
        self.sigma2 = self.R.uniform(
            low=self.sigma2_range[0], high=self.sigma2_range[1]
        )

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        img_t = torch.squeeze(img_t, 0)
        mask = img_t == 0
        low_pass = torch.from_numpy(
            ndimage.gaussian_filter(img_t.numpy(), self.sigma1)
        ).float()
        low_pass2 = torch.from_numpy(
            ndimage.gaussian_filter(img_t.numpy(), self.sigma2)
        ).float()
        high_pass = low_pass - low_pass2
        if self.ignore_zeros:
            high_pass[mask] = 0
        out_t = high_pass.unsqueeze(0)
        out, *_ = convert_to_dst_type(out_t, dst=img, dtype=out_t.dtype)
        return out


class RandomMaskedViews(Transform):
    """Extract two random overlapping views with a fixed overlap shape.

    This variant does not require specifying an overlap strategy (e.g. side-by-side,
    vertical, diagonal). Instead, it selects a single feasible overlap shape during
    initialization and uses it for every sample, ensuring overlap tensors can be
    stacked across a batch.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        input_image_size: Union[Sequence[int], int],
        view_size: Union[Sequence[int], int],
        overlap: float = 0.5,
    ) -> None:
        self.input_image_size = (
            list(input_image_size)
            if isinstance(input_image_size, (list, tuple))
            else [input_image_size]
        )
        self.view_size = (
            list(view_size) if isinstance(view_size, (list, tuple)) else [view_size]
        )
        self.spatial_dims = len(self.input_image_size)

        if isinstance(overlap, (list, tuple)):
            raise ValueError(
                "overlap must be a single float specifying the exact total area/volume overlap."
            )
        self.overlap = float(overlap)

        if len(self.view_size) != self.spatial_dims:
            raise ValueError(
                f"view_size must have {self.spatial_dims} dimensions to match input_image_size, "
                f"got {len(self.view_size)}"
            )

        for i in range(self.spatial_dims):
            if self.view_size[i] > self.input_image_size[i]:
                raise ValueError(
                    f"view_size[{i}] ({self.view_size[i]}) cannot be larger than "
                    f"input_image_size[{i}] ({self.input_image_size[i]})"
                )
        if not (0.0 <= self.overlap <= 1.0):
            raise ValueError(f"overlap must be between 0.0 and 1.0, got {self.overlap}")

        self.view1_start = None
        self.view2_start = None
        self.R = np.random.RandomState()
        self._max_start = [
            size - view for size, view in zip(self.input_image_size, self.view_size)
        ]
        self._view_volume = int(np.prod(self.view_size))

        target_overlap_volume = self.overlap * self._view_volume
        rounded_target = int(round(target_overlap_volume))
        if not math.isclose(
            target_overlap_volume, rounded_target, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                "The requested overlap cannot be represented exactly on the discrete grid: "
                f"overlap={self.overlap} * view_volume={self._view_volume} gives "
                f"{target_overlap_volume}. Choose an overlap whose product with the view "
                "area/volume is an integer."
            )

        self.target_overlap_volume = rounded_target

        if self.target_overlap_volume == 0 and not any(
            max_start >= view_sz
            for max_start, view_sz in zip(self._max_start, self.view_size)
        ):
            raise ValueError(
                "Exact zero overlap is impossible because no spatial axis has enough room to "
                "separate the two views completely."
            )

        self.overlap_shape = self._select_fixed_overlap_shape()
        self._required_abs_offset = tuple(
            view_sz - overlap_len
            for view_sz, overlap_len in zip(self.view_size, self.overlap_shape)
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomMaskedViews":
        if state is not None:
            self.R = state
        elif seed is not None:
            self.R = np.random.RandomState(seed)
        return self

    def _compute_overlap_shape_candidates(self) -> Sequence[Tuple[int, ...]]:
        if self.target_overlap_volume == 0:
            return []

        allowed_lengths = []
        for view_sz, max_start in zip(self.view_size, self._max_start):
            min_overlap = max(1, view_sz - max_start)
            allowed_lengths.append(range(min_overlap, view_sz + 1))

        candidates = []

        def recurse(axis: int, remaining_volume: int, current_lengths: list) -> None:
            if axis == self.spatial_dims - 1:
                last_length = remaining_volume
                if last_length in allowed_lengths[axis]:
                    shape = tuple(current_lengths + [last_length])
                    candidates.append(shape)
                return

            for overlap_len in allowed_lengths[axis]:
                if remaining_volume % overlap_len != 0:
                    continue
                recurse(
                    axis + 1,
                    remaining_volume // overlap_len,
                    current_lengths + [overlap_len],
                )

        recurse(0, self.target_overlap_volume, [])
        return sorted(set(candidates))

    def _select_fixed_overlap_shape(self) -> Sequence[int]:
        if self.target_overlap_volume == 0:
            return [0] * self.spatial_dims

        candidates = self._compute_overlap_shape_candidates()
        if not candidates:
            raise ValueError(
                "No valid pair of views can satisfy the requested exact overlap while staying "
                "within input_image_size."
            )

        # Pick the most isotropic feasible shape; tie-break deterministically.
        def score(shape: Tuple[int, ...]) -> Tuple[float, int, Tuple[int, ...]]:
            mean_len = float(sum(shape)) / float(len(shape))
            spread = sum((length - mean_len) ** 2 for length in shape)
            anisotropy = max(shape) - min(shape)
            # Prefer larger minimum side on ties for stability of overlap crops.
            return spread, anisotropy, tuple(-length for length in shape)

        best_shape = min(candidates, key=score)
        return list(best_shape)

    def _sample_zero_overlap_offsets(self) -> Tuple[int, ...]:
        separating_axes = [
            axis
            for axis, (max_start, view_sz) in enumerate(
                zip(self._max_start, self.view_size)
            )
            if max_start >= view_sz
        ]
        if not separating_axes:
            raise ValueError(
                "Exact zero overlap is not feasible for the current image and view sizes."
            )

        forced_axis = separating_axes[int(self.R.randint(0, len(separating_axes)))]
        offsets = []
        for axis, max_start in enumerate(self._max_start):
            if axis == forced_axis:
                offsets.append(int(self.R.randint(self.view_size[axis], max_start + 1)))
            else:
                offsets.append(int(self.R.randint(0, max_start + 1)))
        return tuple(offsets)

    def _sample_signed_offset(self, abs_offset: Sequence[int]) -> Tuple[int, ...]:
        signed_offset = []
        for delta in abs_offset:
            if delta == 0:
                signed_offset.append(0)
            else:
                sign = -1 if self.R.randint(0, 2) == 0 else 1
                signed_offset.append(sign * int(delta))
        return tuple(signed_offset)

    def _sample_positions_from_offset(self, signed_offset: Sequence[int]) -> None:
        self.view1_start = []
        self.view2_start = []
        for offset_i, max_start_i in zip(signed_offset, self._max_start):
            p1_min = max(0, -offset_i)
            p1_max = min(max_start_i, max_start_i - offset_i)
            if p1_min > p1_max:
                raise ValueError(
                    "Internal error: sampled offset cannot be placed within input_image_size."
                )
            start1 = int(self.R.randint(p1_min, p1_max + 1))
            start2 = start1 + int(offset_i)
            self.view1_start.append(start1)
            self.view2_start.append(start2)

    def randomize(self, data: Optional[Any] = None) -> None:
        if self.target_overlap_volume == 0:
            abs_offset = self._sample_zero_overlap_offsets()
        else:
            abs_offset = self._required_abs_offset
        signed_offset = self._sample_signed_offset(abs_offset)
        self._sample_positions_from_offset(signed_offset)

    def _extract_view(
        self, img: NdarrayOrTensor, start_pos: Sequence[int]
    ) -> NdarrayOrTensor:
        slices = [slice(None)] + [
            slice(start_pos[d], start_pos[d] + self.view_size[d])
            for d in range(self.spatial_dims)
        ]
        return img[tuple(slices)]

    def _create_mask(
        self, view_start: Sequence[int], other_view_start: Sequence[int]
    ) -> np.ndarray:
        coords = np.zeros((self.spatial_dims, 2), dtype=np.int64)
        for axis, view_sz in enumerate(self.view_size):
            start_a = int(view_start[axis])
            end_a = start_a + int(view_sz) - 1
            start_b = int(other_view_start[axis])
            end_b = start_b + int(view_sz) - 1

            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)

            coords[axis, 0] = overlap_start - start_a
            coords[axis, 1] = overlap_end - start_a
        return coords

    def __call__(self, data: dict) -> dict:
        self.randomize()

        if self.view1_start is None or self.view2_start is None:
            return data

        img = data
        img_tensor = convert_to_tensor(img, track_meta=get_track_meta())

        view1 = self._extract_view(img_tensor, self.view1_start)
        view2 = self._extract_view(img_tensor, self.view2_start)

        mask1 = self._create_mask(self.view1_start, self.view2_start)
        mask2 = self._create_mask(self.view2_start, self.view1_start)

        view1_out, *_ = convert_to_dst_type(view1, dst=img, dtype=view1.dtype)
        view2_out, *_ = convert_to_dst_type(view2, dst=img, dtype=view2.dtype)
        mask1_out = torch.from_numpy(mask1).long()
        mask2_out = torch.from_numpy(mask2).long()

        return {
            "view1": view1_out,
            "view2": view2_out,
            "mask1": mask1_out,
            "mask2": mask2_out,
        }
