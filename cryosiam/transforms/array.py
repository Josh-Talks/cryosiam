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
    """
    Extract two random overlapping views from an input image.

    This transform randomly selects two views from a 2D or 3D patch, ensuring they have
    an exact overlap area/volume relative to the size of one view. Each view is cropped from
    the original patch, and masks are generated as overlap coordinates for each view.

    Args:
        input_image_size: size of the input patch (e.g., [64, 64] for 2D or [64, 64, 64] for 3D)
        view_size: size of each view to extract (e.g., [32, 32] for 2D or [32, 32, 32] for 3D)
        overlap: exact overlap fraction between the two views relative to the view area/volume
            (0.0 to 1.0). For example, 0.5 means the two views must overlap by exactly 50%
            of the area/volume of a single view.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        input_image_size: Union[Sequence[int], int],
        view_size: Union[Sequence[int], int],
        overlap: float = 0.5,
        overlap_mode: Literal["side_by_side", "vertical", "diagonal"] = "diagonal",
    ) -> None:
        # Ensure input_image_size and view_size are sequences
        self.input_image_size = (
            list(input_image_size)
            if isinstance(input_image_size, (list, tuple))
            else [input_image_size]
        )
        self.view_size = (
            list(view_size) if isinstance(view_size, (list, tuple)) else [view_size]
        )
        self.spatial_dims = len(self.input_image_size)
        self.overlap_mode = overlap_mode

        if isinstance(overlap, (list, tuple)):
            raise ValueError(
                "overlap must be a single float specifying the exact total area/volume overlap. "
                "Per-axis overlap sequences are not supported."
            )
        self.overlap = float(overlap)

        # Validate dimensions
        if len(self.view_size) != self.spatial_dims:
            raise ValueError(
                f"view_size must have {self.spatial_dims} dimensions to match input_image_size, "
                f"got {len(self.view_size)}"
            )

        # Validate that views fit within the image
        for i in range(self.spatial_dims):
            if self.view_size[i] > self.input_image_size[i]:
                raise ValueError(
                    f"view_size[{i}] ({self.view_size[i]}) cannot be larger than "
                    f"input_image_size[{i}] ({self.input_image_size[i]})"
                )
        if not (0.0 <= self.overlap <= 1.0):
            raise ValueError(f"overlap must be between 0.0 and 1.0, got {self.overlap}")

        # Validate overlap_mode
        if self.overlap_mode not in ["side_by_side", "vertical", "diagonal"]:
            raise ValueError(
                f"overlap_mode must be 'side_by_side', 'vertical', or 'diagonal', "
                f"got {self.overlap_mode}"
            )

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

        # Compute the overlap shape based on overlap_mode
        self._compute_overlap_shape()

        self._feasible_offsets = self._compute_feasible_offsets()
        if self.target_overlap_volume > 0 and not self._feasible_offsets:
            raise ValueError(
                "No valid pair of views can satisfy the requested exact overlap while staying "
                "within input_image_size."
            )
        if self.target_overlap_volume == 0 and not any(
            max_start >= view_sz
            for max_start, view_sz in zip(self._max_start, self.view_size)
        ):
            raise ValueError(
                "Exact zero overlap is impossible because no spatial axis has enough room to "
                "separate the two views completely."
            )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomMaskedViews":
        if state is not None:
            self.R = state
        elif seed is not None:
            self.R = np.random.RandomState(seed)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        """Generate random positions for view1 and view2 with exact overlap shape.

        Since overlap is now fixed in shape, the offset between view1_start and view2_start
        is deterministic. This method randomly samples view1_start within valid ranges,
        then computes view2_start based on the required offset.
        """
        if self.target_overlap_volume == 0:
            abs_offsets = self._sample_zero_overlap_offsets()
            self.view1_start = list(abs_offsets)
            self.view2_start = [
                start + offset
                for start, offset in zip(self.view1_start, (0,) * self.spatial_dims)
            ]
            return

        # Use the required offset (should be the only element)
        required_offset = self._feasible_offsets[0]

        # Randomly sample view1_start such that view2_start is also valid
        self.view1_start = []
        self.view2_start = []

        for i, (offset_i, max_start_i) in enumerate(
            zip(required_offset, self._max_start)
        ):
            # Valid range for p1[i]
            p1_min = max(0, -offset_i)
            p1_max = min(max_start_i, max_start_i - offset_i)

            start1 = int(self.R.randint(p1_min, p1_max + 1))
            start2 = start1 + offset_i

            self.view1_start.append(start1)
            self.view2_start.append(start2)

    def _compute_overlap_shape(self) -> None:
        """Compute the overlap region shape based on overlap_mode.

        This ensures the overlap region has a consistent, fixed shape for all sampled
        view pairs, enabling batching of loss calculations.
        """
        if self.target_overlap_volume == 0:
            # Zero overlap: no region
            self.overlap_shape = [0] * self.spatial_dims
            return

        if self.overlap_mode == "diagonal":
            # Symmetric overlap on all axes
            # For exact overlap_shape with symmetric distribution, we need:
            # overlap_len^spatial_dims = target_overlap_volume
            # However, this may not yield an integer overlap_len.
            # We use the rounded value and verify it's valid.
            overlap_len = int(
                round(self.target_overlap_volume ** (1.0 / self.spatial_dims))
            )
            self.overlap_shape = [overlap_len] * self.spatial_dims

            # Verify the computation is close enough (within rounding)
            computed_volume = int(np.prod(self.overlap_shape))
            if computed_volume != self.target_overlap_volume:
                # Try to find a valid overlap_len by searching nearby values
                found = False
                for candidate in [overlap_len - 1, overlap_len + 1]:
                    if candidate > 0:
                        candidate_volume = candidate**self.spatial_dims
                        relative_error = (
                            abs(self.target_overlap_volume - candidate_volume)
                            / self.target_overlap_volume
                        )
                        if relative_error <= 0.01:  # Within 1% tolerance
                            overlap_len = candidate
                            self.overlap_shape = [overlap_len] * self.spatial_dims
                            found = True
                            break

                if not found:
                    raise ValueError(
                        f"overlap_mode='diagonal' with overlap={self.overlap} cannot produce "
                        f"exact integer dimensions. target_volume={self.target_overlap_volume}, "
                        f"computed_volume={computed_volume}. "
                        f"Use an overlap value where view_size^spatial_dims * overlap is a perfect power."
                    )

        elif self.overlap_mode == "side_by_side":
            # Offset along last (width) axis; full overlap on all other axes
            # overlap_shape = [view_size[0], ..., view_size[-2], overlap_width]
            overlap_width = int(
                self.target_overlap_volume / int(np.prod(self.view_size[:-1]))
            )
            if overlap_width <= 0 or overlap_width > self.view_size[-1]:
                raise ValueError(
                    f"overlap_mode='side_by_side' with overlap={self.overlap} is incompatible. "
                    f"Computed overlap_width={overlap_width}, but view_size[-1]={self.view_size[-1]}. "
                    f"Choose an overlap that divides evenly by {int(np.prod(self.view_size[:-1]))}"
                )
            self.overlap_shape = list(self.view_size[:-1]) + [overlap_width]

        elif self.overlap_mode == "vertical":
            # Offset along first (height/depth) axis; full overlap on all other axes
            # overlap_shape = [overlap_height, view_size[1], ..., view_size[-1]]
            overlap_height = int(
                self.target_overlap_volume / int(np.prod(self.view_size[1:]))
            )
            if overlap_height <= 0 or overlap_height > self.view_size[0]:
                raise ValueError(
                    f"overlap_mode='vertical' with overlap={self.overlap} is incompatible. "
                    f"Computed overlap_height={overlap_height}, but view_size[0]={self.view_size[0]}. "
                    f"Choose an overlap that divides evenly by {int(np.prod(self.view_size[1:]))}"
                )
            self.overlap_shape = [overlap_height] + list(self.view_size[1:])

    def _compute_feasible_offsets(self) -> Sequence[Tuple[int, ...]]:
        """Precompute absolute crop offsets that realize the target overlap shape.

        For a fixed overlap_mode and overlap specification, the offset between views
        must be exactly [view_size[i] - overlap_shape[i], ...] for all valid placements.
        This ensures the overlap region always has the same shape.
        """
        if self.target_overlap_volume == 0:
            return []

        # For the fixed overlap_shape, compute the exact offset required
        # offset[i] = view_size[i] - overlap_shape[i]
        required_offset = tuple(
            view_sz - overlap_len
            for view_sz, overlap_len in zip(self.view_size, self.overlap_shape)
        )

        # Check if this offset is feasible given the input_image_size
        # For view1 at position p1 and view2 at position p2 = p1 + offset:
        # - Both must fit within [0, max_start[i]]
        # - So: max(p1) = max_start[i], min(p2) = 0
        # - If p2 = p1 + offset, then min(p1) = -offset[i] (invalid)
        # - So we need: p1 + offset[i] <= max_start[i]
        # - And: p1 >= 0
        # - Valid range for p1[i]: [max(0, -offset[i]), min(max_start[i], max_start[i] - offset[i])]

        valid = True
        for i, (offset_i, max_start_i) in enumerate(
            zip(required_offset, self._max_start)
        ):
            # p1[i] can range such that p2[i] = p1[i] + offset[i] is also valid
            # Constraints: 0 <= p1[i] <= max_start[i] and 0 <= p1[i] + offset[i] <= max_start[i]
            p1_min = max(0, -offset_i)
            p1_max = min(max_start_i, max_start_i - offset_i)

            if p1_min > p1_max:
                valid = False
                break

        if not valid:
            return []

        # Return the single required offset (randomization of positions happens in randomize())
        return [required_offset]

    def _sample_zero_overlap_offsets(self) -> Tuple[int, ...]:
        """Sample absolute crop offsets that guarantee exactly zero overlap volume."""
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

    def _extract_view(
        self, img: NdarrayOrTensor, start_pos: Sequence[int]
    ) -> NdarrayOrTensor:
        """Extract a view from the image given a start position."""
        slices = [slice(None)] + [
            slice(start_pos[d], start_pos[d] + self.view_size[d])
            for d in range(self.spatial_dims)
        ]
        return img[tuple(slices)]

    def _create_mask(
        self, view_start: Sequence[int], other_view_start: Sequence[int]
    ) -> np.ndarray:
        """Create overlap coordinates [start, end] per axis in the local view frame.

        The output shape is [spatial_dims, 2], matching the coordinate format expected by
        DenseSimSiam.select_overlap_pixels for each sample.
        """
        coords = np.zeros((self.spatial_dims, 2), dtype=np.int64)
        for axis, view_sz in enumerate(self.view_size):
            start_a = int(view_start[axis])
            end_a = start_a + int(view_sz) - 1
            start_b = int(other_view_start[axis])
            end_b = start_b + int(view_sz) - 1

            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)

            # Inclusive coordinates in this view's local reference frame.
            coords[axis, 0] = overlap_start - start_a
            coords[axis, 1] = overlap_end - start_a
        return coords

    def __call__(self, data: dict) -> dict:
        """
        Apply the transform to extract two views.

        Args:
            data: dictionary containing the image data

        Returns:
            dictionary with extracted views and masks
        """
        self.randomize()

        # Ensure we have valid start positions
        if self.view1_start is None or self.view2_start is None:
            return data

        # For now, assume single image in data dict (we'll let the dictionary wrapper handle multiple keys)
        # Extract views
        img = data
        img_tensor = convert_to_tensor(img, track_meta=get_track_meta())

        # Extract the two views
        view1 = self._extract_view(img_tensor, self.view1_start)
        view2 = self._extract_view(img_tensor, self.view2_start)

        # Create overlap coordinate masks in each view's local frame.
        mask1 = self._create_mask(self.view1_start, self.view2_start)
        mask2 = self._create_mask(self.view2_start, self.view1_start)

        # Convert to appropriate type
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
