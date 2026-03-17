import torch
import numpy as np
from scipy import ndimage
from typing import Any, Optional, Tuple, Sequence, Union
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

    This transform randomly selects two views from a 2D or 3D patch, ensuring they overlap
    by a specified amount. Each view is cropped from the original patch, and masks are generated
    to indicate valid regions in each view.

    Args:
        input_image_size: size of the input patch (e.g., [64, 64] for 2D or [64, 64, 64] for 3D)
        view_size: size of each view to extract (e.g., [32, 32] for 2D or [32, 32, 32] for 3D)
        overlap: overlap fraction between the two views (0.0 to 1.0).
                 For example, 0.5 means 50% overlap
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        input_image_size: Union[Sequence[int], int],
        view_size: Union[Sequence[int], int],
        overlap: Union[Sequence[float], float] = 0.5,
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

        # Ensure overlap is a sequence matching spatial dimensions
        if isinstance(overlap, (list, tuple)):
            self.overlap = list(overlap)
        else:
            self.overlap = [overlap] * self.spatial_dims

        # Validate dimensions
        if len(self.view_size) != self.spatial_dims:
            raise ValueError(
                f"view_size must have {self.spatial_dims} dimensions to match input_image_size, "
                f"got {len(self.view_size)}"
            )
        if len(self.overlap) != self.spatial_dims:
            raise ValueError(
                f"overlap must have {self.spatial_dims} dimensions to match input_image_size, "
                f"got {len(self.overlap)}"
            )

        # Validate that views fit within the image
        for i in range(self.spatial_dims):
            if self.view_size[i] > self.input_image_size[i]:
                raise ValueError(
                    f"view_size[{i}] ({self.view_size[i]}) cannot be larger than "
                    f"input_image_size[{i}] ({self.input_image_size[i]})"
                )
            if not (0.0 <= self.overlap[i] <= 1.0):
                raise ValueError(
                    f"overlap must be between 0.0 and 1.0, got {self.overlap[i]}"
                )

        self.view1_start = None
        self.view2_start = None
        self.R = np.random.RandomState()

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomMaskedViews":
        if state is not None:
            self.R = state
        elif seed is not None:
            self.R = np.random.RandomState(seed)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        """Generate random positions for view1 and view2."""
        self.view1_start = []
        self.view2_start = []

        for d in range(self.spatial_dims):
            input_size = self.input_image_size[d]
            view_sz = self.view_size[d]
            ovlp = self.overlap[d]

            # Maximum possible start position for a view to fit within the image
            max_start = input_size - view_sz

            # Randomly select start position for first view
            start1 = self.R.randint(0, max_start + 1)
            self.view1_start.append(start1)

            # Randomly select start position for second view
            # It should be positioned such that it overlaps with view1
            # and is within valid bounds

            # Create a valid range for the second view's start position
            # to ensure overlap
            possible_start2_min = max(
                0, start1 - view_sz + int(np.ceil(ovlp * view_sz))
            )
            possible_start2_max = min(
                max_start, start1 + view_sz - int(np.ceil(ovlp * view_sz))
            )

            # Ensure the range is valid
            if possible_start2_min > possible_start2_max:
                # If overlap constraint is too strict, just pick a random valid position
                start2 = self.R.randint(0, max_start + 1)
            else:
                start2 = self.R.randint(possible_start2_min, possible_start2_max + 1)

            self.view2_start.append(start2)

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
        self, img_shape: Sequence[int], view_start: Sequence[int]
    ) -> np.ndarray:
        """Create a binary mask for the view region (all ones since the entire view is valid)."""
        mask_shape = [1] + self.view_size
        mask = np.ones(mask_shape, dtype=np.uint8)
        return mask

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

        # Create masks (all ones for valid regions)
        mask1 = self._create_mask(img_tensor.shape, self.view1_start)
        mask2 = self._create_mask(img_tensor.shape, self.view2_start)

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
