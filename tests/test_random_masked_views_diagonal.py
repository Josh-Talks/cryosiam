import unittest

from cryosiam.transforms.array import RandomMaskedViews


class TestRandomMaskedViewsDiagonalDirections(unittest.TestCase):
    def setUp(self) -> None:
        self.input_image_size = [64, 64]
        self.view_size = [32, 32]
        # In 2D diagonal mode this yields overlap_shape [16, 16].
        self.overlap = 0.25
        self.transform = RandomMaskedViews(
            input_image_size=self.input_image_size,
            view_size=self.view_size,
            overlap=self.overlap,
            overlap_mode="diagonal",
        ).set_random_state(seed=7)

    def test_diagonal_mode_samples_all_2d_corners(self) -> None:
        """Diagonal mode should allow all sign combinations across axes.

        In 2D this corresponds to top-left, top-right, bottom-left, bottom-right
        relative placement of view2 with respect to view1.
        """
        observed_signs = set()
        expected_signs = {(-1, -1), (-1, 1), (1, -1), (1, 1)}

        for _ in range(600):
            self.transform.randomize()
            dy = self.transform.view2_start[0] - self.transform.view1_start[0]
            dx = self.transform.view2_start[1] - self.transform.view1_start[1]
            sy = -1 if dy < 0 else (1 if dy > 0 else 0)
            sx = -1 if dx < 0 else (1 if dx > 0 else 0)
            observed_signs.add((sy, sx))

        self.assertEqual(observed_signs, expected_signs)

    def test_overlap_shape_is_constant(self) -> None:
        """All randomized pairs should have the same overlap spatial shape."""
        expected_shape = tuple(self.transform.overlap_shape)

        for _ in range(200):
            self.transform.randomize()
            overlap_shape = []
            for axis, view_sz in enumerate(self.view_size):
                s1 = self.transform.view1_start[axis]
                e1 = s1 + view_sz
                s2 = self.transform.view2_start[axis]
                e2 = s2 + view_sz
                overlap_shape.append(min(e1, e2) - max(s1, s2))

            self.assertEqual(tuple(overlap_shape), expected_shape)

    def test_views_stay_within_patch_bounds(self) -> None:
        """Both sampled views must stay inside the original patch."""
        max_start = [i - v for i, v in zip(self.input_image_size, self.view_size)]

        for _ in range(200):
            self.transform.randomize()
            for axis in range(len(self.view_size)):
                self.assertGreaterEqual(self.transform.view1_start[axis], 0)
                self.assertLessEqual(self.transform.view1_start[axis], max_start[axis])
                self.assertGreaterEqual(self.transform.view2_start[axis], 0)
                self.assertLessEqual(self.transform.view2_start[axis], max_start[axis])


class TestRandomMaskedViewsDiagonalDirections3D(unittest.TestCase):
    def setUp(self) -> None:
        self.input_image_size = [64, 64, 64]
        self.view_size = [32, 32, 32]
        # In 3D diagonal mode this yields overlap_shape [16, 16, 16].
        self.overlap = 0.125
        self.transform = RandomMaskedViews(
            input_image_size=self.input_image_size,
            view_size=self.view_size,
            overlap=self.overlap,
            overlap_mode="diagonal",
        ).set_random_state(seed=11)

    def test_diagonal_mode_samples_all_3d_octants(self) -> None:
        """Diagonal mode should allow all sign combinations across 3 axes."""
        observed_signs = set()
        expected_signs = {
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        }

        for _ in range(1200):
            self.transform.randomize()
            dz = self.transform.view2_start[0] - self.transform.view1_start[0]
            dy = self.transform.view2_start[1] - self.transform.view1_start[1]
            dx = self.transform.view2_start[2] - self.transform.view1_start[2]
            sz = -1 if dz < 0 else (1 if dz > 0 else 0)
            sy = -1 if dy < 0 else (1 if dy > 0 else 0)
            sx = -1 if dx < 0 else (1 if dx > 0 else 0)
            observed_signs.add((sz, sy, sx))

        self.assertEqual(observed_signs, expected_signs)

    def test_overlap_shape_is_constant_3d(self) -> None:
        """All randomized 3D pairs should have the same overlap shape."""
        expected_shape = tuple(self.transform.overlap_shape)

        for _ in range(200):
            self.transform.randomize()
            overlap_shape = []
            for axis, view_sz in enumerate(self.view_size):
                s1 = self.transform.view1_start[axis]
                e1 = s1 + view_sz
                s2 = self.transform.view2_start[axis]
                e2 = s2 + view_sz
                overlap_shape.append(min(e1, e2) - max(s1, s2))

            self.assertEqual(tuple(overlap_shape), expected_shape)

    def test_views_stay_within_patch_bounds_3d(self) -> None:
        """Both sampled 3D views must stay inside the original patch."""
        max_start = [i - v for i, v in zip(self.input_image_size, self.view_size)]

        for _ in range(200):
            self.transform.randomize()
            for axis in range(len(self.view_size)):
                self.assertGreaterEqual(self.transform.view1_start[axis], 0)
                self.assertLessEqual(self.transform.view1_start[axis], max_start[axis])
                self.assertGreaterEqual(self.transform.view2_start[axis], 0)
                self.assertLessEqual(self.transform.view2_start[axis], max_start[axis])


if __name__ == "__main__":
    unittest.main()
