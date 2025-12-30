import unittest
import numpy as np

# Assuming the function is in a class named CoordinateTransform
# from your_module import CoordinateTransform
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from waymo_dataset_mtr_p import WaymoDataset

import unittest
import numpy as np
from waymo_dataset_mtr_p import WaymoDataset  # Adjust if your class name differs
import math
import torch

class TestTransformTrajs(unittest.TestCase):

    def setUp(self):
        # Standard dimensions
        self.num_objects = 2
        self.num_timestamps = 5
        self.num_attrs = 7  # [x, y, z, dx, dy, heading, vel_x]
        self.heading_idx = 5

        # Shortcut to the function
        self.func = WaymoDataset.transform_trajs_to_center_coords

    def test_shapes(self):
        """
        Test 1: Verify output dimensions match input dimensions (N, T, A).
        """
        obj_trajs = torch.zeros((self.num_objects, self.num_timestamps, self.num_attrs))
        sdc_xyz = torch.tensor([10.0, 10.0, 0.0])
        sdc_heading = torch.tensor([0.0])

        result = self.func(
            obj_trajs=obj_trajs,
            sdc_xyz=sdc_xyz,
            sdc_heading=sdc_heading,
            heading_index=self.heading_idx
        )

        self.assertEqual(result.shape, obj_trajs.shape,
                         f"Expected shape {obj_trajs.shape}, got {result.shape}")

    def test_translation_logic(self):
        """
        Test 2: Pure Translation.
        SDC at (10, 10), Heading 0.
        Object at (15, 15) should become (5, 5).
        """
        obj_trajs = torch.zeros((1, 1, 6))
        obj_trajs[0, 0, :2] = torch.tensor([15.0, 15.0])

        sdc_xyz = torch.tensor([10.0, 10.0, 0.0])
        sdc_heading = torch.tensor([0.0])

        result = self.func(obj_trajs, sdc_xyz, sdc_heading, heading_index=5)

        expected_pos = torch.tensor([5.0, 5.0])

        # Check X, Y match
        torch.testing.assert_close(result[0, 0, :2], expected_pos, rtol=1e-4, atol=1e-4)

    def test_rotation_logic(self):
        """
        Test 3: Pure Rotation (90 degrees).
        SDC at (0, 0), Heading pi/2 (North).
        Object at (1, 0) (East) should become (0, -1) (SDC's Right).
        Object Heading 0 (East) should become -pi/2 (Relative South).
        """
        obj_trajs = torch.zeros((1, 1, 6))
        obj_trajs[0, 0, :2] = torch.tensor([1.0, 0.0]) # World East
        obj_trajs[0, 0, 5] = 0.0                       # World East Heading

        sdc_xyz = torch.tensor([0.0, 0.0, 0.0])
        sdc_heading = torch.tensor([math.pi / 2])      # SDC Facing North

        result = self.func(obj_trajs, sdc_xyz, sdc_heading, heading_index=5)

        # New Position: (0, -1)
        expected_pos = torch.tensor([0.0, -1.0])
        # New Heading: 0 - pi/2 = -pi/2
        expected_heading = torch.tensor(-math.pi / 2)

        torch.testing.assert_close(result[0, 0, :2], expected_pos, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result[0, 0, 5], expected_heading, rtol=1e-4, atol=1e-4)

    def test_full_transform(self):
        """
        Test 4: Translation + Rotation.
        Object: (11, 10), Heading 0.
        SDC: (10, 10), Heading pi/2.

        1. Translate: (11,10) - (10,10) = (1, 0)
        2. Rotate -pi/2: (1, 0) -> (0, -1)
        """
        obj_trajs = torch.zeros((1, 1, 6))
        obj_trajs[0, 0, :2] = torch.tensor([11.0, 10.0])
        obj_trajs[0, 0, 5] = 0.0

        sdc_xyz = torch.tensor([10.0, 10.0, 0.0])
        sdc_heading = torch.tensor([math.pi / 2])

        result = self.func(obj_trajs, sdc_xyz, sdc_heading, heading_index=5)

        expected_pos = torch.tensor([0.0, -1.0])
        torch.testing.assert_close(result[0, 0, :2], expected_pos, rtol=1e-4, atol=1e-4)

    def test_batch_processing(self):
        """
        Test 5: Multiple objects/timestamps behave consistently.
        """
        # Create 2 objects with identical initial states
        obj_trajs = torch.zeros((2, 1, 6))
        obj_trajs[:, 0, :2] = torch.tensor([15.0, 15.0])

        sdc_xyz = torch.tensor([10.0, 10.0, 0.0])
        sdc_heading = torch.tensor([0.0])

        result = self.func(obj_trajs, sdc_xyz, sdc_heading, heading_index=5)

        # Both objects should be transformed identically
        expected_pos = torch.tensor([5.0, 5.0])
        torch.testing.assert_close(result[0, 0, :2], expected_pos)
        torch.testing.assert_close(result[1, 0, :2], expected_pos)

if __name__ == '__main__':
    unittest.main()