import unittest
import torch
import math
import os
import sys

# 1. Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 2. Import
from waymo_dataset_mtr_p import WaymoDataset

class TestAgentFrames(unittest.TestCase):

    def setUp(self):
        # 2 Objects, 5 Timestamps, 7 Attributes
        self.num_objects = 2
        self.num_timestamps = 5
        self.num_attrs = 7  # [x, y, z, dx, dy, heading, vel_x]

        self.heading_idx = 5
        self.func = WaymoDataset.transform_trajs_to_agent_frames

    def test_shapes(self):
        """
        Test 1: Verify output shapes for all 3 return values.
        """
        obj_trajs = torch.zeros((self.num_objects, self.num_timestamps, self.num_attrs))

        # Expect 3 return values now
        ret_trajs, ret_global_pos, ret_headings = self.func(obj_trajs, heading_index=self.heading_idx)

        # 1. Trajectories: (1, N, T, A)
        expected_traj_shape = (1, self.num_objects, self.num_timestamps, self.num_attrs)
        self.assertEqual(ret_trajs.shape, expected_traj_shape)

        # 2. Global Positions: (1, N, 1, 3) -> As defined by [:, :, -1, None, 0:3]
        expected_pos_shape = (1, self.num_objects, 3)
        self.assertEqual(ret_global_pos.shape, expected_pos_shape)

        # 3. Headings: (1, N)
        expected_heading_shape = (1, self.num_objects)
        self.assertEqual(ret_headings.shape, expected_heading_shape)

    def test_returned_globals_are_correct(self):
        """
        Test 2: Verify the returned global position/heading match the ORIGINAL
        world-frame data at the last timestamp (before the function zeroed them out).
        """
        # Create data with specific end points
        obj_trajs = torch.zeros((1, 2, 7))
        # End point at (10, 10), Heading 90 deg
        obj_trajs[0, -1, 0:3] = torch.tensor([10.0, 10.0, 5.0])
        obj_trajs[0, -1, self.heading_idx] = math.pi / 2

        # Run function
        _, ret_global_pos, ret_headings = self.func(obj_trajs, heading_index=self.heading_idx)

        # Check Positions (should be original 10, 10, 5)
        expected_pos = torch.tensor([10.0, 10.0, 5.0])
        torch.testing.assert_close(ret_global_pos[0, 0], expected_pos)

        # Check Heading (should be original pi/2)
        expected_heading = torch.tensor(math.pi / 2)
        torch.testing.assert_close(ret_headings[0, 0], expected_heading)

    def test_end_point_is_zero(self):
        """
        Test 3: Logic Check. The transformed trajectory should end at (0,0) with 0 heading.
        """
        obj_trajs = torch.randn((self.num_objects, self.num_timestamps, self.num_attrs))

        ret_trajs, _, _ = self.func(obj_trajs, heading_index=self.heading_idx)

        # Check that the last timestamp for every object is 0
        final_pos = ret_trajs[0, :, -1, 0:3]     # x, y, z
        final_heading = ret_trajs[0, :, -1, self.heading_idx]

        torch.testing.assert_close(final_pos, torch.zeros_like(final_pos), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final_heading, torch.zeros_like(final_heading), atol=1e-5, rtol=1e-5)

    def test_translation_logic(self):
        """
        Test 4: Pure Translation (Moving East).
        T=0: (0,0) -> T=1: (10,0).
        Relative to T=1, T=0 is at (-10, 0).
        """
        obj_trajs = torch.zeros((1, 2, 7))
        obj_trajs[0, 0, :2] = torch.tensor([0.0, 0.0])
        obj_trajs[0, 1, :2] = torch.tensor([10.0, 0.0])

        ret_trajs, _, _ = self.func(obj_trajs, heading_index=self.heading_idx)

        expected_start = torch.tensor([-10.0, 0.0])
        torch.testing.assert_close(ret_trajs[0, 0, 0, :2], expected_start, atol=1e-4, rtol=1e-4)

    def test_rotation_logic(self):
        """
        Test 5: Pure Rotation (Moving North).
        T=0: (0,0) -> T=1: (0,10), Heading North.
        Relative to North-facing frame at (0,10), (0,0) is 10m *Behind*.
        Expected T=0: (-10, 0).
        """
        obj_trajs = torch.zeros((1, 2, 7))
        obj_trajs[0, 0, :2] = torch.tensor([0.0, 0.0])
        obj_trajs[0, 1, :2] = torch.tensor([0.0, 10.0])
        obj_trajs[0, 1, self.heading_idx] = math.pi / 2

        ret_trajs, _, _ = self.func(obj_trajs, heading_index=self.heading_idx)

        expected_start = torch.tensor([-10.0, 0.0])
        torch.testing.assert_close(ret_trajs[0, 0, 0, :2], expected_start, atol=1e-4, rtol=1e-4)

    def test_velocity_rotation(self):
        """
        Test 6: Velocity Vector Rotation.
        Moving North (Vy=5) with Heading North.
        Relative to vehicle, this is Forward velocity (Vx=5).
        """
        obj_trajs = torch.zeros((1, 1, 8))
        obj_trajs[0, 0, :2] = torch.tensor([0.0, 10.0])
        obj_trajs[0, 0, self.heading_idx] = math.pi / 2
        obj_trajs[0, 0, 6:8] = torch.tensor([0.0, 5.0])

        ret_trajs, _, _ = self.func(obj_trajs, heading_index=self.heading_idx, rot_vel_index=[6, 7])

        expected_vel = torch.tensor([5.0, 0.0])
        torch.testing.assert_close(ret_trajs[0, 0, 0, 6:8], expected_vel, atol=1e-4, rtol=1e-4)

if __name__ == '__main__':
    unittest.main()