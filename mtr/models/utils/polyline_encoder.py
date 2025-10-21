# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import torch
import torch.nn as nn
from ..utils import common_layers


class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )

        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (B, N_poly, N_pts, C_in): Input polylines.
            polylines_mask (B, N_poly, N_pts): Mask for valid points.

        Returns:
            feature_buffers (B, N_poly, D_out): A feature vector for each polyline.
        """
        # polylines.shape: (B, N_poly, N_pts, C_in)
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape

        # Shape: (N_valid_pts, C_in)
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])

        # The MLP output has shape (N_valid_pts, D_hidden).
        # Shape: (B, N_poly, N_pts, D_hidden)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])

        polylines_feature[polylines_mask] = polylines_feature_valid
        # polylines_feature shape is still (B, N_poly, N_pts, D_hidden)

        # Input shape: (B, N_poly, N_pts, D_hidden)
        # Output shape (pooled_feature): (B, N_poly, D_hidden)
        pooled_feature = polylines_feature.max(dim=2)[0]

        # Input 1 (local): (B, N_poly, N_pts, D_hidden)
        # Input 2 (global): (B, N_poly, N_pts, D_hidden)
        # Output (polylines_feature): (B, N_poly, N_pts, D_hidden * 2)
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # Shape: (N_valid_pts, D_hidden * 2)
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        # The MLP output has shape (N_valid_pts, D_hidden).

        # Shape: (B, N_poly, N_pts, D_hidden)
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])

        feature_buffers[polylines_mask] = polylines_feature_valid
        # feature_buffers shape is still (B, N_poly, N_pts, D_hidden)

        # Input shape: (B, N_poly, N_pts, D_hidden)
        # Output shape (feature_buffers): (B, N_poly, D_hidden)
        feature_buffers = feature_buffers.max(dim=2)[0]

        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers
