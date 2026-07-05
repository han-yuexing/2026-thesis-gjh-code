import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

__all__ = ["FPN", "CrossAttention", "LKA_Attention"]


class FPN(nn.Module):
    def __init__(self, in_channels_list, fpn_out_channels=512):
        super(FPN, self).__init__()

        self.fpn_out_channels = fpn_out_channels

        # Initialize lateral convolutions dynamically based on in_channels_list
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, self.fpn_out_channels, 1)
                for in_channels in in_channels_list
            ]
        )

        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1)
                for _ in range(len(in_channels_list))
            ]
        )

        self.reduce_convs = nn.ModuleList(
            [
                nn.Conv2d(self.fpn_out_channels, in_channels, 1)
                for in_channels in in_channels_list
            ]
        )

    def forward(self, features):
        # 1. Compute lateral features (channel adjustment)
        lateral_features = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]

        # 2. Top-down pathway with feature fusion
        fpn_features = []
        prev_feature = lateral_features[-1]  # Start from the smallest feature (highest level)
        fpn_features.append(prev_feature)    # Add top-most feature directly

        for feature in reversed(lateral_features[:-1]):  # Reverse, exclude the top-most
            # Ensure proper upsampling with bilinear interpolation
            prev_feature = F.interpolate(
                prev_feature, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
            # Add current lateral feature
            prev_feature = prev_feature + feature
            fpn_features.append(prev_feature)

        # 3. Reverse the FPN features to match input order
        fpn_features = list(reversed(fpn_features))

        # 4. Apply 3x3 convolution to refine FPN features
        fpn_features = [
            self.output_convs[i](feature) for i, feature in enumerate(fpn_features)
        ]

        # 5. Optional: Reduce output features back to original input channels
        reduced_fpn_features = [
            reduce_conv(feature)
            for reduce_conv, feature in zip(self.reduce_convs, fpn_features)
        ]

        return reduced_fpn_features


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = dim**-0.5

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv):
        B, C, H, W = x_q.shape

        # Flatten the spatial dimensions
        q = self.query_proj(x_q.flatten(2).transpose(1, 2))  # B, HW, C
        k = self.key_proj(x_kv.flatten(2).transpose(1, 2))  # B, HW, C
        v = self.value_proj(x_kv.flatten(2).transpose(1, 2))  # B, HW, C

        # Reshape for multi-head attention
        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # B, num_heads, HW, C//num_heads
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to the values
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        # Project back to the original dimensions
        x = self.out_proj(x).transpose(1, 2).reshape(B, C, H, W)

        return x


# Define the LKA module
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


# Define the LKA_Attention module
class LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

        self.conv3 = nn.Conv2d(d_model, d_model * 2, 3, padding=1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut  # Residual connection

        x = self.conv3(x)
        return x


if __name__ == "__main__":
    # Set input shape
    input_tensor_1 = torch.randn(
        1, 64, 32, 32
    )  # Input for LR_ASPP and LKA_Attention (Batch, Channels, Height, Width)
    input_tensor_2 = torch.randn(
        1, 128, 16, 16
    )  # Input for CrossAttention (Batch, Channels, Height, Width)
    
    # Test CrossAttention module
    print("\nTesting CrossAttention Module...")
    cross_attention = CrossAttention(dim=128)
    cross_attention_output = cross_attention(input_tensor_2, input_tensor_2)
    print(
        f"CrossAttention Output Shape: {cross_attention_output.shape}"
    )  # Should be [1, 128, 16, 16]

    # Test LKA_Attention module
    print("\nTesting LKA_Attention Module...")
    lka_attention = LKA_Attention(d_model=64)
    lka_attention_output = lka_attention(input_tensor_1)
    print(
        f"LKA_Attention Output Shape: {lka_attention_output.shape}"
    )  # Should be [1, 64, 32, 32]

    print("\nAll tests completed successfully!")