# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observation preprocessing for the GAP policy inside RLinf.

Reproduces, for batched RLinf observations, what GAP's deploy_policy.py does per frame:
DINOv2 features on the (channel-swapped) head-camera RGB at the nearest multiple of 14, and
world-frame 3D centers of the 17x23 patch grid from the head-camera depth map.
"""

import os
import sys

import cv2
import numpy as np
import torch


def compute_patch_centers_world(
    depth_mm, intrinsic_cv, extrinsic_cv, patch_size=14
) -> np.ndarray:
    """World-frame 3D centers [N, 3] of the patch grid (invalid depth -> zeros).

    Same math as GAP's deploy_policy.compute_patch_centers_world / the training-data pipeline.
    depth_mm: [H, W] millimetres; intrinsic_cv: [3, 3]; extrinsic_cv: [3, 4] world-to-camera.
    """
    H, W = depth_mm.shape
    th = max(round(H / patch_size) * patch_size, patch_size)
    tw = max(round(W / patch_size) * patch_size, patch_size)
    ph, pw = th // patch_size, tw // patch_size

    cy = (np.arange(ph) + 0.5) * patch_size
    cx = (np.arange(pw) + 0.5) * patch_size
    gx, gy = np.meshgrid(cx, cy)
    centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    centers[:, 0] *= W / tw
    centers[:, 1] *= H / th

    n = centers.shape[0]
    map_x = np.clip(centers[:, 0], 0, W - 1).astype(np.float32).reshape(1, n)
    map_y = np.clip(centers[:, 1], 0, H - 1).astype(np.float32).reshape(1, n)
    depth_sampled = cv2.remap(
        depth_mm.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR
    ).reshape(n)
    depth_m = depth_sampled / 1000.0
    valid = (depth_m > 0) & np.isfinite(depth_m)

    fx, fy = intrinsic_cv[0, 0], intrinsic_cv[1, 1]
    cx0, cy0 = intrinsic_cv[0, 2], intrinsic_cv[1, 2]
    x_cam = (centers[:, 0] - cx0) * depth_m / fx
    y_cam = (centers[:, 1] - cy0) * depth_m / fy
    pts_cam = np.stack([x_cam, y_cam, depth_m], axis=-1)

    rot = extrinsic_cv[:, :3]
    trans = extrinsic_cv[:, 3]
    pts_world = (pts_cam - trans) @ rot
    pts_world[~valid] = 0.0
    return pts_world.astype(np.float32)


def batch_patch_centers(head_depth, head_intrinsic, head_extrinsic) -> torch.Tensor:
    """[B, H, W], [B, 3, 3], [B, 3, 4] (tensors or arrays) -> [B, N, 3] float32 tensor."""
    depth = np.asarray(head_depth.cpu() if torch.is_tensor(head_depth) else head_depth)
    k = np.asarray(head_intrinsic.cpu() if torch.is_tensor(head_intrinsic) else head_intrinsic)
    e = np.asarray(head_extrinsic.cpu() if torch.is_tensor(head_extrinsic) else head_extrinsic)
    return torch.from_numpy(
        np.stack(
            [compute_patch_centers_world(depth[i], k[i], e[i]) for i in range(depth.shape[0])]
        )
    )


class GapDinoExtractor:
    """DINOv2 ViT-L/14(+reg) patch features for batched uint8 RGB images.

    Matches GAPPolicyWrapper.extract_dino_features: resize to the nearest multiple of 14
    (240x320 -> 238x322), ImageNet normalisation, optional RGB<->BGR swap (the training features
    were extracted from channel-swapped images, see deploy_policy swap_rgb_channels).
    """

    def __init__(
        self,
        gap_root: str,
        weights_path: str | None = None,
        repo_dir: str | None = None,
        model_name: str = "dinov2_vitl14_reg",
        swap_rgb: bool = True,
        device="cuda",
    ):
        if gap_root not in sys.path:
            sys.path.insert(0, gap_root)
        from gap_policy.model.vision.dinov2_encoder import DINOV2

        repo_dir = repo_dir or os.path.join(gap_root, "thirdparty/dinov2")
        weights_path = weights_path or os.path.join(
            gap_root, "pretrained/dinov2_vitl14_reg4_pretrain.pth"
        )
        self.swap_rgb = swap_rgb
        self.device = torch.device(device)
        self.model = (
            DINOV2(
                model_name=model_name,
                repo_dir=repo_dir,
                weights_path=weights_path,
                freeze=True,
            )
            .to(self.device)
            .eval()
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: [B, H, W, 3] uint8 -> [B, 1, N, D] float features."""
        if self.swap_rgb:
            rgb = rgb.flip(-1)
        img = rgb.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        h, w = img.shape[-2:]
        th, tw = max(round(h / 14) * 14, 14), max(round(w / 14) * 14, 14)
        if (h, w) != (th, tw):
            img = torch.nn.functional.interpolate(
                img, size=(th, tw), mode="bilinear", align_corners=False
            )
        img = (img - self.mean) / self.std
        return self.model(img).unsqueeze(1)
