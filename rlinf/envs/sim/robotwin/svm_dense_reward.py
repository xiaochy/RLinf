# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense process reward from a multi-view SVM discriminator f(s, a).

One reward per policy decision: the observation at the first frame of an action chunk plus the
chunk itself are scored, r = lam * clip(logit, -beta, beta) (see SVM-reference/README.md).
The reward is added on top of the sparse success reward by ``RoboTwinEnv.chunk_step``.
"""

import sys

import numpy as np
import torch

CAMERAS = ("head", "left_wrist", "right_wrist")
# Keys the RoboTwin env must expose (see update_obs in robotwin/envs/vector_env.py).
OBS_KEYS = tuple(
    f"{cam}_{suffix}"
    for cam in CAMERAS
    for suffix in ("depth", "intrinsic", "extrinsic")
) + ("ee_state",)


def load_local_dino(repo_dir: str, weights_path: str, device) -> torch.nn.Module:
    """DINOv2 ViT-L/14+reg from a local checkout (no network), as used by GAP and the SVM models."""
    model = torch.hub.load(
        repo_dir,
        "dinov2_vitl14_reg",
        source="local",
        pretrained=True,
        weights=weights_path,
    )
    return model.to(device).eval()


class SvmDenseReward:
    """Scores (observation, action chunk) pairs of a batch of envs with an SVM reward model.

    Args:
        cfg: ``env.<train|eval>.svm_reward`` node (repo_path, ckpt_dir, model, dino_repo_dir,
            dino_weights, lam, beta).
        device: torch device for the SVM and DINOv2 (kept outside the env's own tensors).
    """

    def __init__(self, cfg, device):
        repo_path = str(cfg.repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from svm import MVActionSVMReward

        kind = str(cfg.get("model", "action"))
        if kind != "action":
            raise ValueError(
                f"svm_reward.model must be 'action' (the track model needs tracks the "
                f"env does not produce), got {kind!r}"
            )
        self.device = torch.device(device)
        dino = load_local_dino(
            str(cfg.dino_repo_dir), str(cfg.dino_weights), self.device
        )
        self.scorer = MVActionSVMReward(
            str(cfg.ckpt_dir), device=str(self.device), dino=dino
        )
        self.lam = float(cfg.get("lam", 0.1))
        self.beta = float(cfg.get("beta", 20.0))
        self.batch_size = int(cfg.get("batch_size", 8))

    @staticmethod
    def _np(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    def build_obs(self, extracted_obs: dict, chunk_actions: np.ndarray) -> dict:
        """SVM input dict from RoboTwinEnv's extracted obs (decision-time) and the action chunk."""
        missing = [k for k in OBS_KEYS if k not in extracted_obs]
        if missing:
            raise KeyError(
                f"svm_reward needs {list(OBS_KEYS)} in the env obs (enable data_type depth/"
                f"endpose and collect_wrist_camera); missing {missing}"
            )
        wrist = self._np(extracted_obs["wrist_images"])  # [B, 2, H, W, 3] left, right
        rgb = {
            "head": self._np(extracted_obs["main_images"]),
            "left_wrist": wrist[:, 0],
            "right_wrist": wrist[:, 1],
        }
        obs = {}
        for cam in CAMERAS:
            img = rgb[cam]
            obs[f"{cam}_rgb"] = np.ascontiguousarray(img)
            obs[f"{cam}_depth"] = self._np(extracted_obs[f"{cam}_depth"])
            obs[f"{cam}_K"] = self._np(extracted_obs[f"{cam}_intrinsic"])
            obs[f"{cam}_E"] = self._np(extracted_obs[f"{cam}_extrinsic"])
        obs["state"] = self._np(extracted_obs["ee_state"])
        obs["action"] = np.asarray(chunk_actions, dtype=np.float32)
        return obs

    @torch.no_grad()
    def __call__(self, extracted_obs: dict, chunk_actions: np.ndarray) -> torch.Tensor:
        """[num_envs] dense rewards for one chunk step (before it is executed)."""
        obs = self.build_obs(extracted_obs, chunk_actions)
        logits = self.scorer.logits(obs, chunk=self.batch_size)
        reward = self.lam * np.clip(logits, -self.beta, self.beta)
        return torch.as_tensor(reward, dtype=torch.float32)
