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
"""Dense reward from the track-conditioned student (SVM-cotrain-student, mv_distill).

Per policy decision, for every env: the action chunk about to be executed -> track of the head-camera patches
(action-conditioned dynamics model, subprocess) -> student progress r = V(s, track) - V(s, 0). The reward is
``coef * clip(r, -clip, clip)`` and is added on top of the sparse success reward by ``RoboTwinEnv.chunk_step``.
Replaces ``SvmDenseReward`` (action SVM); the observation keys the env must expose are the same.
"""

import pickle
import struct
import subprocess
import sys

import numpy as np
import torch

from rlinf.envs.sim.robotwin.svm_dense_reward import CAMERAS, OBS_KEYS, load_local_dino

_BACKGROUND = {"table", "wall", "ground"}


def _gripper_joint_names(robot):
    names = set()
    for entry in (robot.left_gripper_name, robot.right_gripper_name):
        names.add(entry["base"])
        for mimic in entry.get("mimic", []):
            names.add(mimic[0])
    return names


def object_patch_mask(task_env, patch_px: np.ndarray, depth_ok: np.ndarray) -> np.ndarray:
    """[391] bool: patches with valid depth that lie on a trackable body (task objects and gripper fingers).

    Same definition as the ``valid`` mask the track models were trained with (arm links, table, wall, ground excluded).
    """
    cam = task_env.cameras.static_camera_list[task_env.cameras.static_camera_name.index("head_camera")]
    seg = cam.get_picture("Segmentation")[..., 1]
    h, w = seg.shape
    px = np.clip(np.round(patch_px[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(patch_px[:, 1]).astype(int), 0, h - 1)
    at_patch = seg[py, px]

    robot = task_env.robot
    arms = {robot.left_entity, robot.right_entity}
    gripper_names = _gripper_joint_names(robot)
    owners = set()
    for art in arms:
        for link in art.get_links():
            if link.joint is not None and link.joint.name in gripper_names:
                owners.add(link.entity.per_scene_id)
    for actor in task_env.scene.get_all_actors():
        if actor.name not in _BACKGROUND:
            owners.add(actor.per_scene_id)
    for art in task_env.scene.get_all_articulations():
        if art not in arms:
            owners.update(link.entity.per_scene_id for link in art.get_links())
    return depth_ok & np.isin(at_patch, np.fromiter(owners, dtype=at_patch.dtype, count=len(owners)))


class TrackClient:
    """Spawns dynamics_track_worker.py once and sends it batches."""

    def __init__(self, python: str, dynamics_ckpt: str, device: str):
        script = __file__.replace("student_dense_reward.py", "dynamics_track_worker.py")
        self.proc = subprocess.Popen(
            [python, script, "--dynamics_ckpt", dynamics_ckpt, "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )

    def predict(self, dino_feat, agent_pos, patch_centers, actions) -> np.ndarray:
        buf = pickle.dumps(
            {"dino_feat": dino_feat, "agent_pos": agent_pos, "patch_centers": patch_centers, "actions": actions},
            protocol=4,
        )
        self.proc.stdin.write(struct.pack(">Q", len(buf)) + buf)
        self.proc.stdin.flush()
        header = self.proc.stdout.read(8)
        if len(header) < 8:
            raise RuntimeError("dynamics_track_worker died (see its stderr in the job log)")
        n, out = struct.unpack(">Q", header)[0], b""
        while len(out) < n:
            out += self.proc.stdout.read(n - len(out))
        resp = pickle.loads(out)
        if "error" in resp:
            raise RuntimeError("dynamics_track_worker failed:\n" + resp["error"])
        return resp["tracks"]

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class StudentDenseReward:
    """Args: cfg: ``env.<train|eval>.student_reward`` node (see env/robotwin_scan_object.yaml); device: torch device."""

    def __init__(self, cfg, device):
        for p in (str(cfg.student_repo), str(cfg.svm_repo)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from mv_distill.reward import MVStudentReward  # noqa: PLC0415
        from svm import preprocess  # noqa: PLC0415

        self.device = torch.device(device)
        self.pp = preprocess
        dino = load_local_dino(str(cfg.dino_repo_dir), str(cfg.dino_weights), self.device)
        self.student = MVStudentReward(str(cfg.student_ckpt), device=str(self.device), dino=dino)
        self.tracker = TrackClient(str(cfg.worker_python), str(cfg.dynamics_ckpt), str(self.device))
        self.clip = float(cfg.get("clip", 5.0))
        self.batch_size = int(cfg.get("batch_size", 4))

    @staticmethod
    def _np(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    def _obs(self, extracted_obs: dict) -> dict:
        missing = [k for k in OBS_KEYS if k not in extracted_obs]
        if missing:
            raise KeyError(f"student_reward needs {list(OBS_KEYS)} in the env obs; missing {missing}")
        wrist = self._np(extracted_obs["wrist_images"])
        rgb = {"head": self._np(extracted_obs["main_images"]), "left_wrist": wrist[:, 0], "right_wrist": wrist[:, 1]}
        obs = {}
        for cam in CAMERAS:
            obs[f"{cam}_rgb"] = np.ascontiguousarray(rgb[cam])
            obs[f"{cam}_depth"] = self._np(extracted_obs[f"{cam}_depth"])
            obs[f"{cam}_K"] = self._np(extracted_obs[f"{cam}_intrinsic"])
            obs[f"{cam}_E"] = self._np(extracted_obs[f"{cam}_extrinsic"])
        obs["state"] = self._np(extracted_obs["ee_state"])
        return obs

    @torch.no_grad()
    def __call__(self, extracted_obs: dict, chunk_actions: np.ndarray, task_envs) -> torch.Tensor:
        """[num_envs] progress rewards for one chunk step (before it is executed); ``task_envs`` are the live task envs."""
        obs = self._obs(extracted_obs)
        actions = np.asarray(chunk_actions, dtype=np.float32)
        B = len(obs["state"])
        head_dino = self.pp.dino_features(self.student.dino, obs["head_rgb"], self.device).cpu().numpy()
        # The dynamics model was trained on DINO features of channel-swapped images (the GAP pipelines' spurious
        # BGR2RGB, see GAP-tte deploy_policy.swap_rgb_channels); the student / SVM read true-RGB features.
        head_dino_dyn = self.pp.dino_features(
            self.student.dino, np.ascontiguousarray(obs["head_rgb"][..., ::-1]), self.device
        ).cpu().numpy()
        centres = np.zeros((B, 391, 3), np.float32)
        valid = np.zeros((B, 391), bool)
        for b in range(B):
            centres[b], ok = self.pp.patch_centres(obs["head_depth"][b], obs["head_K"][b], obs["head_E"][b])
            valid[b] = object_patch_mask(task_envs[b], self.pp.PATCH_PX, ok)
        obs["track"] = self.tracker.predict(head_dino_dyn, obs["state"], centres, actions)[:, :, :16].astype(np.float32)
        obs["track_valid"] = valid
        v, v0 = self.student.values(obs, head_dino=head_dino, chunk=self.batch_size)
        return torch.as_tensor(np.clip(v - v0, -self.clip, self.clip), dtype=torch.float32)
