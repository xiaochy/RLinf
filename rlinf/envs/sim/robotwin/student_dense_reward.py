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

Per policy decision, for every env: the chunk's track of the head-camera patches -> student value.
The reward is ``coef * clip(r, -clip, clip)`` and is added on top of the sparse success reward by
``RoboTwinEnv.chunk_step``.

``reward_mode`` picks what ``r`` is:

- ``raw`` (default): ``r = V(s, track)``. The student predicts its task teacher's logit 16 frames
  ahead, so this IS a value of the state the chunk leads to -- a dense value-based reward, and the
  thing the student was trained to output. One forward pass.
- ``progress``: ``r = V(s, track) - V(s, 0)``, the original form. It was meant to isolate the track's
  contribution by subtracting a track-free baseline, but a zero track never occurs in training (only
  ``valid`` is kept, as in training), so ``V(s, 0)`` is an out-of-distribution extrapolation and the
  subtraction injects whatever the model does off-distribution. Kept so every number recorded before
  2026-11-13 reproduces. Costs a second forward pass.

With ``raw`` the reward carries a large state-dependent offset, which is deliberate: PPO's critic
absorbs the part that depends on s_t and the advantage keeps the part that depends on the chunk.
Subtracting a hand-built baseline is the alternative, and ``progress`` is that alternative measured. Replaces ``SvmDenseReward`` (action SVM); the observation keys
the env must expose are the same.

``track_source`` picks where the track comes from:

- ``dynamics``: predicted from the chunk by the action-conditioned dynamics model in a subprocess. Known
  before the chunk runs, which is what the student was designed for, but predicted tracks are a
  distribution shift away from the simulator ground truth it was trained on.
- ``env``: the simulator's own ground truth, read back after the chunk has run (see ``gt_track``). In
  distribution, and free of the dynamics model's error, at the cost of scoring the chunk after the fact.

Either way the score is a function of the decision-time observation and the chunk, so the reward PPO sees
is still r(s_t, a_t); only ``env`` has to wait for the chunk to finish before it can be computed.
"""

import pickle
import struct
import subprocess
import sys

import numpy as np
import torch

from rlinf.utils.logging import get_logger

from rlinf.envs.sim.robotwin.gt_track import PoseRecorder
from rlinf.envs.sim.robotwin.svm_dense_reward import CAMERAS, OBS_KEYS, load_local_dino

_BACKGROUND = {"table", "wall", "ground"}
_logger = get_logger()


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
        self.track_source = str(cfg.get("track_source", "dynamics"))
        if self.track_source not in ("dynamics", "env"):
            raise ValueError(f"student_reward.track_source must be 'dynamics' or 'env', got {self.track_source!r}")
        self.tracker = (
            TrackClient(str(cfg.worker_python), str(cfg.dynamics_ckpt), str(self.device))
            if self.track_source == "dynamics"
            else None
        )
        self.reward_mode = str(cfg.get("reward_mode", "raw"))
        if self.reward_mode not in ("raw", "progress"):
            raise ValueError(f"student_reward.reward_mode must be 'raw' or 'progress', got {self.reward_mode!r}")
        self.horizon = int(cfg.get("horizon", 16))
        self.clip = float(cfg.get("clip", 5.0))
        self.batch_size = int(cfg.get("batch_size", 4))
        self._recorders = {}
        self._chunks_scored = 0
        self._chunks_flat = 0

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

    def _recorder(self, task_env) -> PoseRecorder:
        rec = self._recorders.get(id(task_env))
        if rec is None:
            rec = PoseRecorder(task_env)
            self._recorders[id(task_env)] = rec
        return rec

    def on_reset(self, task_envs) -> None:
        """Re-read the trackable bodies of every env; a reset rebuilds the scene and invalidates them."""
        if self.track_source != "env":
            return
        for te in task_envs:
            self._recorder(te).refresh()

    @torch.no_grad()
    def begin_chunk(self, extracted_obs: dict, chunk_actions: np.ndarray, task_envs) -> dict:
        """Snapshot what the chunk will be scored against, before it is executed."""
        obs = self._obs(extracted_obs)
        B = len(obs["state"])
        centres = np.zeros((B, 391, 3), np.float32)
        valid = np.zeros((B, 391), bool)
        rows = np.zeros((B, 391), np.int64)
        for b in range(B):
            centres[b], ok = self.pp.patch_centres(obs["head_depth"][b], obs["head_K"][b], obs["head_E"][b])
            if self.track_source == "env":
                rec = self._recorder(task_envs[b])
                rows[b], valid[b] = rec.owner_rows(self.pp.PATCH_PX, ok)
                rec.start_chunk()
            else:
                valid[b] = object_patch_mask(task_envs[b], self.pp.PATCH_PX, ok)
        return {
            "obs": obs,
            "actions": np.asarray(chunk_actions, dtype=np.float32),
            "centres": centres,
            "valid": valid,
            "owner_rows": rows,
        }

    @torch.no_grad()
    def finish_chunk(self, state: dict, task_envs) -> torch.Tensor:
        """[num_envs] progress rewards for the chunk ``begin_chunk`` snapshotted."""
        obs, H = state["obs"], self.horizon
        B = len(obs["state"])
        head_dino = self.pp.dino_features(self.student.dino, obs["head_rgb"], self.device).cpu().numpy()

        if self.track_source == "env":
            track = np.zeros((B, 391, H, 3), np.float32)
            valid = np.zeros((B, 391), bool)
            for b in range(B):
                track[b], valid[b] = self._recorder(task_envs[b]).gt_track(
                    state["centres"][b], state["owner_rows"][b], state["valid"][b], H
                )
        else:
            track = self.tracker.predict(
                head_dino, obs["state"], state["centres"], state["actions"]
            )[:, :, :H].astype(np.float32)
            valid = state["valid"]

        obs["track"] = track
        obs["track_valid"] = valid
        _static = self.reward_mode == "progress"
        v, v0 = self.student.values(obs, head_dino=head_dino, chunk=self.batch_size, static=_static)
        raw = v - v0 if _static else v
        r = np.clip(raw, -self.clip, self.clip)

        # An all-invalid mask or an all-zero track silently turns the dense reward into a constant, so
        # report both for the first few chunks and then periodically.
        self._chunks_scored += 1

        # So does the clip: in ``raw`` mode V carries a large state-dependent offset, so a clip picked for
        # the difference form can push every candidate of a chunk onto the same bound. The reward is then
        # a constant across the batch and says nothing about which action was better -- the chunk is
        # silently wasted. This happened to 17 of 48 logged chunks of the 20261113-115812 n100 run with
        # clip 5.0, and the periodic line above only showed it once every 100 chunks, so count it.
        if B > 1:
            self._chunks_flat += bool(float(r.max() - r.min()) < 1e-6)
            if self._chunks_flat and self._chunks_scored % 100 == 0:
                _logger.warning(
                    "student_reward[%s/%s]: %d of %d chunks (%.0f%%) had every candidate on the same "
                    "clip bound (|clip| = %.1f) and carried no signal; raw spread this chunk %.3f..%.3f",
                    self.track_source, self.reward_mode, self._chunks_flat, self._chunks_scored,
                    100.0 * self._chunks_flat / self._chunks_scored, self.clip,
                    float(raw.min()), float(raw.max()),
                )
        if self._chunks_scored <= 5 or self._chunks_scored % 100 == 0:
            moved = np.abs(track[..., -1, :]).sum(-1) > 1e-4
            _logger.info(
                "student_reward[%s/%s] chunk %d: valid %.1f%% of patches, moved %.1f%%, "
                "|track| max %.4f m, V %.3f..%.3f, V0 %s, r %.3f..%.3f",
                self.track_source, self.reward_mode, self._chunks_scored,
                100 * valid.mean(), 100 * moved.mean(),
                float(np.abs(track).max()), float(v.min()), float(v.max()),
                "n/a" if v0 is None else f"{float(v0.min()):.3f}..{float(v0.max()):.3f}",
                float(r.min()), float(r.max()),
            )
        return torch.as_tensor(r, dtype=torch.float32)
