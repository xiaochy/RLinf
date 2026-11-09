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
"""Ground-truth head-patch tracks from RoboTwin simulator state.

Same construction as GAP's offline ``generate_gt_tracks_rgb.py``: a visible surface point belongs to one
rigid body, so its position h steps later is its chunk-start local coordinate pushed through that body's
pose at h. No tracker, no drift. The track of a chunk is only known once the chunk has run, so
``PoseRecorder`` logs every trackable body's pose per executed action and ``gt_track`` turns the log into
the ``[391, 16, 3]`` displacement field the student reads.
"""

import numpy as np

BACKGROUND_NAMES = {"table", "wall", "ground"}


def gripper_joint_names(robot) -> set:
    """Names of the joints that drive the gripper fingers.

    Link-name suffixes are not reliable across embodiments (the wrist camera mount is also a child of the
    last arm link), so finger links are matched through their driving joint instead.
    """
    names = set()
    for entry in (robot.left_gripper_name, robot.right_gripper_name):
        names.add(entry["base"])
        for mimic in entry.get("mimic", []):
            names.add(mimic[0])
    return names


def trackable_owners(task_env) -> dict:
    """``per_scene_id`` -> pose-bearing body, for every task object and gripper finger link.

    Arm links and the background are left out, matching the ``valid`` mask the track models were trained
    with: the arm's own pose is already in the robot state, and the background never moves.
    """
    arms = {task_env.robot.left_entity, task_env.robot.right_entity}
    finger_joints = gripper_joint_names(task_env.robot)

    owners = {}
    for art in arms:
        for link in art.get_links():
            if link.joint is not None and link.joint.name in finger_joints:
                owners[link.entity.per_scene_id] = link
    for actor in task_env.scene.get_all_actors():
        if actor.name not in BACKGROUND_NAMES:
            owners[actor.per_scene_id] = actor
    for art in task_env.scene.get_all_articulations():
        if art in arms:
            continue
        for link in art.get_links():
            owners[link.entity.per_scene_id] = link

    if not owners:
        raise RuntimeError(
            "no trackable body in the scene: expected task objects and/or gripper finger links, found none"
        )
    return owners


def _transform(body) -> np.ndarray:
    pose = body.get_entity_pose() if hasattr(body, "get_entity_pose") else body.get_pose()
    return pose.to_transformation_matrix()


class PoseRecorder:
    """Per-env log of trackable-body poses, one frame per executed action.

    ``gen_sparse_reward_data`` calls ``task.get_obs()`` exactly once per action it executes (and once more
    to pad a chunk that success or truncation cut short), so wrapping ``get_obs`` yields one pose frame per
    step without changing RoboTwin. ``refresh`` must run after every env reset, because a reset rebuilds
    the scene and invalidates the body handles and their ``per_scene_id``s.
    """

    _WRAPPED = "_rlinf_gt_track_recorder"

    def __init__(self, task_env):
        self.task_env = task_env
        self.refresh()
        self._frames = []
        self._anchor = None
        self._recording = False
        self._install()

    def _install(self):
        task = self.task_env
        if getattr(task, self._WRAPPED, None) is self:
            return
        inner = getattr(task, "get_obs")
        if getattr(task, self._WRAPPED, None) is not None:
            # A previous recorder for this task is already in the chain; reuse its unwrapped get_obs.
            inner = getattr(task, self._WRAPPED)._inner
        self._inner = inner

        def get_obs_recording(*args, **kwargs):
            obs = inner(*args, **kwargs)
            if self._recording:
                self._frames.append(self._poses())
            return obs

        task.get_obs = get_obs_recording
        setattr(task, self._WRAPPED, self)

    def refresh(self):
        """Re-read the scene's trackable bodies; call after every env reset."""
        owners = trackable_owners(self.task_env)
        self.owner_ids = np.array(sorted(owners), dtype=np.int64)
        self._bodies = [owners[pid] for pid in self.owner_ids]
        self._row_of = {pid: i for i, pid in enumerate(self.owner_ids)}

    def _poses(self) -> np.ndarray:
        return np.stack([_transform(b) for b in self._bodies], axis=0).astype(np.float32)

    def owner_rows(self, patch_px: np.ndarray, depth_ok: np.ndarray):
        """``(owner_row [391] int64, valid [391] bool)`` for the head patches at this instant.

        ``owner_row`` indexes ``owner_ids``; it is 0 and meaningless wherever ``valid`` is False.
        """
        cams = self.task_env.cameras
        cam = cams.static_camera_list[cams.static_camera_name.index("head_camera")]
        seg = cam.get_picture("Segmentation")[..., 1]
        h, w = seg.shape
        px = np.clip(np.round(patch_px[:, 0]).astype(int), 0, w - 1)
        py = np.clip(np.round(patch_px[:, 1]).astype(int), 0, h - 1)
        at_patch = seg[py, px]

        valid = depth_ok & np.isin(at_patch, self.owner_ids.astype(at_patch.dtype))
        rows = np.zeros(at_patch.shape, dtype=np.int64)
        for pid, row in self._row_of.items():
            rows[at_patch == pid] = row
        return rows, valid

    def start_chunk(self):
        """Snapshot the chunk's anchor poses and begin logging one frame per executed action."""
        self._anchor = self._poses()
        self._frames = []
        self._recording = True

    def gt_track(self, centres_world: np.ndarray, owner_row: np.ndarray, valid: np.ndarray, horizon: int):
        """``(track [391, horizon, 3] metres, valid [391])`` for the chunk that just ran.

        ``centres_world`` are the patch centres at the anchor frame. A chunk cut short by success or
        truncation holds its last frame for the remaining steps, as the offline generator does.
        """
        self._recording = False
        if self._anchor is None:
            raise RuntimeError("gt_track called before start_chunk")
        frames, anchor = self._frames, self._anchor
        self._anchor = None

        track = np.zeros((len(centres_world), horizon, 3), dtype=np.float32)
        if not frames:
            # Nothing executed (the env was already done): no motion to report.
            return track, np.zeros_like(valid)

        idx = np.where(valid)[0]
        if idx.size == 0:
            return track, valid

        rows = owner_row[idx]
        p_anchor = centres_world[idx].astype(np.float32)
        ones = np.ones((idx.size, 1), dtype=np.float32)
        p_local = np.einsum(
            "nij,nj->ni", np.linalg.inv(anchor[rows]), np.concatenate([p_anchor, ones], axis=-1)
        )[:, :3]
        p_local_h = np.concatenate([p_local, ones], axis=-1)

        for h in range(horizon):
            pose_h = frames[min(h, len(frames) - 1)]
            p_future = np.einsum("nij,nj->ni", pose_h[rows], p_local_h)[:, :3]
            track[idx, h, :] = p_future - p_anchor
        return track, valid
