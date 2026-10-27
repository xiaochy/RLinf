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
"""Subprocess that predicts per-patch 3D tracks with the action-conditioned dynamics model.

Runs in its own process because the dynamics model's `gap_policy` package has the same name as the policy's (see
dsrl-scan-object/dynamics_scorer_worker.py). Requests are length-prefixed pickles on stdin/stdout:
{"dino_feat": [B,391,1024], "agent_pos": [B,16], "patch_centers": [B,391,3], "actions": [B,H,16]} ->
{"tracks": [B,391,H,3]} (metres, world-frame displacement) or {"error": traceback}.
"""

import argparse
import pickle
import struct
import sys
import traceback

import numpy as np
import torch

DYNAMICS_REPO = "/project_data/held/vxiao/Bimanual_Manipulation_Project/dynamics-model-rollout-worldmodel"


def read_msg(stream):
    header = stream.read(8)
    if len(header) < 8:
        return None
    n = struct.unpack(">Q", header)[0]
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed mid-message")
        buf += chunk
    return pickle.loads(buf)


def write_msg(stream, obj):
    buf = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack(">Q", len(buf)))
    stream.write(buf)
    stream.flush()


@torch.no_grad()
def predict(model, req, device):
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)  # noqa: E731
    obs = {
        "agent_pos": t(req["agent_pos"]),
        "patch_centers": t(req["patch_centers"]),
        "dino_features": t(req["dino_feat"]).unsqueeze(1),  # [B, 1, 391, 1024]
    }
    return model.predict_track(obs, action=t(req["actions"]))["track_pred"].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for p in (DYNAMICS_REPO, DYNAMICS_REPO + "/scripts"):
        sys.path.insert(0, p)
    from eval_track import load_model  # noqa: PLC0415

    model = load_model(args.dynamics_ckpt, args.device)
    assert getattr(model, "action_conditioned_world_model", False), "not an action-conditioned dynamics model"
    sys.stderr.write("[dynamics_track_worker] ready\n")
    sys.stderr.flush()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        req = read_msg(stdin)
        if req is None:
            break
        try:
            write_msg(stdout, {"tracks": predict(model, req, args.device)})
        except Exception:
            write_msg(stdout, {"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
