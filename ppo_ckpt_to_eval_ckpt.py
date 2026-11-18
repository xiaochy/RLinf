# Copyright 2026 The RLinf Authors.
"""Repackage a PPO FSDP checkpoint as a GAP-style ``.ckpt`` the eval path can load.

Why this exists: evaluating a PPO checkpoint through ``runner.resume_dir`` is not an
evaluation of that checkpoint. ``embodied_runner.run()`` evaluates *inside* the step loop,
after a rollout and a training update, and the resume restores the lr_scheduler, whose
``base_lrs`` (2e-6) overrides the eval config's ``actor.optim.lr=1e-12`` safety pin. One
real update therefore lands before the eval, computed from ``env.train.total_num_envs=2``
trajectories -- measured at approx_kl -0.061 against 0.0008-0.015 for the genuine training
steps of the same run.

A checkpoint written here goes through ``actor.model.model_path`` with no resume, so the
lr pin holds and the evaluated weights are the checkpoint's own.

The architecture config and the normalizer are taken from the base ``.ckpt`` the arm
trained from: the FSDP save holds neither, and ``gap_flow_rl.py`` reads both from
``model_path``.
"""

import argparse
import sys

import torch

POLICY_PREFIX = "policy."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-ckpt", required=True, help="…/global_step_N/actor/model_state_dict/full_weights.pt")
    ap.add_argument("--base-ckpt", required=True, help="the .ckpt the arm trained from (cfg + normalizer)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    full = torch.load(a.ppo_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(full, dict):
        print(f"expected a state dict in {a.ppo_ckpt}, found {type(full)}", file=sys.stderr)
        return 1
    policy = {k[len(POLICY_PREFIX):]: v for k, v in full.items() if k.startswith(POLICY_PREFIX)}
    if not policy:
        print(f"no '{POLICY_PREFIX}*' tensors in {a.ppo_ckpt}", file=sys.stderr)
        return 1

    base = torch.load(a.base_ckpt, map_location="cpu", weights_only=False)
    for k in ("cfg", "normalizer"):
        if k not in base:
            print(f"base checkpoint {a.base_ckpt} has no '{k}'", file=sys.stderr)
            return 1

    # gap_flow_rl.py loads with strict=False, so a name mismatch would silently leave the
    # base weights in place and the eval would score the wrong policy. Compare the key sets
    # against the base's own, and refuse rather than write a checkpoint that loads as a no-op.
    ref = base.get("ema") or base["model"]
    missing = sorted(set(ref) - set(policy))
    extra = sorted(set(policy) - set(ref))
    if missing or extra:
        print(f"key sets differ from the base: {len(missing)} missing, {len(extra)} extra", file=sys.stderr)
        for k in (missing[:5] + extra[:5]):
            print(f"   {k}", file=sys.stderr)
        return 1

    changed = sum(1 for k in ref if not torch.equal(ref[k].float(), policy[k].float()))
    out = {
        "epoch": int(base.get("epoch", -1)),
        "global_step": int(full.get("global_step", -1)) if "global_step" in full else -1,
        "cfg": base["cfg"],
        "normalizer": base["normalizer"],
        # use_ema defaults to True, so 'ema' is the branch the loader takes; both carry the
        # PPO weights so the result does not depend on that flag.
        "model": policy,
        "ema": policy,
    }
    torch.save(out, a.out)
    print(f"wrote {a.out}")
    print(f"  {len(policy)} policy tensors, {changed} of them differ from the base")
    if changed == 0:
        print("  WARNING: identical to the base -- the eval would re-measure the base", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
