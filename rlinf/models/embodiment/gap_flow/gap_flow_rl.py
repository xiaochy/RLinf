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

"""GAP action-only flow-matching policy as an RLinf PPO policy.

The wrapped ``GAPPolicy`` (from the GAP repository, ``generative_mode='flow_matching'``,
``aux_task='none'``) predicts a velocity field v(x_t, t | obs) with the openpi convention
x_t = t*noise + (1-t)*action (t=1 is noise).  Here each denoising step becomes a Gaussian transition
(``rl_sampler.sample_mean_var``), so the log-probability of a sampled chain is a sum of Gaussian
log-probabilities -- the same construction RLinf uses for pi0 / pi0.5.
"""

import random
import sys
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gap_flow.obs_utils import (
    GapDinoExtractor,
    batch_patch_centers,
)
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi.sampling import rl_sampler


class _StateDictHolder:
    """GAP's set_normalizer() only needs an object exposing state_dict()."""

    def __init__(self, state_dict):
        self._sd = state_dict

    def state_dict(self):
        return self._sd


class GapFlowRL(nn.Module, BasePolicy):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        gf = cfg.get("gap_flow", {})
        self.gap_root = str(gf["gap_root"])
        self.num_steps = int(gf.get("num_steps", 10))
        self.noise_method = str(gf.get("noise_method", "flow_sde"))
        self.noise_level = float(gf.get("noise_level", 0.3))
        self.joint_logprob = bool(gf.get("joint_logprob", False))
        self.ignore_last = bool(gf.get("ignore_last", False))
        self.add_value_head = bool(cfg.get("add_value_head", True))
        self.detach_critic_input = bool(gf.get("detach_critic_input", True))
        self.use_ema = bool(gf.get("use_ema", True))
        self.dino_weights_path = gf.get("dino_weights_path", None)
        self.autocast_bf16 = bool(gf.get("autocast_bf16", True))

        if self.gap_root not in sys.path:
            sys.path.insert(0, self.gap_root)
        import hydra

        ckpt = torch.load(str(cfg.model_path), map_location="cpu")
        gcfg = OmegaConf.create(ckpt["cfg"])
        assert gcfg.policy.generative_mode == "flow_matching", (
            gcfg.policy.generative_mode
        )
        assert gcfg.get("aux_task", "none") == "none", (
            "action-only (aux_task=none) checkpoint required"
        )
        self.policy = hydra.utils.instantiate(gcfg.policy)
        self.policy.set_normalizer(_StateDictHolder(ckpt["normalizer"]))
        weights = ckpt["ema"] if (self.use_ema and "ema" in ckpt) else ckpt["model"]
        missing, unexpected = self.policy.load_state_dict(weights, strict=False)
        self.load_report = {"missing": list(missing), "unexpected": list(unexpected)}

        self.horizon = int(self.policy.horizon)
        self.action_dim = int(self.policy.action_dim)
        self.num_action_chunks = int(
            cfg.get("num_action_chunks", self.policy.n_action_steps)
        )
        assert self.num_action_chunks <= self.horizon
        if self.add_value_head:
            self.value_head = ValueHead(
                input_dim=int(self.policy.feature_dim),
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        self._dino: list = []  # lazily built, kept out of the module tree (frozen, not trained / synced)

    # ------------------------------------------------------------------ helpers
    def train(self, mode: bool = True):
        super().train(mode)
        # No dropout in either rollout or update, otherwise the importance ratio is not 1 at the
        # first update epoch.
        self.policy.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _dino_extractor(self) -> GapDinoExtractor:
        if not self._dino:
            self._dino.append(
                GapDinoExtractor(
                    self.gap_root,
                    weights_path=self.dino_weights_path,
                    device=self.device,
                )
            )
        return self._dino[0]

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.autocast_bf16 and self.device.type == "cuda",
        )

    def _policy_obs_from_env(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Raw RLinf env obs -> GAP policy inputs (DINO features, patch centers, EE state)."""
        dev = self.device
        dino = self._dino_extractor()(env_obs["main_images"])  # [B, 1, N, D]
        centers = batch_patch_centers(
            env_obs["head_depth"], env_obs["head_intrinsic"], env_obs["head_extrinsic"]
        )
        return {
            "dino_features": dino.to(dev).float(),
            "patch_centers": centers.to(dev).float(),
            "agent_pos": env_obs["ee_state"].to(dev).float(),
        }

    def _encode(self, obs: dict[str, torch.Tensor]) -> dict[str, Any]:
        state_xyz, patch_centers = self.policy._extract_rope_coords(obs)
        nobs = self.policy.normalizer.normalize(obs)
        memory, memory_pos, memory_coords, memory_rope_mask = (
            self.policy.encode_observations(
                nobs, state_xyz=state_xyz, patch_centers=patch_centers
            )
        )
        return {
            "memory": memory,
            "memory_pos": memory_pos,
            "memory_coords": memory_coords,
            "memory_rope_mask": memory_rope_mask,
            "state_xyz": state_xyz,
            "patch_centers": patch_centers,
        }

    def _velocity(self, x_t, t, enc, return_hidden=False):
        with self._autocast():
            out = self.policy.predict_velocity(
                x_t, t, return_hidden=return_hidden, **enc
            )
        if return_hidden:
            return out[0].float(), out[1]
        return out.float()

    def _value(self, chains0: torch.Tensor, enc: dict[str, Any]) -> torch.Tensor:
        """V(s) from the decoded action tokens at t=1 (pure-noise input: depends on the observation)."""
        B = chains0.shape[0]
        t1 = torch.ones(B, device=chains0.device, dtype=torch.float32)
        _, hidden = self._velocity(chains0, t1, enc, return_hidden=True)
        pooled = hidden.float().mean(dim=1)
        if self.detach_critic_input:
            pooled = pooled.detach()
        return self.value_head(pooled).reshape(B).float()

    def _decode(self, x0: torch.Tensor) -> torch.Tensor:
        actions = self.policy.normalizer["action"].unnormalize(x0.clamp(-1.0, 1.0))
        return actions[:, : self.num_action_chunks].float()

    # --------------------------------------------------------------- interface
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"GapFlowRL does not support forward_type={forward_type}"
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = True,
        *,
        noise: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
        **kwargs,
    ):
        del kwargs
        return self.predict_from_policy_obs(
            self._policy_obs_from_env(env_obs),
            mode,
            compute_values,
            noise=noise,
            rng=rng,
        )

    @torch.no_grad()
    def predict_from_policy_obs(
        self,
        obs: dict[str, torch.Tensor],
        mode: str = "eval",
        compute_values: bool = True,
        *,
        noise: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ):
        device = self.device
        B = obs["agent_pos"].shape[0]
        enc = self._encode(obs)
        if noise is None:
            noise = torch.randn(
                B,
                self.horizon,
                self.action_dim,
                device=device,
                dtype=torch.float32,
                generator=rng,
            )
        N = self.num_steps
        replay = {
            "agent_pos": obs["agent_pos"].detach().contiguous(),
            "dino_features": obs["dino_features"].detach().half().contiguous(),
            "patch_centers": obs["patch_centers"].detach().contiguous(),
        }

        if mode == "eval":
            x0 = self.policy._flow_integrate(
                noise,
                enc["memory"],
                enc["memory_pos"],
                enc["memory_coords"],
                enc["memory_rope_mask"],
                enc["state_xyz"],
                enc["patch_centers"],
                num_steps=N,
            )
            actions = self._decode(x0)
            return actions, {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": {
                    "action": actions.reshape(B, -1).contiguous(),
                    "model_action": x0.reshape(B, -1).contiguous(),
                    **replay,
                },
                "model_actions": x0,
            }

        if self.joint_logprob:
            denoise_inds = (
                torch.arange(N, device=device)[None].expand(B, -1).contiguous()
            )
        else:
            hi = N - 2 if self.ignore_last else N - 1
            chosen = random.randint(0, max(hi, 0))
            denoise_inds = torch.full((B, N), chosen, device=device, dtype=torch.long)

        timesteps = rl_sampler.get_timesteps(N, device)
        idx_step = torch.empty(B, device=device, dtype=torch.long)
        x_t = noise
        chains = [x_t]
        log_probs = []
        if self.joint_logprob:
            log_probs.append(
                rl_sampler.gaussian_logprob(
                    x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
                )
            )
        for idx in range(N):
            method = (
                self.noise_method
                if idx == int(denoise_inds[0, idx].item())
                else "flow_ode"
            )
            t = timesteps[idx].expand(B).float()
            v = self._velocity(x_t, t, enc)
            idx_step.fill_(idx)
            mean, std = rl_sampler.sample_mean_var(
                x_t.float(),
                v,
                idx_step,
                noise_method=method,
                noise_level=self.noise_level,
                num_steps=N,
            )
            x_next = (
                mean
                + torch.randn(
                    x_t.shape, device=device, dtype=torch.float32, generator=rng
                )
                * std
            )
            log_probs.append(rl_sampler.gaussian_logprob(x_next, mean, std))
            chains.append(x_next)
            x_t = x_next

        x0 = x_t
        chains_t = torch.stack(chains, dim=1).contiguous()
        lp = torch.stack(log_probs, dim=1)[
            :, :, : self.num_action_chunks, : self.action_dim
        ]
        lp = (
            lp.mean(dim=1)
            if self.joint_logprob
            else lp[torch.arange(B, device=device), denoise_inds[:, 0]]
        )
        if compute_values and self.add_value_head:
            prev_values = self._value(chains_t[:, 0], enc)[:, None].contiguous()
        else:
            prev_values = torch.zeros((B, 1), device=device, dtype=torch.float32)
        actions = self._decode(x0)
        forward_inputs = {
            "chains": chains_t,
            "denoise_inds": denoise_inds,
            "action": actions.reshape(B, -1).contiguous(),
            "model_action": x0.reshape(B, -1).contiguous(),
            **replay,
        }
        return actions, {
            "prev_logprobs": lp.float().contiguous(),
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
            "model_actions": x0,
        }

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        """Recompute log-probs (and values) of the stored chain with the current weights."""
        compute_values = kwargs.get("compute_values", True)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        B = chains.shape[0]
        device = chains.device
        obs = {
            "agent_pos": forward_inputs["agent_pos"].float(),
            "dino_features": forward_inputs["dino_features"].float(),
            "patch_centers": forward_inputs["patch_centers"].float(),
        }
        enc = self._encode(obs)
        N = self.num_steps
        timesteps = rl_sampler.get_timesteps(N, device)
        step_logprobs = []
        if self.joint_logprob:
            num_recompute = N
            step_logprobs.append(
                rl_sampler.gaussian_logprob(
                    chains[:, 0].float(),
                    torch.zeros_like(chains[:, 0], dtype=torch.float32),
                    torch.ones_like(chains[:, 0], dtype=torch.float32),
                )
            )
        else:
            num_recompute = 1
        arange_B = torch.arange(B, device=device)
        for idx in range(num_recompute):
            di = denoise_inds[:, idx].to(torch.long)
            pre = chains[arange_B, di].float()
            nxt = chains[arange_B, di + 1].float()
            v = self._velocity(pre, timesteps[di].float(), enc)
            mean, std = rl_sampler.sample_mean_var(
                pre,
                v,
                di,
                noise_method=self.noise_method,
                noise_level=self.noise_level,
                num_steps=N,
            )
            step_logprobs.append(rl_sampler.gaussian_logprob(nxt, mean, std))
        lp = torch.stack(step_logprobs, dim=1)[
            :, :, : self.num_action_chunks, : self.action_dim
        ].float()
        lp = lp.mean(dim=1) if self.joint_logprob else lp[:, 0]
        if compute_values and self.add_value_head:
            values = self._value(chains[:, 0].float(), enc)
        else:
            values = torch.zeros(B, device=device, dtype=torch.float32)
        return {
            "logprobs": lp.contiguous(),
            "values": values.float(),
            "entropy": torch.zeros((B, 1), device=device, dtype=torch.float32),
        }


def get_model(cfg: DictConfig, torch_dtype=None) -> GapFlowRL:
    model = GapFlowRL(cfg)
    if model.load_report["missing"] or model.load_report["unexpected"]:
        print(
            f"[GapFlowRL] load_state_dict: missing={len(model.load_report['missing'])} "
            f"unexpected={len(model.load_report['unexpected'])}"
        )
    return model
