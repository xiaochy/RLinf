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

"""Fold a LoRA finetune's factors into the base weights of a JAX Pi0/Pi05 checkpoint.

openpi stores a LoRA finetune as the frozen base weight plus ``lora_a`` / ``lora_b`` and adds
the product back at call time. The convertors in this package read only the base weight and
load with ``strict=False``, so a LoRA checkpoint has to be merged here first; otherwise the
conversion yields the base model and reports nothing.

Each module's own scaling is reproduced rather than unified, because they differ:
``openpi.models.lora.Einsum`` multiplies the product by ``alpha / rank`` (``alpha /
sqrt(rank)`` under rslora), while ``openpi.models.lora.FeedForward._dot`` applies no scaling.
Both LoRA variants openpi ships set ``alpha == rank``, so the two agree today; a variant that
changed alpha would not.

Factors contract over the last two axes and broadcast over the leading stacked-layer axes, so
each pair merges with one ``matmul``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from rlinf.utils.logging import get_logger

    logger = get_logger()
except ImportError:
    # This module also has to import inside the JAX/openpi environment, which does not
    # carry RLinf's torch stack.
    import logging

    logger = logging.getLogger(__name__)

# rank and alpha, from openpi.models.gemma.get_config. openpi is deliberately not a
# dependency of this package, so its LoRA variants are listed here; the rank is checked
# against the array shapes, and an unlisted variant raises.
_LORA_VARIANTS: dict[str, tuple[int, float]] = {
    "gemma_2b_lora": (16, 16.0),
    "gemma_300m_lora": (32, 32.0),
}

_EINSUM_BASE = "w"
_FFN_BASES = ("gating_einsum", "linear")


def is_lora_variant(variant: str) -> bool:
    return variant.endswith("_lora")


def lora_leaf_paths(tree: Any, prefix: str = "") -> list[str]:
    """Every path in ``tree`` whose final component mentions LoRA."""
    out: list[str] = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if "lora" in str(key):
                out.append(path)
            out.extend(lora_leaf_paths(value, path))
    return out


def unwrap_value_leaves(tree: Any) -> Any:
    """Collapse orbax's ``{"value": array}`` leaf wrapper.

    Checkpoints written by openpi's training loop wrap every leaf, so a weight
    arrives as ``{"kernel": {"value": array}}`` while the converters index
    ``["kernel"]`` and expect the array. Only a single-key ``value`` dict whose
    content is not itself a dict is collapsed, which leaves the SigLIP attention
    parameter that is genuinely *named* ``value`` (it holds ``kernel`` and
    ``bias``) untouched.
    """
    if not isinstance(tree, dict):
        return tree
    if set(tree) == {"value"} and not isinstance(tree["value"], dict):
        return tree["value"]
    return {k: unwrap_value_leaves(v) for k, v in tree.items()}


def _scaling(variant: str, module: str, rank_from_shape: int, *, rslora: bool) -> float:
    """The factor openpi multiplies this module's LoRA product by."""
    if variant not in _LORA_VARIANTS:
        raise ValueError(
            f"{variant!r} is not a LoRA variant this merger knows "
            f"({', '.join(sorted(_LORA_VARIANTS))}). Add its rank and alpha from "
            "openpi.models.gemma.get_config rather than letting the factors be dropped."
        )
    rank, alpha = _LORA_VARIANTS[variant]
    if rank != rank_from_shape:
        raise ValueError(
            f"{variant!r} declares rank {rank} but the checkpoint's factors have rank "
            f"{rank_from_shape}. Refusing to merge with a scaling derived from the wrong rank."
        )
    if module == "ffn":
        # openpi.models.lora.FeedForward._dot applies no scaling at all.
        return 1.0
    return alpha / math.sqrt(rank) if rslora else alpha / rank


def _merge_pair(
    node: dict, base_key: str, a_key: str, b_key: str, scaling: float
) -> tuple[int, float]:
    a = np.asarray(node.pop(a_key), dtype=np.float32)
    b = np.asarray(node.pop(b_key), dtype=np.float32)
    base = np.asarray(node[base_key])
    delta = np.matmul(a, b) * scaling
    if delta.shape != base.shape:
        raise ValueError(
            f"LoRA product for {base_key} has shape {delta.shape}, base weight has "
            f"{base.shape}; the factor layout is not the one this merger assumes."
        )
    merged = base.astype(np.float32) + delta
    node[base_key] = merged if base.dtype == np.float32 else merged.astype(base.dtype)
    # Two magnitudes rather than a ratio: an Einsum's base weight initialises to zeros, and
    # a ratio against that says nothing.
    return int(a.shape[-1]), (float(np.abs(delta).max()), float(np.abs(base).max()))


def merge_lora_params(
    params: dict,
    *,
    paligemma_variant: str,
    action_expert_variant: str,
    rslora: bool = False,
    strict: bool = True,
) -> list[str]:
    """Merge every LoRA pair in ``params`` in place; return one report line per pair.

    ``params`` is the unwrapped JAX tree (``PaliGemma`` / ``action_in_proj`` /
    ... at the top). The action expert's parameters are the ``_1``-suffixed
    siblings (``q_einsum_1``, ``mlp_1``), which is how the expert's variant is
    told apart from PaliGemma's.

    With ``strict``, a factor left over after the walk raises: an unmerged factor
    is exactly the silent base-model conversion this function exists to prevent.
    A checkpoint that carries factors while both variants are non-LoRA also
    raises, instead of dropping them.
    """
    report: list[str] = []
    present = lora_leaf_paths(params)
    if present and not (
        is_lora_variant(paligemma_variant) or is_lora_variant(action_expert_variant)
    ):
        raise ValueError(
            f"the checkpoint carries {len(present)} LoRA parameter(s) (e.g. {present[0]}) but "
            f"the variants given are {paligemma_variant!r} / {action_expert_variant!r}. Pass the "
            "_lora variants, or the finetune would be dropped and the base model converted."
        )
    if not present:
        return report

    def variant_for(name: str) -> str:
        return action_expert_variant if name.endswith("_1") else paligemma_variant

    def walk(node: Any, name: str, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "lora_a" in node and "lora_b" in node:
            if _EINSUM_BASE not in node:
                raise ValueError(
                    f"{path} has LoRA factors but no {_EINSUM_BASE!r} base weight"
                )
            variant = variant_for(name)
            rank = int(np.asarray(node["lora_a"]).shape[-1])
            scaling = _scaling(variant, "attn", rank, rslora=rslora)
            _, (d_max, w_max) = _merge_pair(
                node, _EINSUM_BASE, "lora_a", "lora_b", scaling
            )
            report.append(
                f"{path}/w  rank={rank} scaling={scaling:g} "
                f"max|delta|={d_max:.3e} max|w|={w_max:.3e}"
            )
        for base in _FFN_BASES:
            a_key, b_key = f"{base}_lora_a", f"{base}_lora_b"
            if a_key in node and b_key in node:
                if base not in node:
                    raise ValueError(
                        f"{path} has LoRA factors but no {base!r} base weight"
                    )
                variant = variant_for(name)
                rank = int(np.asarray(node[a_key]).shape[-1])
                scaling = _scaling(variant, "ffn", rank, rslora=rslora)
                _, (d_max, w_max) = _merge_pair(node, base, a_key, b_key, scaling)
                report.append(
                    f"{path}/{base}  rank={rank} scaling={scaling:g} "
                    f"max|delta|={d_max:.3e} max|w|={w_max:.3e}"
                )
        for key, value in node.items():
            walk(value, str(key), f"{path}/{key}" if path else str(key))

    walk(params, "", "")

    leftover = lora_leaf_paths(params)
    if leftover and strict:
        raise ValueError(
            f"{len(leftover)} LoRA parameter(s) were not merged: {leftover[:6]}. "
            "Converting now would write the base weights and drop the finetune."
        )
    logger.info("merged %d LoRA pair(s)", len(report))
    for line in report:
        logger.info("  %s", line)
    return report
