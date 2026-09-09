"""MiniCPM5 AWQ backbone loading and lightweight LoRA adaptation.

The local MiniCPM5 checkpoint used by this project was produced with a newer
``compressed-tensors`` schema than the runtime currently understands.  This
module therefore validates that exact schema and dequantizes each packed
linear weight once into a frozen execution-only tensor.  It does *not* claim
to train the packed integer weights: only explicit LoRA parameters are
trainable.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .safety import SidecarSafetyLimits, categorical_policy_kl, tensor_linf_drift


LOGGER = logging.getLogger(__name__)
PathLike = Union[str, Path]


def _as_shape(value: Union[torch.Tensor, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, torch.Tensor):
        items = value.detach().cpu().tolist()
    else:
        items = list(value)
    if len(items) != 2:
        raise ValueError(f"weight_shape must have two entries, got {items!r}")
    shape = (int(items[0]), int(items[1]))
    if min(shape) <= 0:
        raise ValueError(f"weight_shape must be positive, got {shape}")
    return shape


def dequantize_compressed_int_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_shape: Union[torch.Tensor, Sequence[int]],
    *,
    group_size: int = 32,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Decode a 4-bit ``compressed-tensors`` asymmetric groupwise weight.

    The checkpoint layout is:

    - ``weight_packed[out, in/8]``: eight input-axis nibbles per int32;
    - ``weight_scale[out, in/group_size]``;
    - ``weight_zero_point[out/8, in/group_size]``: eight output-axis
      zero-points per int32.

    The returned matrix satisfies ``W[o, i] = (q[o, i] - z[o, g]) s[o, g]``.
    This function deliberately accepts only the format that has been verified
    against the bundled MiniCPM5 checkpoint.
    """

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if weight_packed.dtype != torch.int32:
        raise TypeError("weight_packed must use torch.int32")
    if weight_zero_point.dtype != torch.int32:
        raise TypeError("weight_zero_point must use torch.int32")
    if not weight_scale.is_floating_point():
        raise TypeError("weight_scale must be floating point")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")

    out_features, in_features = _as_shape(weight_shape)
    if in_features % group_size != 0:
        raise ValueError("input width must be divisible by group_size")
    if in_features % 8 != 0 or out_features % 8 != 0:
        raise ValueError("verified 4-bit layout requires both dimensions divisible by 8")
    groups = in_features // group_size
    if tuple(weight_packed.shape) != (out_features, in_features // 8):
        raise ValueError(
            "weight_packed shape mismatch: expected "
            f"{(out_features, in_features // 8)}, got {tuple(weight_packed.shape)}"
        )
    if tuple(weight_scale.shape) != (out_features, groups):
        raise ValueError(
            f"weight_scale shape mismatch: expected {(out_features, groups)}, "
            f"got {tuple(weight_scale.shape)}"
        )
    if tuple(weight_zero_point.shape) != (out_features // 8, groups):
        raise ValueError(
            "weight_zero_point shape mismatch: expected "
            f"{(out_features // 8, groups)}, got {tuple(weight_zero_point.shape)}"
        )
    devices = {weight_packed.device, weight_scale.device, weight_zero_point.device}
    if len(devices) != 1:
        raise ValueError("packed weights, scales, and zero-points must share a device")

    shifts_last = torch.arange(
        0, 32, 4, device=weight_packed.device, dtype=torch.int32
    )
    quantized = torch.bitwise_and(
        torch.bitwise_right_shift(weight_packed.unsqueeze(-1), shifts_last), 0xF
    ).reshape(out_features, in_features)

    shifts_out = shifts_last.view(1, 8, 1)
    zero_points = torch.bitwise_and(
        torch.bitwise_right_shift(weight_zero_point.unsqueeze(1), shifts_out), 0xF
    ).reshape(out_features, groups)

    quantized = quantized.reshape(out_features, groups, group_size).float()
    scales = weight_scale.float().unsqueeze(-1)
    zeros = zero_points.float().unsqueeze(-1)
    return ((quantized - zeros) * scales).reshape(out_features, in_features).to(dtype)


def validate_minicpm_awq_checkpoint(model_path: PathLike) -> Dict[str, Any]:
    """Validate the local checkpoint and return normalized model metadata."""

    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    tensor_path = root / "model.safetensors"
    tokenizer_path = root / "tokenizer.json"
    for required in (config_path, tensor_path, tokenizer_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required local checkpoint file is missing: {required}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("model_type") != "llama" or "LlamaForCausalLM" not in raw.get(
        "architectures", []
    ):
        raise ValueError("Expected a LlamaForCausalLM-compatible MiniCPM5 checkpoint")

    quant = raw.get("quantization_config")
    if not isinstance(quant, dict):
        raise ValueError("Checkpoint has no quantization_config")
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("Only compressed-tensors checkpoints are supported")
    if quant.get("format") != "pack-quantized":
        raise ValueError("Only pack-quantized checkpoints are supported")
    groups = quant.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("quantization_config.config_groups is empty")

    schemes = []
    for group in groups.values():
        if not isinstance(group, dict) or "weights" not in group:
            raise ValueError("Every quantization group must define a weight scheme")
        schemes.append(group["weights"])
    for scheme in schemes:
        expected = {
            "num_bits": 4,
            "group_size": 32,
            "strategy": "group",
            "symmetric": False,
            "type": "int",
        }
        mismatches = {
            key: (scheme.get(key), value)
            for key, value in expected.items()
            if scheme.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Unsupported compressed weight scheme: {mismatches}")

    rope = raw.get("rope_parameters") or {}
    return {
        "model_path": str(root),
        "config": raw,
        "hidden_size": int(raw["hidden_size"]),
        "num_hidden_layers": int(raw["num_hidden_layers"]),
        "group_size": 32,
        "num_bits": 4,
        "rope_theta": float(rope.get("rope_theta", raw.get("rope_theta", 10000.0))),
        "tensor_path": str(tensor_path),
    }


class LoRALinear(nn.Module):
    """A frozen linear map plus a trainable low-rank residual.

    ``base`` keeps its execution dtype (FP16 for the MiniCPM backbone), while
    ``lora_A`` and ``lora_B`` stay in FP32 for stable small online updates.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")

        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        low_rank_input = self.dropout(x).to(dtype=self.lora_A.dtype)
        delta = F.linear(F.linear(low_rank_input, self.lora_A), self.lora_B)
        return base_output + (self.scaling * delta).to(dtype=base_output.dtype)


class MiniCPMLoRAEncoder(nn.Module):
    """Frozen MiniCPM5 feature extractor with LoRA on one decoder block."""

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Any,
        *,
        base_model_path: PathLike,
        layer_index: int,
        target_modules: Sequence[str] = ("q_proj", "v_proj"),
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.base_model_path = str(Path(base_model_path).expanduser().resolve())
        self.layer_index = int(layer_index)
        self.target_modules = tuple(str(name) for name in target_modules)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)

        if not hasattr(backbone, "layers"):
            raise TypeError("backbone must expose Llama decoder layers")
        if not (0 <= self.layer_index < len(backbone.layers)):
            raise IndexError("layer_index out of range")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")

        self.backbone.requires_grad_(False)
        attention = backbone.layers[self.layer_index].self_attn
        for module_name in self.target_modules:
            base = getattr(attention, module_name, None)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"self_attn.{module_name} is not an nn.Linear")
            wrapped = LoRALinear(
                base,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            ).to(device=base.weight.device)
            setattr(attention, module_name, wrapped)
        self.backbone.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def execution_dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    def named_lora_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if name.endswith("lora_A") or name.endswith("lora_B"):
                yield name, parameter

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.named_lora_parameters())

    @property
    def frozen_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not (name.endswith("lora_A") or name.endswith("lora_B"))
        )

    def encode_prompts(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        max_length: int = 256,
        require_lora_grad: bool = False,
    ) -> torch.Tensor:
        """Return the final valid-token hidden state as a FP32 ``[B, D]`` tensor."""

        if isinstance(prompts, str):
            batch = [prompts]
        else:
            batch = list(prompts)
        if not batch or any(not isinstance(prompt, str) for prompt in batch):
            raise ValueError("prompts must contain at least one string")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        context = torch.enable_grad() if require_lora_grad else torch.no_grad()
        with context:
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            columns = torch.arange(output.size(1), device=self.device).unsqueeze(0)
            last_indices = (columns * attention_mask).max(dim=1).values
            rows = torch.arange(output.size(0), device=self.device)
            features = output[rows, last_indices].float()
        return features

    def zero_lora_grad(self) -> None:
        for _, parameter in self.named_lora_parameters():
            parameter.grad = None

    def step_from_external_gradient(
        self,
        features: torch.Tensor,
        grad_features: torch.Tensor,
        *,
        lr: float,
        orthogonal_memory: Optional[Any] = None,
        remember_gradient: bool = False,
        project_with_memory: bool = True,
        max_grad_norm: Optional[float] = 1.0,
        max_update_norm: Optional[float] = None,
        constancy_closure: Optional[Callable[[], torch.Tensor]] = None,
        safety_limits: Optional[SidecarSafetyLimits] = None,
    ) -> Dict[str, Any]:
        """Backpropagate an explicit MQR input gradient and update only LoRA.

        When supplied, ``constancy_closure`` is evaluated immediately before
        and after the candidate update.  It should return frozen-backbone probe
        logits (or features when only an infinity-norm limit is used).  A
        violated safety limit restores both LoRA tensors and OGD memory.
        Recurrent-state and Cayley-transition limits belong to the MQR agent;
        this LoRA-only bridge rejects them instead of silently ignoring them.
        """

        if lr <= 0:
            raise ValueError("lr must be positive")
        if features.shape != grad_features.shape:
            raise ValueError("features and grad_features must have the same shape")
        if not features.requires_grad:
            raise ValueError("features must come from encode_prompts(require_lora_grad=True)")
        if remember_gradient and orthogonal_memory is None:
            raise ValueError("remember_gradient requires orthogonal_memory")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if max_update_norm is not None and (
            not math.isfinite(float(max_update_norm)) or max_update_norm <= 0
        ):
            raise ValueError("max_update_norm must be finite and positive or None")
        if safety_limits is not None and not isinstance(
            safety_limits, SidecarSafetyLimits
        ):
            raise TypeError("safety_limits must be a SidecarSafetyLimits or None")
        limits = safety_limits or SidecarSafetyLimits()
        unsupported_limits = []
        if limits.max_state_linf_drift is not None:
            unsupported_limits.append("max_state_linf_drift")
        if limits.max_transition_fro_drift is not None:
            unsupported_limits.append("max_transition_fro_drift")
        if unsupported_limits:
            joined = ", ".join(unsupported_limits)
            raise ValueError(
                "MiniCPM LoRA constancy cannot evaluate "
                f"{joined}; use policy-KL/output limits or audit the MQR "
                "state and transition in the agent transaction"
            )
        constancy_required = (
            limits.max_policy_kl is not None
            or limits.max_output_linf_drift is not None
        )
        if constancy_required and constancy_closure is None:
            raise ValueError("constancy safety limits require constancy_closure")
        if constancy_closure is not None and not callable(constancy_closure):
            raise TypeError("constancy_closure must be callable or None")
        if (
            constancy_closure is not None
            and orthogonal_memory is not None
            and not all(hasattr(orthogonal_memory, name) for name in ("snapshot", "restore"))
        ):
            raise TypeError("atomic constancy rollback requires snapshot/restore OGD memory")

        constancy_before: Optional[torch.Tensor] = None
        if constancy_closure is not None:
            with torch.no_grad():
                constancy_before = constancy_closure()
            if not isinstance(constancy_before, torch.Tensor):
                raise TypeError("constancy_closure must return a tensor")
            constancy_before = constancy_before.detach().clone()

        self.zero_lora_grad()
        torch.autograd.backward(
            features,
            grad_tensors=grad_features.to(device=features.device, dtype=features.dtype),
        )
        named = list(self.named_lora_parameters())
        missing = [name for name, parameter in named if parameter.grad is None]
        if missing:
            raise RuntimeError(f"LoRA parameters are disconnected from the feature graph: {missing}")
        if any(not bool(torch.isfinite(parameter.grad).all()) for _, parameter in named):
            self.zero_lora_grad()
            raise FloatingPointError("LoRA gradient contains NaN or Inf")

        raw_euclidean_norm = math.sqrt(
            sum(float(parameter.grad.float().square().sum().item()) for _, parameter in named)
        )
        clip_scale = 1.0
        if max_grad_norm is not None and raw_euclidean_norm > max_grad_norm:
            clip_scale = float(max_grad_norm) / (raw_euclidean_norm + 1e-12)
        entries = [
            (name, parameter.grad.detach() * clip_scale, float(lr))
            for name, parameter in named
        ]
        parameter_snapshot = {
            name: parameter.detach().clone() for name, parameter in named
        }
        memory_snapshot = (
            orthogonal_memory.snapshot()
            if orthogonal_memory is not None
            and hasattr(orthogonal_memory, "snapshot")
            else None
        )

        projection_applied = bool(orthogonal_memory is not None and project_with_memory)
        if projection_applied:
            projected, stats = orthogonal_memory.project_preconditioned(entries)
        else:
            projected = {name: gradient for name, gradient, _ in entries}
            whitened_norm = math.sqrt(
                sum(float(lr) * float(gradient.square().sum().item()) for _, gradient, _ in entries)
            )
            stats = {
                "rank": float(orthogonal_memory.rank) if orthogonal_memory is not None else 0.0,
                "raw_norm": whitened_norm,
                "projected_norm": whitened_norm,
                "retained_norm": 1.0 if whitened_norm > 0 else 0.0,
                "max_abs_overlap": 0.0,
            }

        memory_added = False
        if remember_gradient:
            memory_added = bool(orthogonal_memory.observe(entries))

        unclipped_update_norm = math.sqrt(
            sum(
                (float(lr) ** 2) * float(projected[name].square().sum().item())
                for name, _parameter in named
            )
        )
        update_scale = 1.0
        if max_update_norm is not None and unclipped_update_norm > max_update_norm:
            update_scale = float(max_update_norm) / (unclipped_update_norm + 1e-12)

        safety_violations = []
        constancy_linf: Optional[float] = None
        constancy_kl: Optional[float] = None
        rolled_back = False
        with torch.no_grad():
            for name, parameter in named:
                gradient = projected[name]
                parameter.add_(gradient, alpha=-float(lr) * update_scale)
            try:
                if constancy_closure is not None:
                    constancy_after = constancy_closure()
                    if not isinstance(constancy_after, torch.Tensor):
                        raise TypeError("constancy_closure must return a tensor")
                    constancy_after = constancy_after.detach()
                    assert constancy_before is not None
                    constancy_linf = tensor_linf_drift(
                        constancy_before, constancy_after
                    )
                    if limits.max_policy_kl is not None:
                        constancy_kl = categorical_policy_kl(
                            constancy_before, constancy_after
                        )
                    if not math.isfinite(constancy_linf) or (
                        constancy_kl is not None and not math.isfinite(constancy_kl)
                    ):
                        safety_violations.append("nonfinite_constancy")
                    if (
                        limits.max_output_linf_drift is not None
                        and constancy_linf > limits.max_output_linf_drift
                    ):
                        safety_violations.append("output_linf_drift")
                    if (
                        limits.max_policy_kl is not None
                        and constancy_kl is not None
                        and constancy_kl > limits.max_policy_kl
                    ):
                        safety_violations.append("policy_kl")
            except Exception:
                for name, parameter in named:
                    parameter.copy_(parameter_snapshot[name])
                if memory_snapshot is not None:
                    orthogonal_memory.restore(memory_snapshot)
                self.zero_lora_grad()
                raise

            rolled_back = bool(safety_violations)
            if rolled_back:
                for name, parameter in named:
                    parameter.copy_(parameter_snapshot[name])
                if memory_snapshot is not None:
                    orthogonal_memory.restore(memory_snapshot)
                memory_added = False
        self.zero_lora_grad()
        retained = float(stats["retained_norm"])
        ogd_capacity_warning = bool(
            projection_applied
            and float(stats.get("raw_norm", 0.0)) > 0.0
            and retained < limits.ogd_retained_warning_threshold
        )
        candidate_update_norm = unclipped_update_norm * update_scale
        update_norm = 0.0 if rolled_back else candidate_update_norm
        return {
            "raw_grad_norm": raw_euclidean_norm,
            "clip_scale": clip_scale,
            "update_clip_scale": update_scale,
            "candidate_update_norm": candidate_update_norm,
            "update_norm": update_norm,
            "did_update": bool(not rolled_back and update_norm > 0.0),
            "ogd_rank": int(orthogonal_memory.rank) if orthogonal_memory is not None else 0,
            "ogd_retained_norm": retained,
            "ogd_capacity_warning": ogd_capacity_warning,
            "ogd_max_abs_overlap": float(stats["max_abs_overlap"]),
            "ogd_projection_applied": projection_applied,
            "ogd_memory_added": memory_added,
            "constancy_checked": constancy_closure is not None,
            "constancy_output_linf_drift": constancy_linf,
            "constancy_policy_kl": constancy_kl,
            "safety_passed": not rolled_back,
            "safety_violations": list(safety_violations),
            "rolled_back": rolled_back,
        }

    def adapter_state_dict(self) -> Dict[str, Any]:
        return {
            "format": "mqr-minicpm-lora-v1",
            "metadata": {
                "base_model_path": self.base_model_path,
                "layer_index": self.layer_index,
                "target_modules": list(self.target_modules),
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "hidden_size": self.hidden_size,
            },
            "state": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.named_lora_parameters()
            },
        }

    def load_adapter_state_dict(self, payload: Mapping[str, Any], *, strict: bool = True) -> None:
        if payload.get("format") != "mqr-minicpm-lora-v1":
            raise ValueError("Unsupported adapter checkpoint format")
        metadata = payload.get("metadata", {})
        expected = {
            "layer_index": self.layer_index,
            "target_modules": list(self.target_modules),
            "rank": self.lora_rank,
            "hidden_size": self.hidden_size,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Adapter metadata mismatch: {mismatches}")
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("Adapter checkpoint has no tensor state")

        current = dict(self.named_lora_parameters())
        missing = sorted(set(current) - set(state))
        unexpected = sorted(set(state) - set(current))
        if strict and (missing or unexpected):
            raise ValueError(f"Adapter keys mismatch; missing={missing}, unexpected={unexpected}")
        with torch.no_grad():
            for name, parameter in current.items():
                if name not in state:
                    continue
                value = state[name]
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise ValueError(f"Adapter tensor {name!r} has an incompatible shape")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def save_adapter(self, path: PathLike) -> None:
        torch.save(self.adapter_state_dict(), Path(path))

    def load_adapter(self, path: PathLike, *, strict: bool = True) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.load_adapter_state_dict(payload, strict=strict)


def _normalized_llama_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(raw)
    config.pop("quantization_config", None)
    config.pop("transformers_version", None)
    rope = config.pop("rope_parameters", None)
    if isinstance(rope, Mapping):
        config["rope_theta"] = float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))
        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            config["rope_scaling"] = dict(rope)
    config["use_cache"] = False
    return config


def load_minicpm_awq_encoder(
    model_path: PathLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float16,
    layer_index: int = -1,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    attention_backend: str = "sdpa",
) -> MiniCPMLoRAEncoder:
    """Load the verified local AWQ checkpoint as a frozen dequantized backbone.

    Loading is performed parameter-by-parameter from safetensors so a second
    full FP16 state dictionary is never materialized.  ``lm_head`` is omitted:
    this component is a hidden-state encoder for the external MQR action head.
    """

    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if attention_backend not in ("eager", "sdpa"):
        raise ValueError('attention_backend must be "eager" or "sdpa"')
    info = validate_minicpm_awq_checkpoint(model_path)
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    try:
        from safetensors import safe_open
        from transformers import AutoTokenizer, LlamaConfig, LlamaModel
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    except ImportError as exc:
        raise RuntimeError("Loading MiniCPM requires transformers and safetensors") from exc

    config = LlamaConfig(**_normalized_llama_config(info["config"]))
    config._attn_implementation = attention_backend
    with torch.device("meta"):
        backbone = LlamaModel(config).to(dtype=dtype)
    backbone.to_empty(device=target_device)
    # ``to_empty`` cannot retain a meta-created non-persistent RoPE buffer.
    backbone.rotary_emb = LlamaRotaryEmbedding(config, device=target_device)
    backbone.requires_grad_(False)

    tensor_path = info["tensor_path"]
    LOGGER.info(
        "Loading MiniCPM5 backbone from %s as frozen %s tensors on %s",
        tensor_path,
        str(dtype).removeprefix("torch."),
        target_device,
    )
    loaded = 0
    with safe_open(tensor_path, framework="pt", device="cpu") as archive:
        available = set(archive.keys())
        with torch.no_grad():
            for name, parameter in backbone.named_parameters():
                checkpoint_name = "model." + name
                if checkpoint_name in available:
                    value = archive.get_tensor(checkpoint_name).to(
                        device=target_device, dtype=parameter.dtype
                    )
                elif checkpoint_name.endswith(".weight"):
                    base = checkpoint_name[: -len(".weight")]
                    required = [
                        base + ".weight_packed",
                        base + ".weight_scale",
                        base + ".weight_zero_point",
                        base + ".weight_shape",
                    ]
                    missing = [key for key in required if key not in available]
                    if missing:
                        raise KeyError(f"Missing compressed tensors for {checkpoint_name}: {missing}")
                    value = dequantize_compressed_int_weight(
                        archive.get_tensor(required[0]),
                        archive.get_tensor(required[1]),
                        archive.get_tensor(required[2]),
                        archive.get_tensor(required[3]),
                        group_size=info["group_size"],
                        dtype=parameter.dtype,
                    ).to(target_device)
                else:
                    raise KeyError(f"No checkpoint tensor for backbone parameter {checkpoint_name}")
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"Shape mismatch for {checkpoint_name}: checkpoint {tuple(value.shape)}, "
                        f"model {tuple(parameter.shape)}"
                    )
                parameter.copy_(value)
                loaded += 1
                del value

    resolved_layer = int(layer_index)
    if resolved_layer < 0:
        resolved_layer += int(config.num_hidden_layers)
    encoder = MiniCPMLoRAEncoder(
        backbone,
        AutoTokenizer.from_pretrained(
            info["model_path"], local_files_only=True, trust_remote_code=False
        ),
        base_model_path=info["model_path"],
        layer_index=resolved_layer,
        target_modules=target_modules,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    LOGGER.info(
        "Loaded %d backbone tensors; LoRA trainable parameters: %d",
        loaded,
        encoder.trainable_parameter_count,
    )
    return encoder


__all__ = [
    "LoRALinear",
    "MiniCPMLoRAEncoder",
    "dequantize_compressed_int_weight",
    "load_minicpm_awq_encoder",
    "validate_minicpm_awq_checkpoint",
]
