#!/usr/bin/env python3
"""Tests for Go rules, compressed-weight decoding, and the LoRA/MQR bridge."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from mqr import OnlineMultiRingClassifier, OrthogonalGradientMemory, SidecarSafetyLimits
from mqr.go import (
    BLACK,
    WHITE,
    GoBoard,
    GoTrainingExample,
    HeuristicGoTeacher,
    action_to_gtp,
    format_go_prompt,
    generate_basic_go_dataset,
    gtp_to_action,
)
from mqr.minicpm import (
    LoRALinear,
    MiniCPMLoRAEncoder,
    dequantize_compressed_int_weight,
    validate_minicpm_awq_checkpoint,
)
from mqr.sayuri import SayuriGTPClient, discover_sayuri_paths
from experiments.minicpm_go_online import (
    _advance_context_memory_schedule,
    _replay_board,
    _restore_stream_state,
)
from experiments.minicpm_go_real_games import (
    CONDITIONS,
    PassGatedSayuriPolicyTeacher,
    _evaluate_probes,
    _generate_probe_examples,
    _play_game,
    _probe_curve_trends,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _pack_last_axis(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int64)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return (values.reshape(*values.shape[:-1], -1, 8) << shifts).sum(dim=-1).to(torch.int32)


def _pack_output_axis(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int64)
    out_features, groups = values.shape
    shifts = torch.arange(0, 32, 4, dtype=torch.int64).view(1, 8, 1)
    return (values.reshape(out_features // 8, 8, groups) << shifts).sum(dim=1).to(torch.int32)


def test_compressed_int4_decode_exact() -> None:
    torch.manual_seed(1)
    quantized = torch.randint(0, 16, (16, 64), dtype=torch.int32)
    zero_points = torch.randint(0, 16, (16, 2), dtype=torch.int32)
    scales = torch.rand(16, 2, dtype=torch.float32) + 0.01
    packed = _pack_last_axis(quantized)
    packed_zeros = _pack_output_axis(zero_points)

    decoded = dequantize_compressed_int_weight(
        packed, scales, packed_zeros, (16, 64), group_size=32, dtype=torch.float32
    )
    expected = (
        quantized.reshape(16, 2, 32).float() - zero_points.float().unsqueeze(-1)
    ) * scales.unsqueeze(-1)
    torch.testing.assert_close(decoded, expected.reshape(16, 64), rtol=0.0, atol=0.0)


def test_lora_identity_then_external_update() -> None:
    torch.manual_seed(2)
    base = nn.Linear(6, 5, bias=False)
    module = LoRALinear(base, rank=3, alpha=6.0)
    x = torch.randn(4, 6)
    initial = module(x)
    torch.testing.assert_close(initial, base(x), rtol=0.0, atol=0.0)

    loss = module(x).square().mean()
    loss.backward()
    with torch.no_grad():
        module.lora_B.add_(module.lora_B.grad, alpha=-0.1)
    changed = module(x)
    assert not torch.equal(initial, changed)


class _TinyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = _TinyAttention(width)


class _TinyBackbone(nn.Module):
    def __init__(self, width: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(width)])
        self.config = SimpleNamespace(hidden_size=width)


def _tiny_encoder(seed: int = 3) -> MiniCPMLoRAEncoder:
    torch.manual_seed(seed)
    return MiniCPMLoRAEncoder(
        _TinyBackbone(),
        tokenizer=None,
        base_model_path=".",
        layer_index=0,
        target_modules=("q_proj", "v_proj"),
        lora_rank=2,
        lora_alpha=4.0,
    )


def _tiny_features(encoder: MiniCPMLoRAEncoder, x: torch.Tensor) -> torch.Tensor:
    attention = encoder.backbone.layers[0].self_attn
    return attention.q_proj(x) + attention.v_proj(x)


def test_external_gradient_updates_lora_with_ogd() -> None:
    encoder = _tiny_encoder()
    memory = OrthogonalGradientMemory(max_rank=2)
    x = torch.randn(3, 6)
    features = _tiny_features(encoder, x)
    before = features.detach().clone()
    info = encoder.step_from_external_gradient(
        features,
        torch.ones_like(features),
        lr=0.05,
        orthogonal_memory=memory,
        remember_gradient=True,
    )
    after = _tiny_features(encoder, x).detach()
    assert info["update_norm"] > 0
    assert info["ogd_memory_added"] and memory.rank == 1
    assert not torch.equal(before, after)


def test_lora_constancy_guard_rolls_back_adapter_and_ogd_atomically() -> None:
    encoder = _tiny_encoder(seed=31)
    memory = OrthogonalGradientMemory(max_rank=2)
    x = torch.randn(3, 6)
    features = _tiny_features(encoder, x)
    output_before = features.detach().clone()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in encoder.named_lora_parameters()
    }

    for unsupported in (
        SidecarSafetyLimits(max_state_linf_drift=0.1),
        SidecarSafetyLimits(max_transition_fro_drift=0.1),
    ):
        try:
            encoder.step_from_external_gradient(
                features,
                torch.ones_like(features),
                lr=0.2,
                safety_limits=unsupported,
            )
        except ValueError as exc:
            assert "cannot evaluate" in str(exc)
        else:
            raise AssertionError("unsupported LoRA safety limits must fail closed")

    info = encoder.step_from_external_gradient(
        features,
        torch.ones_like(features),
        lr=0.2,
        orthogonal_memory=memory,
        remember_gradient=True,
        max_update_norm=0.05,
        constancy_closure=lambda: _tiny_features(encoder, x),
        safety_limits=SidecarSafetyLimits(max_output_linf_drift=0.0),
    )
    assert info["constancy_checked"] is True
    assert info["rolled_back"] is True
    assert info["did_update"] is False
    assert info["candidate_update_norm"] <= 0.05 + 1e-7
    assert info["update_norm"] == 0.0
    assert "output_linf_drift" in info["safety_violations"]
    assert memory.rank == 0 and memory.dimension == 0
    for name, parameter in encoder.named_lora_parameters():
        torch.testing.assert_close(
            parameter, parameters_before[name], rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(
        _tiny_features(encoder, x), output_before, rtol=0.0, atol=0.0
    )


def test_adapter_checkpoint_roundtrip() -> None:
    source = _tiny_encoder(seed=4)
    x = torch.randn(2, 6)
    features = _tiny_features(source, x)
    source.step_from_external_gradient(features, torch.randn_like(features), lr=0.03)
    payload = source.adapter_state_dict()

    restored = _tiny_encoder(seed=4)
    restored.load_adapter_state_dict(payload)
    for (name_a, value_a), (name_b, value_b) in zip(
        source.named_lora_parameters(), restored.named_lora_parameters()
    ):
        assert name_a == name_b
        torch.testing.assert_close(value_a, value_b, rtol=0.0, atol=0.0)


def test_capture_suicide_and_superko() -> None:
    capture = GoBoard.from_stones(3, black=[1, 3, 7], white=[4], to_play=BLACK)
    result = capture.play(5)
    assert result.captures == 1 and capture.board[4] == 0

    suicide = GoBoard.from_stones(3, white=[1, 3, 5, 7], to_play=BLACK)
    assert not suicide.is_legal(4)

    # A classical one-stone ko. Black captures at C3; White's B3 recapture
    # would restore both the stones and Black-to-play situation.
    ko = GoBoard.from_stones(
        4,
        black=[1, 4, 9],
        white=[2, 5, 7, 10],
        to_play=BLACK,
    )
    capture_result = ko.play(6)
    assert capture_result.captures == 1
    assert not ko.is_legal(5)


def test_pass_scoring_and_coordinates() -> None:
    board = GoBoard(5)
    assert board.play(board.pass_action).game_over is False
    assert board.play(board.pass_action).game_over is True

    territory = GoBoard.from_stones(3, black=[1, 3, 5, 7], komi=0.0)
    score = territory.score()
    assert score["black"] == 9.0 and score["white"] == 0.0
    assert territory.winner() == BLACK

    for action in (0, 4, 20, 24, 25):
        assert gtp_to_action(action_to_gtp(action, 5), 5) == action
    assert action_to_gtp(gtp_to_action("J9", 9), 9) == "J9"


def test_teacher_dataset_is_legal_and_deterministic() -> None:
    first = generate_basic_go_dataset(20, seed=7)
    second = generate_basic_go_dataset(20, seed=7)
    teacher = HeuristicGoTeacher()
    for left, right in zip(first, second):
        assert left.board.position_key() == right.board.position_key()
        assert left.target_action == right.target_action
        assert left.board.is_legal(left.target_action)
        assert teacher.select_move(left.board) == left.target_action


def test_go_prompt_rule_ablation_is_label_free() -> None:
    board = GoBoard(5)
    for action in (12, 6, 13):
        board.play(action)
    rules = format_go_prompt(board, prompt_mode="rules")
    wrong_rules = format_go_prompt(board, prompt_mode="wrong-rules")
    board_only = format_go_prompt(board, prompt_mode="board-only")
    assert board.render() in rules and board.render() in wrong_rules
    assert board.render() in board_only
    assert "规则：" in rules and "禁止自杀" in rules
    assert "规则：" in wrong_rules and "允许自杀" in wrong_rules
    assert "禁止自杀" not in wrong_rules
    assert "规则：" not in board_only and "禁止自杀" not in board_only
    assert "white" in rules and "white" in board_only
    try:
        format_go_prompt(board, prompt_mode="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown prompt modes must be rejected")


def test_preview_step_is_read_only_and_matches_deferred_update() -> None:
    torch.manual_seed(71)
    learner = OnlineMultiRingClassifier(
        5,
        7,
        4,
        num_rings=1,
        lr=0.02,
        carry_state=True,
        ring_kwargs={"relaxation_steps": 4, "lora_rank": 2, "readout_dim": 7},
    )
    x = torch.randn(1, 5)
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in learner.named_parameters()
    }
    preview = learner.preview_step(x, context_id="black")
    assert learner._states == [None]
    assert int(learner.ring_usage.sum().item()) == 0
    assert int(learner.routing_key_counts.sum().item()) == 0
    for name, parameter in learner.named_parameters():
        torch.testing.assert_close(parameter, parameters_before[name], rtol=0.0, atol=0.0)

    committed = learner.online_step(x, torch.tensor([2]), context_id="black")
    torch.testing.assert_close(committed["logits"], preview["logits"], rtol=0.0, atol=0.0)
    assert learner._states[0] is not None
    state_before = learner._states[0].h.clone()
    usage_before = learner.ring_usage.clone()
    keys_before = learner.routing_keys.clone()
    learner.preview_step(-x, context_id="black")
    torch.testing.assert_close(learner._states[0].h, state_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(learner.ring_usage, usage_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(learner.routing_keys, keys_before, rtol=0.0, atol=0.0)


def test_online_input_gradient_is_prequential() -> None:
    torch.manual_seed(8)
    model = OnlineMultiRingClassifier(
        6,
        8,
        4,
        num_rings=1,
        lr=0.02,
        carry_state=False,
        ring_kwargs={
            "alpha": 0.3,
            "relaxation_steps": 20,
            "lora_rank": 3,
            "readout_dim": 8,
        },
    )
    x = torch.randn(2, 6)
    before = model.rings[0](x)
    info = model.online_step(
        x, torch.tensor([1, 3]), context_id="go", return_grad_x=True
    )
    torch.testing.assert_close(info["logits"], before, rtol=0.0, atol=0.0)
    assert info["grad_x"].shape == x.shape
    assert bool(torch.isfinite(info["grad_x"]).all())
    unlabeled = model.online_step(x, None, context_id="go", return_grad_x=True)
    assert unlabeled["grad_x"] is None


def test_complete_mqr_update_respects_trust_region() -> None:
    torch.manual_seed(9)
    model = OnlineMultiRingClassifier(
        6,
        8,
        4,
        num_rings=1,
        lr=1.0,
        max_update_norm=0.01,
        carry_state=False,
        ring_kwargs={
            "alpha": 0.3,
            "relaxation_steps": 20,
            "lora_rank": 3,
            "readout_dim": 8,
        },
    )
    info = model.online_step(torch.randn(3, 6), torch.tensor([0, 1, 2]))
    assert info["unclipped_update_norm"] > 0.01
    assert info["update_norm"] <= 0.01 + 1e-7
    assert 0.0 < info["update_clip_scale"] < 1.0


def test_local_checkpoint_metadata() -> None:
    path = "/home/spikebai/checkpoints/MiniCPM5-1B-AWQ-INT4"
    info = validate_minicpm_awq_checkpoint(path)
    assert info["hidden_size"] == 1536
    assert info["num_hidden_layers"] == 24
    assert info["num_bits"] == 4 and info["group_size"] == 32


def test_sayuri_gtp_rules_policy_and_search() -> None:
    try:
        binary, weights = discover_sayuri_paths()
    except FileNotFoundError as exc:
        logging.warning("Skipping optional Sayuri integration test: %s", exc)
        return
    examples = generate_basic_go_dataset(4, seed=11)
    with SayuriGTPClient(
        binary,
        weights,
        board_size=5,
        komi=5.5,
        threads=1,
        playouts=4,
    ) as client:
        assert client.engine_name == "Sayuri"
        for example in examples:
            comparison = client.compare_legal_moves(example.board)
            assert comparison == {"local_only": set(), "sayuri_only": set()}
        policy = client.raw_policy(examples[-1].board)
        assert len(policy) == examples[-1].board.action_size
        assert bool(torch.isfinite(torch.tensor(policy)).all())
        assert examples[-1].board.is_legal(client.select_move(examples[-1].board, mode="policy"))
        assert examples[-1].board.is_legal(client.select_move(examples[-1].board, mode="mcts"))


def test_pass_gated_sayuri_policy_teacher() -> None:
    class FakeClient:
        @staticmethod
        def raw_policy(board: GoBoard):
            values = [0.0] * board.action_size
            values[board.pass_action] = 1.0
            for action in board.legal_moves(include_pass=False):
                values[action] = 0.5 - action * 1e-3
            return values

    teacher = PassGatedSayuriPolicyTeacher(FakeClient(), min_pass_occupancy=0.5)
    empty = GoBoard(5)
    assert teacher.select_move(empty) == 0
    occupied = GoBoard.from_stones(5, black=range(13), to_play=WHITE)
    assert teacher.select_move(occupied) == occupied.pass_action
    try:
        PassGatedSayuriPolicyTeacher(FakeClient(), min_pass_occupancy=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid pass thresholds must be rejected")


def test_real_game_probe_split_is_exact_and_deterministic() -> None:
    class DeterministicMixedTeacher:
        @staticmethod
        def select_move(board: GoBoard) -> int:
            signature = sum(
                (index + 1) * stone for index, stone in enumerate(board.board)
            ) + board.to_play
            if signature % 4 == 0:
                return board.pass_action
            legal = board.legal_moves(include_pass=False)
            return legal[signature % len(legal)] if legal else board.pass_action

    kwargs = {
        "count": 20,
        "size": 5,
        "komi": 5.5,
        "seed": 313,
        "pass_fraction": 0.25,
    }
    first = _generate_probe_examples(
        teacher=DeterministicMixedTeacher(), **kwargs
    )
    second = _generate_probe_examples(
        teacher=DeterministicMixedTeacher(), **kwargs
    )
    assert sum(item.target_action == item.board.pass_action for item in first) == 5
    assert sum(item.target_action != item.board.pass_action for item in first) == 15
    assert len({item.board.position_key() for item in first}) == len(first)
    assert [item.board.position_key() for item in first] == [
        item.board.position_key() for item in second
    ]
    assert [item.target_action for item in first] == [
        item.target_action for item in second
    ]


def test_probe_metrics_separate_placement_pass_and_raw_legality() -> None:
    class IndexedEncoder:
        @staticmethod
        def encode_prompts(prompts, *, max_length: int, require_lora_grad: bool):
            del max_length, require_lora_grad
            return torch.eye(len(prompts), dtype=torch.float32)

    class TableLearner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "table",
                torch.tensor(
                    [
                        [4.0, 0.0, 0.0, 0.0, -1.0],
                        [4.0, 3.0, 0.0, 0.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, 4.0],
                        [4.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )

        def forward(self, features: torch.Tensor, *, context_id: str):
            del context_id
            return self.table[int(torch.argmax(features[0]).item())].unsqueeze(0)

    empty = GoBoard(2)
    occupied_zero = GoBoard.from_stones(2, black=[0], to_play=WHITE)
    pass_a = GoBoard.from_stones(2, black=[0], to_play=WHITE)
    pass_b = GoBoard.from_stones(2, black=[1], to_play=WHITE)
    examples = [
        GoTrainingExample(empty, 0),
        GoTrainingExample(occupied_zero, 1),
        GoTrainingExample(pass_a, pass_a.pass_action),
        GoTrainingExample(pass_b, pass_b.pass_action),
    ]
    learner = TableLearner()
    metrics = _evaluate_probes(
        IndexedEncoder(),
        learner,
        examples,
        prompt_mode="rules",
        max_length=64,
        batch_size=4,
    )
    probabilities = torch.softmax(learner.table, dim=1)[:, 4]
    expected_brier = torch.mean(
        (probabilities - torch.tensor([0.0, 0.0, 1.0, 1.0])).square()
    )
    assert metrics["placement_target_count"] == 2
    assert metrics["pass_target_count"] == 2
    assert metrics["conditional_placement_agreement"] == 0.5
    assert metrics["false_pass_rate_on_placement_targets"] == 0.0
    assert metrics["pass_recall"] == 0.5
    assert metrics["pass_precision"] == 1.0
    assert metrics["raw_legal_rate"] == 0.75
    assert metrics["raw_illegal_occupied_rate"] == 0.25
    assert abs(metrics["pass_brier"] - float(expected_brier.item())) < 1e-7


def test_probe_curve_trend_arithmetic() -> None:
    curve = [
        {
            "after_train_games": 0,
            "loss": 3.0,
            "conditional_placement_loss": 2.0,
            "conditional_placement_agreement": 0.10,
            "false_pass_rate_on_placement_targets": 0.30,
            "pass_brier": 0.25,
            "raw_legal_rate": 0.50,
            "raw_teacher_agreement": 0.20,
        },
        {
            "after_train_games": 2,
            "loss": 2.0,
            "conditional_placement_loss": 1.5,
            "conditional_placement_agreement": 0.20,
            "false_pass_rate_on_placement_targets": 0.20,
            "pass_brier": 0.20,
            "raw_legal_rate": 0.60,
            "raw_teacher_agreement": 0.30,
        },
        {
            "after_train_games": 4,
            "loss": 1.0,
            "conditional_placement_loss": 1.0,
            "conditional_placement_agreement": 0.15,
            "false_pass_rate_on_placement_targets": 0.10,
            "pass_brier": 0.15,
            "raw_legal_rate": 0.70,
            "raw_teacher_agreement": 0.40,
        },
    ]
    trends = _probe_curve_trends(curve)
    assert abs(trends["loss"]["slope_per_training_game"] + 0.5) < 1e-12
    assert trends["loss"]["favorable_step_fraction"] == 1.0
    assert abs(
        trends["conditional_placement_agreement"]["slope_per_training_game"]
        - 0.0125
    ) < 1e-12
    assert (
        trends["conditional_placement_agreement"]["favorable_step_fraction"]
        == 0.5
    )


def test_lora_only_game_path_has_exact_zero_mqr_parameter_drift() -> None:
    class TinyOnlineEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora = nn.Parameter(torch.zeros(4))
            self.register_buffer("base", torch.tensor([1.0, -1.0, 0.5, -0.5]))

        def encode_prompts(
            self, prompts, *, max_length: int, require_lora_grad: bool
        ) -> torch.Tensor:
            del max_length, require_lora_grad
            return (self.base + self.lora).unsqueeze(0).expand(len(prompts), -1)

        def step_from_external_gradient(
            self,
            features: torch.Tensor,
            gradient: torch.Tensor,
            *,
            lr: float,
            orthogonal_memory,
            remember_gradient: bool,
            project_with_memory: bool,
            max_grad_norm: float,
        ):
            del features, orthogonal_memory, remember_gradient
            del project_with_memory, max_grad_norm
            update = gradient.detach().mean(dim=0)
            with torch.no_grad():
                self.lora.add_(update, alpha=-float(lr))
            norm = float(torch.linalg.vector_norm(update).item())
            return {
                "raw_grad_norm": norm,
                "update_norm": float(lr) * norm,
                "ogd_rank": 0,
                "ogd_retained_norm": 1.0,
                "ogd_memory_added": False,
            }

    class FirstLegalTeacher:
        @staticmethod
        def select_move(board: GoBoard) -> int:
            legal = board.legal_moves(include_pass=False)
            return legal[0] if legal else board.pass_action

    torch.manual_seed(317)
    learner = OnlineMultiRingClassifier(
        4,
        4,
        5,
        num_rings=2,
        lr=0.05,
        carry_state=True,
        ring_kwargs={"relaxation_steps": 3, "lora_rank": 2, "readout_dim": 4},
    )
    encoder = TinyOnlineEncoder()
    before_mqr = {
        name: parameter.detach().clone()
        for name, parameter in learner.named_parameters()
    }
    before_lora = encoder.lora.detach().clone()
    args = SimpleNamespace(
        board_size=2,
        komi=0.0,
        max_game_moves=2,
        max_length=64,
        ogd_rank=0,
        remember_every=1,
        pass_update_scale=1.0,
        lora_lr=0.1,
        max_lora_grad_norm=1.0,
    )
    condition = CONDITIONS["rules-lora-only"]
    assert condition.update_lora and not condition.update_mqr
    summary, records = _play_game(
        encoder,
        learner,
        FirstLegalTeacher(),
        args=args,
        condition=condition,
        lora_memory=OrthogonalGradientMemory(0),
        context_counts={},
        opening_history=[],
        student_color=BLACK,
        game_index=0,
        phase="unit-test",
        learn=True,
        collect_records=True,
    )
    assert summary["moves_played"] == 2 and len(records) == 2
    for name, parameter in learner.named_parameters():
        torch.testing.assert_close(parameter, before_mqr[name], rtol=0.0, atol=0.0)
    assert not torch.equal(encoder.lora, before_lora)
    assert all(record["mqr_update_norm"] == 0.0 for record in records)
    assert any(record["lora_update_norm"] > 0.0 for record in records)


def test_sayuri_discovery_matches_setup_layout() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        managed_binary = root / ".external/sayuri-build/sayuri"
        in_tree_binary = root / "UsedCode/Sayuri/build/sayuri"
        managed_weight = root / ".external/sayuri-weights/managed.bin.txt"
        in_tree_weight = root / "UsedCode/Sayuri/weights/upstream.bin.txt"
        for path in (managed_binary, in_tree_binary, managed_weight, in_tree_weight):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
        binary, weights = discover_sayuri_paths(root)
        assert binary == managed_binary.resolve()
        assert weights == managed_weight.resolve()


def test_context_local_memory_schedule_has_no_color_aliasing() -> None:
    counts = {}
    remembered = {"black": [], "white": []}
    for step in range(24):
        context = "black" if step % 2 == 0 else "white"
        ordinal, remember = _advance_context_memory_schedule(
            counts,
            context,
            ogd_rank=8,
            remember_every=6,
        )
        if remember:
            remembered[context].append(ordinal)
    assert counts == {"black": 12, "white": 12}
    assert remembered == {"black": [6, 12], "white": [6, 12]}


def test_stream_resume_keeps_board_and_ring_state_aligned() -> None:
    board = GoBoard(5, komi=5.5)
    for action in (12, 6, 13, board.pass_action):
        board.play(action)
    restored = _replay_board(5, 5.5, board.move_history)
    assert restored.board == board.board
    assert restored.to_play == board.to_play
    assert restored.position_history == board.position_history

    learner = OnlineMultiRingClassifier(
        4,
        6,
        3,
        num_rings=2,
        lr=0.01,
        carry_state=True,
        ring_kwargs={"relaxation_steps": 2, "lora_rank": 2, "readout_dim": 6},
    )
    learner.online_step(torch.randn(1, 4), None, context_id="black")
    assert learner._states[0] is not None
    args = SimpleNamespace(
        stream_mode="game",
        board_size=5,
        komi=5.5,
        teacher="sayuri-policy",
        rollout_policy="teacher",
        seed=2026,
        dataset_size=8,
        sayuri_playouts=16,
    )
    state = {
        "version": 1,
        "steps_completed": 4,
        "stream_mode": "game",
        "board_size": 5,
        "komi": 5.5,
        "teacher": "sayuri-policy",
        "rollout_policy": "teacher",
        "seed": 2026,
        "dataset_size": 8,
        "sayuri_playouts": 16,
        "board_move_history": list(board.move_history),
        "context_sample_counts": {"black": 2, "white": 2},
        "completed_games": [],
    }
    resumed_board, offset, counts, completed = _restore_stream_state(state, args, learner)
    assert resumed_board.board == board.board and offset == 4
    assert counts == {"black": 2, "white": 2} and completed == []
    assert learner._states[0] is not None

    legacy_board, legacy_offset, _, _ = _restore_stream_state({}, args, learner)
    assert legacy_offset == 0 and not legacy_board.move_history
    assert learner._states == [None, None]


def run_all_tests() -> None:
    tests = [
        test_compressed_int4_decode_exact,
        test_lora_identity_then_external_update,
        test_external_gradient_updates_lora_with_ogd,
        test_lora_constancy_guard_rolls_back_adapter_and_ogd_atomically,
        test_adapter_checkpoint_roundtrip,
        test_capture_suicide_and_superko,
        test_pass_scoring_and_coordinates,
        test_teacher_dataset_is_legal_and_deterministic,
        test_go_prompt_rule_ablation_is_label_free,
        test_preview_step_is_read_only_and_matches_deferred_update,
        test_online_input_gradient_is_prequential,
        test_complete_mqr_update_respects_trust_region,
        test_local_checkpoint_metadata,
        test_sayuri_discovery_matches_setup_layout,
        test_context_local_memory_schedule_has_no_color_aliasing,
        test_stream_resume_keeps_board_and_ring_state_aligned,
        test_pass_gated_sayuri_policy_teacher,
        test_real_game_probe_split_is_exact_and_deterministic,
        test_probe_metrics_separate_placement_pass_and_raw_legality,
        test_probe_curve_trend_arithmetic,
        test_lora_only_game_path_has_exact_zero_mqr_parameter_drift,
        test_sayuri_gtp_rules_policy_and_search,
    ]
    for test in tests:
        logging.info("Running %s...", test.__name__)
        test()
        logging.info("✓ %s", test.__name__)
    logging.info("Go/MiniCPM test results: %d passed, 0 failed", len(tests))


if __name__ == "__main__":
    run_all_tests()
