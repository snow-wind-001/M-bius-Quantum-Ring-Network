"""Read-only checkpoint interventions; no fitting or replacement of formal results."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go_memory_research import build
from mqr.agent import GoMultiHeadOutput
from mqr.go import GoBoard, HeuristicGoTeacher
from mqr.go_agent import generate_go_agent_trajectories

SEEDS = (401, 409, 419, 431, 443)
METHODS = ('transport1', 'transport1_guard')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zero_state(agent):
    ref = next(agent.parameters())
    return agent.core.zero_state(1, device=ref.device, dtype=ref.dtype)


def summary(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]} | {'positions': len(rows)} if rows else {}


@torch.no_grad()
def interventions(agent, encoder, board, state):
    x = encoder.encode_board(board)
    full, state = agent._transition(x, state, slow_write=True)
    reset, _ = agent._transition(x, zero_state(agent), slow_write=True)
    p, l, passing, value = agent.spatial_skip_heads(x)
    base = GoMultiHeadOutput(p, l, passing, value.tanh(), x.new_zeros(1, agent.latent_dim),
                            legality_policy_scale=agent.legality_policy_scale)
    no_prior = GoMultiHeadOutput(full.placement_logits, full.legality_logits,
                                full.pass_logit, full.value, full.latent, legality_policy_scale=0.0)
    hidden = agent.spatial_skip_heads.spatial_features(x)
    hidden *= 1 + agent.ring_channel_gain(full.latent)[:, :, None, None]
    modulated_p = agent.spatial_skip_heads.placement(hidden).flatten(1)
    modulated_l = agent.spatial_skip_heads.legality(hidden).flatten(1)
    no_query = GoMultiHeadOutput(modulated_p, modulated_l, full.pass_logit, full.value,
                                full.latent, legality_policy_scale=agent.legality_policy_scale)
    outputs = {'full': full, 'reset_history': reset, 'spatial_base': base,
               'no_learned_legality': no_prior, 'no_position_query': no_query}
    legal = torch.tensor(board.legal_moves())
    target = HeuristicGoTeacher().select_move(board)
    results, actions = {}, {}
    for name, output in outputs.items():
        scores = output.policy_logits[0]
        action = int(legal[scores[legal].argmax()])
        actions[name] = action
        results[name] = {'joint_nll': float(-scores[target]),
                         'teacher_agreement': float(action == target),
                         'raw_legal': float(board.is_legal(int(scores.argmax()))),
                         'pass_chosen': float(action == board.pass_action),
                         'value_abs': float(output.value.abs().max())}
    for name in outputs:
        results[name]['action_differs_from_full'] = float(actions[name] != actions['full'])
    query = agent.point_query(hidden).flatten(2).transpose(1, 2)
    attention = (query @ agent.ring_keys.T / np.sqrt(agent.query_dim)).softmax(-1)
    results['full']['normalized_attention_entropy'] = float((-(attention * attention.clamp_min(1e-30).log()).sum(-1) / np.log(attention.size(-1))).mean())
    results['full']['max_attention_weight'] = float(attention.max(-1).values.mean())
    direct = full.placement_logits - modulated_p
    gain = modulated_p - p
    results['full']['position_query_centered_logit_rms'] = float((direct - direct.mean(-1, keepdim=True)).square().mean().sqrt())
    results['full']['channel_gain_centered_logit_rms'] = float((gain - gain.mean(-1, keepdim=True)).square().mean().sqrt())
    return results, actions, state, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='analysis/results/ccfa_go_engineering_audit.json')
    output_path = Path(parser.parse_args().output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    evidence = {'post_hoc': True, 'training_updates': 0,
                'interpretation': 'Fixed-checkpoint interventions on the original observations; not retrained baselines or new match-win claims.',
                'source_sha256': {str(p): digest(p) for p in sorted(Path('mqr').glob('*.py'))},
                'diagnostic_sha256': digest(Path(__file__)),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'completed': False, 'runs': []}
    teacher = HeuristicGoTeacher()
    for seed in SEEDS:
        formal_path = Path(f'analysis/results/go_constraint_online_seed{seed}.json')
        formal = json.loads(formal_path.read_text())
        for method in METHODS:
            path = Path(f'checkpoints/go_constraint_v1/{method}-{seed}.pt')
            original_hash = digest(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            args = argparse.Namespace(**checkpoint['config'])
            agent, encoder = build(args, 'conditional_guard', seed)
            agent.load_state_dict(checkpoint['session']['agent'])
            agent.eval()
            row = {'seed': seed, 'method': method, 'checkpoint_sha256': original_hash,
                   'formal_sha256': digest(formal_path), 'heldout': {}, 'matches': {}}
            original = next(r for r in formal['runs'] if r['method'] == method)
            for phase, offset, prefix in (('a', 23000, 0), ('b', 24000, 10)):
                data = generate_go_agent_trajectories(args.test_games, seed=seed + offset,
                    size=args.size, komi=2.5, recorded_moves=24,
                    start_random_moves=prefix, random_move_probability=0.5)
                records = defaultdict(list)
                for trace in data:
                    state = zero_state(agent)
                    for example in trace.examples:
                        metrics, _, state, target = interventions(agent, encoder, example.board, state)
                        assert target == example.target_action
                        for name, values in metrics.items(): records[name].append(values)
                row['heldout'][phase] = {name: summary(values) for name, values in records.items()}
                assert abs(row['heldout'][phase]['full']['joint_nll'] - original['after_b'][phase]['joint_nll']) < 2e-6
                assert abs(row['heldout'][phase]['spatial_base']['joint_nll'] - original['before'][phase]['joint_nll']) < 2e-6
            records, tactics = defaultdict(list), defaultdict(list)
            history_effect = defaultdict(int)
            actual_tactics = defaultdict(int)
            predicted_moves = 0
            for game in original['matches']['games']:
                board, state = GoBoard(args.size, komi=2.5), zero_state(agent)
                for ply, played in enumerate(game['moves']):
                    metrics, actions, state, target = interventions(agent, encoder, board, state)
                    fresh = board.copy()
                    fresh.position_history = {fresh.position_key()}
                    history_effect['positions'] += 1
                    history_effect['legal_set_changes_without_history'] += int(board.legal_moves() != fresh.legal_moves())
                    history_effect['teacher_changes_without_history'] += int(target != teacher.select_move(fresh))
                    if board.to_play == game['student_color'] and ply >= 4:
                        assert actions['full'] == played, (seed, method, ply, actions['full'], played)
                        predicted_moves += 1
                        actual_tactics['student_moves'] += 1
                        if target != board.pass_action:
                            _, target_captures, target_liberties = board._simulate_stone(target)
                            actual_tactics['teacher_non_capturing_self_atari'] += int(target_captures == 0 and target_liberties == 1)
                        if played != board.pass_action:
                            actual_tactics['student_placements'] += 1
                            if ply + 1 < len(game['moves']):
                                future = board.copy()
                                future.play(played)
                                future.play(game['moves'][ply + 1])
                                actual_tactics['placed_stone_captured_by_next_reply'] += int(future.board[played] == 0)
                        for name, values in metrics.items(): records[name].append(values)
                        legal = board.legal_moves(include_pass=False)
                        max_captures = max((board._simulate_stone(a)[1] for a in legal), default=0)
                        for name, action in actions.items():
                            captures, liberties = (0, 0) if action == board.pass_action else board._simulate_stone(action)[1:]
                            tactics[name].append({'capture_opportunity': float(max_captures > 0),
                                'missed_available_capture': float(max_captures > 0 and captures == 0),
                                'noncapturing_self_atari': float(action != board.pass_action and captures == 0 and liberties == 1),
                                'pass_while_teacher_places': float(action == board.pass_action and target != board.pass_action),
                                'place_while_teacher_passes': float(action != board.pass_action and target == board.pass_action),
                                'score_below_teacher': float(teacher.score_move(board, action) + 1e-6 < teacher.score_move(board, target)),
                                'plies_at_least_24': float(ply >= 24)})
                    board.play(played)
                assert board.game_over and board.winner() == game['winner']
            row['matches'] = {name: summary(values) | {'tactics': summary(tactics[name])} for name, values in records.items()}
            row['history_dependence'] = dict(history_effect)
            row['actual_tactics'] = dict(actual_tactics)
            row['exact_reproduced_student_moves'] = predicted_moves
            anchors = checkpoint['session']['protection']['memory']['records']
            anchor_rows = []
            for group in anchors:
                for record in group:
                    logp = record['reference_log_probs']
                    anchor_rows.append({'reference_teacher_probability': float(logp[record['action']].exp()),
                                        'reference_raw_argmax_matches_teacher': float(int(logp.argmax()) == record['action']),
                                        'prefix_steps': len(record['features'])})
            row['anchor_reference'] = summary(anchor_rows)
            row['parameter_groups'] = {prefix: sum(p.numel() for name,p in agent.named_parameters() if p.requires_grad and name.startswith(prefix))
                                     for prefix in ('core.angle_controllers.', 'core.input_', 'core.unitary_params.', 'spatial_skip_heads.', 'ring_keys', 'ring_channel_gain.', 'point_correction.')}
            assert digest(path) == original_hash
            evidence['runs'].append(row)
            output_path.write_text(json.dumps(evidence, indent=2) + '\n')
            print(seed, method, 'verified student moves', predicted_moves, 'history', dict(history_effect), flush=True)
    assert len(evidence['runs']) == len(SEEDS) * len(METHODS)
    assert evidence['source_sha256'] == {str(p): digest(p) for p in sorted(Path('mqr').glob('*.py'))}
    evidence['completed'] = True
    output_path.write_text(json.dumps(evidence, indent=2) + '\n')
    print('Completed all 10 immutable-checkpoint interventions', flush=True)


if __name__ == '__main__':
    main()
