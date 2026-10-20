"""Audit the separate Go13 task-order follow-up, including every full game."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import sys
import time

# Match the experiment's BLAS load order for exact float32 replay. Importing
# NumPy first changes a diagnostic dot-product reduction by about 2.4e-7.
import torch
import numpy as np
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import go13_algorithm as main
from experiments import go13_transfer as experiment
from analysis.go13_verify import verify_game, paired
from mqr import GoBoard, HeuristicGoTeacher
from mqr.go import DefensiveGoTeacher
from mqr.go_online import GoOnlineSession
from mqr.go_research import history_output, queried_logits
from mqr.go_search import policy_value_search

OUT = ROOT / 'analysis/results'


def load(name):
    return json.loads((OUT / (name + '.json')).read_text())


def signatures(records):
    return {repr((main.boards(r)[-1].board, main.boards(r)[-1].to_play, r['query'])) for r in records}


def audit_seed(seed):
    torch.set_num_threads(1)
    path = experiment.DIRECTORY / f'prepared-{seed}.pt'
    p = torch.load(path, weights_only=True)
    original = torch.load(ROOT / f'checkpoints/go13_v1/prepared-{seed}.pt', weights_only=True)
    assert p['source'] == experiment.source_hashes()
    assert p['data']['old_train'] == original['data']['b_train']
    assert p['data']['old_test'] == original['data']['b_test']
    previous = set().union(*(signatures(rows) for rows in original['data'].values()))
    assert not previous & signatures(p['data']['new_train'] + p['data']['new_test'])
    assert not signatures(p['data']['new_train']) & signatures(p['data']['new_test'])
    for split, count in (('new_train', 2048), ('new_test', 256)):
        rows = p['data'][split]
        assert len(rows) == count
        batch = main.materialize(rows)
        assert batch.labels.tolist() == [0., 1.] * (count // 2)
        assert torch.equal(batch.features[-1, ::2], batch.features[-1, 1::2])
        for row in rows:
            history = main.boards(row)
            assert len(history) == 2 and len(history[0].position_history) == 1
            for point, color in enumerate(history[0].board):
                if color:
                    assert history[0]._group_and_liberties(history[0].board, point)[1]
    assert 0 < len(p['anchor_indices']) <= 16
    margins = []
    for geometry in ('orth', 'similar'):
        parent_path = ROOT / f'checkpoints/go13_v1/learned-{geometry}_free-{seed}.pt'
        parent = torch.load(parent_path, weights_only=True)
        assert main.sha(parent_path) == p['parent_checkpoints'][geometry]
        agent, _ = main.build(seed, geometry)
        agent.load_state_dict(p['models'][geometry])
        assert main.parameter_hash(agent) == parent['result']['final_parameter_hash']
        optimizer = p['optimizers'][geometry]
        assert optimizer['param_groups'] == parent['learner']['optimizer']['param_groups']
        for key, state in optimizer['state'].items():
            assert int(state['step']) == 192
            for name, value in state.items():
                torch.testing.assert_close(value, parent['learner']['optimizer']['state'][key][name], rtol=0, atol=0)
        old = main.materialize(p['data']['old_train'])
        prediction = torch.tensor(main.evaluate(agent, old)['logits'])
        margin = (2 * old.labels - 1) * prediction
        margins.append(margin)
        assert bool((margin[p['anchor_indices']] > .02).all())
        torch.testing.assert_close(p['anchor_floors'][geometry], .5 * margin[p['anchor_indices']], rtol=0, atol=0)
    minimum = torch.stack(margins).amin(0).reshape(-1, 2).amin(1)
    pairs = [int(i) for i in minimum.argsort(descending=True) if minimum[i] > .02][:8]
    assert p['anchor_indices'] == [2 * i + j for i in pairs for j in (0, 1)]
    return {'seed': seed, 'verified': True, 'anchors': len(p['anchor_indices']),
            'prepared_sha256': main.sha(path), 'optimizer_steps_carried': 192}


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def audit_run(method, seed):
    torch.set_num_threads(1)
    learning_path = OUT / f'go13_transfer_learning_{method}_seed{seed}.json'
    games_path = OUT / f'go13_transfer_games_{method}_seed{seed}.json'
    run = json.loads(learning_path.read_text())
    games = json.loads(games_path.read_text())
    assert run['completed'] and games['completed']
    assert run['source'] == games['source'] == experiment.source_hashes()
    assert run['seed'] == games['seed'] == seed and run['method'] == games['method'] == method
    prepared, agent, encoder, batches, learner = experiment.restore(seed, method)
    initial_hash = main.parameter_hash(agent)
    for task in ('old', 'new'):
        assert main.evaluate(agent, batches[task + '_test']) == run['before'][task]
    first = learner.step(batches['new_train'].subset(slice(0, 16)))
    expected_first = run['updates'][0]
    assert first == expected_first, (method, seed, {
        key: (first[key], expected_first[key]) for key in first if first[key] != expected_first[key]})
    assert len(run['updates']) == 128 and sum(u['feedback'] for u in run['updates']) == 2048
    for index, update in enumerate(run['updates']):
        logits = torch.tensor(update['pre_logits'])
        labels = batches['new_train'].labels[index * 16:index * 16 + 16]
        assert len(logits) == 16
        assert abs(float(F.binary_cross_entropy_with_logits(logits, labels)) - update['pre_loss']) < 1e-6
        assert float(((logits >= 0) == labels.bool()).float().mean()) == update['pre_accuracy']
        assert update['step_norm'] <= update['raw_step_norm'] + 1e-5
        assert update['backtracks'] <= 8
        if method.endswith(('_current', '_stale')):
            assert update['actual_anchor_violations'] == 0
    path = experiment.DIRECTORY / f'learned-{method}-{seed}.pt'
    assert main.sha(path) == run['checkpoint_sha256'] == games['checkpoint_sha256']
    saved = torch.load(path, weights_only=True)
    learner.load_state_dict(saved['learner'])
    assert main.parameter_hash(agent) == run['parameter_hash'] == games['parameter_hash']
    if method.endswith('_frozen'):
        assert main.parameter_hash(agent) == initial_hash
    expected_steps = 192 if method.endswith('_frozen') else 320
    assert all(int(s['step']) == expected_steps for s in saved['learner']['optimizer']['state'].values())
    for task in ('old', 'new'):
        assert main.evaluate(agent, batches[task + '_test']) == run['after'][task]
    for intervention in ('reset', 'swap', 'no_query'):
        assert main.evaluate(agent, batches['new_test'], intervention=intervention) == run['interventions'][intervention]
    assert run['interventions']['reset']['accuracy'] == .5
    anchors = batches['old_train'].subset(prepared['anchor_indices'])
    with torch.no_grad():
        margins = (2 * anchors.labels - 1) * queried_logits(history_output(agent, anchors.features), anchors.queries)
    torch.testing.assert_close(margins, torch.tensor(run['anchor_margins_after']), rtol=0, atol=0)
    assert int((margins < torch.tensor(run['anchor_floors']) - 1e-5).sum()) == run['anchor_violations']
    assert len(games['games']) == 24
    session = GoOnlineSession(agent, encoder, update_every=16, project_with_memory=False)
    reproduced = 0
    query_changes = 0
    for index, game in enumerate(games['games']):
        assert game['index'] == index and game['student_color'] == (1 if index % 2 == 0 else -1)
        assert game['opening'] == main.prior.opening(seed + 111000 + index // 2, 10, (4, 24, 60)[(index // 2) % 3])
        assert game['moves'][:len(game['opening'])] == game['opening']
        teacher = DefensiveGoTeacher() if index // 2 % 2 else HeuristicGoTeacher()
        verify_game(game, teacher=teacher, color=game['student_color'])
        if index < 2:
            session.reset_game()
            board = GoBoard(10, komi=5.5)
            for ply, action in enumerate(game['moves']):
                observation = session.observe(board, learn=False)
                if ply >= len(game['opening']) and board.to_play == game['student_color']:
                    result = policy_value_search(agent, encoder, board, observation['state'],
                                                 observation['policy_logits'], simulations=16, max_depth=8)
                    assert result['action'] == action
                    with torch.no_grad():
                        state = observation['state']
                        no_query = agent._query_output(agent.core.readout_state(state), state,
                                                       encoder.encode_board(board), include_query=False)
                        query_changes += int(int(no_query.policy_logits.argmax()) != observation['raw_action'])
                    reproduced += 1
                board.play(action)
    assert main.parameter_hash(agent) == run['parameter_hash']
    audit = {'seed': seed, 'method': method, 'verified': True, 'complete_games': 24,
             'matches_reproduced': 2, 'decisions_reproduced': reproduced,
             'raw_query_action_changes_in_reproduced_decisions': query_changes,
             'constraint_tensor_bytes': tensor_bytes(saved['learner']['constraint']),
             'optimizer_tensor_bytes': tensor_bytes(saved['learner']['optimizer']),
             'learning_sha256': main.sha(learning_path), 'games_sha256': main.sha(games_path),
             'verifier_sha256': main.sha(__file__)}
    main.prior.save_json(audit, OUT / f'go13_transfer_audit_{method}_seed{seed}.json')
    return audit


def summarize(audits, seed_audits):
    table = {}
    runs_by_method = {}
    for method in main.METHODS:
        runs = [load(f'go13_transfer_learning_{method}_seed{seed}') for seed in main.SEEDS]
        gs = [load(f'go13_transfer_games_{method}_seed{seed}') for seed in main.SEEDS]
        runs_by_method[method] = runs
        mean = lambda fn: float(np.mean([fn(r) for r in runs]))
        updates = [u for r in runs for u in r['updates']]
        games = [g for row in gs for g in row['games']]
        table[method] = {
            'old_before_accuracy': mean(lambda r: r['before']['old']['accuracy']),
            'old_after_accuracy': mean(lambda r: r['after']['old']['accuracy']),
            'old_forgetting_loss': mean(lambda r: r['after']['old']['loss'] - r['before']['old']['loss']),
            'new_before_accuracy': mean(lambda r: r['before']['new']['accuracy']),
            'new_after_accuracy': mean(lambda r: r['after']['new']['accuracy']),
            'new_learning_gain_loss': mean(lambda r: r['before']['new']['loss'] - r['after']['new']['loss']),
            'new_pair_accuracy': mean(lambda r: r['after']['new']['both_correct'] / r['after']['new']['pairs']),
            'prequential_new_loss': float(np.mean([u['pre_loss'] for u in updates])),
            'attempted_updates': 0 if method.endswith('_frozen') else len(updates),
            'accepted_updates': sum(u['accepted'] for u in updates),
            'active_constraint_updates': sum(u['active'] > 0 for u in updates),
            'raw_anchor_violation_updates': sum(u['raw_anchor_violations'] > 0 for u in updates),
            'actual_anchor_violations': sum(r['anchor_violations'] for r in runs),
            'backtracked_updates': sum(u['backtracks'] > 0 for u in updates),
            'anchors': sum(len(r['anchor_indices']) for r in runs),
            'mean_step_norm': float(np.mean([u['step_norm'] for u in updates])),
            'mean_raw_step_norm': float(np.mean([u['raw_step_norm'] for u in updates])),
            'mean_first_order_max_error': float(np.mean([u['first_order_max_error'] for u in updates])),
            'mean_normal_age': float(np.mean([u['normal_age'] for u in updates])),
            'constraint_refreshes': sum(r['constraint_refreshes'] for r in runs),
            'learning_seconds': sum(r['seconds'] for r in runs),
            'wins': sum(g['win'] for g in games), 'games': len(games),
            'mean_margin': float(np.mean([g['margin'] for g in games])),
            'per_seed': [{'seed': r['seed'], 'old_before': r['before']['old']['accuracy'],
                          'old_after': r['after']['old']['accuracy'], 'new_after': r['after']['new']['accuracy'],
                          'wins': sum(g['win'] for g in gr['games'])} for r, gr in zip(runs, gs)]}
        for intervention in ('reset', 'swap', 'no_query'):
            table[method][intervention + '_accuracy'] = mean(lambda r: r['interventions'][intervention]['accuracy'])
    comparisons = {}
    metrics = {
        'old_retention_accuracy_change': lambda r: r['after']['old']['accuracy'] - r['before']['old']['accuracy'],
        'new_learning_accuracy_change': lambda r: r['after']['new']['accuracy'] - r['before']['new']['accuracy'],
        'old_forgetting_loss': lambda r: r['after']['old']['loss'] - r['before']['old']['loss'],
        'new_learning_gain_loss': lambda r: r['before']['new']['loss'] - r['after']['new']['loss']}
    for left, right in (('orth_current', 'orth_free'), ('similar_current', 'similar_free'),
                        ('orth_current', 'similar_current'), ('orth_free', 'similar_free'),
                        ('orth_current', 'orth_stale'), ('orth_current', 'orth_norm')):
        comparisons[left + '_minus_' + right] = {
            metric: paired([fn(a) - fn(b) for a, b in zip(runs_by_method[left], runs_by_method[right])])
            for metric, fn in metrics.items()}
    result = {'verified': True, 'source': experiment.source_hashes(), 'verifier_sha256': main.sha(__file__),
              'table': table, 'comparisons': comparisons, 'seed_audits': seed_audits, 'run_audits': audits,
              'complete_games': sum(a['complete_games'] for a in audits),
              'matches_reproduced': sum(a['matches_reproduced'] for a in audits),
              'decisions_reproduced': sum(a['decisions_reproduced'] for a in audits),
              'primary_results_replaced': False, 'same_seed_cohort': True}
    assert result['complete_games'] == 840
    main.prior.save_json(result, OUT / 'go13_transfer_summary.json')
    print(json.dumps({'verified': True, 'complete_games': 840, 'decisions_reproduced': result['decisions_reproduced']}), flush=True)


def run(workers):
    locked = load('go13_transfer_source')
    assert locked['source'] == experiment.source_hashes()
    pending = {(m, s) for m in main.METHODS for s in main.SEEDS}
    running = {}
    audits = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            for key in sorted(pending):
                if len(running) >= workers:
                    break
                method, seed = key
                if not (OUT / f'go13_transfer_games_{method}_seed{seed}.json').exists():
                    continue
                path = OUT / f'go13_transfer_audit_{method}_seed{seed}.json'
                if path.exists():
                    cached = json.loads(path.read_text())
                    if (cached['verifier_sha256'] == main.sha(__file__)
                            and cached['games_sha256'] == main.sha(OUT / f'go13_transfer_games_{method}_seed{seed}.json')
                            and cached['learning_sha256'] == main.sha(OUT / f'go13_transfer_learning_{method}_seed{seed}.json')):
                        audits.append(cached)
                        pending.remove(key)
                        continue
                running[pool.submit(audit_run, *key)] = key
                pending.remove(key)
            if running:
                done, _ = wait(running, timeout=2, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    audits.append(row)
                    running.pop(future)
                    print(json.dumps({'audited': row['method'], 'seed': row['seed'],
                                      'decisions': row['decisions_reproduced']}), flush=True)
            else:
                time.sleep(2)
    summarize(audits, [audit_seed(seed) for seed in main.SEEDS])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    run(parser.parse_args().workers)
