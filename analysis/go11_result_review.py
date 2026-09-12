"""Derive descriptive score tails and seed dispersion from verified Go 11 runs."""

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis" / "results"
SEEDS = (701, 709, 719, 727, 733)
METHODS = (
    "mixed_ogd3", "selfplay_ogd3", "selfplay_cone3",
    "selfplay_cone6", "selfplay_free6", "frozen6",
)


def main() -> None:
    """Check result provenance and write post hoc summaries without new games."""
    inputs: dict[str, str] = {}

    def read(name: str) -> dict:
        path = RESULTS / name
        content = path.read_bytes()
        inputs[name] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    summary = read("go11_summary.json")
    assert summary["verified"]
    for name, digest in summary["source"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    methods = {}
    initial = {}
    for method in METHODS:
        games = []
        seed_wins = []
        initial[method] = []
        for seed in SEEDS:
            run = read(f"go11_{method}_seed{seed}.json")
            assert run["completed"] and run["source"] == summary["source"]
            rows = [g for g in run["games"] if g["phase"] == "search_eval"]
            assert len(rows) == 48
            for game in rows:
                assert game["terminated"] and game["consecutive_passes"] == 2
                assert bool(game["win"]) == (game["margin"] > 0)
                assert game["margin"] == game["score"]["margin_black"] * game["student_color"]
            initial[method].append((run["base_sha256"], run["probes"]["before"]))
            games.extend(rows)
            seed_wins.append(sum(int(g["win"]) for g in rows))
        wins = [g["margin"] for g in games if g["win"]]
        losses = [g["margin"] for g in games if not g["win"]]
        table = summary["tables"][method]["phases"]["search_eval"]
        assert table["wins"] == len(wins) and table["wins_per_seed"] == seed_wins
        assert abs(table["margin"] - statistics.mean(g["margin"] for g in games)) < 1e-12
        methods[method] = {
            "games": len(games), "wins": len(wins), "losses": len(losses),
            "wins_per_seed": seed_wins,
            "sample_sd_of_seed_win_rates_pp": statistics.stdev(n / 48 for n in seed_wins) * 100,
            "mean_winning_margin": statistics.mean(wins),
            "mean_losing_margin": statistics.mean(losses),
            "losses_by_at_least_50_points": sum(m <= -50 for m in losses),
            "minimum_margin": min(losses),
        }
    for group in (METHODS[:3], METHODS[3:]):
        assert all(initial[m] == initial[group[0]] for m in group)
    output = {
        "verified": True,
        "scope": "Post hoc descriptive score tails and sample dispersion; no new games, training, or configuration selection.",
        "methods": methods,
        "initialization_and_before_probes_identical_within_ring_count": True,
        "input_sha256": inputs,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (RESULTS / "go11_result_review.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"verified": True, "methods": methods}))


if __name__ == "__main__":
    main()
