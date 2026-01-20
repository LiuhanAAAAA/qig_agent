# tools/summarize_run.py
import json
from pathlib import Path
from collections import Counter
import statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_results_jsonl(run_dir: Path) -> Path:
    # ✅ 自动找 results*.jsonl
    cands = sorted(run_dir.glob("*.jsonl"))
    for p in cands:
        if p.name.startswith("results") and p.suffix == ".jsonl":
            return p
    raise FileNotFoundError(f"No jsonl found in {run_dir}")


def read_jsonl(p: Path):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    # 你只需要改这里的 run_dir
    RUN_DIR = PROJECT_ROOT / "runs" / "20260116_111541"
    results_file = find_results_jsonl(RUN_DIR)
    print(f"Using results file: {results_file}")

    rows = read_jsonl(results_file)
    if not rows:
        print("Empty results.jsonl")
        return

    # 兼容：没有 accepted 就默认 rank<=999
    accepted_rows = []
    for r in rows:
        if "accepted" in r:
            if r["accepted"]:
                accepted_rows.append(r)
        else:
            accepted_rows.append(r)

    scores = [float(r.get("score", 0.0)) for r in rows]
    cal_scores = [float(r.get("calibrated_score", -1.0)) for r in rows if "calibrated_score" in r]

    # failure tags
    tag_counter = Counter()
    for r in rows:
        for t in (r.get("tags") or []):
            tag_counter[str(t)] += 1

    print("=== Run Summary ===")
    print(f"run_dir: {RUN_DIR}")
    print(f"num_rows: {len(rows)}")
    print(f"accepted: {len(accepted_rows)}")
    print(f"accept_rate: {len(accepted_rows) / max(1, len(rows)):.3f}")

    print("\n=== Scores ===")
    print(f"score_mean: {statistics.mean(scores):.4f}")
    print(f"score_min : {min(scores):.4f}")
    print(f"score_max : {max(scores):.4f}")

    if cal_scores:
        print(f"calibrated_mean: {statistics.mean(cal_scores):.4f}")
        print(f"calibrated_min : {min(cal_scores):.4f}")
        print(f"calibrated_max : {max(cal_scores):.4f}")
    else:
        print("calibrated_score: (not available)")

    print("\n=== Failure Tag Distribution ===")
    if tag_counter:
        for k, v in tag_counter.most_common(30):
            print(f"{k:<20} {v}")
    else:
        print("(none)")

    # prompt length
    prompt_lens = [len((r.get("prompt") or "")) for r in rows]
    print("\n=== Prompt Length (chars) ===")
    print(f"mean: {statistics.mean(prompt_lens):.1f}")
    print(f"min : {min(prompt_lens)}")
    print(f"max : {max(prompt_lens)}")


if __name__ == "__main__":
    main()
