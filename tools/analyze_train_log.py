import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics as stats

LOG_PATH = r"D:\LiuHanProject\qig_agent\runs\gepa_ppo\20260123_151159\train_log.jsonl"  

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def main():
    rows = read_jsonl(LOG_PATH)
    print(f"[Loaded] rows={len(rows)}")

    # overall metrics
    rewards = [r["reward"] for r in rows]
    prompt_scores = [r.get("prompt_score") for r in rows]
    image_scores = [r.get("image_score") for r in rows]

    hard_fail = [r.get("hard_fail", False) for r in rows]
    hard_fail_rate = sum(1 for x in hard_fail if x) / max(1, len(rows))

    image_eval_rate = sum(1 for x in image_scores if x is not None) / max(1, len(rows))

    print("\n=== Overall ===")
    print(f"mean_reward      = {safe_mean(rewards):.4f}")
    print(f"mean_prompt_score= {safe_mean(prompt_scores):.4f}")
    print(f"mean_image_score = {safe_mean([x for x in image_scores if x is not None]) or 0:.4f}")
    print(f"hard_fail_rate   = {hard_fail_rate:.2%}")
    print(f"image_eval_rate  = {image_eval_rate:.2%}")

    # best per query
    best_by_query = {}
    for r in rows:
        q = r["user_query"]
        if q not in best_by_query or r["reward"] > best_by_query[q]["reward"]:
            best_by_query[q] = r

    print("\n=== Best per user_query ===")
    for q, r in best_by_query.items():
        print("\n---")
        print("query:", q)
        print("best_reward:", r["reward"])
        print("prompt_score:", r["prompt_score"], "image_score:", r["image_score"])
        print("prompt:", r["prompt"].replace("\n", " ")[:300])
        if r.get("img_paths"):
            print("img:", r["img_paths"][0])

    # tag stats
    tag_counter = Counter()
    for r in rows:
        for t in r.get("tags_prompt", []):
            tag_counter["prompt:" + t] += 1
        for t in r.get("tags_image", []):
            tag_counter["image:" + t] += 1

    print("\n=== Top tags ===")
    for k, v in tag_counter.most_common(20):
        print(f"{k:30s} {v}")

if __name__ == "__main__":
    main()
