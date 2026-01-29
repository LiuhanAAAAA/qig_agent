import os
from src.task_spec import TaskSpec
from src.utils.io import make_run_dir
from src.utils.logger import log_info
from src.graph import build_graph

def main():
    spec = TaskSpec.load("configs/miaotu_avatar.yaml").raw
    run_dir = make_run_dir()

    log_info(f"Run dir = {run_dir}")

    graph = build_graph()

    init_state = {
        "run_dir": run_dir,
        "task_spec": spec,
        "user_query": "healing cute avatar for a pet-themed account",
        "retrieved_prompts": [],
        "prompt_candidates": [],
        "prompt_eval": [],
        "selected_prompts": [],
        "images": [],
        "image_eval": [],
        "accepted_topk": [],
        "iter": 0,
        "done": False,
        "logs": []
    }

    final = graph.invoke(init_state)
    log_info("############### Finished. Check outputs in runs/ folder.")
    log_info(f"TopK = {len(final.get('accepted_topk', []))}")

if __name__ == "__main__":
    main()
