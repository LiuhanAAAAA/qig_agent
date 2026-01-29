# src/graph.py
import os
import uuid
from typing import Dict, Any, List

from langgraph.graph import StateGraph, START, END

from src.generators.prompt_generator import generate_prompt_candidates
from src.generators.prompt_mutation import mutate_prompt
from src.generators.image_generator import ImageGenerator

from src.evaluators.prompt_evaluator import eval_prompts
from src.evaluators.image_evaluator import eval_images
from src.evaluators.taxonomy import autofix_prompt

from src.memory.prompt_bank import PromptBank
from src.utils.logger import log_info, log_warn
from src.utils.io import append_jsonl
from src.utils.excel_export import export_urls_to_excel
from pathlib import Path
from pathlib import Path
from src.memory.prompt_bank import PromptBank
from src.utils.logger import log_info



# -------------------------
# Nodes
# -------------------------

def node_retrieve_memory(state: dict) -> dict:
    spec = state["task_spec"]

    # 需要 sqlite 文件路径
    db_path = str(Path(state["run_dir"]) / "prompt_bank.sqlite")
    bank = PromptBank(db_path)

    topk = int(spec.get("memory", {}).get("top_k_retrieve", 3))
    if not spec.get("memory", {}).get("enabled", True):
        state["retrieved_prompts"] = []
        return state

    task_name = spec.get("task_name", "default")
    query = state["user_query"]

    retrieved = bank.retrieve_similar(task_name=task_name, query=query, top_k=topk)

    state["retrieved_prompts"] = retrieved
    log_info(f"[INFO] Memory retrieved {len(retrieved)} prompts")
    return state



def node_generate_prompts(state: Dict[str, Any]) -> Dict[str, Any]:
    spec = state["task_spec"]
    retrieved = state.get("retrieved_prompts", [])

    k = int(spec.get("generation", {}).get("prompt_candidates_k", 6))

    # Expand Agent + Policy Agent（失败统计驱动先验规避）
    prompts = generate_prompt_candidates(
        spec=spec,
        user_query=state["user_query"],
        k=k,
        retrieved=retrieved,
        run_dir=state["run_dir"],
    )

    # Prompt mutation（可选）
    mut_cfg = spec.get("prompt_mutation", {})
    if mut_cfg.get("enabled", False):
        m = int(mut_cfg.get("mutations_per_prompt", 2))
        mutated = []
        for p in prompts:
            mutated.extend(mutate_prompt(p, spec, m=m))
        prompts = prompts + mutated

    state["prompt_candidates"] = prompts
    log_info(f"Generated prompt candidates: {len(prompts)}")
    return state


def node_eval_prompts(state: Dict[str, Any]) -> Dict[str, Any]:
    spec = state["task_spec"]
    pe = eval_prompts(spec, state["prompt_candidates"])
    state["prompt_eval"] = pe
    return state


def node_select_prompts(state: Dict[str, Any]) -> Dict[str, Any]:
    spec = state["task_spec"]
    th = float(spec.get("thresholds", {}).get("prompt_min_score", 0.60))

    good = [x for x in state["prompt_eval"] if x["score"] >= th]
    good = sorted(good, key=lambda x: x["score"], reverse=True)


    selected = [x.get("rewrite", x["prompt"]) for x in good[: int(spec["generation"]["prompt_candidates_k"])]]

    if not selected:
        # fallback：至少留一个（也要保证有 Negative prompt）
        selected = [state["prompt_eval"][0].get("rewrite", state["prompt_candidates"][0])]

    state["selected_prompts"] = selected
    log_info(f"Selected prompts: {len(selected)}")
    return state


def node_generate_images(state: Dict[str, Any]) -> Dict[str, Any]:
    spec = state["task_spec"]
    gen = ImageGenerator(spec)
    
    # gen = ImageGenerator()
    out_dir = os.path.join(state["run_dir"], "images")
    os.makedirs(out_dir, exist_ok=True)

    gen_cfg = spec.get("generation", {})
    n = int(gen_cfg.get("images_per_prompt_n", gen_cfg.get("images_per_prompt", 1)))

    images = []
    for idx, prompt in enumerate(state["selected_prompts"]):
        subdir = os.path.join(out_dir, f"p{idx}")
        os.makedirs(subdir, exist_ok=True)
        paths = gen.generate(prompt, out_dir=subdir, n=n)

        for p in paths:
            images.append({"prompt": prompt, "image_path": p})

    state["images"] = images
    log_info(f"Generated images: {len(images)}")
    return state


def node_eval_images(state: Dict[str, Any]) -> Dict[str, Any]:
    spec = state["task_spec"]

    eval_rows = []
    for row in state["images"]:
        prompt = row["prompt"]
        image_path = row["image_path"]
        # eval_images 返回 list[dict]，这里输入单张，所以取 [0]
        res = eval_images(spec, prompt, [image_path])[0]
        eval_rows.append({
            "prompt": prompt,
            "image_path": image_path,
            "score": float(res["score"]),
            "tags": res["tags"],
            "hard_fail": bool(res["hard_fail"]),
            "metrics": res["metrics"],
        })

    state["image_eval"] = eval_rows
    return state


def node_decide_loop(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    如果 top1 分数不够 → 自动修复 prompt 再循环
    """
    spec = state["task_spec"]
    min_score = float(spec.get("thresholds", {}).get("image_min_score", 0.68))

    good = [x for x in state["image_eval"] if not x["hard_fail"]]
    top1 = max(good, key=lambda x: x["score"]) if good else None

    if top1 and top1["score"] >= min_score:
        state["done"] = True
        return state

    if state["iter"] + 1 >= int(spec.get("generation", {}).get("max_iters", 2)):
        state["done"] = True
        return state

    state["iter"] += 1
    state["done"] = False

    if top1:
        fixed = autofix_prompt(spec, top1["prompt"], top1["tags"])
        # 下一轮优先尝试 fixed prompt
        state["prompt_candidates"] = [fixed] + state.get("prompt_candidates", [])
        log_info(f"AutoFix applied, iter={state['iter']} (tags={top1['tags']})")
    else:
        log_warn("All images hard-failed, will retry with new prompts.")

    return state


def node_accept_and_store(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    选 topK，并写入 memory + results.jsonl
    """
    spec = state["task_spec"]
    run_dir = state["run_dir"]

    candidates = [x for x in state["image_eval"] if not x["hard_fail"]]
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    topk = candidates[: int(spec.get("generation", {}).get("final_topk", 8))]
    state["accepted_topk"] = topk

    results_path = os.path.join(run_dir, "results.jsonl")
    img_th = float(spec.get("thresholds", {}).get("image_min_score", 0.68))

    for i, r in enumerate(topk):
        # accepted label：用于 reward calibrator 训练
        accepted = (not r["hard_fail"]) and (r["score"] >= img_th)

        append_jsonl(results_path, {
            "rank": i + 1,
            "score": float(r["score"]),
            "accepted": bool(accepted),
            "prompt": r["prompt"],
            "image_path": r["image_path"],
            "tags": r["tags"],
            "metrics": r["metrics"],
        })

    # 写入 memory（存高分，并记录 failure tags / 修复历史）
    mem_cfg = spec.get("memory", {})
    if mem_cfg.get("enabled", False):
        # bank = PromptBank(run_dir)
        bank = PromptBank(str(Path(run_dir) / "prompt_bank.sqlite"))

        task_name = spec.get("task_name", "default")
        min_store = float(mem_cfg.get("min_score_to_store", 0.70))

        for r in topk:
            if r["score"] >= min_store:
                bank.add(
                    task_name=task_name,
                    prompt=r["prompt"],
                    score=float(r["score"]),
                    failure_tags=r["tags"],
                    gen_params={"model": "sdxl", "n": spec.get("generation", {}).get("images_per_prompt_n", 1)},
                    fixed_prompt="", 
                )

    log_info(f"Accepted topK: {len(topk)}")
    return state


def node_export_excel(state: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = state["run_dir"]
    rows = []
    for i, r in enumerate(state["accepted_topk"]):
        rows.append({
            "rank": i + 1,
            "score": float(r["score"]),
            "image_path": r["image_path"],
            "prompt": r["prompt"],
            "tags": ",".join(r.get("tags", [])),
        })
    out_path = os.path.join(run_dir, "deliver_urls.xlsx")
    export_urls_to_excel(rows, out_path)
    log_info(f"Exported Excel: {out_path}")
    return state


def route_after_decide(state: Dict[str, Any]) -> str:
    if state.get("done", False):
        return "accept_store"
    return "generate_prompts"


# -------------------------
# Graph
# -------------------------
def build_graph():
    g = StateGraph(dict)

    g.add_node("retrieve_memory", node_retrieve_memory)
    g.add_node("generate_prompts", node_generate_prompts)
    g.add_node("eval_prompts", node_eval_prompts)
    g.add_node("select_prompts", node_select_prompts)
    g.add_node("generate_images", node_generate_images)
    g.add_node("eval_images", node_eval_images)
    g.add_node("decide_loop", node_decide_loop)
    g.add_node("accept_store", node_accept_and_store)
    g.add_node("export_excel", node_export_excel)

    g.add_edge(START, "retrieve_memory")
    g.add_edge("retrieve_memory", "generate_prompts")
    g.add_edge("generate_prompts", "eval_prompts")
    g.add_edge("eval_prompts", "select_prompts")
    g.add_edge("select_prompts", "generate_images")
    g.add_edge("generate_images", "eval_images")
    g.add_edge("eval_images", "decide_loop")

    g.add_conditional_edges("decide_loop", route_after_decide)

    g.add_edge("accept_store", "export_excel")
    g.add_edge("export_excel", END)

    return g.compile()
