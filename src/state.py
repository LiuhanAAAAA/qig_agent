from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    run_dir: str
    task_spec: Dict[str, Any]

    user_query: str

    retrieved_prompts: List[Dict[str, Any]]
    prompt_candidates: List[str]
    prompt_eval: List[Dict[str, Any]]
    selected_prompts: List[str]

    images: List[Dict[str, Any]]
    image_eval: List[Dict[str, Any]]
    accepted_topk: List[Dict[str, Any]]

    iter: int
    done: bool
    logs: List[Dict[str, Any]]
