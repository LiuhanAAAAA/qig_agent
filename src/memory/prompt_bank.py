# src/prompt_bank.py
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import uuid

from sentence_transformers import SentenceTransformer
import numpy as np

from src.memory.vector_index import FaissIndex
import json
from collections import Counter
from typing import Dict, Any


class PromptBank:
    def __init__(self, db_path: str):
        p = Path(db_path)
        if p.suffix == "" or p.is_dir():
            p = p / "prompt_bank.sqlite"
        self.db_path = str(p)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

        # embedding 模型
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # faiss index
        self.index_path = self.db_path + ".faiss"
        self.dim = 384
        self.index = FaissIndex(self.dim, self.index_path)

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS prompt_bank (
            id TEXT PRIMARY KEY,
            task_name TEXT,
            query TEXT,
            prompt TEXT,
            score REAL,
            failure_tags TEXT,
            gen_params TEXT,
            fixed_prompt TEXT
        )
        """)
        self.conn.commit()

    def _embed(self, text: str) -> np.ndarray:
        v = self.embedder.encode([text], normalize_embeddings=True)
        return v.astype("float32")

    def upsert_record(
        self,
        task_name: str,
        query: str,
        prompt: str,
        score: float,
        failure_tags: str = "",
        gen_params: str = "{}",
        fixed_prompt: str = ""
    ):
        _id = str(uuid.uuid4())

        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO prompt_bank (id, task_name, query, prompt, score, failure_tags, gen_params, fixed_prompt) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_id, task_name, query, prompt, float(score), failure_tags, gen_params, fixed_prompt)
        )
        self.conn.commit()

        vec = self._embed(query)
        self.index.add(vecs=vec, ids=[_id])

    def retrieve_similar(self, task_name: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        qv = self._embed(query)
        ids = self.index.search(qv, top_k=top_k)
        if not ids:
            return []

        cur = self.conn.cursor()
        out = []
        for _id in ids:
            cur.execute("SELECT id, task_name, query, prompt, score, failure_tags, gen_params, fixed_prompt FROM prompt_bank WHERE id=?", (_id,))
            row = cur.fetchone()
            if not row:
                continue
            if row[1] != task_name:
                continue
            out.append({
                "id": row[0],
                "task_name": row[1],
                "query": row[2],
                "prompt": row[3],
                "score": row[4],
                "failure_tags": row[5] or "",
                "gen_params": row[6] or "{}",
                "fixed_prompt": row[7] or ""
            })
        return out


    def stats_for_task(self, task_name: str) -> Dict[str, Any]:
        """
        统计某个 task 下 failure_tags 的分布，用于 prior avoidance / policy agent。
        返回：
          {
            "total": int,
            "tag_counts": {tag: count},
            "top_tags": [(tag, count), ...]
          }
        """
        cur = self.conn.cursor()
        cur.execute(
            "SELECT failure_tags FROM prompt_bank WHERE task_name=?",
            (task_name,)
        )
        rows = cur.fetchall()

        counter = Counter()
        total = 0
        for (tags_str,) in rows:
            total += 1
            if not tags_str:
                continue

            # failure_tags 可能是 "a,b,c" 或 json list，做兼容
            tags = []
            s = tags_str.strip()
            if not s:
                continue

            if s.startswith("[") and s.endswith("]"):
                try:
                    tags = json.loads(s)
                except Exception:
                    tags = []
            else:
                tags = [t.strip() for t in s.split(",") if t.strip()]

            for t in tags:
                counter[t] += 1

        tag_counts = dict(counter)
        top_tags = counter.most_common(10)

        return {
            "total": total,
            "tag_counts": tag_counts,
            "top_tags": top_tags
        }



