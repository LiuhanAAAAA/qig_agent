import os
import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []

        if os.path.exists(path):
            self.load()

    def add(self, vecs: np.ndarray, ids):
        self.index.add(vecs.astype("float32"))
        self.ids.extend(ids)
        self.save()

    def search(self, q: np.ndarray, top_k: int):
        D, I = self.index.search(q.astype("float32"), top_k)
        result_ids = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.ids):
                continue
            result_ids.append(self.ids[idx])
        return result_ids

    def save(self):
        faiss.write_index(self.index, self.path)
        # 保存 ids
        with open(self.path + ".ids", "w", encoding="utf-8") as f:
            for _id in self.ids:
                f.write(str(_id) + "\n")

    def load(self):
        self.index = faiss.read_index(self.path)
        self.ids = []
        with open(self.path + ".ids", "r", encoding="utf-8") as f:
            for line in f:
                self.ids.append(line.strip())
