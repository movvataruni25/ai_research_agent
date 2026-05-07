import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import KNOWLEDGE_BASE_DIR, TOP_K_RESULTS, CHUNK_SIZE


class RAGPipeline:
    """TF-IDF based retrieval pipeline over a folder of .txt knowledge files."""

    def __init__(self):
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        self.tfidf_matrix = None
        self._load_and_index()

    def _chunk_text(self, text: str, source: str) -> list[tuple]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks, current, current_len = [], [], 0
        for sentence in sentences:
            if current_len + len(sentence) > CHUNK_SIZE and current:
                chunks.append((" ".join(current), source))
                current, current_len = [sentence], len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence)
        if current:
            chunks.append((" ".join(current), source))
        return chunks

    def _load_and_index(self):
        kb_dir = Path(KNOWLEDGE_BASE_DIR)
        if not kb_dir.exists():
            print(f"[RAG] Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
            return

        all_chunks: list[tuple] = []
        for filepath in sorted(kb_dir.glob("*.txt")):
            text = filepath.read_text(encoding="utf-8")
            all_chunks.extend(self._chunk_text(text, filepath.name))

        if not all_chunks:
            print("[RAG] No documents found.")
            return

        self.chunks = [c[0] for c in all_chunks]
        self.sources = [c[1] for c in all_chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        print(f"[RAG] Indexed {len(self.chunks)} chunks from {len(set(self.sources))} file(s).")

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        if not self.chunks or self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            {"content": self.chunks[i], "source": self.sources[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0.01
        ]
