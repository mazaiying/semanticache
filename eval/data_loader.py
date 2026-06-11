"""
Data Loader for SemantiCache Benchmark

Supports:
  - ShareGPT (lmsys/chatbot_arena_conversations or local JSON)
  - LMSYS-Chat-1M
  - RAG Synthetic Dataset (generated locally)

Usage:
    loader = DataLoader("sharegpt", num_requests=1000)
    requests = loader.load()
    # returns List[Dict] with keys: token_ids, semantic_vector, tenant_id, text
"""

import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for SemantiCache benchmarks.
    Falls back to synthetic data if dataset not available.
    """

    SUPPORTED = ["sharegpt", "lmsys", "rag_synthetic", "synthetic"]

    def __init__(
        self,
        dataset: str = "sharegpt",
        num_requests: int = 500,
        seed: int = 42,
        cache_dir: str = "~/.cache/semanticache_data",
        tokenizer_name: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        assert dataset in self.SUPPORTED, f"Dataset must be one of {self.SUPPORTED}"
        self.dataset = dataset
        self.num_requests = num_requests
        self.seed = seed
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_name = tokenizer_name
        self.embedding_model_name = embedding_model
        self._embedder = None
        self._tokenizer = None

    def load(self) -> List[Dict]:
        """Load and return request list."""
        logger.info(f"Loading dataset: {self.dataset} ({self.num_requests} requests)")
        if self.dataset == "sharegpt":
            return self._load_sharegpt()
        elif self.dataset == "lmsys":
            return self._load_lmsys()
        elif self.dataset == "rag_synthetic":
            return self._load_rag_synthetic()
        else:
            return self._load_synthetic()

    # ─────────────────────────────────────────
    # ShareGPT
    # ─────────────────────────────────────────

    def _load_sharegpt(self) -> List[Dict]:
        """Load ShareGPT conversations from HuggingFace."""
        try:
            from datasets import load_dataset
            logger.info("Downloading ShareGPT dataset from HuggingFace...")
            ds = load_dataset(
                "lmsys/lmsys-chat-1m",  # publicly available
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            prompts = []
            for item in ds:
                if len(prompts) >= self.num_requests * 2:
                    break
                conv = item.get("conversation", [])
                if conv and len(conv) > 0:
                    # Take first human turn
                    first_human = next(
                        (m["content"] for m in conv if m.get("role") == "user"), None
                    )
                    if first_human and len(first_human) > 20:
                        prompts.append(first_human)

            logger.info(f"Loaded {len(prompts)} prompts from LMSYS-Chat")
            return self._prompts_to_requests(prompts[: self.num_requests])

        except Exception as e:
            logger.warning(f"Failed to load ShareGPT: {e}. Falling back to synthetic.")
            return self._load_synthetic()

    # ─────────────────────────────────────────
    # LMSYS-Chat-1M
    # ─────────────────────────────────────────

    def _load_lmsys(self) -> List[Dict]:
        """Same as ShareGPT (using lmsys-chat-1m)."""
        return self._load_sharegpt()

    # ─────────────────────────────────────────
    # RAG Synthetic Dataset
    # ─────────────────────────────────────────

    def _load_rag_synthetic(self) -> List[Dict]:
        """
        Generate RAG-style synthetic dataset:
        Same document, different question wordings → high semantic similarity.
        Perfect for testing semantic hit rate.
        """
        rng = random.Random(self.seed)

        # Base documents (simulated)
        documents = [
            "The transformer architecture uses self-attention to process sequences in parallel.",
            "Large language models are trained on massive text corpora using next-token prediction.",
            "KV Cache stores key-value pairs from attention layers to avoid recomputation.",
            "Retrieval-augmented generation combines LLMs with external knowledge retrieval.",
            "Multi-tenant serving allows multiple users to share the same model instance.",
            "Semantic similarity measures how close two text snippets are in meaning.",
            "GPU memory bandwidth is the primary bottleneck in LLM inference systems.",
            "Flash Attention reduces memory usage by computing attention in tiles.",
            "Quantization reduces model precision from FP16 to INT8 or INT4.",
            "Speculative decoding uses a small draft model to speed up generation.",
        ]

        # Paraphrase templates
        question_templates = [
            "请介绍一下：{}",
            "帮我解释：{}",
            "{}是什么意思？",
            "请用简单的语言描述{}",
            "关于{}，你能告诉我什么？",
            "{}的原理是什么？",
            "请详细说明{}",
            "{}有什么重要性？",
        ]

        prompts = []
        for _ in range(self.num_requests):
            doc = rng.choice(documents)
            template = rng.choice(question_templates)
            prompt = template.format(doc[:30])  # Use first 30 chars as topic
            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} RAG synthetic prompts")
        return self._prompts_to_requests(prompts)

    # ─────────────────────────────────────────
    # Pure Synthetic (fallback)
    # ─────────────────────────────────────────

    def _load_synthetic(self) -> List[Dict]:
        """Fallback: pure synthetic with controllable similarity."""
        from eval.run_benchmark import generate_synthetic_requests
        return generate_synthetic_requests(
            num_requests=self.num_requests,
            semantic_similarity_ratio=0.4,
            seed=self.seed,
        )

    # ─────────────────────────────────────────
    # Shared: prompts -> requests
    # ─────────────────────────────────────────

    def _prompts_to_requests(self, prompts: List[str]) -> List[Dict]:
        """Convert text prompts to request dicts with token_ids and embeddings."""
        embedder = self._get_embedder()
        tokenizer = self._get_tokenizer()

        logger.info(f"Encoding {len(prompts)} prompts...")
        embeddings = embedder.encode(
            prompts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Project to 128-dim (matching HSI embedding_dim)
        if embeddings.shape[1] != 128:
            rng = np.random.default_rng(42)
            proj = rng.normal(0, 1, (128, embeddings.shape[1])).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=1, keepdims=True)
            embeddings = (proj @ embeddings.T).T
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        requests = []
        rng = random.Random(self.seed)
        for i, (prompt, emb) in enumerate(
            tqdm(zip(prompts, embeddings), total=len(prompts), desc="Building requests")
        ):
            if tokenizer:
                token_ids = tokenizer.encode(prompt, add_special_tokens=True)[:512]
            else:
                # Fake token IDs based on character codes
                token_ids = [ord(c) % 32000 for c in prompt[:128]]

            requests.append({
                "request_id": i,
                "text": prompt,
                "token_ids": token_ids,
                "semantic_vector": emb.astype(np.float32),
                "tenant_id": f"tenant_{rng.randint(0, 3)}",
                "type": "real",
            })

        return requests

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_tokenizer(self):
        if self._tokenizer is None and self.tokenizer_name:
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer
