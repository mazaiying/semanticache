"""
Hierarchical Semantic Index (HSI)

Two-level index for KV Cache lookup:
  Level 1: Exact hash match (compatible with vLLM Prefix Caching)
  Level 2: Approximate LSH semantic match
"""

import hashlib
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheBlock:
    """Represents a cached KV block with its semantic fingerprint."""
    block_id: str                        # Unique block identifier
    exact_hash: str                      # Exact token-level hash
    semantic_vector: np.ndarray          # Compressed semantic embedding
    token_ids: List[int]                 # Original token IDs
    kv_ref: any                          # Reference to actual KV tensors (in TSM)
    access_count: int = 0
    last_access: float = 0.0
    tenant_id: Optional[str] = None
    sensitivity: str = "public"          # "public" | "private"


class LSHIndex:
    """
    Locality-Sensitive Hashing index for approximate nearest neighbor search.
    Uses random projection LSH for cosine similarity.
    """

    def __init__(self, dim: int, num_tables: int = 8, num_bits: int = 12):
        self.dim = dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        # Random projection matrices for each hash table
        self.projections = [
            np.random.randn(num_bits, dim).astype(np.float32)
            for _ in range(num_tables)
        ]
        # Hash tables: hash_value -> list of block_ids
        self.tables: List[Dict[int, List[str]]] = [
            {} for _ in range(num_tables)
        ]

    def _hash(self, vec: np.ndarray, table_idx: int) -> int:
        """Compute LSH hash for a vector using table table_idx."""
        proj = self.projections[table_idx] @ vec
        bits = (proj > 0).astype(int)
        return int("".join(map(str, bits)), 2)

    def insert(self, block_id: str, vec: np.ndarray):
        """Insert a block into all hash tables."""
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        for i, table in enumerate(self.tables):
            h = self._hash(vec, i)
            table.setdefault(h, []).append(block_id)

    def query(self, vec: np.ndarray, top_k: int = 10) -> List[str]:
        """Return candidate block IDs that share at least one hash bucket."""
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        candidates = set()
        for i, table in enumerate(self.tables):
            h = self._hash(vec, i)
            candidates.update(table.get(h, []))
        return list(candidates)[:top_k]

    def remove(self, block_id: str):
        """Remove a block from all hash tables."""
        for table in self.tables:
            for key in list(table.keys()):
                if block_id in table[key]:
                    table[key].remove(block_id)
                if not table[key]:
                    del table[key]


class HierarchicalSemanticIndex:
    """
    Two-level hierarchical index for KV Cache lookup.

    Lookup strategy:
      1. Exact hash match -> return block immediately (zero overhead)
      2. LSH semantic match -> return candidates for QBR verification
      3. Miss -> return None (recompute needed)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        lsh_num_tables: int = 8,
        lsh_num_bits: int = 12,
    ):
        self.embedding_dim = embedding_dim
        # Level 1: exact hash map
        self.exact_index: Dict[str, CacheBlock] = {}
        # Level 2: LSH approximate index
        self.lsh_index = LSHIndex(embedding_dim, lsh_num_tables, lsh_num_bits)
        # block_id -> CacheBlock mapping
        self.blocks: Dict[str, CacheBlock] = {}

        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "total_queries": 0,
        }

    def _compute_exact_hash(self, token_ids: List[int]) -> str:
        """Compute exact hash from token IDs (compatible with vLLM APC)."""
        token_bytes = str(token_ids).encode()
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def _extract_semantic_vector(self, kv_data: np.ndarray) -> np.ndarray:
        """
        Extract compressed semantic vector from KV tensors.
        Uses mean-pooled projection of K matrix.
        This is done in-place during KV computation, adding negligible overhead.

        Args:
            kv_data: KV tensor of shape [num_layers, 2, seq_len, head_dim]
                     or pre-computed mean embedding of shape [embedding_dim]
        Returns:
            Compressed semantic vector of shape [embedding_dim]
        """
        if kv_data.ndim == 1 and len(kv_data) == self.embedding_dim:
            return kv_data  # Already extracted
        # Take K matrices, mean-pool across layers and heads
        k_matrices = kv_data[:, 0, :, :]  # [layers, seq_len, head_dim]
        pooled = k_matrices.mean(axis=(0, 2))  # [seq_len]
        # Project to embedding_dim via random linear projection (pre-computed)
        if not hasattr(self, "_proj"):
            np.random.seed(42)
            self._proj = np.random.randn(
                self.embedding_dim, pooled.shape[0]
            ).astype(np.float32)
        vec = self._proj @ pooled.astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def insert(
        self,
        token_ids: List[int],
        semantic_vector: np.ndarray,
        kv_ref: any,
        tenant_id: Optional[str] = None,
        sensitivity: str = "public",
    ) -> str:
        """Insert a new KV block into both indexes."""
        exact_hash = self._compute_exact_hash(token_ids)
        block_id = f"{exact_hash}_{tenant_id or 'global'}"

        block = CacheBlock(
            block_id=block_id,
            exact_hash=exact_hash,
            semantic_vector=semantic_vector,
            token_ids=token_ids,
            kv_ref=kv_ref,
            tenant_id=tenant_id,
            sensitivity=sensitivity,
        )

        self.blocks[block_id] = block
        self.exact_index[exact_hash] = block
        # Only index public blocks in LSH (private blocks excluded for privacy)
        if sensitivity == "public":
            self.lsh_index.insert(block_id, semantic_vector)

        logger.debug(f"Inserted block {block_id}")
        return block_id

    def lookup(
        self,
        token_ids: List[int],
        query_vector: np.ndarray,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
        enable_semantic: bool = True,
        enforce_access: bool = True,
    ) -> Tuple[Optional[CacheBlock], str]:
        """
        Hierarchical lookup: exact first, then semantic.

        Returns:
            (block, hit_type) where hit_type in {"exact", "semantic", "miss"}
        """
        self._stats["total_queries"] += 1
        import time

        # --- Level 1: Exact match ---
        exact_hash = self._compute_exact_hash(token_ids)
        if exact_hash in self.exact_index:
            block = self.exact_index[exact_hash]
            # Tenant isolation check
            if not enforce_access or self._is_accessible(block, tenant_id):
                block.access_count += 1
                block.last_access = time.time()
                self._stats["exact_hits"] += 1
                logger.debug(f"Exact hit: {block.block_id}")
                return block, "exact"

        # --- Level 2: Semantic LSH match ---
        if not enable_semantic:
            self._stats["misses"] += 1
            return None, "miss"

        candidate_ids = self.lsh_index.query(query_vector, top_k=top_k)
        best_block = None
        best_sim = -1.0

        for cid in candidate_ids:
            block = self.blocks.get(cid)
            if block is None:
                continue
            if enforce_access and not self._is_accessible(block, tenant_id):
                continue
            sim = self._cosine_similarity(query_vector, block.semantic_vector)
            if sim > best_sim:
                best_sim = sim
                best_block = block

        if best_block is not None:
            self._stats["semantic_hits"] += 1
            logger.debug(
                f"Semantic hit: {best_block.block_id} (sim={best_sim:.3f})"
            )
            return best_block, "semantic"

        self._stats["misses"] += 1
        return None, "miss"

    def _is_accessible(
        self, block: CacheBlock, tenant_id: Optional[str]
    ) -> bool:
        """Check if tenant can access this block (TIL enforcement)."""
        if block.sensitivity == "public":
            return True
        # Private blocks: only accessible by the owning tenant
        return block.tenant_id == tenant_id

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def remove(self, block_id: str):
        """Remove a block from all indexes."""
        block = self.blocks.pop(block_id, None)
        if block:
            self.exact_index.pop(block.exact_hash, None)
            self.lsh_index.remove(block_id)

    def get_stats(self) -> Dict:
        """Return cache hit/miss statistics."""
        total = self._stats["total_queries"]
        if total == 0:
            return self._stats
        return {
            **self._stats,
            "exact_hit_rate": self._stats["exact_hits"] / total,
            "semantic_hit_rate": self._stats["semantic_hits"] / total,
            "miss_rate": self._stats["misses"] / total,
            "total_hit_rate": (
                self._stats["exact_hits"] + self._stats["semantic_hits"]
            ) / total,
        }
