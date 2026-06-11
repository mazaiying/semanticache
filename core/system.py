"""
SemantiCache - Main System Entry Point

Integrates HSI + QBR + TSM + TIL into a unified KV Cache management system.
Designed to be plugged into vLLM's cache management layer.
"""

import time
import logging
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

from .hsi import HierarchicalSemanticIndex, CacheBlock
from .qbr import QualityBoundedReuse, ReuseDecision
from .tsm import TieredStorageManager, StorageConfig
from .til import TenantIsolationLayer

logger = logging.getLogger(__name__)


class SemantiCache:
    """
    Unified SemantiCache system.

    Request flow:
      1. lookup(token_ids, semantic_vec, tenant_id)
         → HSI.lookup() → exact/semantic/miss
      2. If semantic hit: QBR.decide(similarity)
         → approve/reject
      3. If approved: TSM.fetch(block_id)
         → get KV tensors from appropriate tier
      4. If miss or rejected: recompute KV, then store()
      5. TIL enforces isolation at every step
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        lsh_num_tables: int = 8,
        lsh_num_bits: int = 12,
        similarity_threshold: float = 0.85,
        adaptive_qbr: bool = True,
        storage_config: Optional[StorageConfig] = None,
    ):
        if storage_config is None:
            storage_config = StorageConfig()

        self.hsi = HierarchicalSemanticIndex(
            embedding_dim=embedding_dim,
            lsh_num_tables=lsh_num_tables,
            lsh_num_bits=lsh_num_bits,
        )
        self.qbr = QualityBoundedReuse(
            similarity_threshold=similarity_threshold,
            adaptive=adaptive_qbr,
        )
        self.tsm = TieredStorageManager(storage_config)
        self.til = TenantIsolationLayer()

        self._request_log: List[Dict] = []

    def lookup(
        self,
        token_ids: List[int],
        semantic_vector: np.ndarray,
        tenant_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Main lookup interface.

        Returns:
            (kv_data, info) where kv_data is None on miss/rejection
            info contains hit_type, similarity, decision, storage_tier, latency_ms
        """
        t0 = time.perf_counter()

        # 1. HSI lookup
        block, hit_type = self.hsi.lookup(
            token_ids, semantic_vector, tenant_id=tenant_id
        )

        if block is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            return None, {
                "hit_type": "miss",
                "similarity": 0.0,
                "decision": "miss",
                "storage_tier": None,
                "latency_ms": latency_ms,
            }

        # 2. Compute similarity
        if hit_type == "exact":
            similarity = 1.0
        else:
            similarity = self.hsi._cosine_similarity(
                semantic_vector, block.semantic_vector
            )

        # 3. QBR decision
        decision: ReuseDecision = self.qbr.decide(
            similarity_score=similarity,
            hit_type=hit_type,
        )

        if not decision.allow_reuse:
            latency_ms = (time.perf_counter() - t0) * 1000
            return None, {
                "hit_type": hit_type,
                "similarity": similarity,
                "decision": "rejected",
                "reason": decision.reason,
                "storage_tier": None,
                "latency_ms": latency_ms,
            }

        # 4. TIL access check
        if not self.til.check_access(
            block_id=block.block_id,
            block_sensitivity=block.sensitivity,
            block_tenant_id=block.tenant_id,
            requesting_tenant_id=tenant_id,
        ):
            latency_ms = (time.perf_counter() - t0) * 1000
            return None, {
                "hit_type": hit_type,
                "similarity": similarity,
                "decision": "access_denied",
                "storage_tier": None,
                "latency_ms": latency_ms,
            }

        # 5. Fetch from TSM
        kv_data, storage_tier = self.tsm.fetch(block.block_id)

        latency_ms = (time.perf_counter() - t0) * 1000

        info = {
            "hit_type": hit_type,
            "similarity": similarity,
            "decision": "approved",
            "storage_tier": storage_tier,
            "latency_ms": latency_ms,
            "block_id": block.block_id,
        }

        self._request_log.append(info)
        return kv_data, info

    def store(
        self,
        token_ids: List[int],
        semantic_vector: np.ndarray,
        kv_data: Any,
        tenant_id: Optional[str] = None,
        context_label: Optional[str] = None,
        kv_size_gb: float = 0.001,
    ) -> str:
        """Store newly computed KV block in all indexes."""
        sensitivity = self.til.classify_block(
            block_id="",
            token_ids=token_ids,
            context_label=context_label,
            tenant_id=tenant_id,
        )

        block_id = self.hsi.insert(
            token_ids=token_ids,
            semantic_vector=semantic_vector,
            kv_ref=None,
            tenant_id=tenant_id,
            sensitivity=sensitivity,
        )

        self.til.register_block(block_id, sensitivity, tenant_id)
        self.tsm.store(block_id, kv_data, size_gb=kv_size_gb)

        return block_id

    def get_stats(self) -> Dict:
        """Aggregate statistics from all components."""
        hsi_stats = self.hsi.get_stats()
        qbr_stats = self.qbr.get_stats()
        tsm_stats = self.tsm.get_stats()
        til_stats = self.til.get_stats()

        return {
            "hsi": hsi_stats,
            "qbr": qbr_stats,
            "tsm": tsm_stats,
            "til": til_stats,
            "overall": {
                "total_requests": hsi_stats.get("total_queries", 0),
                "overall_hit_rate": hsi_stats.get("total_hit_rate", 0),
                "qbr_approval_rate": qbr_stats.get("approval_rate", 0),
                "current_tau": qbr_stats.get("current_tau", 0),
            },
        }
