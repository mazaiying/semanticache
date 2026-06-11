"""
Tiered Storage Manager (TSM)

Three-tier KV Cache storage:
  L1: GPU HBM     (~40-80 GB, μs access)
  L2: CPU RAM     (~256 GB, ms access)
  L3: NVMe SSD    (~TB, 10ms access)

Migration decisions based on cost model:
  migrate_benefit = P(rehit) * recompute_cost - migration_bandwidth_cost
"""

import time
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    gpu_capacity_gb: float = 40.0
    cpu_capacity_gb: float = 128.0
    ssd_capacity_gb: float = 500.0
    eviction_policy: str = "lru"


@dataclass
class TierStats:
    capacity_gb: float
    used_gb: float = 0.0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    migrations_in: int = 0
    migrations_out: int = 0

    @property
    def utilization(self) -> float:
        return self.used_gb / max(self.capacity_gb, 1e-8)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache:
    """Simple LRU cache with capacity limit (in GB)."""

    def __init__(self, capacity_gb: float):
        self.capacity_bytes = int(capacity_gb * 1e9)
        self._cache: OrderedDict = OrderedDict()  # block_id -> (data, size_bytes)
        self._used_bytes: int = 0

    def get(self, block_id: str) -> Optional[Any]:
        if block_id not in self._cache:
            return None
        self._cache.move_to_end(block_id)  # LRU update
        return self._cache[block_id][0]

    def put(self, block_id: str, data: Any, size_bytes: int) -> list:
        """Insert block; returns list of tuples: (evicted_id, evicted_data, evicted_size)."""
        evicted = []
        if block_id in self._cache:
            self._used_bytes -= self._cache[block_id][1]
            del self._cache[block_id]

        while self._used_bytes + size_bytes > self.capacity_bytes and self._cache:
            evicted_id, (evicted_data, evicted_size) = self._cache.popitem(last=False)
            self._used_bytes -= evicted_size
            evicted.append((evicted_id, evicted_data, evicted_size))

        self._cache[block_id] = (data, size_bytes)
        self._used_bytes += size_bytes
        return evicted

    def remove(self, block_id: str) -> bool:
        if block_id not in self._cache:
            return False
        _, size = self._cache.pop(block_id)
        self._used_bytes -= size
        return True

    @property
    def used_gb(self) -> float:
        return self._used_bytes / 1e9

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)


class TieredStorageManager:
    """
    Three-tier storage manager for KV Cache blocks.

    Cost model for migration decisions:
        benefit = P(rehit) * recompute_cost_ms - migration_cost_ms
        Migrate L1->L2 when benefit < 0 (cold block)
        Promote L2->L1 when benefit > promotion_threshold

    Bandwidth assumptions:
        GPU->CPU: ~50 GB/s (PCIe 4.0 x16)
        CPU->SSD: ~7 GB/s (NVMe PCIe 4.0)
    """

    # Bandwidth (GB/s) and latency (ms) constants
    GPU_CPU_BW = 50.0    # GB/s
    CPU_SSD_BW = 7.0     # GB/s
    GPU_LATENCY_MS = 0.001
    CPU_LATENCY_MS = 1.0
    SSD_LATENCY_MS = 10.0

    def __init__(self, config: StorageConfig):
        self.config = config
        self.l1 = LRUCache(config.gpu_capacity_gb)
        self.l2 = LRUCache(config.cpu_capacity_gb)
        self.l3 = LRUCache(config.ssd_capacity_gb)

        self._stats = {
            "l1": TierStats(config.gpu_capacity_gb),
            "l2": TierStats(config.cpu_capacity_gb),
            "l3": TierStats(config.ssd_capacity_gb),
        }
        # Access frequency tracking for P(rehit) estimation
        self._access_log: Dict[str, list] = {}

    def store(self, block_id: str, kv_data: Any, size_gb: float = 0.001):
        """Store a new KV block, starting in GPU L1."""
        size_bytes = int(size_gb * 1e9)
        evicted = self.l1.put(block_id, kv_data, size_bytes)
        self._stats["l1"].used_gb = self.l1.used_gb

        # Cascade evicted blocks to L2
        for eid, edata, esize in evicted:
            self._stats["l1"].evictions += 1
            self._demote_l1_to_l2(eid, edata, esize)

        self._access_log[block_id] = [time.time()]
        logger.debug(f"Stored block {block_id} in L1 (GPU)")

    def fetch(self, block_id: str) -> Tuple[Optional[Any], str]:
        """
        Fetch KV data for block_id, checking tiers in order.
        Returns (data, tier) where tier in {"l1", "l2", "l3", "miss"}.
        Automatically promotes blocks on access.
        """
        t = time.time()
        self._access_log.setdefault(block_id, []).append(t)

        # Check L1 (GPU)
        data = self.l1.get(block_id)
        if data is not None:
            self._stats["l1"].hits += 1
            return data, "l1"
        self._stats["l1"].misses += 1

        # Check L2 (CPU RAM) → promote to L1
        data = self.l2.get(block_id)
        if data is not None:
            self._stats["l2"].hits += 1
            self._promote_to_l1(block_id, data)
            return data, "l2"
        self._stats["l2"].misses += 1

        # Check L3 (SSD) → promote to L2
        data = self.l3.get(block_id)
        if data is not None:
            self._stats["l3"].hits += 1
            self._promote_to_l2(block_id, data)
            return data, "l3"
        self._stats["l3"].misses += 1

        return None, "miss"

    def evict(self, block_id: str):
        """Explicitly evict a block from all tiers."""
        self.l1.remove(block_id)
        self.l2.remove(block_id)
        self.l3.remove(block_id)
        self._access_log.pop(block_id, None)

    def should_migrate(self, block_id: str, current_tier: str) -> Optional[str]:
        """
        Cost-model-based migration decision.
        Returns target tier or None if no migration needed.
        """
        p_rehit = self._estimate_rehit_probability(block_id)
        recompute_cost_ms = 100.0  # Estimated for 4K context on A100

        if current_tier == "l1":
            # Consider demoting to L2
            size_gb = 0.001  # ~1MB per block estimate
            migration_cost_ms = size_gb / self.GPU_CPU_BW * 1000
            benefit = p_rehit * recompute_cost_ms - migration_cost_ms
            if benefit < 0:
                return "l2"  # Demote cold block

        elif current_tier == "l2":
            # Consider further demoting to L3
            size_gb = 0.001
            migration_cost_ms = size_gb / self.CPU_SSD_BW * 1000
            benefit = p_rehit * recompute_cost_ms - migration_cost_ms
            if benefit < 0:
                return "l3"

        return None  # Stay in current tier

    def _estimate_rehit_probability(self, block_id: str) -> float:
        """
        Estimate P(block will be accessed again) using recency and frequency.
        Simple exponential decay model.
        """
        accesses = self._access_log.get(block_id, [])
        if not accesses:
            return 0.0
        freq = len(accesses)
        recency = time.time() - accesses[-1]
        # Exponential decay: P = freq_factor * exp(-recency / decay_rate)
        decay_rate = 3600.0  # 1 hour
        return min(1.0, (freq / 10.0) * np.exp(-recency / decay_rate))

    def _promote_to_l1(self, block_id: str, data: Any):
        """Move block from L2/L3 to L1 (GPU)."""
        self.l2.remove(block_id)
        evicted = self.l1.put(block_id, data, int(0.001 * 1e9))
        self._stats["l1"].migrations_in += 1
        for eid, edata, esize in evicted:
            self._demote_l1_to_l2(eid, edata, esize)

    def _promote_to_l2(self, block_id: str, data: Any):
        """Move block from L3 to L2 (CPU RAM)."""
        self.l3.remove(block_id)
        evicted = self.l2.put(block_id, data, int(0.001 * 1e9))
        self._stats["l2"].migrations_in += 1
        for eid, edata, esize in evicted:
            self._demote_l2_to_l3(eid, edata, esize)

    def _demote_l1_to_l2(self, block_id: str, data: Any, size_bytes: int):
        """Demote block from L1 (GPU) to L2 (CPU)."""
        evicted = self.l2.put(block_id, data, size_bytes)
        self._stats["l2"].migrations_in += 1
        for eid, edata, esize in evicted:
            self._demote_l2_to_l3(eid, edata, esize)

    def _demote_l2_to_l3(self, block_id: str, data: Any, size_bytes: int):
        """Demote block from L2 (CPU) to L3 (SSD)."""
        evicted = self.l3.put(block_id, data, size_bytes)
        self._stats["l3"].migrations_in += 1
        for eid, edata, esize in evicted:
            self.evict(eid)  # Fully evict from L3

    def get_stats(self) -> Dict:
        return {
            "l1": {
                "used_gb": self.l1.used_gb,
                "capacity_gb": self.config.gpu_capacity_gb,
                "utilization": self.l1.used_gb / max(self.config.gpu_capacity_gb, 1e-8),
                "hit_rate": self._stats["l1"].hit_rate,
                "blocks": len(self.l1),
            },
            "l2": {
                "used_gb": self.l2.used_gb,
                "capacity_gb": self.config.cpu_capacity_gb,
                "utilization": self.l2.used_gb / max(self.config.cpu_capacity_gb, 1e-8),
                "hit_rate": self._stats["l2"].hit_rate,
                "blocks": len(self.l2),
            },
            "l3": {
                "used_gb": self.l3.used_gb,
                "capacity_gb": self.config.ssd_capacity_gb,
                "utilization": self.l3.used_gb / max(self.config.ssd_capacity_gb, 1e-8),
                "hit_rate": self._stats["l3"].hit_rate,
                "blocks": len(self.l3),
            },
        }
