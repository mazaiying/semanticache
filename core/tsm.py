"""
Physical tiered storage for SemantiCache KV blocks.

The manager supports three policies over the same physical backends:

* single_lru: GPU-only LRU; an eviction deletes the block.
* tiered_lru: GPU -> pinned CPU -> NVMe with LRU victims.
* benefit: the same physical hierarchy with benefit-density victims.

L1 stores tensors on the configured accelerator, L2 stores page-locked host
tensors when CUDA is available, and L3 stores one serialized file per block.
The NVMe directory may also be placed on a mounted remote filesystem.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    gpu_capacity_gb: float = 20.0
    cpu_capacity_gb: float = 64.0
    ssd_capacity_gb: float = 200.0
    eviction_policy: str = "tiered_lru"
    device: str = "cuda"
    ssd_path: Optional[str] = None
    pin_cpu_memory: bool = True
    cleanup_ssd_on_close: bool = True
    rehit_alpha: float = 0.1
    rehit_decay_seconds: float = 3600.0
    gpu_cpu_bandwidth_gbps: float = 50.0
    cpu_ssd_bandwidth_gbps: float = 7.0

    def normalized_policy(self) -> str:
        aliases = {
            "lru": "tiered_lru",
            "single": "single_lru",
            "single-tier-lru": "single_lru",
            "tiered": "tiered_lru",
            "benefit_density": "benefit",
        }
        policy = aliases.get(self.eviction_policy, self.eviction_policy)
        if policy not in {"single_lru", "tiered_lru", "benefit"}:
            raise ValueError(
                "eviction_policy must be single_lru, tiered_lru, or benefit"
            )
        return policy


@dataclass
class BlockRecord:
    block_id: str
    size_bytes: int
    prefill_cost_ms: float
    tier: str
    access_count: int = 1
    last_access: float = field(default_factory=time.time)
    payload: Any = None
    disk_path: Optional[str] = None


@dataclass
class TierStats:
    capacity_gb: float
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    migrations_in: int = 0
    migrations_out: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    read_ms: list = field(default_factory=list)
    write_ms: list = field(default_factory=list)


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.nelement() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _map_payload(value: Any, tensor_fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {k: _map_payload(v, tensor_fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_payload(v, tensor_fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_map_payload(v, tensor_fn) for v in value)
    return value


class TieredStorageManager:
    """Exclusive physical hierarchy for KV payloads."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.policy = config.normalized_policy()
        self.device = self._resolve_device(config.device)
        self._tiers: Dict[str, OrderedDict[str, BlockRecord]] = {
            "l1": OrderedDict(),
            "l2": OrderedDict(),
            "l3": OrderedDict(),
        }
        self._records: Dict[str, BlockRecord] = {}
        self._used_bytes = {"l1": 0, "l2": 0, "l3": 0}
        self._capacity_bytes = {
            "l1": int(config.gpu_capacity_gb * 1e9),
            "l2": int(config.cpu_capacity_gb * 1e9),
            "l3": int(config.ssd_capacity_gb * 1e9),
        }
        self._stats = {
            "l1": TierStats(config.gpu_capacity_gb),
            "l2": TierStats(config.cpu_capacity_gb),
            "l3": TierStats(config.ssd_capacity_gb),
        }
        self._on_final_evict: Optional[Callable[[str], None]] = None
        self._closed = False

        root = config.ssd_path
        if root is None:
            root = str(Path(tempfile.gettempdir()) / "semanticache_nvme")
        self._ssd_root = Path(root).expanduser().resolve()
        self._ssd_dir = self._ssd_root / (
            f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self._ssd_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; L1 uses CPU for functional testing.")
            return torch.device("cpu")
        return device

    def set_eviction_callback(self, callback: Callable[[str], None]) -> None:
        self._on_final_evict = callback

    def store(
        self,
        block_id: str,
        kv_data: Any,
        size_gb: Optional[float] = None,
        prefill_cost_ms: float = 0.0,
    ) -> None:
        if kv_data is None:
            raise ValueError("TSM requires the physical KV payload, not a placeholder")
        self.evict(block_id, notify=False)

        measured_bytes = _tensor_bytes(kv_data)
        declared_bytes = int(size_gb * 1e9) if size_gb is not None else 0
        size_bytes = measured_bytes or declared_bytes
        if size_bytes <= 0:
            raise ValueError("Unable to determine KV payload size")

        t0 = time.perf_counter()
        payload = self._to_l1(kv_data)
        elapsed_ms = self._elapsed_ms(t0, synchronize=True)
        self._stats["l1"].bytes_written += size_bytes
        self._stats["l1"].write_ms.append(elapsed_ms)

        record = BlockRecord(
            block_id=block_id,
            size_bytes=size_bytes,
            prefill_cost_ms=float(prefill_cost_ms),
            tier="l1",
            payload=payload,
        )
        self._records[block_id] = record
        self._insert("l1", record)
        self._enforce_capacity("l1")

    def fetch(self, block_id: str) -> Tuple[Optional[Any], str]:
        record = self._records.get(block_id)
        if record is None:
            for stats in self._stats.values():
                stats.misses += 1
            return None, "miss"

        source_tier = record.tier
        record.access_count += 1
        record.last_access = time.time()
        self._stats[source_tier].hits += 1
        self._tiers[source_tier].move_to_end(block_id)

        if source_tier == "l1":
            return record.payload, "l1"

        if source_tier == "l2":
            host_payload = record.payload
            t0 = time.perf_counter()
            gpu_payload = self._to_l1(host_payload)
            elapsed_ms = self._elapsed_ms(t0, synchronize=True)
            self._stats["l2"].bytes_read += record.size_bytes
            self._stats["l2"].read_ms.append(elapsed_ms)
            self._promote_to_l1(record, gpu_payload)
            return gpu_payload, "l2"

        t0 = time.perf_counter()
        host_payload = self._read_disk(record)
        disk_ms = self._elapsed_ms(t0)
        self._stats["l3"].bytes_read += record.size_bytes
        self._stats["l3"].read_ms.append(disk_ms)

        t1 = time.perf_counter()
        gpu_payload = self._to_l1(host_payload)
        upload_ms = self._elapsed_ms(t1, synchronize=True)
        self._stats["l2"].bytes_read += record.size_bytes
        self._stats["l2"].read_ms.append(upload_ms)
        self._promote_to_l1(record, gpu_payload)
        return gpu_payload, "l3"

    def evict(self, block_id: str, notify: bool = True) -> bool:
        record = self._records.pop(block_id, None)
        if record is None:
            return False
        self._remove_from_tier(record)
        if record.disk_path:
            Path(record.disk_path).unlink(missing_ok=True)
        if notify and self._on_final_evict is not None:
            self._on_final_evict(block_id)
        return True

    def _promote_to_l1(self, record: BlockRecord, gpu_payload: Any) -> None:
        source = record.tier
        if record.size_bytes > self._capacity_bytes["l1"]:
            return
        self._remove_from_tier(record)
        if record.disk_path:
            Path(record.disk_path).unlink(missing_ok=True)
            record.disk_path = None
        record.payload = gpu_payload
        self._make_room("l1", record.size_bytes, protected=record.block_id)
        if record.block_id not in self._records:
            return
        self._insert("l1", record)
        self._stats[source].migrations_out += 1
        self._stats["l1"].migrations_in += 1

    def _make_room(
        self, tier: str, incoming_bytes: int, protected: Optional[str] = None
    ) -> None:
        capacity = self._capacity_bytes[tier]
        if capacity <= 0:
            while self._tiers[tier]:
                self._demote_or_drop(self._select_victim(tier, protected))
            return
        while self._used_bytes[tier] + incoming_bytes > capacity:
            victim = self._select_victim(tier, protected)
            if victim is None:
                break
            self._demote_or_drop(victim)

    def _enforce_capacity(self, tier: str) -> None:
        capacity = self._capacity_bytes[tier]
        while self._used_bytes[tier] > capacity and self._tiers[tier]:
            self._demote_or_drop(self._select_victim(tier))

    def _select_victim(
        self, tier: str, protected: Optional[str] = None
    ) -> Optional[BlockRecord]:
        candidates = [
            record
            for block_id, record in self._tiers[tier].items()
            if block_id != protected
        ]
        if not candidates:
            return None
        if self.policy != "benefit":
            return candidates[0]
        return min(candidates, key=lambda record: self._benefit_density(record, tier))

    def _benefit_density(self, record: BlockRecord, tier: str) -> float:
        age = max(0.0, time.time() - record.last_access)
        probability = min(
            1.0,
            self.config.rehit_alpha
            * record.access_count
            * math.exp(-age / max(self.config.rehit_decay_seconds, 1e-9)),
        )
        size_gb = record.size_bytes / 1e9
        if tier == "l1":
            migration_ms = (
                size_gb / max(self.config.gpu_cpu_bandwidth_gbps, 1e-9) * 1000
            )
        else:
            migration_ms = (
                size_gb / max(self.config.cpu_ssd_bandwidth_gbps, 1e-9) * 1000
            )
        return (
            probability * record.prefill_cost_ms - migration_ms
        ) / max(size_gb, 1e-12)

    def _demote_or_drop(self, record: Optional[BlockRecord]) -> None:
        if record is None:
            return
        source = record.tier
        self._remove_from_tier(record)
        self._stats[source].evictions += 1
        self._stats[source].migrations_out += 1

        if self.policy == "single_lru" or source == "l3":
            self._records.pop(record.block_id, None)
            if record.disk_path:
                Path(record.disk_path).unlink(missing_ok=True)
            if self._on_final_evict is not None:
                self._on_final_evict(record.block_id)
            return

        if source == "l1":
            t0 = time.perf_counter()
            record.payload = self._to_l2(record.payload)
            elapsed_ms = self._elapsed_ms(t0, synchronize=True)
            self._stats["l2"].bytes_written += record.size_bytes
            self._stats["l2"].write_ms.append(elapsed_ms)
            if record.block_id in self._records:
                self._insert("l2", record)
                self._stats["l2"].migrations_in += 1
                self._enforce_capacity("l2")
            return

        t0 = time.perf_counter()
        path = self._write_disk(record.block_id, record.payload)
        elapsed_ms = self._elapsed_ms(t0)
        record.payload = None
        record.disk_path = str(path)
        self._stats["l3"].bytes_written += record.size_bytes
        self._stats["l3"].write_ms.append(elapsed_ms)
        if record.block_id in self._records:
            self._insert("l3", record)
            self._stats["l3"].migrations_in += 1
            self._enforce_capacity("l3")

    def _insert(self, tier: str, record: BlockRecord) -> None:
        record.tier = tier
        self._tiers[tier][record.block_id] = record
        self._tiers[tier].move_to_end(record.block_id)
        self._used_bytes[tier] += record.size_bytes

    def _remove_from_tier(self, record: BlockRecord) -> None:
        tier = record.tier
        if self._tiers[tier].pop(record.block_id, None) is not None:
            self._used_bytes[tier] -= record.size_bytes

    def _to_l1(self, payload: Any) -> Any:
        return _map_payload(
            payload,
            lambda tensor: tensor.detach().to(
                device=self.device, non_blocking=False
            ).contiguous(),
        )

    def _to_l2(self, payload: Any) -> Any:
        use_pin = (
            self.config.pin_cpu_memory
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

        def convert(tensor: torch.Tensor) -> torch.Tensor:
            host = tensor.detach().to(device="cpu", non_blocking=False).contiguous()
            return host.pin_memory() if use_pin and not host.is_pinned() else host

        return _map_payload(payload, convert)

    def _write_disk(self, block_id: str, payload: Any) -> Path:
        digest = hashlib.sha256(block_id.encode("utf-8")).hexdigest()
        path = self._ssd_dir / f"{digest}.pt"
        torch.save(self._to_l2(payload), path)
        return path

    def _read_disk(self, record: BlockRecord) -> Any:
        if record.disk_path is None:
            raise RuntimeError(f"L3 record {record.block_id} has no backing file")
        try:
            payload = torch.load(
                record.disk_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            payload = torch.load(record.disk_path, map_location="cpu")
        return self._to_l2(payload)

    def _elapsed_ms(self, started: float, synchronize: bool = False) -> float:
        if synchronize and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - started) * 1000

    @staticmethod
    def _summary(values: list) -> Dict[str, float]:
        if not values:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(array.mean()),
            "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)),
        }

    def get_stats(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for tier in ("l1", "l2", "l3"):
            stats = self._stats[tier]
            used_gb = self._used_bytes[tier] / 1e9
            capacity_gb = stats.capacity_gb
            result[tier] = {
                "backend": {
                    "l1": str(self.device),
                    "l2": "pinned_cpu"
                    if self.device.type == "cuda" and self.config.pin_cpu_memory
                    else "cpu",
                    "l3": f"file:{self._ssd_dir}",
                }[tier],
                "used_gb": used_gb,
                "capacity_gb": capacity_gb,
                "utilization": used_gb / max(capacity_gb, 1e-12),
                "blocks": len(self._tiers[tier]),
                "hits": stats.hits,
                "misses": stats.misses,
                "evictions": stats.evictions,
                "migrations_in": stats.migrations_in,
                "migrations_out": stats.migrations_out,
                "bytes_read": stats.bytes_read,
                "bytes_written": stats.bytes_written,
                "read_latency": self._summary(stats.read_ms),
                "write_latency": self._summary(stats.write_ms),
            }
        result["policy"] = self.policy
        result["physical_cuda"] = self.device.type == "cuda"
        result["live_blocks"] = len(self._records)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.config.cleanup_ssd_on_close:
            shutil.rmtree(self._ssd_dir, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
