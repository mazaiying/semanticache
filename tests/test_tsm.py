import tempfile
import unittest

import torch

from core.tsm import StorageConfig, TieredStorageManager


def payload(value: float):
    return {
        "kv_list": [
            (
                torch.full((1024,), value, dtype=torch.float32),
                torch.full((1024,), value + 1, dtype=torch.float32),
            )
        ],
        "last_token_id": 7,
    }


class PhysicalTieredStorageTest(unittest.TestCase):
    def test_tiered_lru_uses_disk_and_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = TieredStorageManager(
                StorageConfig(
                    gpu_capacity_gb=0.000009,
                    cpu_capacity_gb=0.000009,
                    ssd_capacity_gb=0.001,
                    eviction_policy="tiered_lru",
                    device="cpu",
                    ssd_path=directory,
                )
            )
            manager.store("a", payload(1.0), prefill_cost_ms=10)
            manager.store("b", payload(2.0), prefill_cost_ms=10)
            manager.store("c", payload(3.0), prefill_cost_ms=10)

            before = manager.get_stats()
            self.assertEqual(before["l1"]["blocks"], 1)
            self.assertEqual(before["l2"]["blocks"], 1)
            self.assertEqual(before["l3"]["blocks"], 1)

            restored, source = manager.fetch("a")
            self.assertEqual(source, "l3")
            self.assertEqual(restored["last_token_id"], 7)
            self.assertTrue(
                torch.equal(
                    restored["kv_list"][0][0],
                    torch.full((1024,), 1.0, dtype=torch.float32),
                )
            )
            after = manager.get_stats()
            self.assertGreater(after["l3"]["bytes_read"], 0)
            manager.close()

    def test_single_lru_finally_evicts(self):
        evicted = []
        with tempfile.TemporaryDirectory() as directory:
            manager = TieredStorageManager(
                StorageConfig(
                    gpu_capacity_gb=0.000009,
                    cpu_capacity_gb=0.0,
                    ssd_capacity_gb=0.0,
                    eviction_policy="single_lru",
                    device="cpu",
                    ssd_path=directory,
                )
            )
            manager.set_eviction_callback(evicted.append)
            manager.store("a", payload(1.0), prefill_cost_ms=10)
            manager.store("b", payload(2.0), prefill_cost_ms=10)
            restored, source = manager.fetch("a")
            self.assertIsNone(restored)
            self.assertEqual(source, "miss")
            self.assertEqual(evicted, ["a"])
            manager.close()


if __name__ == "__main__":
    unittest.main()
