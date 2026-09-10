"""CPU-only policy tests for the Stage 5 replica router."""

import unittest

from feno_rt.runtime import (
    CacheAwareReplicaRouter,
    ReplicaRouterConfig,
    RoundRobinRouter,
    WorkerRouteState,
)


def state(
    worker_id,
    *,
    healthy=True,
    queue_depth=0,
    free_ratio=0.8,
    has_replica=True,
    residency=None,
):
    total = 1_000
    return WorkerRouteState(
        worker_id=worker_id,
        device=f"cuda:{worker_id[-1]}",
        healthy=healthy,
        queue_depth=queue_depth,
        free_memory_bytes=int(total * free_ratio),
        total_memory_bytes=total,
        has_medium_replica=has_replica,
        cache_residency=residency or {},
    )


class Stage5ReplicaRouterTest(unittest.TestCase):
    def test_round_robin_rotates_over_healthy_workers_only(self):
        router = RoundRobinRouter()
        states = [
            state("gpu-0"),
            state("gpu-1", healthy=False),
            state("gpu-2"),
        ]
        selected = [router.select(states).worker_id for _ in range(5)]
        self.assertEqual(selected, ["gpu-0", "gpu-2", "gpu-0", "gpu-2", "gpu-0"])

    def test_cache_locality_can_outweigh_moderate_queue_pressure(self):
        router = CacheAwareReplicaRouter()
        cached = state(
            "gpu-0",
            queue_depth=2,
            residency={
                "medium": True,
                "decoder": True,
                "geometry": True,
                "wavelet": True,
            },
        )
        idle = state("gpu-1", residency={"medium": True})
        decision = router.select([cached, idle])
        self.assertEqual(decision.worker_id, "gpu-0")
        self.assertEqual(decision.cache_hits, 4)
        self.assertGreater(decision.locality_score, decision.queue_penalty)

    def test_memory_headroom_filters_a_nearly_full_worker(self):
        router = CacheAwareReplicaRouter(
            ReplicaRouterConfig(minimum_free_memory_ratio=0.1)
        )
        nearly_full = state(
            "gpu-0",
            free_ratio=0.01,
            residency={"medium": True, "decoder": True, "geometry": True},
        )
        available = state("gpu-1", free_ratio=0.5, residency={"medium": True})
        self.assertEqual(router.select([nearly_full, available]).worker_id, "gpu-1")

    def test_batch_boundary_penalty_avoids_an_extra_partial_batch(self):
        router = CacheAwareReplicaRouter()
        full = state(
            "gpu-0",
            queue_depth=32,
            residency={"medium": True, "geometry": True},
        )
        available = state(
            "gpu-1", queue_depth=30, residency={"medium": True}
        )
        full = WorkerRouteState(**{**full.__dict__, "batch_capacity": 32})
        available = WorkerRouteState(
            **{**available.__dict__, "batch_capacity": 32}
        )
        self.assertEqual(router.select([full, available]).worker_id, "gpu-1")

    def test_stable_affinity_can_preserve_a_hot_prefix(self):
        router = CacheAwareReplicaRouter()
        affinity = WorkerRouteState(
            **{
                **state(
                    "gpu-0",
                    queue_depth=32,
                    residency={"medium": True, "geometry": True},
                ).__dict__,
                "batch_capacity": 32,
                "affinity_match": True,
            }
        )
        idle = WorkerRouteState(
            **{
                **state("gpu-1", queue_depth=30, residency={"medium": True}).__dict__,
                "batch_capacity": 32,
            }
        )
        self.assertEqual(router.select([affinity, idle]).worker_id, "gpu-0")

    def test_missing_replica_penalty_prefers_local_medium(self):
        router = CacheAwareReplicaRouter()
        local = state("gpu-0", queue_depth=2, has_replica=True)
        remote = state("gpu-1", queue_depth=0, has_replica=False)
        decision = router.select([local, remote])
        self.assertEqual(decision.worker_id, "gpu-0")
        self.assertEqual(decision.missing_replica_penalty, 0.0)

    def test_no_healthy_worker_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no healthy GPU worker"):
            CacheAwareReplicaRouter().select([state("gpu-0", healthy=False)])

    def test_invalid_router_config_is_rejected(self):
        with self.assertRaises(ValueError):
            ReplicaRouterConfig(queue_penalty=-1.0)
        with self.assertRaises(ValueError):
            ReplicaRouterConfig(minimum_free_memory_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
