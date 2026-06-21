import sys
import unittest
from pathlib import Path


# src 모듈을 저장소 루트에서 직접 가져옵니다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.vllm_load_controller import (
    AdaptiveConcurrencyController,
    VllmLoadMetrics,
    build_vllm_metrics_url,
    parse_vllm_load_metrics,
)


class VllmLoadControllerTests(unittest.IsolatedAsyncioTestCase):
    def test_build_metrics_url_removes_v1_suffix(self):
        self.assertEqual(
            build_vllm_metrics_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/metrics",
        )
        self.assertEqual(
            build_vllm_metrics_url("https://example.com/api/v1/"),
            "https://example.com/api/metrics",
        )

    def test_parse_metrics_sums_multiple_engine_series(self):
        payload = """
vllm:num_requests_running{engine="0",model_name="model"} 3.0
vllm:num_requests_running{engine="1",model_name="model"} 4.0
vllm:num_requests_waiting{engine="0",model_name="model"} 1.0
vllm:num_requests_waiting{engine="1",model_name="model"} 2.0
"""

        self.assertEqual(
            parse_vllm_load_metrics(payload),
            VllmLoadMetrics(running=7, waiting=3),
        )

    async def test_controller_increases_and_decreases_limit(self):
        loads = iter([
            VllmLoadMetrics(running=16, waiting=0),
            VllmLoadMetrics(running=18, waiting=5),
            VllmLoadMetrics(running=15, waiting=2),
        ])

        async def read_load():
            return next(loads)

        controller = AdaptiveConcurrencyController(
            read_load=read_load,
            min_limit=1,
            initial_limit=16,
            max_limit=64,
            target_waiting=2,
            poll_interval_sec=0,
        )

        self.assertEqual(await controller.refresh(), 18)
        self.assertEqual(await controller.refresh(), 15)
        self.assertEqual(await controller.refresh(), 15)

    async def test_controller_uses_fixed_limit_when_metrics_fail(self):
        async def read_load():
            raise OSError("metrics unavailable")

        controller = AdaptiveConcurrencyController(
            read_load=read_load,
            min_limit=1,
            initial_limit=16,
            max_limit=64,
            target_waiting=2,
            poll_interval_sec=0,
        )

        self.assertEqual(await controller.refresh(), 16)

if __name__ == "__main__":
    unittest.main()
