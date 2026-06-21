import asyncio
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from core.logging_utils import log_ocr_event


# vLLM Prometheus 출력에서 실행 중/대기 중 요청 게이지만 추출합니다.
_LOAD_METRIC_PATTERN = re.compile(
    r"^(?P<name>vllm(?::|_)num_requests_(?P<kind>running|waiting))"
    r"(?:\{[^}]*\})?\s+(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass(frozen=True, slots=True)
class VllmLoadMetrics:
    running: int
    waiting: int


def build_vllm_metrics_url(base_url: str) -> str:
    # OpenAI 호환 API의 /v1 경로를 같은 서버의 /metrics 경로로 바꿉니다.
    normalized_url = base_url.strip().rstrip("/")
    parts = urlsplit(normalized_url)
    path = parts.path.removesuffix("/v1").rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/metrics", "", ""))


def parse_vllm_load_metrics(payload: str) -> VllmLoadMetrics:
    # 구버전 호환 이름이 함께 노출될 수 있으므로 콜론 이름을 우선합니다.
    values_by_name: dict[str, list[float]] = {}
    for line in payload.splitlines():
        match = _LOAD_METRIC_PATTERN.match(line.strip())
        if match is None:
            continue
        values_by_name.setdefault(match.group("name"), []).append(float(match.group("value")))

    def metric_total(kind: str) -> int:
        colon_name = f"vllm:num_requests_{kind}"
        underscore_name = f"vllm_num_requests_{kind}"
        values = values_by_name.get(colon_name) or values_by_name.get(underscore_name)
        if values is None:
            raise ValueError(f"vLLM {kind} 메트릭이 없습니다.")
        return max(0, int(round(sum(values))))

    return VllmLoadMetrics(
        running=metric_total("running"),
        waiting=metric_total("waiting"),
    )


class VllmMetricsClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout_sec: float = 1.0):
        self.metrics_url = build_vllm_metrics_url(base_url)
        self.api_key = (api_key or "").strip()
        self.timeout_sec = max(0.1, timeout_sec)

    async def read_load(self) -> VllmLoadMetrics:
        # 표준 라이브러리의 동기 HTTP 호출은 이벤트 루프 밖에서 실행합니다.
        return await asyncio.to_thread(self._read_load_sync)

    def _read_load_sync(self) -> VllmLoadMetrics:
        headers = {"Accept": "text/plain"}
        if self.api_key and self.api_key != "dummy_key":
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.metrics_url, headers=headers)
        with urlopen(request, timeout=self.timeout_sec) as response:
            payload = response.read().decode("utf-8", errors="replace")
        return parse_vllm_load_metrics(payload)


class AdaptiveConcurrencyController:
    def __init__(
        self,
        read_load: Callable[[], Awaitable[VllmLoadMetrics]],
        min_limit: int,
        initial_limit: int,
        max_limit: int,
        target_waiting: int,
        poll_interval_sec: float,
        enabled: bool = True,
        component: str = "vllm",
        task_id: str | None = None,
    ):
        self._read_load = read_load
        self._initial_limit = max(1, initial_limit)
        self._min_limit = max(1, min(min_limit, self._initial_limit))
        self._max_limit = max(self._initial_limit, max_limit)
        self._target_waiting = max(0, target_waiting)
        self._poll_interval_sec = max(0.0, poll_interval_sec)
        self._enabled = enabled
        self._component = component
        self._task_id = task_id
        self._limit = self._initial_limit
        self._last_poll_at = float("-inf")
        self._last_status_log_at = float("-inf")
        self._has_successful_read = False
        self._warned_metrics_failure = False

    @property
    def limit(self) -> int:
        return self._limit

    async def refresh(self) -> int:
        # 폴링 주기 안에서는 마지막 제한값을 재사용해 메트릭 서버 부하를 제한합니다.
        now = time.monotonic()
        if not self._enabled or now - self._last_poll_at < self._poll_interval_sec:
            return self._limit
        self._last_poll_at = now

        try:
            load = await self._read_load()
        except Exception as exc:
            # 메트릭을 사용할 수 없으면 기존 고정 동시성을 유지합니다.
            if not self._warned_metrics_failure or now - self._last_status_log_at >= 30.0:
                log_ocr_event(
                    self._component,
                    f"메트릭 조회 실패: 고정 동시성 {self._initial_limit} 사용, 오류={exc}",
                    self._task_id,
                )
                self._last_status_log_at = now
                self._warned_metrics_failure = True
            self._limit = self._initial_limit
            return self._limit

        previous_limit = self._limit
        recovered = self._warned_metrics_failure
        self._warned_metrics_failure = False
        if load.waiting < self._target_waiting:
            self._limit = min(
                self._max_limit,
                self._limit + max(1, self._target_waiting - load.waiting),
            )
        elif load.waiting > self._target_waiting:
            self._limit = max(
                self._min_limit,
                self._limit - max(1, load.waiting - self._target_waiting),
            )

        # 제한 변경, 메트릭 복구, 최초 조회와 30초 heartbeat를 한 줄로 기록합니다.
        should_log = (
            not self._has_successful_read
            or recovered
            or previous_limit != self._limit
            or now - self._last_status_log_at >= 30.0
        )
        self._has_successful_read = True
        if should_log:
            log_ocr_event(
                self._component,
                (
                    f"부하 상태: running={load.running}, waiting={load.waiting}, "
                    f"동시성={previous_limit}->{self._limit}"
                ),
                self._task_id,
            )
            self._last_status_log_at = now
        return self._limit
