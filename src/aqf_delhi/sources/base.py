"""Resilient HTTP primitives: endpoint failover + bounded retry.

Every source connector shares one failover contract:
  * try base endpoints in configured order;
  * retry each up to N times with exponential backoff;
  * raise :class:`AllEndpointsFailed` only when every candidate is down.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import requests

from aqf_delhi.config import OpsConfig

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """A fetch that exhausted all retry/failover candidates."""


@dataclass
class FetchResult:
    status_code: int
    used_endpoint: str
    attempts: int
    content: bytes
    headers: dict
    text_encoding: str | None = None

    @property
    def text(self) -> str:
        return self.content.decode(self.text_encoding or "utf-8", errors="replace")


def _sleep_backoff(attempt: int, backoff: float) -> None:
    time.sleep(backoff * (2**attempt))


def fetch_with_failover(
    base_endpoints: Iterable[str],
    path: str,
    *,
    ops: OpsConfig,
    method: str = "GET",
    params: dict | None = None,
    headers: dict | None = None,
    body: bytes | None = None,
    json_body: dict | None = None,
    stream: bool = False,
    timeout: float | None = None,
    chunked: bool = False,
    verify: bool = True,
) -> FetchResult:
    """Try ``base_endpoints`` in order; per candidate retry with backoff."""
    from aqf_delhi.sources.base import FetchError  # noqa: F401  (re-export convenience)

    total_attempts = 0
    last_exc: Exception | None = None
    timeout = timeout or ops.request_timeout_seconds

    for endpoint in base_endpoints:
        url = endpoint.rstrip("/") + "/" + path.lstrip("/")
        for attempt in range(ops.retries + 1):
            total_attempts += 1
            try:
                if chunked:
                    pass
                resp = requests.request(
                    method.upper(),
                    url,
                    params=params,
                    headers=headers,
                    data=body,
                    json=json_body,
                    timeout=timeout,
                    stream=stream,
                    verify=verify,
                )
                if resp.status_code >= 500 or resp.status_code in (429, 403, 408):
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code} from {url}", response=resp
                    )
                return FetchResult(
                    status_code=resp.status_code,
                    used_endpoint=endpoint,
                    attempts=total_attempts,
                    content=resp.content,
                    headers=dict(resp.headers),
                    text_encoding=resp.encoding,
                )
            except requests.RequestException as exc:  # includes HTTPError
                last_exc = exc
                logger.warning(
                    "endpoint=%s attempt=%d failed: %s", endpoint, attempt, exc
                )
                if attempt < ops.retries:
                    _sleep_backoff(attempt, ops.retry_backoff_seconds)

    raise FetchError(
        f"all endpoints failed for path={path!r} "
        f"after {total_attempts} attempt(s); last error: {last_exc}"
    )


def decode_csv(result: FetchResult) -> list[dict]:
    """Parse a CSV FetchResult into a list of row dicts."""
    import csv
    import io

    reader = csv.DictReader(io.StringIO(result.text))
    return [dict(row) for row in reader]


def extract_channel(endpoint_url: str, init_utc, lead_h: int) -> str:
    """Deterministic object-store partition key for a forecast field."""
    return f"{endpoint_url}/{init_utc:%Y%m%d/%H}/{lead_h:03d}"


def getstream(
    endpoints: Iterable[str],
    path: str,
    *,
    ops: OpsConfig,
    write_to: Callable[[requests.Response], None],
) -> str:
    """Stream download with failover; returns the used endpoint."""
    from aqf_delhi.sources.base import FetchError  # noqa: F401

    last_exc: Exception | None = None
    for endpoint in endpoints:
        url = endpoint.rstrip("/") + "/" + path.lstrip("/")
        for attempt in range(ops.retries + 1):
            try:
                with requests.get(url, timeout=ops.request_timeout_seconds, stream=True) as resp:
                    resp.raise_for_status()
                    write_to(resp)
                return endpoint
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < ops.retries:
                    _sleep_backoff(attempt, ops.retry_backoff_seconds)
    raise FetchError(f"stream download failed for {path!r}: {last_exc}")