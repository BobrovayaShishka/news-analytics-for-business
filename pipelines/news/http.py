"""HTTP-клиент: таймауты, повторы, условные запросы, лимит размера и запрет внутренних адресов."""

from __future__ import annotations

import ipaddress
import socket
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx

from alfa.data.interfaces import IngestError

_REDIRECTS = {301, 302, 303, 307, 308}
_RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class FetchResult:
    url: str
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    truncated: bool = False

    @property
    def not_modified(self) -> bool:
        return self.status == 304


def _blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not ip.is_global


def assert_http_scheme(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestError("разрешены только http и https")


def assert_public_http_url(url: str) -> None:
    """Проверка до запроса и после каждого редиректа, уже по разрешённому IP."""
    assert_http_scheme(url)
    host = urlparse(url).hostname or ""
    lowered = host.lower().rstrip(".")
    if lowered in {"localhost", "localhost.localdomain"} or lowered.endswith((".local", ".internal", ".localhost")):
        raise IngestError("адрес отклонён")
    try:
        literal = ipaddress.ip_address(lowered)
    except ValueError:
        literal = None
    if literal is not None:
        if _blocked(literal):
            raise IngestError("адрес отклонён")
        return
    try:
        infos = socket.getaddrinfo(lowered, None)
    except socket.gaierror as exc:
        raise IngestError("адрес отклонён") from exc
    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    if not addresses or any(_blocked(ip) for ip in addresses):
        raise IngestError("адрес отклонён")


class HttpxTransport:
    def __init__(self, settings: dict, sleeper=time.sleep):
        timeout = float(settings["timeout_s"])
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            follow_redirects=False,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
            headers={"User-Agent": settings["user_agent"]},
        )
        self._settings = settings
        self._sleeper = sleeper
        self._next_at: dict[str, float] = {}

    def close(self) -> None:
        self._client.close()

    def get(self, url: str, conditional: dict | None = None) -> FetchResult:
        headers = {"Accept": "text/html,application/xml,application/rss+xml,application/json;q=0.9,*/*;q=0.8"}
        if conditional:
            if conditional.get("etag"):
                headers["If-None-Match"] = conditional["etag"]
            if conditional.get("last_modified"):
                headers["If-Modified-Since"] = conditional["last_modified"]
        delay = float(self._settings["backoff_s"])
        attempts = max(1, int(self._settings["max_retries"]))
        last_error = "запрос не выполнен"
        for attempt in range(attempts):
            self._pace(url)
            try:
                with self._client.stream("GET", url, headers=headers) as response:
                    if response.status_code in _RETRY_STATUS and attempt + 1 < attempts:
                        response.read()
                        self._sleeper(delay)
                        delay *= 2
                        last_error = f"HTTP {response.status_code}"
                        continue
                    body, truncated = self._read_limited(response)
                    return FetchResult(
                        url=str(response.url),
                        status=response.status_code,
                        body=body,
                        headers={key.lower(): value for key, value in response.headers.items()},
                        truncated=truncated,
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = str(exc)
                if attempt + 1 >= attempts:
                    break
                self._sleeper(delay)
                delay *= 2
        return FetchResult(url=url, status=0, error=last_error)

    def _pace(self, url: str) -> None:
        host = urlparse(url).netloc.lower()
        rps = float(self._settings["per_domain_rps"] or 1)
        interval = 1.0 / rps if rps > 0 else 0
        now = time.monotonic()
        earliest = self._next_at.get(host, 0.0)
        if earliest > now:
            self._sleeper(earliest - now)
            now = time.monotonic()
        self._next_at[host] = now + interval

    def _read_limited(self, response: httpx.Response) -> tuple[bytes, bool]:
        limit = int(self._settings["max_response_bytes"])
        chunks: list[bytes] = []
        size = 0
        truncated = False
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            if size + len(chunk) > limit:
                chunks.append(chunk[: max(0, limit - size)])
                truncated = True
                break
            chunks.append(chunk)
            size += len(chunk)
        return b"".join(chunks), truncated


def fetch_checked(url: str, transport, settings: dict, *, user_supplied: bool, conditional: dict | None = None) -> FetchResult:
    """Редиректы обходятся вручную. Для адреса пользователя каждый хоп проверяется на SSRF."""
    current = url
    redirects = 0
    limit = int(settings["max_redirects"])
    while True:
        if user_supplied or redirects:
            assert_public_http_url(current)
        else:
            assert_http_scheme(current)
        result = transport.get(current, conditional=conditional if redirects == 0 else None)
        location = result.headers.get("location")
        if result.status in _REDIRECTS and location:
            redirects += 1
            if redirects > limit:
                raise IngestError("слишком много перенаправлений")
            current = urljoin(current, location)
            continue
        result.url = current
        return result
