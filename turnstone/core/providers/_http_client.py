"""SDK HTTP clients whose ``close()`` wakes a body read blocked in another thread.

:meth:`ModelRegistry.reload` closes a provider's SDK client from the admin
thread while a worker may be blocked reading a streamed response. Closing the
socket's descriptor does not interrupt that read: on Linux it ends with the
server's next byte or FIN, on macOS not even then, so the worker waits out the
read timeout (600 s by default). ``shutdown(2)`` wakes the read on both at once,
so these clients shut each socket down before closing it, and the blocked read
fails immediately with the transport error ``transport_guarded`` already treats
as retryable.
"""

from __future__ import annotations

import contextlib
import functools
import socket
import threading
from typing import TYPE_CHECKING, Any

import httpcore2
import httpx2
import structlog

if TYPE_CHECKING:
    import ssl
    from collections.abc import Iterable, Iterator

log = structlog.get_logger(__name__)


class _ShutdownStream(httpcore2.NetworkStream):
    """The backend's own stream, except that ``close()`` shuts the socket down first.

    The descriptor is closed only once no read or write is in progress. A read that
    has not yet started waiting when the descriptor closes waits on nothing until
    its timeout, which on macOS caught a few percent of reads closed just after they
    began. So a close that finds I/O in progress leaves the descriptor to the last
    call out, which the shutdown has already woken.
    """

    def __init__(self, inner: httpcore2.NetworkStream) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._io_in_progress = 0
        self._close_pending = False

    @contextlib.contextmanager
    def _io(self) -> Iterator[None]:
        with self._lock:
            self._io_in_progress += 1
        try:
            yield
        finally:
            with self._lock:
                self._io_in_progress -= 1
                if self._close_pending and not self._io_in_progress:
                    self._close_pending = False
                    self._inner.close()

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        with self._io():
            return self._inner.read(max_bytes, timeout)

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        with self._io():
            self._inner.write(buffer, timeout)

    def close(self) -> None:
        sock = self._inner.get_extra_info("socket")
        if isinstance(sock, socket.socket):
            # ENOTCONN once the peer has gone, EBADF once closed; neither stops the close.
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        with self._lock:
            if self._io_in_progress:
                self._close_pending = True
            else:
                self._inner.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore2.NetworkStream:
        return _ShutdownStream(self._inner.start_tls(ssl_context, server_hostname, timeout))

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)


class _ShutdownBackend(httpcore2.NetworkBackend):
    def __init__(self, inner: httpcore2.NetworkBackend) -> None:
        self._inner = inner

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.NetworkStream:
        stream = self._inner.connect_tcp(
            host, port, timeout=timeout, local_address=local_address, socket_options=socket_options
        )
        return _ShutdownStream(stream)

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.NetworkStream:
        stream = self._inner.connect_unix_socket(
            path, timeout=timeout, socket_options=socket_options
        )
        return _ShutdownStream(stream)


def _connection_pool(transport: httpx2.BaseTransport) -> Any:
    if not isinstance(transport, httpx2.HTTPTransport):
        raise TypeError(f"expected an httpx2.HTTPTransport, got {type(transport).__name__}")
    # httpx2 keeps its httpcore2 pool, and the pool its network backend, as private
    # attributes; neither has a public seam for swapping the backend after construction.
    pool = transport._pool
    if not isinstance(pool._network_backend, httpcore2.NetworkBackend):
        raise TypeError(f"unexpected network backend {type(pool._network_backend).__name__}")
    return pool


@functools.cache
def _warn_left_unwrapped(reason: str) -> None:
    log.warning(
        "providers.http_client: closing a model client will not wake a blocked stream read",
        reason=reason,
    )


def shutdown_on_close(client: httpx2.Client) -> httpx2.Client:
    """Make ``client.close()`` wake reads blocked on its connections in other threads.

    Covers the default transport and every proxy mount, including the ones the
    SDKs build from the proxy environment variables. Call it before the client's
    first request: a connection keeps the backend it was opened with.

    If an httpx2 or httpcore2 release moves the private attributes this relies on,
    the client is returned as the SDK built it, with one warning, rather than failing
    every model. tests/test_provider_http_client.py pins the shape so CI notices.
    """
    try:
        pools = [
            _connection_pool(transport)
            for transport in (client._transport, *client._mounts.values())
            if transport is not None
        ]
    except (AttributeError, TypeError) as exc:
        _warn_left_unwrapped(f"{type(exc).__name__}: {exc}")
        return client
    for pool in pools:
        if not isinstance(pool._network_backend, _ShutdownBackend):
            pool._network_backend = _ShutdownBackend(pool._network_backend)
    return client
