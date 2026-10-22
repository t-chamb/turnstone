"""Turnstone's SDK HTTP clients wake a blocked read when closed from another thread."""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

import anthropic
import httpcore2
import httpx2
import openai
import pytest

from turnstone.core.providers._http_client import (
    _ShutdownBackend,
    _ShutdownStream,
    _warn_left_unwrapped,
    shutdown_on_close,
)


def _backends(client: httpx2.Client) -> list[httpcore2.NetworkBackend]:
    transports = [client._transport, *client._mounts.values()]
    return [t._pool._network_backend for t in transports if isinstance(t, httpx2.HTTPTransport)]


@pytest.mark.parametrize(
    "default_client", [openai.DefaultHttpxClient, anthropic.DefaultHttpxClient]
)
def test_default_transport_and_proxy_mounts_shut_down_on_close(
    monkeypatch: pytest.MonkeyPatch, default_client: type[httpx2.Client]
) -> None:
    """Every transport the SDK's default client builds, including the proxy mounts
    it derives from the environment, gets the shutdown-on-close backend; a bump of
    httpx2 or httpcore2 that moves the pool or backend attribute fails here."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    client = shutdown_on_close(default_client())
    try:
        backends = _backends(client)
        assert len(backends) >= 3, "default transport plus one mount per proxy variable"
        assert all(isinstance(backend, _ShutdownBackend) for backend in backends)
    finally:
        client.close()


def test_shutdown_on_close_is_idempotent() -> None:
    client = shutdown_on_close(shutdown_on_close(openai.DefaultHttpxClient()))
    try:
        (backend,) = _backends(client)
        assert isinstance(backend, _ShutdownBackend)
        assert not isinstance(backend._inner, _ShutdownBackend)
    finally:
        client.close()


def _unwrapped_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "will not wake" in r.getMessage()]


def test_an_unknown_transport_leaves_the_client_as_built(caplog: pytest.LogCaptureFixture) -> None:
    """A transport this module can't wrap fails safe to the SDK's own client, warned once."""
    _warn_left_unwrapped.cache_clear()
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200))
    with caplog.at_level(logging.WARNING, logger="turnstone.core.providers._http_client"):
        for _ in range(2):
            client = httpx2.Client(transport=transport)
            assert shutdown_on_close(client) is client
            assert client.get("http://model.invalid/").status_code == 200
    [message] = _unwrapped_warnings(caplog)
    assert "MockTransport" in message


def test_a_moved_private_attribute_leaves_the_client_as_built(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An httpx2 release that renames the pool's backend wraps nothing, not some transports."""
    _warn_left_unwrapped.cache_clear()
    client = httpx2.Client(mounts={"http://proxy.invalid": httpx2.HTTPTransport()})
    default_backend = client._transport._pool._network_backend
    renamed = next(iter(client._mounts.values()))._pool
    renamed._backend = renamed.__dict__.pop("_network_backend")
    try:
        with caplog.at_level(logging.WARNING, logger="turnstone.core.providers._http_client"):
            assert shutdown_on_close(client) is client
        assert client._transport._pool._network_backend is default_backend
        [message] = _unwrapped_warnings(caplog)
        assert "AttributeError" in message
    finally:
        renamed._network_backend = renamed._backend
        client.close()


class _HeldRead(httpcore2.NetworkStream):
    """A stream whose read blocks until released, recording when it is closed."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self.reading = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.reading.set()
        self.release.wait(timeout=5.0)
        return b""

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        raise AssertionError("not written")

    def close(self) -> None:
        self.closed.set()

    def get_extra_info(self, info: str) -> Any:
        return self._sock if info == "socket" else None


def test_close_during_a_read_shuts_down_now_and_closes_when_the_read_returns() -> None:
    """Closing the descriptor under a read that has not yet started waiting leaves it
    waiting on nothing until its timeout, so the close is left to the read on its way
    out; the shutdown, which is what wakes the read, still happens at once."""
    ours, peer = socket.socketpair()
    peer.settimeout(5.0)
    inner = _HeldRead(ours)
    stream = _ShutdownStream(inner)
    reader = threading.Thread(target=stream.read, args=(1,))
    reader.start()
    try:
        assert inner.reading.wait(timeout=5.0)
        stream.close()
        assert peer.recv(1) == b"", "the shutdown reached the peer"
        assert not inner.closed.is_set(), "the descriptor closed under the read"
        inner.release.set()
        reader.join(timeout=5.0)
        assert inner.closed.is_set()
    finally:
        inner.release.set()
        reader.join(timeout=5.0)
        ours.close()
        peer.close()


def test_close_with_no_read_in_progress_closes_at_once_and_again_harmlessly() -> None:
    ours, peer = socket.socketpair()
    inner = _HeldRead(ours)
    stream = _ShutdownStream(inner)
    try:
        stream.close()
        assert inner.closed.is_set()
        ours.close()
        stream.close()
    finally:
        ours.close()
        peer.close()


def test_close_wakes_a_read_blocked_in_another_thread() -> None:
    """Closing the stream ends a read blocked in another thread at once, while the
    peer is still connected and silent. Without the shutdown the read waits for
    the peer on Linux and for the read timeout on macOS."""
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(5.0)
    port = listener.getsockname()[1]
    release_peer = threading.Event()

    def serve() -> None:
        conn, _ = listener.accept()
        with conn:
            release_peer.wait(timeout=5.0)

    server = threading.Thread(target=serve)
    server.start()
    stream = _ShutdownBackend(httpcore2.SyncBackend()).connect_tcp("127.0.0.1", port, timeout=5.0)
    outcome: list[tuple[str, float]] = []

    def read() -> None:
        started = time.monotonic()
        try:
            data = stream.read(1, timeout=5.0)
        except httpcore2.ReadError:
            outcome.append(("read_error", time.monotonic() - started))
        else:
            outcome.append(("eof" if data == b"" else "data", time.monotonic() - started))

    reader = threading.Thread(target=read)
    reader.start()
    try:
        time.sleep(0.2)  # let the reader enter the blocking socket read
        stream.close()
        reader.join(timeout=5.0)
    finally:
        release_peer.set()
        reader.join(timeout=5.0)
        server.join(timeout=5.0)
        listener.close()
    assert outcome, "the read never returned"
    ((kind, elapsed),) = outcome
    # The kernel reports the shutdown either as end-of-stream or as an error on
    # the closed descriptor, depending on which call the reader was in.
    assert kind in {"read_error", "eof"}
    assert elapsed < 2.0
