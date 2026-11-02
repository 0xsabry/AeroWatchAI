"""AeroWatch AI — opt-in offline mode (Phase 10).

The application needs no network at runtime: models are local files,
history is a local database, and no external AI API exists (Phase 6).
Offline mode makes that guarantee enforceable: when enabled (env
``AEROWATCH_OFFLINE=1``), outbound socket connections to non-loopback
addresses raise immediately, so any accidental external call fails
loudly instead of silently phoning home. Loopback stays allowed —
Streamlit's own local server and browser channel use it.

Default is OFF: enabling is an explicit operator choice, and the guard
is installed once, at application start, before models load.
"""

from __future__ import annotations

import os
import socket

_ENABLED = False
_original_create_connection = socket.create_connection
_original_connect = socket.socket.connect

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_loopback(address) -> bool:
    try:
        host = address[0]
    except (TypeError, IndexError):
        return False
    return host in _LOOPBACK_HOSTS or str(host).startswith("127.")


def _guarded_create_connection(address, *args, **kwargs):
    if not _is_loopback(address):
        raise OSError(
            f"AeroWatch offline mode: blocked outbound connection to "
            f"{address!r}. The application requires no network access; "
            "unset AEROWATCH_OFFLINE to allow it."
        )
    return _original_create_connection(address, *args, **kwargs)


def _guarded_connect(self, address):
    if not _is_loopback(address):
        raise OSError(
            f"AeroWatch offline mode: blocked outbound connection to "
            f"{address!r}."
        )
    return _original_connect(self, address)


def enable_offline_mode() -> bool:
    """Install the guard. Returns True if it is active afterwards."""
    global _ENABLED
    if _ENABLED:
        return True
    socket.create_connection = _guarded_create_connection
    socket.socket.connect = _guarded_connect
    _ENABLED = True
    return True


def disable_offline_mode() -> None:
    global _ENABLED
    socket.create_connection = _original_create_connection
    socket.socket.connect = _original_connect
    _ENABLED = False


def offline_requested() -> bool:
    return os.environ.get("AEROWATCH_OFFLINE", "").strip() in ("1", "true", "yes")


def is_enabled() -> bool:
    return _ENABLED
