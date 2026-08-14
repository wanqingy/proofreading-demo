"""Raise the HTTP connection-pool ceiling for the GCS/CAVE clients.

Tube builds were bottlenecked by ``urllib3``'s "Connection pool is full, discarding
connection: storage.googleapis.com. Connection pool size: 10" -- not by our own thread count.
The real concurrency is ``fill_branch(workers=N)`` x *chunks-per-cutout*, because CloudVolume
fans every cutout out over its underlying chunks itself
(``cloudvolume/datasource/precomputed/image/rx.py`` -> ``schedule_jobs(concurrency=20)``), and
our 64^3 boxes span several native source chunks. That lands well above the ``pool_maxsize=10``
default (``requests.adapters.DEFAULT_POOLSIZE``) inside every ``google.cloud.storage.Client``
that ``cloudfiles`` creates.

Exceeding the pool doesn't fail requests -- urllib3 discards and re-opens the connection, so
each excess fetch pays a fresh TCP+TLS handshake. Raising the ceiling keeps the parallelism we
want and just lets the pool hold it.
"""

from __future__ import annotations

DEFAULT_POOL_SIZE = 64

_tuned = False


def tune_connection_pool(size: int = DEFAULT_POOL_SIZE) -> None:
    """Make new ``requests`` sessions use a ``size``-connection pool.

    MUST run before any client is constructed: a ``Session`` binds its ``HTTPAdapter`` (and the
    adapter its ``PoolManager``) at creation, so patching afterwards leaves existing clients on
    the old ceiling.

    Patches ``HTTPAdapter.__init__`` rather than assigning ``requests.adapters.DEFAULT_POOLSIZE``
    -- that constant is only read as a *default argument value*, which Python binds when
    ``requests.adapters`` is imported, so reassigning it later silently does nothing.

    Process-wide by design: this raises a ceiling (idle connections just sit in the pool), so
    letting CAVE's sessions benefit too is intended, not a side effect to avoid.
    """
    global _tuned
    if _tuned:
        return
    import requests.adapters

    orig_init = requests.adapters.HTTPAdapter.__init__

    def _init(self, *args, pool_connections=size, pool_maxsize=size, **kwargs):
        orig_init(self, *args, pool_connections=pool_connections, pool_maxsize=pool_maxsize, **kwargs)

    requests.adapters.HTTPAdapter.__init__ = _init
    _tuned = True
