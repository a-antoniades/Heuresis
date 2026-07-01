"""Shared memory primitives for QD agent campaigns.

One SQLite DB per campaign (``exp.dir/memory.db``) with sqlite-vec for
semantic search. Framework writes experiment rows; agents append
learnings + query both tables via read-only views.

Memory is opt-in: experiments that want it import :class:`MemoryStore`
and add ``MEMORY`` to their workspace ``tools``. Nothing is loaded at
``import heuresis`` time (google-genai + sqlite-vec live here).
"""

from heuresis.memory.protocol import MemoryIngest
from heuresis.memory.store import MemoryStore

__all__ = ["MemoryIngest", "MemoryStore"]
