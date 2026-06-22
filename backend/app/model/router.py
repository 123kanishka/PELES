from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict

from app.core.config import get_settings
from app.model.base import RankingBackend, RankingResult
from app.model.gemma_backend import get_gemma_backend
from app.model.heuristic import HeuristicBackend

logger = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    pass


class _LRUCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._store: OrderedDict[str, RankingResult] = OrderedDict()

    @staticmethod
    def _key(prompt: str, response_a: str, response_b: str) -> str:
        digest = hashlib.sha256()
        for part in (prompt, response_a, response_b):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def get(self, prompt: str, response_a: str, response_b: str) -> RankingResult | None:
        key = self._key(prompt, response_a, response_b)
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, prompt: str, response_a: str, response_b: str, result: RankingResult) -> None:
        key = self._key(prompt, response_a, response_b)
        self._store[key] = result
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)


class RankingRouter:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._heuristic = HeuristicBackend()
        self._cache = (
            _LRUCache(self.settings.response_cache_size)
            if self.settings.enable_response_cache
            else None
        )
        self._active_backend: RankingBackend | None = None

    def _resolve_backend(self) -> RankingBackend:
        if self._active_backend is not None:
            return self._active_backend

        mode = self.settings.model_backend
        gemma = get_gemma_backend()

        if mode == "heuristic":
            backend = self._heuristic
        elif mode == "gemma":
            if not gemma.is_available():
                raise BackendUnavailableError(
                    "PELES_MODEL_BACKEND=gemma but the Gemma-2 LoRA backend is "
                    "unavailable (no CUDA device, missing deps, or no adapter "
                    "configured via PELES_LORA_ADAPTER_PATH)."
                )
            backend = gemma
        else:
            if gemma.is_available():
                backend = gemma
            else:
                logger.info(
                    "Gemma-2 LoRA backend unavailable; serving with the "
                    "heuristic ranking backend instead."
                )
                backend = self._heuristic

        self._active_backend = backend
        return backend

    @property
    def active_backend_name(self) -> str:
        return self._resolve_backend().name

    def rank(self, prompt: str, response_a: str, response_b: str) -> RankingResult:
        if self._cache is not None:
            cached = self._cache.get(prompt, response_a, response_b)
            if cached is not None:
                return cached

        backend = self._resolve_backend()
        result = backend.rank(prompt, response_a, response_b)

        if self._cache is not None:
            self._cache.put(prompt, response_a, response_b, result)
        return result


_router: RankingRouter | None = None


def get_router() -> RankingRouter:
    global _router
    if _router is None:
        _router = RankingRouter()
    return _router
