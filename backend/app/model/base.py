from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    MODEL_A = "model_a"
    MODEL_B = "model_b"
    TIE = "tie"


@dataclass(frozen=True)
class RankingResult:
    verdict: Verdict
    prob_a: float
    prob_b: float
    prob_tie: float
    backend: str
    latency_ms: float


class RankingBackend(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def rank(self, prompt: str, response_a: str, response_b: str) -> RankingResult:
        ...
