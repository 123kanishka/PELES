from __future__ import annotations

import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.model.base import RankingBackend, RankingResult, Verdict

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _repetition_penalty(tokens: list[str], n: int = 3) -> float:
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def _length_prior(tokens: list[str], saturation: int = 220) -> float:
    return float(np.log1p(min(len(tokens), saturation)) / np.log1p(saturation))


class HeuristicBackend(RankingBackend):
    name = "heuristic"

    def __init__(self, *, temperature: float = 1.6, tie_band: float = 0.07) -> None:
        self.temperature = temperature
        self.tie_band = tie_band

    def is_available(self) -> bool:
        return True

    def _relevance_scores(self, prompt: str, response_a: str, response_b: str) -> tuple[float, float]:
        corpus = [prompt, response_a, response_b]
        try:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return 0.0, 0.0
        sims = cosine_similarity(matrix[0], matrix[1:3])[0]
        return float(sims[0]), float(sims[1])

    def _response_score(self, prompt: str, response: str, relevance: float) -> float:
        tokens = _tokenize(response)
        diversity = _lexical_diversity(tokens)
        repetition = _repetition_penalty(tokens)
        length = _length_prior(tokens)
        return (
            0.50 * relevance
            + 0.20 * diversity
            + 0.15 * length
            - 0.15 * repetition
        )

    def rank(self, prompt: str, response_a: str, response_b: str) -> RankingResult:
        start = time.perf_counter()

        rel_a, rel_b = self._relevance_scores(prompt, response_a, response_b)
        score_a = self._response_score(prompt, response_a, rel_a)
        score_b = self._response_score(prompt, response_b, rel_b)

        logits = np.array([score_a, score_b]) / self.temperature
        exp = np.exp(logits - logits.max())
        p_a, p_b = (exp / exp.sum()).tolist()

        if abs(p_a - p_b) < self.tie_band:
            margin_recovered = self.tie_band - abs(p_a - p_b)
            prob_tie = float(margin_recovered)
            scale = 1.0 - prob_tie
            p_a, p_b = p_a * scale, p_b * scale
        else:
            prob_tie = 0.0

        if prob_tie >= max(p_a, p_b):
            verdict = Verdict.TIE
        elif p_a > p_b:
            verdict = Verdict.MODEL_A
        else:
            verdict = Verdict.MODEL_B

        latency_ms = (time.perf_counter() - start) * 1000
        return RankingResult(
            verdict=verdict,
            prob_a=round(p_a, 4),
            prob_b=round(p_b, 4),
            prob_tie=round(prob_tie, 4),
            backend=self.name,
            latency_ms=round(latency_ms, 3),
        )
