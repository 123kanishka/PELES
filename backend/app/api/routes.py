from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.model.base import Verdict
from app.model.router import BackendUnavailableError, get_router
from app.schemas import EvaluationRequest, EvaluationResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    try:
        active = get_router().active_backend_name
        status = "ok"
    except BackendUnavailableError:
        active = "unavailable"
        status = "degraded"
    return HealthResponse(
        status=status,
        active_backend=active,
        configured_mode=settings.model_backend,
    )


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    try:
        result = get_router().rank(payload.prompt, payload.response_a, payload.response_b)
    except BackendUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    winner_name = {
        Verdict.MODEL_A: payload.model_a_name,
        Verdict.MODEL_B: payload.model_b_name,
        Verdict.TIE: "Tie",
    }[result.verdict]

    return EvaluationResponse(
        winner_name=winner_name,
        verdict=result.verdict,
        prob_model_a=result.prob_a,
        prob_model_b=result.prob_b,
        prob_tie=result.prob_tie,
        backend=result.backend,
        latency_ms=result.latency_ms,
    )
