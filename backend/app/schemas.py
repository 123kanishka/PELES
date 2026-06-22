from pydantic import BaseModel, ConfigDict, Field

from app.model.base import Verdict


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_a_name: str = Field(..., min_length=1, max_length=120, examples=["GPT-4o"])
    model_b_name: str = Field(..., min_length=1, max_length=120, examples=["Claude 3.5 Sonnet"])
    response_a: str = Field(..., min_length=1, examples=["Paris is the capital of France."])
    response_b: str = Field(..., min_length=1, examples=["The capital of France is Paris, located on the Seine."])
    prompt: str = Field(
        default="",
        max_length=8000,
        description="Original prompt both responses were generated for. Optional but improves ranking quality.",
    )


class EvaluationResponse(BaseModel):
    winner_name: str
    verdict: Verdict
    prob_model_a: float
    prob_model_b: float
    prob_tie: float
    backend: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    active_backend: str
    configured_mode: str
