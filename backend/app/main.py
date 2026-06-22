import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "%s v%s starting (mode=%s)", settings.app_name, settings.api_version, settings.model_backend
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    description=(
        "Predictive LLM response ranking service. Wraps a LoRA-tuned, "
        "4-bit-quantized Gemma-2-9B sequence classifier (with a CPU "
        "heuristic fallback) behind a REST API for programmatic "
        "downstream consumption."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {"service": settings.app_name, "version": settings.api_version, "docs": "/docs"}
