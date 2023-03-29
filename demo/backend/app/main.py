from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Dict, List

import torch
from coordgen.models import AutoModelForCoordinationGeneration
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    model_name: str = "t5-small"
    device: str = "cpu"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    model = AutoModelForCoordinationGeneration.from_pretrained(
        settings.model_name, device=torch.device(settings.device)
    )
    app.state.models = {}
    app.state.models["coordgen"] = model
    yield
    app.state.models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/info")
async def get_info() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "model_name": settings.model_name,
    }


class Span(BaseModel):
    start: int
    end: int


class GenerateRequest(BaseModel):
    text: str
    start: int
    end: int


class GenerateResponse(BaseModel):
    text: str
    cc: Span
    conjuncts: List[Span]


@app.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    batch = [(request.text, (request.start, request.end))]
    results = app.state.models["coordgen"].generate(batch)
    raw, coord = results[0]
    return GenerateResponse(
        text=raw,
        cc=Span(start=coord.cc[0], end=coord.cc[1]),
        conjuncts=[Span(start=conj[0], end=conj[1]) for conj in coord.conjuncts],
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> Response:
    if str(exc).startswith("CUDA out of memory."):
        return JSONResponse({"detail": "Service Unavailable"}, status_code=503)
    raise exc
