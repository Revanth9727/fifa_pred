from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # project root .env

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from wcpredict.web.service import PredictionService


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="World Cup Predictor", version="0.1.0")
service = PredictionService()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def _no_icon() -> Response:
    return Response(status_code=204)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return service.health()


@app.get("/api/metrics")
def metrics() -> dict:
    return service.metrics()


@app.post("/api/retrain")
def retrain(n_runs: int | None = None, seed: int | None = None) -> dict:
    return service.start_retrain_job(n_runs=n_runs, seed=seed)


@app.get("/api/dashboard")
def dashboard() -> dict:
    return service.dashboard()


@app.post("/api/results/refresh")
def refresh_results(n_runs: int | None = None, seed: int | None = None) -> dict:
    return service.start_refresh_job(n_runs=n_runs, seed=seed)


@app.get("/api/upcoming")
def upcoming_matches() -> dict:
    return service.get_upcoming_matches()


@app.get("/api/results/job")
def refresh_job() -> dict:
    return service.job_status()


@app.post("/api/simulate")
def simulate(n_runs: int | None = None, seed: int | None = None) -> dict:
    return service.start_simulation_job(n_runs=n_runs, seed=seed)


@app.get("/api/live-tournament")
def live_tournament() -> dict:
    return service.load_live_state()


@app.post("/api/live-tournament/reset")
def reset_live_tournament() -> dict:
    return service.reset_live_state()


@app.post("/api/live-tournament/matches/{match_id}/simulate")
def simulate_live_match(match_id: str) -> dict:
    return service.simulate_live_match(match_id)


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    return service.chat(req.message, req.history)


def main() -> None:
    uvicorn.run(
        "wcpredict.web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
