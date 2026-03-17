"""Routes for running translation experiments and serving log files."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from yaduha.experiments import RESULTS_DIR, ExperimentConfig, ExperimentResult, run_experiment

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResult)
async def run_experiment_endpoint(config: ExperimentConfig):
    """Run a translation experiment across all configured providers, models, and sentences.

    All events are logged to ``results/<name>.jsonl``. The response includes
    per-sentence results for every (provider, model) combination.
    """
    try:
        return run_experiment(config)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/logs")
async def list_logs():
    """List available JSONL log files in the results directory."""
    if not RESULTS_DIR.exists():
        return {"files": []}
    return {"files": sorted(f.name for f in RESULTS_DIR.glob("*.jsonl"))}


@router.get("/logs/{filename}")
async def get_log(filename: str):
    """Serve a JSONL log file as plain text."""
    if not filename.endswith(".jsonl") or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    log_path = RESULTS_DIR / filename
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    return PlainTextResponse(log_path.read_text(encoding="utf-8"))
