from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_db, require_api_key
from app.core.config import get_settings
from app.db.models import BacktestRun, DatasetManifest
from app.schemas.backtest import BacktestRequest, BacktestRunRead
from app.services.backtest_service import execute_backtest
from app.services.model_service import get_model_record
from app.workers.tasks import run_backtest_task

router = APIRouter(prefix="/backtests", tags=["backtests"])


def _execute_local(run_id: str) -> None:
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        execute_backtest(db, run_id)


@router.post("", response_model=BacktestRunRead, status_code=status.HTTP_202_ACCEPTED)
def create_backtest(
    payload: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> BacktestRunRead:
    if db.get(DatasetManifest, payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND", "message": payload.dataset_id})
    try:
        model = get_model_record(db, payload.model_version)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": str(exc)}) from exc
    configuration = payload.model_dump(exclude={"dataset_id", "model_version"})
    run = BacktestRun(dataset_id=payload.dataset_id, model_version_id=model.id, configuration=configuration)
    db.add(run)
    db.commit()
    db.refresh(run)
    if get_settings().task_always_eager:
        background_tasks.add_task(_execute_local, run.id)
    else:
        run_backtest_task.delay(run.id)
    return BacktestRunRead.model_validate(run)


@router.get("/{run_id}", response_model=BacktestRunRead)
def get_backtest(
    run_id: str,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> BacktestRunRead:
    run = db.get(BacktestRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "BACKTEST_NOT_FOUND", "message": run_id})
    return BacktestRunRead.model_validate(run)
