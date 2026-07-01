from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_db, require_api_key
from app.schemas.prediction import ExplanationResponse, PredictionRequest, PredictionResponse
from app.services.prediction_service import generate_prediction

router = APIRouter(tags=["predictions"])


@router.post("/predict-market", response_model=PredictionResponse)
def predict_market(
    payload: PredictionRequest,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> PredictionResponse:
    try:
        return PredictionResponse.model_validate(generate_prediction(db, payload))
    except LookupError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_OR_STATION_UNAVAILABLE", "message": str(exc), "production_trading_allowed": False},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "PREDICTION_REJECTED", "message": str(exc)}) from exc


@router.post("/explain-prediction", response_model=ExplanationResponse)
def explain_prediction(
    payload: PredictionRequest,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> ExplanationResponse:
    try:
        result = generate_prediction(db, payload)
    except LookupError as exc:
        raise HTTPException(status_code=503, detail={"code": "MODEL_UNAVAILABLE", "message": str(exc)}) from exc
    distribution = result["prediction_distribution"]
    return ExplanationResponse(
        model_version=result["model_version"],
        predicted_value=result["predicted_value"],
        top_positive_drivers=result["top_positive_drivers"],
        top_negative_drivers=result["top_negative_drivers"],
        model_disagreement=distribution.get("sigma"),
        station_bias=None,
        historical_analogues=result["historical_analogues"],
        warnings=result["warnings"],
    )
