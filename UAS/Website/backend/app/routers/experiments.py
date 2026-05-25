from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any

from app.services.experiment_service import run_experiment, list_experiments, get_experiment
from app.registry import DEFAULT_GAMING_FEATURES

router = APIRouter()


class ExperimentRequest(BaseModel):
    experiment_name: str = "Eksperimen Agglomerative Clustering"
    dataset_id: str
    preprocessing_id: str = "ordinal_standard"
    model_id: str = "agglomerative"
    feature_columns: list[str] = Field(default_factory=lambda: DEFAULT_GAMING_FEATURES)
    model_params: dict[str, Any] = Field(default_factory=lambda: {
        "n_clusters": 4,
        "metric": "euclidean",
        "linkage": "ward"
    })
    dendrogram_sample_size: int = 100


@router.post("/run")
def run(payload: ExperimentRequest):
    try:
        return run_experiment(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
def get_experiments():
    return list_experiments()


@router.get("/{experiment_id}")
def get_experiment_detail(experiment_id: str):
    try:
        return get_experiment(experiment_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
