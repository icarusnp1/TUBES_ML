from fastapi import APIRouter

from app.registry import PREPROCESSORS, CLUSTER_MODELS, DEFAULT_GAMING_FEATURES

router = APIRouter()


@router.get("/preprocessors")
def list_preprocessors():
    return [
        {
            "id": item.id,
            "name": item.name,
            "description": item.description
        }
        for item in PREPROCESSORS.values()
    ]


@router.get("/models")
def list_models():
    return [
        {
            "id": item.id,
            "name": item.name,
            "description": item.description
        }
        for item in CLUSTER_MODELS.values()
    ]


@router.get("/default-features")
def get_default_features():
    return {
        "default_gaming_features": DEFAULT_GAMING_FEATURES,
        "mandatory_encoding": {
            "GameDifficulty": {
                "Easy": 0,
                "Medium": 1,
                "Hard": 2,
                "reason": "Easy < Medium < Hard sehingga dipakai ordinal encoding."
            }
        }
    }
