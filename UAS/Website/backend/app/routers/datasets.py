from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.dataset_service import save_dataset, list_datasets, preview_dataset

router = APIRouter()


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    try:
        return save_dataset(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
def get_datasets():
    return list_datasets()


@router.get("/{dataset_id}/preview")
def get_dataset_preview(dataset_id: str, limit: int = 10):
    try:
        return preview_dataset(dataset_id, limit)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
