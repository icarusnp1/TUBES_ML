from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import STORAGE_DIR
from app.routers import datasets, experiments, registry

app = FastAPI(
    title="Clustering Experiment Platform",
    description="Platform eksperimen clustering: dataset + preprocessing + model + evaluasi.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=STORAGE_DIR), name="files")

app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(registry.router, prefix="/api/registry", tags=["Registry"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])


@app.get("/")
def root():
    return {
        "message": "Clustering Experiment Platform API",
        "docs": "/docs",
        "status": "running"
    }
