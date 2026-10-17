"""Create the FastAPI application and register its API routers."""

from fastapi import FastAPI

from app.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.routers import alerts, dashboard, farms, predictions, weather

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)


@app.get("/")
def root():
    return {
        "message": "MicroFarmCast backend is running",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# Register the API feature routers.
app.include_router(farms.router)
app.include_router(weather.router)
app.include_router(predictions.router)
app.include_router(alerts.router)
app.include_router(dashboard.router)