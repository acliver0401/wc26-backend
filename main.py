"""
WC26 World Cup 2026 Prediction System — FastAPI Backend.

Features:
  - Ensemble ML predictor (RF + GB + ExtraTrees + Logistic)
  - Environmental factors: elevation, live weather, humidity, flight fatigue
  - Injury & team-status features
  - APScheduler for periodic data refresh (weather → injuries → predictions)
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.api import router as api_router
from services.scheduler import start_scheduler, shutdown_scheduler, run_initial_refresh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("wc26")


# ---------------------------------------------------------------------------
# FastAPI lifespan — replaces deprecated on_event("startup") / ("shutdown")
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: run initial data refresh + start background scheduler.
       Shutdown: gracefully stop scheduler."""
    logger.info("=== WC26 Backend starting ===")

    # 1. Run the full pipeline once so data exists immediately
    try:
        summary = await run_initial_refresh()
        logger.info("Initial refresh: %s", summary)
    except Exception:
        logger.exception("Initial refresh failed; API will serve fallback data.")

    # 2. Start periodic scheduler (default: every 6 hours)
    #    Change cron_expr to "0 2 * * *" for daily at 02:00 local time.
    start_scheduler(cron_expr="0 */6 * * *")

    logger.info("=== WC26 Backend ready ===")
    yield  # application runs here

    # Shutdown
    logger.info("=== WC26 Backend shutting down ===")
    shutdown_scheduler()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WC26 世界杯预测系统",
    description=(
        "RandomForest + GradientBoosting + ExtraTrees + Logistic Ensemble "
        "with Environmental Features (elevation / live weather / humidity / "
        "precipitation / flight fatigue) and Injury/Form factors. "
        "Data refreshed every 6 hours via APScheduler."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# CORS: allow local dev + Vercel production origins.
# Set CORS_ORIGINS env var to add custom domains, comma-separated.
cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://wc26-frontend-tau.vercel.app",
]
extra = os.getenv("CORS_ORIGINS", "").strip()
if extra:
    cors_origins.extend(origin.strip() for origin in extra.split(",") if origin.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
async def root():
    return {"service": "WC26 Prediction API", "version": "2.1.0", "docs": "/docs"}
