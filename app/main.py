import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import get_settings

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info(
        "jd-role-classifier starting",
        env=settings.app_env,
        mock_nlp=settings.mock_nlp,
        load_classifier=settings.load_classifier,
    )
    yield
    log.info("jd-role-classifier shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="JD Role Classifier",
        description="NLP skill extraction + O*NET SOC role classification for job descriptions",
        version="0.1.0",
        lifespan=lifespan,
    )

    from app.routers.health import router as health_router
    from app.routers.jd import router as jd_router
    from app.routers.evaluate import router as evaluate_router

    app.include_router(health_router)
    app.include_router(jd_router)
    app.include_router(evaluate_router)

    return app


app = create_app()
