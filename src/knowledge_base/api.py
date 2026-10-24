import logging
import os
from contextlib import asynccontextmanager
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .pipeline import KnowledgeBase

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    collection_name: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=20000)


def create_app(settings=None):
    kb = KnowledgeBase(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await run_in_threadpool(kb.telemetry.flush)

    app = FastAPI(
        title="Smart Knowledge Base for Agents",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.kb = kb
    # CORS lets the static frontend (frontend/index.html) call the API; it adds no routes.
    origins = os.getenv("KB_CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080,http://localhost:8766,http://127.0.0.1:8766")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )

    @app.post("/index")
    async def index(file: UploadFile = File(...), collection_name: str = Form(...)):
        try:
            content = await file.read(kb.settings.max_zip_bytes + 1)
            return await run_in_threadpool(kb.index_zip, content, collection_name)
        except (ValueError, zipfile.BadZipFile) as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            logger.exception("Indexing failed")
            raise HTTPException(503, "Indexing failed; see server logs") from e
        finally:
            await file.close()

    @app.post("/query")
    def query(request: QueryRequest):
        try:
            return kb.query(request.collection_name, request.query)
        except FileNotFoundError as e:
            raise HTTPException(404, "unknown collection") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            logger.exception("Query failed")
            raise HTTPException(503, "Query failed; check model configuration and server logs") from e

    return app
