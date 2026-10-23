import logging
from contextlib import asynccontextmanager
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

    app = FastAPI(title="Smart Knowledge Base for Agents", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.kb = kb

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
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            logger.exception("Query failed")
            raise HTTPException(503, "Query failed; check model configuration and server logs") from e

    return app
