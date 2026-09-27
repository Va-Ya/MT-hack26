"""Single-origin Render entrypoint; the ML/backend modules remain independent."""
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.app import app as api
api.servers=[{'url':'/api','description':'Same-origin Backend and ML'}]
api.openapi_schema=None

@asynccontextmanager
async def lifespan(app):
    if os.getenv('DEMO_ENABLED','0')=='1' and not os.getenv('INGEST_TOKEN'):
        raise RuntimeError('Public demo requires a private INGEST_TOKEN for operator endpoints')
    async with api.router.lifespan_context(api):
        yield

app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
app.mount('/api',api)
site=Path(os.getenv('FRONTEND_DIR','dashboard/dist')).resolve()
documentation=Path('docs/site')
if documentation.is_dir():app.mount('/documentation',StaticFiles(directory=documentation,html=True))

@app.get('/{path:path}',include_in_schema=False)
def frontend(path:str):
    candidate=(site/path).resolve()
    if not candidate.is_relative_to(site):raise HTTPException(404)
    if candidate.is_file():return FileResponse(candidate)
    if path.startswith(('assets/','api/','documentation/')) or '.' in Path(path).name:
        raise HTTPException(404)
    index=site/'index.html'
    if not index.is_file():raise HTTPException(503,'Build dashboard first')
    return FileResponse(index)
