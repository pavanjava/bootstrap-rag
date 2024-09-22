from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from api_routes.apis import router
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import time

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
allowed_origins = [
    "*"
]

app = FastAPI(
    title="My FastAPI Application",
    description="This is a FastAPI implementation for RAG application with Swagger UI configurations.",
    version="1.0.0",
    docs_url="/documentation",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "M K Pavan Kumar",
        "linkedin": "https://www.linkedin.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    terms_of_service="https://www.yourwebsite.com/terms/",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


# Custom OpenAPI schema generation (optional)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="RAG APIs",
        version="1.0.0",
        description="This is a custom OpenAPI schema with additional metadata.",
        routes=app.routes,
        tags=[
            {
                "name": "rag",
                "description": "Operations for RAG query.",
            }
        ],
    )
    # Modify openapi_schema as needed
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        logger.info(f"Incoming request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        raise e


# Request Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Processed in {process_time:.4f} seconds")
    return response


# Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


if __name__ == "__main__":
    uvicorn.run(
        "apis:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        workers=1,
    )
