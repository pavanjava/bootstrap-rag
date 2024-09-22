from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from api_routes.apis import router
import uvicorn

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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        workers=1,
    )
