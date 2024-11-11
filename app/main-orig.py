from fastapi import FastAPI
from app.api.api_router import vexo_api_router

app = FastAPI(title="Vexoo API Documentation", version="1.1.0")

# Include the API routes
app.include_router(vexo_api_router, prefix="/api/v1", tags=["Pre-Beta Version 1.1.0"])

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)