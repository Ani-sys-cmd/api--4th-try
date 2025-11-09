# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import routers (these files will be created next)
from routes.upload_routes import router as upload_router
from routes.scan_routes import router as scan_router
from routes.test_routes import router as test_router
from routes.heal_routes import router as heal_router

app = FastAPI(
    title="Hybrid Agentic API Tester",
    description="Local FastAPI backend for scanning uploaded projects, generating API tests, running RL/self-heal loops.",
    version="0.1.0",
)

# CORS - allow local frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}

# Include routers
app.include_router(upload_router, prefix="/api")
app.include_router(scan_router, prefix="/api")
app.include_router(test_router, prefix="/api")
app.include_router(heal_router, prefix="/api")

if __name__ == "__main__":
    # default run for local dev; run with uvicorn backend.main:app --reload --port 8000 in production/dev
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
