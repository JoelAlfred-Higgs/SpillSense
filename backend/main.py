from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import spill
from routers import vessels
from routers import environment
from routers import satellite


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Oil Spill Detection API",
    description="Backend API for oil spill detection, vessel monitoring, environmental conditions, and satellite data",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# INCLUDE ROUTERS
# ============================================================

# Oil Spill API
app.include_router(spill.router)

# Vessel API
app.include_router(vessels.router)

# Environmental API
app.include_router(environment.router)

# Satellite API
app.include_router(satellite.router)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/", tags=["System"])
def root():

    return {
        "message": "Oil Spill Detection API is running",
        "status": "online",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", tags=["System"])
def health_check():

    return {
        "status": "healthy"
    }