from fastapi import APIRouter

router = APIRouter(
    prefix="/api/satellite",
    tags=["Satellite"]
)

@router.get("/")
def get_satellite():

    return {
        "satellite": "Sentinel-1",
        "mission": "Copernicus Sentinel-1",
        "sensor": "SAR",
        "mode": "IW",
        "polarization": "VV",
        "orbit": "Ascending",
        "resolution_m": 10,
        "acquisition_date": "2026-08-27",
        "processing": {
            "preprocessing": True,
            "oil_detection": True,
            "segmentation": True
        },
        "status": "AVAILABLE"
    }