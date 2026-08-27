from fastapi import APIRouter

router = APIRouter(
    prefix="/api/environment",
    tags=["Environment"]
)

@router.get("/")
def get_environment():

    return {
        "location": {
            "latitude": 18.9550,
            "longitude": 72.7450
        },
        "wind": {
            "speed_knots": 14,
            "direction_deg": 225,
            "direction": "SW"
        },
        "ocean_current": {
            "speed_knots": 0.8,
            "direction_deg": 240,
            "direction": "WSW"
        },
        "wave": {
            "height_m": 1.4
        },
        "temperature": {
            "water_celsius": 28.4
        },
        "conditions": "Moderate",
        "drift_direction": "WSW",
        "data_source": "Environmental Model",
        "timestamp": "2026-08-27T10:00:00Z"
    }