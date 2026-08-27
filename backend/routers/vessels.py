from fastapi import APIRouter


# ============================================================
# CREATE ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/vessels",
    tags=["Vessels"]
)


# ============================================================
# GET ALL VESSELS
# ============================================================

@router.get("/")
def get_vessels():

    vessels = [

        {
            "mmsi": "419000001",
            "name": "MV Ocean Star",
            "latitude": 18.9550,
            "longitude": 72.7450,
            "speed": 12,
            "course": 135,
            "timestamp": "2025-05-05T10:30:00",
            "time_since_passing_min": 4,
            "direction_match": True,
            "distance_from_spill_km": 0.59,
            "likelihood": 85,
            "risk": "HIGH"
        },

        {
            "mmsi": "419000002",
            "name": "MV Blue Horizon",
            "latitude": 18.9700,
            "longitude": 72.7800,
            "speed": 15,
            "course": 250,
            "timestamp": "2025-05-05T10:25:00",
            "time_since_passing_min": 15,
            "direction_match": True,
            "distance_from_spill_km": 3.35,
            "likelihood": 70,
            "risk": "HIGH"
        },

        {
            "mmsi": "419000003",
            "name": "MV Sea Pearl",
            "latitude": 19.0200,
            "longitude": 72.7000,
            "speed": 10,
            "course": 80,
            "timestamp": "2025-05-05T10:10:00",
            "time_since_passing_min": 30,
            "direction_match": False,
            "distance_from_spill_km": 9.11,
            "likelihood": 30,
            "risk": "LOW"
        },

        {
            "mmsi": "419000004",
            "name": "MV Coastal Express",
            "latitude": 18.9300,
            "longitude": 72.7700,
            "speed": 18,
            "course": 320,
            "timestamp": "2025-05-05T10:22:00",
            "time_since_passing_min": 8,
            "direction_match": False,
            "distance_from_spill_km": 3.05,
            "likelihood": 35,
            "risk": "LOW"
        }

    ]

    return {
        "count": len(vessels),
        "vessels": vessels
    }


# ============================================================
# GET SINGLE VESSEL
# ============================================================

@router.get("/{mmsi}")
def get_vessel(mmsi: str):

    vessels = get_vessels()["vessels"]

    for vessel in vessels:

        if vessel["mmsi"] == mmsi:
            return vessel

    return {
        "error": "Vessel not found",
        "mmsi": mmsi
    }