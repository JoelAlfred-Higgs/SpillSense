import json
import os
from pathlib import Path


# ============================================================
# ML DATA LOCATION
# ============================================================

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "ml"
    / "oil_spill_data.json"
)


# ============================================================
# LOAD ML DATA
# ============================================================

def load_spill_data():

    data_path = Path(
        os.getenv(
            "SPILL_DATA_PATH",
            DEFAULT_DATA_PATH
        )
    )

    if not data_path.exists():

        return {
            "project": "Oil Spill Detection",
            "model": {
                "name": "U-Net",
                "input": "Sentinel-1 SAR",
                "threshold": 0.5
            },
            "total_scenes": 0,
            "scenes": []
        }

    with open(
        data_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# GET ALL SPILLS
# ============================================================

def get_all_spills():

    data = load_spill_data()

    return data.get(
        "scenes",
        []
    )


# ============================================================
# GET ONE SCENE
# ============================================================

def get_spill_by_scene(scene):

    spills = get_all_spills()

    for spill in spills:

        if spill.get("scene") == scene:

            return spill

    return None


# ============================================================
# SPILL INFORMATION
# ============================================================

def get_spill_information():

    spills = get_all_spills()

    if not spills:

        return {
            "status": "NO_DATA",
            "total_scenes": 0,
            "message": "No ML prediction data available."
        }

    latest = max(
        spills,
        key=lambda x: x.get(
            "date",
            ""
        )
    )

    return {

        "status": (
            "DETECTED"
            if latest.get(
                "oil_detected",
                False
            )
            else "NOT_DETECTED"
        ),

        "scene": latest.get(
            "scene"
        ),

        "date": latest.get(
            "date"
        ),

        "area_km2": latest.get(
            "spill_area_km2",
            0
        ),

        "oil_percentage": latest.get(
            "oil_percentage",
            0
        ),

        "mean_oil_probability": latest.get(
            "mean_oil_probability",
            0
        ),

        "max_oil_probability": latest.get(
            "max_oil_probability",
            0
        ),

        "center": latest.get(
            "center"
        ),

        "bounding_box": latest.get(
            "bounding_box"
        ),

        "satellite": "Sentinel-1",

        "data": "GRD SAR"
    }


# ============================================================
# OIL SPILL DETECTION
# ============================================================

def detect_oil_spill():

    spills = get_all_spills()

    if not spills:

        return {
            "status": "NO_DATA",
            "total_scenes": 0,
            "detection_method": "U-Net semantic segmentation",
            "satellite": "Sentinel-1",
            "product": "GRD",
            "input": "SAR"
        }

    detected_scenes = [
        spill
        for spill in spills
        if spill.get(
            "oil_detected",
            False
        )
    ]

    return {

        "status": (
            "DETECTED"
            if detected_scenes
            else "NOT_DETECTED"
        ),

        "total_scenes": len(
            spills
        ),

        "detected_scenes": len(
            detected_scenes
        ),

        "detection_method":
            "U-Net semantic segmentation",

        "satellite":
            "Sentinel-1",

        "product":
            "GRD",

        "input":
            "SAR"
    }


# ============================================================
# TOTAL SPILL AREA
# ============================================================

def calculate_spill_area():

    spills = get_all_spills()

    detected_spills = [
        spill
        for spill in spills
        if spill.get(
            "oil_detected",
            False
        )
    ]

    total_area = sum(

        float(
            spill.get(
                "spill_area_km2",
                0
            )
        )

        for spill in detected_spills

    )

    return {

        "area_km2":
            round(
                total_area,
                4
            )

    }


# ============================================================
# SEVERITY
# ============================================================

def determine_severity(area_km2):

    if area_km2 >= 100:

        return "HIGH"

    elif area_km2 >= 25:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# GET GEOJSON
# ============================================================

def get_spill_boundary(scene=None):

    spills = get_all_spills()

    if not spills:

        return None

    # Specific scene requested

    if scene:

        spill = get_spill_by_scene(
            scene
        )

        if not spill:

            return None

    else:

        # Latest scene

        spill = max(
            spills,
            key=lambda x:
                x.get(
                    "date",
                    ""
                )
        )

    return spill.get(
        "geojson"
    )