from fastapi import APIRouter, HTTPException

from services.spill_detection_service import (
    get_all_spills,
    get_spill_by_scene,
    get_spill_information,
    detect_oil_spill,
    calculate_spill_area,
    determine_severity,
    get_spill_boundary
)


router = APIRouter(
    prefix="/api/spill",
    tags=["Oil Spill Detection"]
)


# ============================================================
# ALL SPILLS
# ============================================================

@router.get("/")
def get_spills():

    spills = get_all_spills()

    scenes = []

    for spill in spills:

        scenes.append({
            "scene": spill.get("scene"),
            "date": spill.get("date"),
            "oil_detected": spill.get("oil_detected", False),
            "oil_pixels": spill.get("oil_pixels", 0),
            "spill_area_km2": spill.get("spill_area_km2", 0),
            "oil_percentage": spill.get("oil_percentage", 0),
            "mean_oil_probability": spill.get(
                "mean_oil_probability", 0
            ),
            "max_oil_probability": spill.get(
                "max_oil_probability", 0
            ),
            "center": spill.get("center"),
            "bounding_box": spill.get("bounding_box")
        })

    return {
        "total_scenes": len(scenes),
        "scenes": scenes
    }


# ============================================================
# SUMMARY
# ============================================================

@router.get("/summary/latest")
def get_latest_summary():

    return get_spill_information()


# ============================================================
# DETECTION STATUS
# ============================================================

@router.get("/detection/status")
def detection_status():

    return detect_oil_spill()


# ============================================================
# TOTAL AREA
# ============================================================

@router.get("/area/total")
def total_area():

    result = calculate_spill_area()

    result["severity"] = determine_severity(
        result["area_km2"]
    )

    return result


# ============================================================
# SINGLE SCENE
# ============================================================

@router.get("/{scene}")
def get_single_scene(scene: str):

    spill = get_spill_by_scene(scene)

    if spill is None:

        raise HTTPException(
            status_code=404,
            detail="Scene not found"
        )

    return spill


# ============================================================
# GEOJSON
# ============================================================

@router.get("/{scene}/geojson")
def scene_geojson(scene: str):

    geometry = get_spill_boundary(scene)

    if geometry is None:

        raise HTTPException(
            status_code=404,
            detail="GeoJSON not found for scene"
        )

    return geometry