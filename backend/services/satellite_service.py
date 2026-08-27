# ============================================================
# SATELLITE SERVICE
# ============================================================
# Handles satellite information and Sentinel-1 SAR data
# processing for the oil spill detection system.
# ============================================================


# ============================================================
# GET SATELLITE INFORMATION
# ============================================================

def get_satellite_information():

    satellite_data = {

        "satellite": "Sentinel-1",

        "mission": {
            "agency": "European Space Agency (ESA)",
            "mission_type": "Earth Observation",
            "sensor": "Synthetic Aperture Radar (SAR)"
        },

        "data": {
            "product": "GRD",
            "resolution_m": 10,
            "polarization": [
                "VV",
                "VH"
            ],
            "acquisition_mode": "IW"
        },

        "image": {
            "latitude": 18.9500,
            "longitude": 72.7500,
            "acquisition_date": "2026-08-27",
            "cloud_cover_percent": 0
        },

        "processing": {
            "preprocessing": [
                "Thermal Noise Removal",
                "Radiometric Calibration",
                "Terrain Correction"
            ],
            "detection_method": "SAR-based oil spill detection",
            "status": "PROCESSED"
        }

    }

    return satellite_data


# ============================================================
# GET SATELLITE DATASET INFORMATION
# ============================================================

def get_sentinel1_dataset():

    dataset = {

        "satellite": "Sentinel-1",
        "sensor": "SAR",
        "product": "GRD",
        "acquisition_mode": "IW",

        "polarization": [
            "VV",
            "VH"
        ],

        "resolution_m": 10,

        "data_type": "Synthetic Aperture Radar",

        "status": "AVAILABLE"

    }

    return dataset


# ============================================================
# SATELLITE DATA STATUS
# ============================================================

def get_satellite_status():

    return {

        "satellite": "Sentinel-1",
        "status": "ONLINE",
        "data_source": "Sentinel-1 SAR",
        "processing_status": "READY"

    }