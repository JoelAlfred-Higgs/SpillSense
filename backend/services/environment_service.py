# ============================================================
# ENVIRONMENT SERVICE
# ============================================================
# Handles environmental conditions around the detected
# oil spill.
# ============================================================


# ============================================================
# GET ENVIRONMENTAL INFORMATION
# ============================================================

def get_environment_information():

    environment_data = {

        "location": {
            "latitude": 18.9500,
            "longitude": 72.7500
        },

        "wind": {
            "speed_kmph": 18,
            "direction": 135,
            "direction_name": "Southeast"
        },

        "sea": {
            "wave_height_m": 1.4,
            "wave_direction": 120,
            "current_speed_kmph": 2.8,
            "current_direction": 110
        },

        "weather": {
            "temperature_c": 29,
            "humidity_percent": 78,
            "condition": "Partly Cloudy"
        },

        "visibility": {
            "distance_km": 8.5
        },

        "spill_effect": {
            "drift_direction": "Southeast",
            "estimated_drift_speed_kmph": 2.8,
            "environmental_risk": "HIGH"
        }

    }

    return environment_data


# ============================================================
# GET WIND INFORMATION
# ============================================================

def get_wind_information():

    wind_data = {

        "speed_kmph": 18,

        "direction": 135,

        "direction_name": "Southeast"

    }

    return wind_data


# ============================================================
# GET SEA CONDITIONS
# ============================================================

def get_sea_conditions():

    sea_data = {

        "wave_height_m": 1.4,

        "wave_direction": 120,

        "current_speed_kmph": 2.8,

        "current_direction": 110

    }

    return sea_data


# ============================================================
# GET WEATHER INFORMATION
# ============================================================

def get_weather_information():

    weather_data = {

        "temperature_c": 29,

        "humidity_percent": 78,

        "condition": "Partly Cloudy",

        "visibility_km": 8.5

    }

    return weather_data


# ============================================================
# GET SPILL DRIFT INFORMATION
# ============================================================

def get_spill_drift_information():

    drift_data = {

        "drift_direction": "Southeast",

        "estimated_drift_speed_kmph": 2.8,

        "environmental_risk": "HIGH"

    }

    return drift_data