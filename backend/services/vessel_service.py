# ============================================================
# VESSEL SERVICE
# ============================================================
# Handles vessel information and vessel risk analysis
# around a detected oil spill.
#
# NOTE:
# The current vessel data is sample/mock data.
# Later, this service can be connected to real vessel
# tracking/AIS data.
# ============================================================


# ============================================================
# GET VESSEL INFORMATION
# ============================================================

def get_vessel_information():

    vessels = [

        {
            "name": "MV Ocean Star",
            "latitude": 18.9550,
            "longitude": 72.7450,
            "speed": 12,
            "course": 135,
            "time_since_passing_min": 4,
            "direction_match": True,
            "distance_from_spill_km": 0.59,
            "likelihood": 85,
            "risk": "HIGH"
        },

        {
            "name": "MV Blue Horizon",
            "latitude": 18.9700,
            "longitude": 72.7800,
            "speed": 15,
            "course": 250,
            "time_since_passing_min": 15,
            "direction_match": True,
            "distance_from_spill_km": 3.35,
            "likelihood": 70,
            "risk": "HIGH"
        },

        {
            "name": "MV Sea Pearl",
            "latitude": 19.0200,
            "longitude": 72.7000,
            "speed": 10,
            "course": 80,
            "time_since_passing_min": 30,
            "direction_match": False,
            "distance_from_spill_km": 9.11,
            "likelihood": 30,
            "risk": "LOW"
        },

        {
            "name": "MV Coastal Express",
            "latitude": 18.9300,
            "longitude": 72.7700,
            "speed": 18,
            "course": 320,
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
# GET NEARBY VESSELS
# ============================================================

def get_nearby_vessels():

    vessel_data = get_vessel_information()

    nearby_vessels = []

    for vessel in vessel_data["vessels"]:

        if vessel["distance_from_spill_km"] <= 5:

            nearby_vessels.append(vessel)

    return {
        "count": len(nearby_vessels),
        "vessels": nearby_vessels
    }


# ============================================================
# CALCULATE VESSEL RISK
# ============================================================

def calculate_vessel_risk(distance_from_spill_km,
                          direction_match,
                          time_since_passing_min):

    score = 0


    # --------------------------------------------------------
    # Distance factor
    # --------------------------------------------------------

    if distance_from_spill_km <= 1:

        score += 50

    elif distance_from_spill_km <= 5:

        score += 30

    elif distance_from_spill_km <= 10:

        score += 10


    # --------------------------------------------------------
    # Direction factor
    # --------------------------------------------------------

    if direction_match:

        score += 30


    # --------------------------------------------------------
    # Time factor
    # --------------------------------------------------------

    if time_since_passing_min <= 5:

        score += 20

    elif time_since_passing_min <= 15:

        score += 10


    # --------------------------------------------------------
    # Determine likelihood and risk
    # --------------------------------------------------------

    likelihood = min(score, 100)


    if likelihood >= 70:

        risk = "HIGH"

    elif likelihood >= 40:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    return {
        "likelihood": likelihood,
        "risk": risk
    }


# ============================================================
# ANALYZE VESSEL
# ============================================================

def analyze_vessel(vessel):

    risk_result = calculate_vessel_risk(

        vessel["distance_from_spill_km"],

        vessel["direction_match"],

        vessel["time_since_passing_min"]

    )


    result = vessel.copy()

    result["likelihood"] = risk_result["likelihood"]

    result["risk"] = risk_result["risk"]


    return result


# ============================================================
# ANALYZE ALL VESSELS
# ============================================================

def analyze_all_vessels():

    vessel_data = get_vessel_information()

    analyzed_vessels = []


    for vessel in vessel_data["vessels"]:

        analyzed_vessel = analyze_vessel(vessel)

        analyzed_vessels.append(analyzed_vessel)


    return {
        "count": len(analyzed_vessels),
        "vessels": analyzed_vessels
    }