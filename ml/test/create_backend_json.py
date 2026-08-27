import os
import json
import csv


COORDINATES_CSV = "spill_coordinates.csv"
RESULTS_CSV = "oil_spill_results.csv"
GEOJSON_DIR = "geojson"

OUTPUT_FILE = "oil_spill_data.json"


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


coordinates = read_csv(COORDINATES_CSV)
results = read_csv(RESULTS_CSV)


results_lookup = {
    row["Scene"]: row
    for row in results
}


scenes = []


for coord in coordinates:

    scene = coord["Scene"]

    result = results_lookup.get(scene, {})

    geojson_path = os.path.join(
        GEOJSON_DIR,
        f"{scene}.geojson"
    )

    geometry = None

    if os.path.exists(geojson_path):

        with open(
            geojson_path,
            "r",
            encoding="utf-8"
        ) as file:

            geometry = json.load(file)


    scene_data = {

        "scene": scene,

        "date": coord["Date"],

        "oil_detected": True,

        "oil_pixels": int(
            coord["Oil Pixels"]
        ),

        "spill_area_km2": float(
            result.get(
                "Spill Area (km2)",
                0
            )
        ),

        "oil_percentage": float(
            result.get(
                "Oil Percentage",
                0
            )
        ),

        "mean_oil_probability": float(
            result.get(
                "Mean Oil Probability",
                0
            )
        ),

        "max_oil_probability": float(
            result.get(
                "Maximum Oil Probability",
                0
            )
        ),

        "center": {

            "latitude": float(
                coord["Center Latitude"]
            ),

            "longitude": float(
                coord["Center Longitude"]
            )

        },

        "bounding_box": {

            "north": float(
                coord["North"]
            ),

            "south": float(
                coord["South"]
            ),

            "east": float(
                coord["East"]
            ),

            "west": float(
                coord["West"]
            )

        },

        "geojson": geometry

    }


    scenes.append(scene_data)


data = {

    "project": "Oil Spill Detection",

    "model": {

        "name": "U-Net",

        "input": "Sentinel-1 SAR",

        "threshold": 0.5

    },

    "total_scenes": len(scenes),

    "scenes": scenes

}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        data,
        file,
        indent=2
    )


print("=" * 70)
print("BACKEND JSON CREATED")
print("=" * 70)

print(
    f"Scenes: {len(scenes)}"
)

print(
    f"Output: {os.path.abspath(OUTPUT_FILE)}"
)

print("\nDONE.")