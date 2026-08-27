import os
import csv
import numpy as np
import rasterio


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = r"images"
PREDICTION_DIR = r"predictions"

OUTPUT_CSV = r"oil_spill_results.csv"


# ============================================================
# FIND SCENES
# ============================================================

scenes = sorted([

    name for name in os.listdir(PREDICTION_DIR)

    if os.path.isdir(
        os.path.join(
            PREDICTION_DIR,
            name
        )
    )

])


print(
    f"Found {len(scenes)} prediction scenes."
)


# ============================================================
# RESULTS
# ============================================================

results = []


for scene in scenes:

    print(
        f"Processing: {scene}"
    )


    # --------------------------------------------------------
    # Prediction mask
    # --------------------------------------------------------

    mask_path = os.path.join(

        PREDICTION_DIR,

        scene,

        "prediction_mask.tif"

    )


    if not os.path.exists(mask_path):

        print(
            "  Prediction mask not found."
        )

        continue


    # --------------------------------------------------------
    # Read mask
    # --------------------------------------------------------

    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(1)

        transform = src.transform

        crs = src.crs

        width = src.width

        height = src.height


    # --------------------------------------------------------
    # Area calculation
    # --------------------------------------------------------

    oil_pixels = int(
        np.sum(mask > 0)
    )


    pixel_width = abs(
        transform.a
    )

    pixel_height = abs(
        transform.e
    )


    pixel_area = (

        pixel_width
        *
        pixel_height

    )


    area_m2 = (

        oil_pixels
        *
        pixel_area

    )


    area_km2 = (

        area_m2
        /
        1_000_000

    )


    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    # Extract date from scene name

    date = scene.replace(
        "_",
        "-"
    )


    # Known scene names

    if scene == "2018_09_26":

        date = "2018-09-26"

    elif scene.startswith(
        "2018_12_19"
    ):

        date = "2018-12-19"

    elif scene == "20191015":

        date = "2019-10-15"

    elif scene == "20200224_b":

        date = "2020-02-24"

    elif scene == "20200319b":

        date = "2020-03-19"


    # --------------------------------------------------------
    # Add result
    # --------------------------------------------------------

    results.append({

        "Scene": scene,

        "Date": date,

        "Width": width,

        "Height": height,

        "CRS": str(crs),

        "Oil Pixels": oil_pixels,

        "Pixel Area (m2)": round(
            pixel_area,
            2
        ),

        "Spill Area (m2)": round(
            area_m2,
            2
        ),

        "Spill Area (km2)": round(
            area_km2,
            4
        )

    })


# ============================================================
# WRITE CSV
# ============================================================

fieldnames = [

    "Scene",

    "Date",

    "Width",

    "Height",

    "CRS",

    "Oil Pixels",

    "Pixel Area (m2)",

    "Spill Area (m2)",

    "Spill Area (km2)"

]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 70)
print("OIL SPILL RESULTS")
print("=" * 70)


for result in results:

    print(
        f"{result['Date']} | "
        f"{result['Scene']} | "
        f"{result['Spill Area (km2)']:.4f} km²"
    )


print("\n")
print(
    "CSV created:"
)

print(
    os.path.abspath(
        OUTPUT_CSV
    )
)

print("\nDONE.")