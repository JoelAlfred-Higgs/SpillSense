import os
import csv
import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Transformer


# ============================================================
# SETTINGS
# ============================================================

PREDICTION_DIR = r"predictions"

OUTPUT_CSV = r"spill_coordinates.csv"


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


# ============================================================
# PROCESS EACH SCENE
# ============================================================

for scene in scenes:

    print(
        f"\nProcessing: {scene}"
    )


    mask_path = os.path.join(

        PREDICTION_DIR,

        scene,

        "prediction_mask.tif"

    )


    if not os.path.exists(mask_path):

        print(
            "Prediction mask not found."
        )

        continue


    # ========================================================
    # READ MASK
    # ========================================================

    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(1)

        transform = src.transform

        crs = src.crs

        width = src.width

        height = src.height


    # ========================================================
    # FIND OIL PIXELS
    # ========================================================

    rows, cols = np.where(
        mask > 0
    )


    if len(rows) == 0:

        print(
            "No oil detected."
        )

        continue


    # ========================================================
    # PIXEL CENTER COORDINATES
    # ========================================================

    xs, ys = xy(
        transform,
        rows,
        cols
    )


    xs = np.array(xs)
    ys = np.array(ys)


    # ========================================================
    # BOUNDING BOX IN PROJECTED CRS
    # ========================================================

    min_x = xs.min()
    max_x = xs.max()

    min_y = ys.min()
    max_y = ys.max()


    # ========================================================
    # CENTROID
    # ========================================================

    center_x = xs.mean()
    center_y = ys.mean()


    # ========================================================
    # CONVERT TO LAT/LON
    # ========================================================

    transformer = Transformer.from_crs(

        crs,

        "EPSG:4326",

        always_xy=True

    )


    center_lon, center_lat = transformer.transform(

        center_x,

        center_y

    )


    west_lon, south_lat = transformer.transform(

        min_x,

        min_y

    )


    east_lon, north_lat = transformer.transform(

        max_x,

        max_y

    )


    # ========================================================
    # DATE
    # ========================================================

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

    else:

        date = scene


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "Scene": scene,

        "Date": date,

        "Oil Pixels": len(rows),

        "Center Latitude": round(
            center_lat,
            6
        ),

        "Center Longitude": round(
            center_lon,
            6
        ),

        "North": round(
            north_lat,
            6
        ),

        "South": round(
            south_lat,
            6
        ),

        "East": round(
            east_lon,
            6
        ),

        "West": round(
            west_lon,
            6
        )

    })


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Oil pixels: {len(rows):,}"
    )

    print(
        f"Center: "
        f"{center_lat:.6f}, "
        f"{center_lon:.6f}"
    )

    print(
        f"North: {north_lat:.6f}"
    )

    print(
        f"South: {south_lat:.6f}"
    )

    print(
        f"East: {east_lon:.6f}"
    )

    print(
        f"West: {west_lon:.6f}"
    )


# ============================================================
# WRITE CSV
# ============================================================

fieldnames = [

    "Scene",

    "Date",

    "Oil Pixels",

    "Center Latitude",

    "Center Longitude",

    "North",

    "South",

    "East",

    "West"

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
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)

print(
    "SPILL COORDINATES COMPLETE"
)

print("=" * 70)

print(
    f"CSV created: "
    f"{os.path.abspath(OUTPUT_CSV)}"
)

print("\nDONE.")