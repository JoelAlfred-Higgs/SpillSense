import rasterio
import matplotlib.pyplot as plt

file = r"YOUR_TIF_FILE_PATH"

with rasterio.open(file) as src:
    image = src.read(1)

plt.figure(figsize=(12, 6))
plt.imshow(image, cmap="gray")
plt.colorbar(label="Sigma0 VV (dB)")
plt.title("Sentinel-1 SAR — Sigma0 VV")
plt.axis("off")
plt.show()