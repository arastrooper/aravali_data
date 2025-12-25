import rasterio
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from scipy.ndimage import minimum_filter, binary_dilation
from rasterio.warp import calculate_default_transform, reproject, Resampling

# --- CONFIGURATION ---
INPUT_FILE = 'aravalli_dem.hgt'
OUTPUT_IMAGE = 'evidence_map_aravalli.png'

# The "Kill Shot" Logic
ECOLOGICAL_THRESHOLD = 20
GOVT_THRESHOLD = 100
SEARCH_RADIUS_M = 1500


def generate_evidence():
    print(f"1. Loading Elevation Data from {INPUT_FILE}...")

    with rasterio.open(INPUT_FILE) as src:
        dst_crs = 'EPSG:32643'

        src_crs = src.crs
        if src_crs is None:
            src_crs = 'EPSG:4326'

        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        dem_data = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dem_data,
            src_transform=src.transform,
            src_crs=src_crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        print("2. Calculating 'Local Relief' (The Scientific Hack)...")
        pixel_size = float(transform[0])
        if pixel_size <= 0:
            raise ValueError(f"Invalid pixel size after reprojection: {pixel_size}")

        filter_size = int(round(SEARCH_RADIUS_M / pixel_size))
        filter_size = max(3, filter_size)

        filled = np.where(np.isnan(dem_data), np.nanmax(dem_data), dem_data)
        base_level = minimum_filter(filled, size=filter_size, mode='reflect')
        base_level = np.where(np.isnan(dem_data), np.nan, base_level)

        hill_height = dem_data - base_level

        print("3. Applying the '500m Cluster Rule'...")

        mask_govt_peaks = hill_height >= GOVT_THRESHOLD

        buffer_pixels = int(round(500 / pixel_size))
        buffer_pixels = max(1, buffer_pixels)
        y, x = np.ogrid[-buffer_pixels:buffer_pixels + 1, -buffer_pixels:buffer_pixels + 1]
        mask_structure = (x ** 2 + y ** 2) <= buffer_pixels ** 2

        mask_protected_zone = binary_dilation(mask_govt_peaks, structure=mask_structure)

        mask_ecological = hill_height >= ECOLOGICAL_THRESHOLD

        mask_destroyed = np.logical_and(mask_ecological, ~mask_protected_zone)

        total_hill_pixels = int(np.sum(mask_ecological))
        destroyed_pixels = int(np.sum(mask_destroyed))
        protected_pixels = int(np.sum(np.logical_and(mask_ecological, mask_protected_zone)))

        percent_lost = 0.0
        if total_hill_pixels > 0:
            percent_lost = (destroyed_pixels / total_hill_pixels) * 100

        print("\n" + "=" * 40)
        print("RESULTS (WITH 500m BUFFER APPLIED):")
        print(f"Total Ecological Hills:       {total_hill_pixels:,} pixels")
        print(f"Protected (Peaks + Buffer):   {protected_pixels:,} pixels")
        print(f"TRUE DESTROYED AREA:          {destroyed_pixels:,} pixels")
        print(f"TRUE PERCENTAGE LOSS:         {percent_lost:.2f}%")
        print("=" * 40 + "\n")

        return mask_destroyed, transform, dst_crs


def plot_map(mask, transform, crs):
    print("4. Drawing the Map (This takes a moment)...")
    fig, ax = plt.subplots(figsize=(12, 12))

    plot_data = mask.astype(float)
    plot_data[plot_data == 0] = np.nan

    bounds = rasterio.transform.array_bounds(mask.shape[0], mask.shape[1], transform)
    extents = [bounds[0], bounds[2], bounds[1], bounds[3]]

    ax.imshow(plot_data, cmap='Reds_r', extent=extents, alpha=0.8, zorder=10)

    print("   Downloading Satellite Imagery...")
    try:
        cx.add_basemap(
            ax,
            crs=crs,
            source=cx.providers.Esri.WorldImagery,
            attribution=False,
        )
    except Exception:
        print("   (Could not fetch satellite tiles. Check internet.)")

    ax.set_title(
        "THE ARAVALLI 'DEATH ZONE' MAP\nRed Areas = Hills Stripped of Protection (<100m)",
        fontsize=15,
        color='darkred',
        weight='bold',
    )

    ax.set_axis_off()

    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"DONE! Map saved as: {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    try:
        mask, trans, crs = generate_evidence()
        plot_map(mask, trans, crs)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{INPUT_FILE}'. Did you download and rename it?")
