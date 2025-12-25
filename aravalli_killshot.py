import rasterio
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from scipy.ndimage import minimum_filter, binary_dilation
from rasterio.warp import calculate_default_transform, reproject, Resampling

# --- CONFIGURATION ---
INPUT_FILE = 'aravalli_dem.hgt'
OUTPUT_IMAGE = 'evidence_map_aravalli.png'
COMPARISON_IMAGE = 'evidence_map_aravalli_comparison.png'

# The "Kill Shot" Logic
ECOLOGICAL_THRESHOLD = 20
GOVT_THRESHOLD = 100
SEARCH_RADIUS_M = 1500


def _print_stats(header: str, stats: dict) -> None:
    print(f"Total Ecological Hills:       {stats['total']:,} pixels")
    print(f"Protected Area:               {stats['protected']:,} pixels")
    print(f"Destroyed Area:               {stats['destroyed']:,} pixels")
    print(f"Percentage Loss:              {stats['percent']:.2f}%")
    print("=" * 40 + "\n")


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

        print("3. Applying protection rules...")

        mask_ecological = hill_height >= ECOLOGICAL_THRESHOLD
        mask_govt_peaks = hill_height >= GOVT_THRESHOLD

        mask_destroyed_direct = np.logical_and(mask_ecological, ~mask_govt_peaks)

        buffer_pixels = int(round(500 / pixel_size))
        buffer_pixels = max(1, buffer_pixels)
        y, x = np.ogrid[-buffer_pixels:buffer_pixels + 1, -buffer_pixels:buffer_pixels + 1]
        mask_structure = (x ** 2 + y ** 2) <= buffer_pixels ** 2

        mask_protected_zone = binary_dilation(mask_govt_peaks, structure=mask_structure)
        mask_destroyed_buffer = np.logical_and(mask_ecological, ~mask_protected_zone)

        total_hill_pixels = int(np.sum(mask_ecological))
        protected_direct = int(np.sum(mask_govt_peaks))
        destroyed_direct = int(np.sum(mask_destroyed_direct))
        protected_buffer = int(np.sum(np.logical_and(mask_ecological, mask_protected_zone)))
        destroyed_buffer = int(np.sum(mask_destroyed_buffer))

        percent_direct = (destroyed_direct / total_hill_pixels) * 100 if total_hill_pixels else 0.0
        percent_buffer = (destroyed_buffer / total_hill_pixels) * 100 if total_hill_pixels else 0.0

        print("\n" + "=" * 40)
        print("RESULTS (NO BUFFER / GOVT RULE ONLY):")
        _print_stats(
            "RESULTS (NO BUFFER / GOVT RULE ONLY):",
            {
                "total": total_hill_pixels,
                "protected": protected_direct,
                "destroyed": destroyed_direct,
                "percent": percent_direct,
            },
        )

        print("=" * 40)
        print("RESULTS (WITH 500m BUFFER APPLIED):")
        _print_stats(
            "RESULTS (WITH 500m BUFFER APPLIED):",
            {
                "total": total_hill_pixels,
                "protected": protected_buffer,
                "destroyed": destroyed_buffer,
                "percent": percent_buffer,
            },
        )

        return {
            "mask_buffer": mask_destroyed_buffer,
            "mask_direct": mask_destroyed_direct,
            "transform": transform,
            "crs": dst_crs,
            "stats_buffer": {
                "total": total_hill_pixels,
                "protected": protected_buffer,
                "destroyed": destroyed_buffer,
                "percent": percent_buffer,
            },
            "stats_direct": {
                "total": total_hill_pixels,
                "protected": protected_direct,
                "destroyed": destroyed_direct,
                "percent": percent_direct,
            },
        }


def plot_map(mask, transform, crs, output_path=OUTPUT_IMAGE, title=None):
    print(f"4. Drawing the Map ({output_path})...")
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

    map_title = (
        title
        or "THE ARAVALLI 'DEATH ZONE' MAP\nRed Areas = Hills Stripped of Protection (<100m)"
    )
    ax.set_title(map_title, fontsize=15, color='darkred', weight='bold')

    ax.set_axis_off()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"DONE! Map saved as: {output_path}")
    plt.show()


def plot_comparison(mask_direct, mask_buffer, transform, crs, output_path=COMPARISON_IMAGE):
    print(f"5. Drawing side-by-side comparison map ({output_path})...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    bounds = rasterio.transform.array_bounds(mask_direct.shape[0], mask_direct.shape[1], transform)
    extents = [bounds[0], bounds[2], bounds[1], bounds[3]]

    configs = [
        (mask_direct, "Destruction (Govt Rule Only)"),
        (mask_buffer, "Destruction (500m Buffer Applied)"),
    ]

    for ax, (mask, subtitle) in zip(axes, configs):
        plot_data = mask.astype(float)
        plot_data[plot_data == 0] = np.nan
        ax.imshow(plot_data, cmap='Reds_r', extent=extents, alpha=0.8, zorder=10)
        try:
            cx.add_basemap(
                ax,
                crs=crs,
                source=cx.providers.Esri.WorldImagery,
                attribution=False,
            )
        except Exception:
            print("   (Could not fetch satellite tiles for comparison map. Check internet.)")
        ax.set_title(subtitle, fontsize=14, color='darkred', weight='bold')
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"DONE! Comparison map saved as: {output_path}")
    plt.show()


if __name__ == "__main__":
    try:
        results = generate_evidence()
        plot_map(
            results["mask_buffer"],
            results["transform"],
            results["crs"],
            output_path=OUTPUT_IMAGE,
            title="THE ARAVALLI 'DEATH ZONE' MAP\nRed Areas = Hills Outside 500m Buffers",
        )
        plot_comparison(
            results["mask_direct"],
            results["mask_buffer"],
            results["transform"],
            results["crs"],
            output_path=COMPARISON_IMAGE,
        )
    except FileNotFoundError:
        print(f"ERROR: Could not find '{INPUT_FILE}'. Did you download and rename it?")
