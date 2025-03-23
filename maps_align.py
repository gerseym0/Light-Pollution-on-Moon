import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
import numpy as np

# Paths to input files
elevation_path = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test3\elevation200.tif'   # Elevation map
slope_path     = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test3\slope200.tif'         # Slope map
minerals_path  = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test3\mineral200.tif'       # Mineral map

# Paths to output files for aligned elevation and slope maps
elevation_aligned_path = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test3\elevation_aligned.tif'
slope_aligned_path     = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test3\slope_aligned.tif'
# The mineral map remains unchanged

print("Reading target parameters from the mineral map...")
with rasterio.open(minerals_path) as mineral_src:
    target_crs = mineral_src.crs
    target_transform = mineral_src.transform
    target_width = mineral_src.width
    target_height = mineral_src.height
    print("Target CRS:", target_crs)
    print("Target GeoTransform:", target_transform)
    print(f"Target size: {target_width} x {target_height}")

def reproject_raster(src_path, dst_path, target_crs, target_transform, target_width, target_height, block_size=1024):
    """Reproject the raster using tile-based processing"""
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': target_transform,
            'width': target_width,
            'height': target_height,
            'driver': 'GTiff'  # Save as GeoTIFF, change if necessary
        })
        print(f"\nCreating output file: {dst_path}")
        with rasterio.open(dst_path, 'w', **profile) as dst:
            # Use WarpedVRT for reprojection with the given parameters
            with WarpedVRT(src,
                           crs=target_crs,
                           transform=target_transform,
                           width=target_width,
                           height=target_height,
                           resampling=Resampling.nearest) as vrt:
                for row in range(0, target_height, block_size):
                    num_rows = min(block_size, target_height - row)
                    window = Window(0, row, target_width, num_rows)
                    data = vrt.read(window=window)
                    dst.write(data, window=window)
                    print(f"Processed rows {row} - {row + num_rows} for file {dst_path}")
    print(f"Reprojection complete for file: {src_path} -> {dst_path}")

print("\nStarting reprojection of the elevation map...")
reproject_raster(elevation_path, elevation_aligned_path, target_crs, target_transform, target_width, target_height, block_size=1024)
print("\nElevation map reprojected and saved.")

print("\nStarting reprojection of the slope map...")
reproject_raster(slope_path, slope_aligned_path, target_crs, target_transform, target_width, target_height, block_size=1024)
print("\nSlope map reprojected and saved.")

print("\nProcessing complete. All aligned files have been saved.")
