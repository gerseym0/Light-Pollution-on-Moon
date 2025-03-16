import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
import numpy as np

# Paths to input files (specify your own paths)
elevation_path = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\elevation200m.tif'   # Elevation map
slopes_path = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\slope200m.tif'         # Slope map
minerals_path = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\mineral200m.tif'      # Mineral map

# Paths to output files
elevation_aligned_path = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\elevation_align.tif'
slopes_aligned_path = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\slope_align.tif'
minerals_aligned_path = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\mineral_align.tif'

# Target projection: Moon Equirectangular (WKT)
target_crs = (
    'PROJCS["Equirectangular Moon",'
    'GEOGCS["GCS_Moon",'
    'DATUM["D_Moon",'
    'SPHEROID["Moon_localRadius",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
    'PROJECTION["Equirectangular"],'
    'PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],'
    'AXIS["Northing",NORTH]]'
)

# Determine the shift for elevation and slope maps:
# To shift by the meridian, we split the image horizontally (by the number of columns)
with rasterio.open(elevation_path) as elev_src:
    width = elev_src.width
    # shift_pixels is the number of pixels to shift (half the width)
    shift_pixels = width // 2
    # Get the original geotransform and pixel size (assumes square pixel of 200 m)
    orig_transform = elev_src.transform
    pixel_size = orig_transform.a  # Pixel size in X (200 m)
    # Compute the new geotransform: shift in X by shift_pixels * pixel_size
    new_transform = orig_transform * rasterio.Affine.translation(-shift_pixels, 0)

print(f"Calculated shift_pixels = {shift_pixels} and new geotransform for shifting.")

# Function to shift (roll) the raster horizontally in blocks (tiles)
def shift_raster(src_path, dst_path, shift_pixels, block_size=1024):
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        # Update the geotransform: apply horizontal shift in X
        profile.update(transform=src.transform * rasterio.Affine.translation(-shift_pixels, 0))
        
        with rasterio.open(dst_path, 'w', **profile) as dst:
            # Process the image in blocks (by rows)
            for row in range(0, src.height, block_size):
                num_rows = min(block_size, src.height - row)
                window = Window(col_off=0, row_off=row, width=src.width, height=num_rows)
                data = src.read(window=window)
                # np.roll shifts data along axis=2 (columns)
                data_shifted = np.roll(data, shift=shift_pixels, axis=2)
                dst.write(data_shifted, window=window)
    print(f"Finished shifting raster: {src_path} -> {dst_path}")

# Shift elevation and slope maps
print("Starting to shift elevation map...")
shift_raster(elevation_path, elevation_aligned_path, shift_pixels, block_size=1024)
print("Elevation map shifted and saved.\n")

print("Starting to shift slope map...")
shift_raster(slopes_path, slopes_aligned_path, shift_pixels, block_size=1024)
print("Slope map shifted and saved.\n")

# For the mineral map, we need to reproject it to the target projection and match the dimensions and transform
# of the aligned elevation/slope maps.
# We use WarpedVRT for tiled processing.
def reproject_raster(src_path, dst_path, target_crs, target_transform, target_width, target_height, block_size=1024):
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': target_transform,
            'width': target_width,
            'height': target_height
        })
        with rasterio.open(dst_path, 'w', **profile) as dst:
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
    print(f"Finished reprojecting raster: {src_path} -> {dst_path}")

# For the mineral map, use parameters from the aligned elevation map
with rasterio.open(elevation_aligned_path) as aligned_elev:
    target_transform = aligned_elev.transform
    target_width = aligned_elev.width
    target_height = aligned_elev.height

print("Starting to reproject mineral map...")
# Reproject the mineral map
reproject_raster(minerals_path, minerals_aligned_path, target_crs, target_transform, target_width, target_height, block_size=1024)
print("Mineral map reprojected and saved.\n")

print("Processing complete. Aligned files have been saved.")
