import numpy as np
from osgeo import gdal

def shift_dem(input_path, output_path, tile_size=1024):
    print("Opening input dataset...")
    input_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    band = input_ds.GetRasterBand(1)
    
    xsize = input_ds.RasterXSize
    ysize = input_ds.RasterYSize
    split_x = xsize // 2
    nodata = band.GetNoDataValue()
    
    print(f"Image dimensions: {xsize} x {ysize}")
    print(f"Calculated split (in pixels): {split_x}")
    
    print("Creating output dataset using ISIS3 driver...")
    driver = gdal.GetDriverByName("ISIS3")
    output_ds = driver.Create(output_path, xsize, ysize, 1, band.DataType)
    
    # Copy metadata and projection from input dataset
    output_ds.SetMetadata(input_ds.GetMetadata())
    output_ds.SetProjection(input_ds.GetProjection())
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    
    output_band = output_ds.GetRasterBand(1)
    # Set scale and offset (for conversion from RAW to Physical values)
    output_band.SetScale(0.5)
    output_band.SetOffset(1737400.0)
    output_band.SetNoDataValue(nodata)
    
    print("Starting tile-based processing...")
    # Process the image in tiles
    for y in range(0, ysize, tile_size):
        ysz = min(tile_size, ysize - y)
        for x in range(0, xsize, tile_size):
            xsz = min(tile_size, xsize - x)
            # Calculate the source x offset with horizontal shift
            src_x = (x + split_x) % xsize
            
            # If the tile wraps around the image edge, split and concatenate parts
            if src_x + xsz > xsize:
                part1_width = xsize - src_x
                part2_width = xsz - part1_width
                
                print(f"Tile at (x={x}, y={y}) wraps around. Reading two parts: width1={part1_width}, width2={part2_width}.")
                part1 = band.ReadAsArray(src_x, y, part1_width, ysz)
                part2 = band.ReadAsArray(0, y, part2_width, ysz)
                raw_tile = np.concatenate((part1, part2), axis=1)
            else:
                raw_tile = band.ReadAsArray(src_x, y, xsz, ysz)
            
            # Normalize data: replace nodata with NaN
            raw_tile = np.where(raw_tile == nodata, np.nan, raw_tile)
            # Clip the values to the tile's range if there are any finite values
            if np.any(np.isfinite(raw_tile)):
                np.clip(raw_tile, a_min=np.nanmin(raw_tile), a_max=np.nanmax(raw_tile), out=raw_tile)
                
            output_band.WriteArray(raw_tile, x, y)
            print(f"Processed tile at (x={x}, y={y}) of size ({xsz} x {ysz}).")
    
    # Set additional metadata items
    output_ds.SetMetadataItem("MinimumLongitude", "0")
    output_ds.SetMetadataItem("MaximumLongitude", "360")
    
    input_ds = None
    output_ds = None
    print("Processing complete. Output dataset saved.")


def verify_output(input_path, output_path):
    def print_stats(ds, label):
        band = ds.GetRasterBand(1)
        # Read a 1000x1000 block from the top-left for statistics
        arr = band.ReadAsArray(0, 0, 1000, 1000)
        print(f"\n{label} (RAW):")
        print(f"Min: {np.nanmin(arr)}, Max: {np.nanmax(arr)}")
        physical = arr * band.GetScale() + band.GetOffset()
        print(f"\n{label} (Physical):")
        print(f"Min: {np.nanmin(physical)}, Max: {np.nanmax(physical)}")
    
    print("Verifying input dataset:")
    input_ds = gdal.Open(input_path)
    print_stats(input_ds, "Input")
    
    print("\nVerifying output dataset:")
    output_ds = gdal.Open(output_path)
    print_stats(output_ds, "Output")
    input_ds = None
    output_ds = None


# Example usage:
input_file = r'C:\Users\Ildar\Desktop\Moonpol\data prep\WAC_GLD100_V1.0_GLOBAL_with_LOLA_30M_POLE.16bit.lp.demprep.cub'
output_file = r'C:\Users\Ildar\Desktop\Moonpol\data prep\shifted_output.cub'

print("Starting DEM shift operation...")
shift_dem(input_file, output_file)
print("\nDEM shift operation complete. Now verifying output...")
verify_output(input_file, output_file)
