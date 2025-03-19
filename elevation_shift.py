import numpy as np
from osgeo import gdal

def shift_dem(input_path, output_path, tile_size=1024):
    input_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    band = input_ds.GetRasterBand(1)
    
    xsize = input_ds.RasterXSize
    ysize = input_ds.RasterYSize
    split_x = xsize // 2
    nodata = band.GetNoDataValue()
    
    driver = gdal.GetDriverByName("ISIS3")
    output_ds = driver.Create(output_path, xsize, ysize, 1, band.DataType)
    
    output_ds.SetMetadata(input_ds.GetMetadata())
    output_ds.SetProjection(input_ds.GetProjection())
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    
    output_band = output_ds.GetRasterBand(1)
    output_band.SetScale(0.5)
    output_band.SetOffset(1737400.0)
    output_band.SetNoDataValue(nodata)
    
    for y in range(0, ysize, tile_size):
        ysz = min(tile_size, ysize - y)
        
        for x in range(0, xsize, tile_size):
            xsz = min(tile_size, xsize - x)
            src_x = (x + split_x) % xsize
            
            if src_x + xsz > xsize:
                part1_width = xsize - src_x
                part2_width = xsz - part1_width
                
                part1 = band.ReadAsArray(src_x, y, part1_width, ysz)
                part2 = band.ReadAsArray(0, y, part2_width, ysz)
                raw_tile = np.concatenate((part1, part2), axis=1)
            else:
                raw_tile = band.ReadAsArray(src_x, y, xsz, ysz)
            
            # Нормализация данных
            raw_tile = np.where(raw_tile == nodata, np.nan, raw_tile)
            if np.any(np.isfinite(raw_tile)):
                np.clip(raw_tile, 
                       a_min=np.nanmin(raw_tile), 
                       a_max=np.nanmax(raw_tile), 
                       out=raw_tile)
                
            output_band.WriteArray(raw_tile, x, y)
    
    output_ds.SetMetadataItem("MinimumLongitude", "0")
    output_ds.SetMetadataItem("MaximumLongitude", "360")
    
    input_ds = None
    output_ds = None

def verify_output(input_path, output_path):
    # Проверяем сохранение значений
    def print_stats(ds, name):
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray(0, 0, 1000, 1000)
        print(f"\n{name} (RAW):")
        print(f"Min: {np.nanmin(arr)}, Max: {np.nanmax(arr)}")
        
        physical = arr * band.GetScale() + band.GetOffset()
        print(f"\n{name} (Physical):")
        print(f"Min: {np.nanmin(physical)}, Max: {np.nanmax(physical)}")
    
    print("Проверка входного файла:")
    input_ds = gdal.Open(input_path)
    print_stats(input_ds, "Input")
    
    print("\nПроверка выходного файла:")
    output_ds = gdal.Open(output_path)
    print_stats(output_ds, "Output")

# Пример использования
input_file = r'C:\Users\Ildar\Desktop\Moonpol\data prep\WAC_GLD100_V1.0_GLOBAL_with_LOLA_30M_POLE.16bit.lp.demprep.cub'
output_file = r'C:\Users\Ildar\Desktop\Moonpol\data prep\shifted_output.cub'

shift_dem(input_file, output_file)
verify_output(input_file, output_file)