import os
import math
import logging
import numpy as np
from osgeo import gdal, osr
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class LunarRasterProcessor:
    def __init__(self, reference_path=None, target_res=200):
        """
        Если задан reference_path, используем его геотрансформацию и проекцию в качестве эталона.
        """
        self.target_res = target_res
        if reference_path:
            ds_ref = gdal.Open(reference_path)
            if ds_ref is None:
                raise ValueError(f"Не удалось открыть референсный файл: {reference_path}")
            self.ref_gt = ds_ref.GetGeoTransform()
            self.ref_proj = ds_ref.GetProjection()
            self.ref_size = (ds_ref.RasterXSize, ds_ref.RasterYSize)
            ds_ref = None
        else:
            # Если reference_path не задан, создаём глобальную сетку на основе луны
            self.ref_gt = (-100.0, target_res, 0, 2729300.0, 0, -target_res)
            self.ref_proj = self._create_projection()
            # Вычисляем размер из геотрансформации и bounds (примерно)
            self.ref_size = (54583, 27293)

    def _create_projection(self):
        """Создает WKT для Equirectangular Moon с нулевыми смещениями"""
        wkt = '''
        PROJCS["Moon_Equirectangular",
            GEOGCS["GCS_Moon",
                DATUM["D_Moon",
                    SPHEROID["Moon",1737400,0]],
            PRIMEM["Reference_Meridian",0],
            UNIT["degree",0.0174532925199433]],
        PROJECTION["Equirectangular"],
        PARAMETER["standard_parallel_1",0],
        PARAMETER["central_meridian",0],
        PARAMETER["false_easting",0],
        PARAMETER["false_northing",0],
        UNIT["metre",1]]
        '''
        return wkt.strip()

    def create_tile_grid(self, tile_size=4096):
        tiles = []
        full_size_x, full_size_y = self.ref_size
        num_tiles_x = math.ceil(full_size_x / tile_size)
        num_tiles_y = math.ceil(full_size_y / tile_size)
        
        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                xoff = tx * tile_size
                yoff = ty * tile_size
                xsize = min(tile_size, full_size_x - xoff)
                ysize = min(tile_size, full_size_y - yoff)
                
                # Вычисляем bounds на основе референсной геотрансформации
                gt = self.ref_gt
                min_x = gt[0] + xoff * gt[1]
                max_y = gt[3] + yoff * gt[5]  # gt[5] отрицательный
                
                bounds = [
                    min_x,
                    max_y + ysize * abs(gt[5]),
                    min_x + xsize * gt[1],
                    max_y
                ]
                tile = {
                    'tx': tx,
                    'ty': ty,
                    'xoff': xoff,
                    'yoff': yoff,
                    'xsize': xsize,
                    'ysize': ysize,
                    'bounds': bounds,
                    'geotransform': (
                        min_x, gt[1], 0, max_y, 0, gt[5]
                    ),
                    'projection_wkt': self.ref_proj
                }
                tiles.append(tile)
        return tiles

    def process_dataset(self, in_path, out_path, resample_alg=gdal.GRA_NearestNeighbour, tile_size=4096):
        """Обрабатывает входной растр, приводя его к референсной геометрии и проекции"""
        tiles = self.create_tile_grid(tile_size)
        temp_dir = os.path.join(os.path.dirname(out_path), 'tiles')
        os.makedirs(temp_dir, exist_ok=True)

        tasks = []
        for tile in tiles:
            tile_path = os.path.join(
                temp_dir,
                f"{os.path.basename(out_path)}_tile_{tile['tx']}_{tile['ty']}.tif"
            )
            tasks.append((in_path, tile_path, tile, resample_alg, self.ref_proj))

        with Pool(cpu_count(), initializer=self._init_worker) as pool:
            results = pool.map(self._process_tile, tasks)

        # Фильтруем успешно обработанные тайлы
        valid_tiles = [r[0] for r in results if r[1]]
        if not valid_tiles:
            raise RuntimeError("Ни один из тайлов не обработался корректно.")

        vrt_path = out_path + '.vrt'
        vrt = gdal.BuildVRT(vrt_path, valid_tiles)
        vrt.FlushCache()

        gdal.Translate(out_path, vrt_path,
                       creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'])
        os.remove(vrt_path)
        return out_path

    @staticmethod
    def _init_worker():
        gdal.AllRegister()

    @staticmethod
    def _process_tile(args):
        in_path, out_path, tile, resample_alg, proj_wkt = args
        try:
            ds = gdal.Warp('', in_path,
                           format='MEM',
                           outputBounds=tile['bounds'],
                           xRes=tile['geotransform'][1],
                           yRes=abs(tile['geotransform'][5]),
                           dstSRS=proj_wkt,
                           resampleAlg=resample_alg)
            if ds is None:
                raise RuntimeError("gdal.Warp вернул None")
            
            driver = gdal.GetDriverByName('GTiff')
            tile_ds = driver.Create(
                out_path, 
                tile['xsize'], 
                tile['ysize'], 
                ds.RasterCount,
                ds.GetRasterBand(1).DataType,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            tile_ds.SetGeoTransform(tile['geotransform'])
            tile_ds.SetProjection(proj_wkt)
            
            for b in range(ds.RasterCount):
                arr = ds.GetRasterBand(b+1).ReadAsArray()
                tile_ds.GetRasterBand(b+1).WriteArray(arr)
            
            tile_ds.FlushCache()
            tile_ds = None
            ds = None
            return (out_path, True)
        except Exception as e:
            logging.error(f"Tile error {out_path}: {str(e)}")
            return (out_path, False)

    def verify_alignment(self, raster_paths):
        params = []
        for path in raster_paths:
            ds = gdal.Open(path)
            params.append({
                'size': (ds.RasterXSize, ds.RasterYSize),
                'gt': ds.GetGeoTransform(),
                'proj': ds.GetProjection()
            })
            ds = None
        
        base = params[0]
        for i, p in enumerate(params[1:]):
            if p['size'] != base['size']:
                raise ValueError(f"Size mismatch: {raster_paths[0]} vs {raster_paths[i+1]}")
            if not np.allclose(p['gt'], base['gt'], atol=1e-6):
                raise ValueError(f"Geotransform mismatch: {raster_paths[0]} vs {raster_paths[i+1]}")
            if p['proj'] != base['proj']:
                raise ValueError(f"Projection mismatch: {raster_paths[0]} vs {raster_paths[i+1]}")
        
        logging.info("All rasters aligned perfectly!")

if __name__ == "__main__":
    # Используем карту высот в качестве референсной для выравнивания
    ref_file = r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\elevation200m.tif'
    config = {
        'datasets': [
            {
                'type': 'elevation',
                'input': ref_file,
                'output': r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\aligned_elevation.tif',
                'resample': gdal.GRA_NearestNeighbour
            },
            {
                'type': 'minerals',
                'input': r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\mineral200m.tif',
                'output': r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\aligned_mineral.tif',
                'resample': gdal.GRA_NearestNeighbour
            },
            {
                'type': 'slope',
                'input': r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\slope200m.tif',
                'output': r'C:\Users\Ildar\Desktop\Moonpol\data prep\test2\aligned_slope.tif',
                'resample': gdal.GRA_NearestNeighbour
            }
        ],
        'tile_size': 4096
    }

    processor = LunarRasterProcessor(reference_path=ref_file)

    try:
        processed_files = []
        for dataset in config['datasets']:
            logging.info(f"Processing {dataset['type']}...")
            result = processor.process_dataset(
                dataset['input'],
                dataset['output'],
                dataset['resample'],
                config['tile_size']
            )
            processed_files.append(result)
        
        processor.verify_alignment(processed_files)
        logging.info("All datasets processed successfully!")

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
