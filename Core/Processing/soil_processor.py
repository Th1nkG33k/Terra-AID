
import numpy as np
import rasterio

from pathlib import Path
from rasterio.warp import reproject, Resampling



# =================================================================
# Loads + resamples soil rasters to match S2 tile profile.
# =================================================================
class SoilProcessor:

    def __init__(self, cfg, path_manager):
        self.cfg = cfg
        self.pm = path_manager   # <-- injected PathManager
        

    def load_and_resample(self, soil_path: Path, target_profile: dict) -> np.ndarray:
        soil_path = Path(soil_path)

        with rasterio.open(soil_path) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_count = src.count

            dst_crs = target_profile["crs"]
            dst_transform = target_profile["transform"]
            dst_height = target_profile["height"]
            dst_width = target_profile["width"]

            soil_resampled = np.zeros((src_count, dst_height, dst_width), dtype=np.float32)

            for b in range(1, src_count + 1):
                band = src.read(b)

                reproject(source=band,
                        destination=soil_resampled[b - 1],
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                )

        return soil_resampled
