from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# =================================================================
#  Handles DEM loading + resampling to match S2 tile profile.
# =================================================================
class DEMProcessor:

    def __init__(self, cfg, path_manager):

        self.cfg = cfg
        self.pm = path_manager   # PathManager injected by DatasetProcessingManager


    # ---------------------------------------------------------
    #    Load DEM and resample it to match the Sentinel‑2 tile grid.
    #
    #    Returns:
    #        dem_resampled: 2D float32 array (height, width)
    # ---------------------------------------------------------
    def load_and_resample(self, dem_path: Path, target_profile: dict) -> np.ndarray:

        dem_path = Path(dem_path)

        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            src_transform = src.transform
            src_crs = src.crs

            dst_crs = target_profile["crs"]
            dst_transform = target_profile["transform"]
            dst_height = target_profile["height"]
            dst_width = target_profile["width"]

            dem_resampled = np.zeros((dst_height, dst_width), dtype=np.float32)

            reproject(source=dem,
                      destination=dem_resampled,
                      src_transform=src_transform,
                      src_crs=src_crs,
                      dst_transform=dst_transform,
                      dst_crs=dst_crs,
                      resampling=Resampling.bilinear,
            )

        return dem_resampled
