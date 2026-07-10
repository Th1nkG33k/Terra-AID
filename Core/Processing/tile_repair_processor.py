
import numpy as np
import rasterio

from pathlib import Path
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from datetime import datetime

# ====================================================================
#    Ensures all tiles in a dataset have:
#        - matching CRS
#        - matching dimensions
#        - matching band count
#
#    This processor does NOT assume any folder structure.
#    PathManager + DatasetConfig supply all paths.
# ====================================================================
class TileRepairProcessor:

    def __init__(self, cfg, path_manager):

        self.cfg = cfg
        self.pm = path_manager

        self.tiles_root = cfg.processed_path
        self.log_path = self.pm.logs_dir / f"{cfg.name}_tile_repair.log"


    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    def _log(self, msg):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
        
        print(msg)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _get_raster_info(self, path):

        with rasterio.open(path) as src:
            return {"width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "crs": src.crs,
                    "transform": src.transform,
                    "dtype": src.dtypes[0],
                    "meta": src.meta
            }

    def _resample_to_match(self, src_path, dst_path, ref_info):

        with rasterio.open(src_path) as src:

            data = src.read(out_shape=(src.count,
                                       ref_info["height"],
                                       ref_info["width"]
                            ),
                            resampling=Resampling.bilinear
            )

            new_transform = src.transform * src.transform.scale(
                (src.width / ref_info["width"]),
                (src.height / ref_info["height"])
            )

            meta = src.meta.copy()
            meta.update({"height": ref_info["height"],
                         "width": ref_info["width"],
                         "transform": new_transform,
                         "crs": ref_info["crs"]
            })

            with rasterio.open(dst_path, "w", **meta) as dst:
                dst.write(data)

    def _reproject_to_match(self, src_path, dst_path, ref_info):

        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(src.crs, ref_info["crs"],
                                                                   src.width, src.height,
                                                                   *src.bounds
            )

            meta = src.meta.copy()
            meta.update({"crs": ref_info["crs"],
                         "transform": transform,
                         "width": width,
                         "height": height
            })

            with rasterio.open(dst_path, "w", **meta) as dst:

                for i in range(1, src.count + 1):
                
                    reproject(source=rasterio.band(src, i),
                              destination=rasterio.band(dst, i),
                              src_transform=src.transform,
                              src_crs=src.crs,
                              dst_transform=transform,
                              dst_crs=ref_info["crs"],
                              resampling=Resampling.bilinear
                    )

    # ---------------------------------------------------------
    # Main repair logic
    # ---------------------------------------------------------
    def run(self):

        # Reset log
        self.log_path.unlink(missing_ok=True)
        self._log("=== TILE REPAIR STARTED ===")

        tile_dirs = sorted([p for p in self.tiles_root.iterdir() if p.is_dir()])

        # Collect all .tif files inside each tile folder
        image_files = []

        for tdir in tile_dirs:
            image_files.extend(sorted(tdir.glob("*.tif")))

        if not image_files:
            self._log("ERROR: No tiles found.")
            return

        # ---------------------------------------------------------
        # Select reference tile
        # ---------------------------------------------------------
        ref_info = None
        
        for f in image_files:
            try:
                ref_info = self._get_raster_info(f)
                self._log(f"Reference tile selected: {f}")
                break
        
            except:
                continue

        if ref_info is None:
            self._log("ERROR: No valid reference tile found.")
            return

        # ---------------------------------------------------------
        # Process all tiles
        # ---------------------------------------------------------
        ok_count = fixed_count = skipped_count = error_count = 0

        for img_path in image_files:

            try:
                info = self._get_raster_info(img_path)
            
            except:
                self._log(f"ERROR: Cannot read {img_path.name}")
                error_count += 1
                continue

            # Band mismatch → skip
            if info["count"] != ref_info["count"]:

                self._log(f"SKIP: Band mismatch → {img_path.name}")
                skipped_count += 1
                continue

            # CRS mismatch → reproject
            if info["crs"] != ref_info["crs"]:

                self._log(f"FIX: CRS mismatch → {img_path.name}")
                tmp = img_path.with_suffix(".tmp.tif")
                self._reproject_to_match(img_path, tmp, ref_info)
                tmp.replace(img_path)
                info = self._get_raster_info(img_path)
                fixed_count += 1

            # Dimension mismatch → resample
            if info["width"] != ref_info["width"] or info["height"] != ref_info["height"]:
                
                self._log(f"FIX: Dimension mismatch → {img_path.name}")
                tmp = img_path.with_suffix(".tmp.tif")
                self._resample_to_match(img_path, tmp, ref_info)
                tmp.replace(img_path)
                fixed_count += 1
                continue

            self._log(f"OK: {img_path.name}")
            ok_count += 1

        # ---------------------------------------------------------
        # Summary
        # ---------------------------------------------------------
        self._log("=== TILE REPAIR SUMMARY ===")
        self._log(f"OK tiles: {ok_count}")
        self._log(f"Fixed tiles: {fixed_count}")
        self._log(f"Skipped (band mismatch): {skipped_count}")
        self._log(f"Errors: {error_count}")
        self._log("=== PROCESS COMPLETE ===")
