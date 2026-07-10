from pathlib import Path
import json
import datetime
import rasterio


# =================================================================
#    Builds a dataset manifest (JSONL) from processed tiles.
#    Uses DatasetConfig + PathManager for all paths.
# =================================================================
class ManifestBuilder:

    def __init__(self, cfg, path_manager):
        self.cfg = cfg
        self.pm = path_manager

        # Where processed tiles live
        self.tiles_root = cfg.processed_path

        # Where manifest output goes
        self.manifest_dir = self.pm.manifest_dir
        self.manifest_log = self.manifest_dir / f"{cfg.name}_manifest_log.txt"
        self.manifest_jsonl = self.manifest_dir / f"{cfg.name}_dataset_index.jsonl"

        self.project_root = self.pm.project_root

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    def _log(self, msg: str):
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.manifest_log.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
        print(msg)

    # ---------------------------------------------------------
    # Raster info helper
    # ---------------------------------------------------------
    def _raster_info(self, path: Path):

        if not path.exists():
            return {}
        
        with rasterio.open(path) as src:
            return {"width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "crs": str(src.crs),
                    "dtype": src.dtypes[0],
            }

    # ---------------------------------------------------------
    # Build a single tile record
    # ---------------------------------------------------------
    def _build_record(self, tile_dir: Path):

        meta_path = tile_dir / "metadata.json"
        
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        tile_id = meta["tile_id"]

        def rel(p: Path):
            return str(p.resolve().relative_to(self.project_root))

        return {"tile_id": tile_id,
                "tile_dir": rel(tile_dir),

                "s2_path": rel(tile_dir / "S2_stack.tif"),
                "indices_path": rel(tile_dir / "indices.tif"),
                "dem_path": rel(tile_dir / "DEM.tif"),
                "soil_path": rel(tile_dir / "SOIL.tif"),
                "qc_path": rel(tile_dir / "QC.tif"),

                "s2_raster": self._raster_info(tile_dir / "S2_stack.tif"),
                "indices_raster": self._raster_info(tile_dir / "indices.tif"),
                "dem_raster": self._raster_info(tile_dir / "DEM.tif"),
                "soil_raster": self._raster_info(tile_dir / "SOIL.tif"),
                "qc_raster": self._raster_info(tile_dir / "QC.tif"),

                # metadata passthrough
                "projection": meta.get("projection"),
                "s2_date": meta.get("s2_date"),
                "s2_cloud_pct": meta.get("s2_cloud_pct"),
                "bands": meta.get("bands"),
                "scale_m": meta.get("scale_m"),
                "bbox": meta.get("bbox"),
        }

    # ---------------------------------------------------------
    # Build manifest for entire dataset
    # ---------------------------------------------------------
    def build(self):

        # Reset log
        self.manifest_log.unlink(missing_ok=True)
        self._log(f"Building manifest for dataset: {self.cfg.name}")

        records = []

        tile_dirs = sorted([p for p in self.tiles_root.iterdir() if p.is_dir()])

        for tile_dir in tile_dirs:

            try:
                rec = self._build_record(tile_dir)
                records.append(rec)
                self._log(f"[OK] Added tile {rec['tile_id']}")
            
            except Exception as e:
                self._log(f"[ERROR] {tile_dir}: {e}")

        # Write JSONL
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        with self.manifest_jsonl.open("w", encoding="utf-8") as f:
            
            for r in records:
                f.write(json.dumps(r) + "\n")

        self._log(f"[DONE] Wrote {len(records)} records → {self.manifest_jsonl}")
