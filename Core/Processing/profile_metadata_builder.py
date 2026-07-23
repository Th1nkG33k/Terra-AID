import csv
import json
import numpy as np
import rasterio

from pathlib import Path


# ============================================================
# PROFILE METADATA BUILDER
#
# Builds profile-driven dataset metadata after processing.
# This writes the tile manifest and channel statistics used by
# training, prediction, and profile compatibility checks.
# ============================================================
class ProfileMetadataBuilder:

    def __init__(self, cfg):

        self.cfg = cfg
        self.dataset_root = Path(cfg.paths.root)
        self.tiles_root = Path(cfg.processed_path)
        self.manifest_path = self.dataset_root / "tile_manifest.csv"
        self.stats_path = self.dataset_root / "channel_stats.json"


    # ---------------------------------------------------------
    # Return canonical processed tile directories.
    # ---------------------------------------------------------
    def _tile_dirs(self):

        return sorted(
            (d for d in self.tiles_root.glob("tile *") if d.is_dir()),
            key=lambda d: int(d.name.removeprefix("tile ")),
        )


    # ---------------------------------------------------------
    # Load tile metadata if available.
    # ---------------------------------------------------------
    def _load_metadata(self, tile_dir: Path) -> dict:

        meta_path = tile_dir / "metadata.json"

        if not meta_path.exists():
            return {}

        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}


    # ---------------------------------------------------------
    # Return simple raster details for manifest rows.
    # ---------------------------------------------------------
    def _raster_summary(self, path: Path) -> dict:

        if not path.exists():
            return {"exists": False, "bands": 0, "width": None, "height": None}

        with rasterio.open(path) as src:
            return {"exists": True,
                    "bands": src.count,
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
            }


    # ---------------------------------------------------------
    # Build one manifest row for a processed tile.
    # ---------------------------------------------------------
    def _build_manifest_row(self, tile_dir: Path) -> dict:

        meta = self._load_metadata(tile_dir)
        model_input = tile_dir / "model_input.tif"
        valid_mask = tile_dir / "valid_mask.tif"
        ground_truth = self._find_ground_truth(tile_dir)

        raster = self._raster_summary(model_input)
        profile = getattr(self.cfg, "profile", None)

        return {"tile_id": meta.get("tile_id", tile_dir.name),
                "tile_dir": str(tile_dir),
                "role": getattr(self.cfg, "role", "mixed"),
                "structure": getattr(self.cfg, "structure", "aoi_grid"),
                "profile_name": getattr(profile, "name", None),
                "num_input_channels": raster.get("bands", 0),
                "expected_input_channels": getattr(profile, "num_input_channels", None),
                "input_channels": "|".join(meta.get("model_input_channels", getattr(profile, "input_channels", []) or [])),
                "has_model_input": model_input.exists(),
                "has_s2_stack": (tile_dir / "S2_stack.tif").exists(),
                "has_indices": (tile_dir / "indices.tif").exists(),
                "has_rgb": (tile_dir / "RGB.tif").exists(),
                "has_qc": (tile_dir / "QC.tif").exists(),
                "has_valid_mask": valid_mask.exists(),
                "has_dem": (tile_dir / "DEM.tif").exists(),
                "has_soil": (tile_dir / "SOIL.tif").exists(),
                "has_ground_truth": ground_truth is not None,
                "valid_pixel_fraction": meta.get("valid_pixel_fraction"),
                "nan_fraction": meta.get("model_input_nan_fraction"),
                "width": raster.get("width"),
                "height": raster.get("height"),
                "status": "valid" if model_input.exists() else "missing_model_input",
                "excluded_reason": "" if model_input.exists() else "model_input.tif not found",
        }


    # ---------------------------------------------------------
    # Locate ground-truth masks using strict, unambiguous names.
    # ---------------------------------------------------------
    def _find_ground_truth(self, tile_dir: Path):

        candidates = ["ground_truth.tif", "GroundTruth.tif", "labels.tif", "label.tif", "mask.tif", "GT.tif"]

        for name in candidates:
            path = tile_dir / name
            if path.exists():
                return path

        return None


    # ---------------------------------------------------------
    # Write tile_manifest.csv to the dataset project folder.
    # ---------------------------------------------------------
    def write_manifest(self) -> Path:

        rows = [self._build_manifest_row(tile_dir) for tile_dir in self._tile_dirs()]
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["tile_id",
                      "tile_dir",
                      "role",
                      "structure",
                      "profile_name",
                      "num_input_channels",
                      "expected_input_channels",
                      "input_channels",
                      "has_model_input",
                      "has_s2_stack",
                      "has_indices",
                      "has_rgb",
                      "has_qc",
                      "has_valid_mask",
                      "has_dem",
                      "has_soil",
                      "has_ground_truth",
                      "valid_pixel_fraction",
                      "nan_fraction",
                      "width",
                      "height",
                      "status",
                      "excluded_reason",
        ]

        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return self.manifest_path


    # ---------------------------------------------------------
    # Compute channel stats from all model_input.tif files.
    # ---------------------------------------------------------
    def write_channel_stats(self) -> Path:

        channel_values = []
        channel_names = []
        channel_totals = []
        channel_nans = []

        for tile_dir in self._tile_dirs():

            model_input = tile_dir / "model_input.tif"

            if not model_input.exists():
                continue

            with rasterio.open(model_input) as src:
                arr = src.read().astype(np.float32)
                descriptions = list(src.descriptions or [])

            if not channel_values:
                channel_values = [[] for _ in range(arr.shape[0])]
                channel_names = [d if d else f"channel_{i + 1}" for i, d in enumerate(descriptions)]
                channel_totals = [0 for _ in range(arr.shape[0])]
                channel_nans = [0 for _ in range(arr.shape[0])]

            for i in range(min(arr.shape[0], len(channel_values))):
                band = arr[i]
                finite_mask = np.isfinite(band)
                valid = band[finite_mask]
                channel_totals[i] += int(band.size)
                channel_nans[i] += int(band.size - np.count_nonzero(finite_mask))
                if valid.size:
                    channel_values[i].append(valid.reshape(-1))

        channels = []

        for i, values in enumerate(channel_values):

            if values:
                flat = np.concatenate(values)
                p25 = float(np.percentile(flat, 25))
                p75 = float(np.percentile(flat, 75))
                iqr = p75 - p25

                record = {"name": channel_names[i] if i < len(channel_names) else f"channel_{i + 1}",
                          "index": i,
                          "count": int(flat.size),
                          "mean": float(np.mean(flat)),
                          "std": float(np.std(flat)),
                          "median": float(np.median(flat)),
                          "p2": float(np.percentile(flat, 2)),
                          "p25": p25,
                          "p75": p75,
                          "p98": float(np.percentile(flat, 98)),
                          "iqr": float(iqr),
                          "min": float(np.min(flat)),
                          "max": float(np.max(flat)),
                          "nan_fraction": float(channel_nans[i] / channel_totals[i]) if channel_totals[i] else None,
                }

            else:
                record = {"name": channel_names[i] if i < len(channel_names) else f"channel_{i + 1}",
                          "index": i,
                          "count": 0,
                          "mean": None,
                          "std": None,
                          "median": None,
                          "p2": None,
                          "p25": None,
                          "p75": None,
                          "p98": None,
                          "iqr": None,
                          "min": None,
                          "max": None,
                          "nan_fraction": None,
                }

            channels.append(record)

        stats = {"dataset": self.cfg.dataset_name,
                 "profile": getattr(getattr(self.cfg, "profile", None), "name", None),
                 "num_input_channels": len(channels),
                 "normalisation_recommendation": "robust_zscore",
                 "channels": channels,
        }

        self.stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return self.stats_path


    # ---------------------------------------------------------
    # Build all profile metadata outputs.
    # ---------------------------------------------------------
    def run(self) -> dict:

        manifest = self.write_manifest()
        stats = self.write_channel_stats()

        return {"manifest": manifest,
                "channel_stats": stats,
        }
