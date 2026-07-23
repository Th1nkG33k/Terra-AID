import json
import numpy as np
import rasterio

from pathlib import Path
from Core.Processing.profile_metadata_builder import ProfileMetadataBuilder


# =================================================================
#    Rebuild model_input.tif files from already-processed tiles without
#    downloading anything again.
#
#    It is mainly used after the channel policy changes, for example to remove
#    SCL from model inputs while keeping it available as QC/mask data.
# =================================================================
class ModelInputRestacker:

    def __init__(self, cfg):

        self.cfg = cfg
        self.tiles_root = Path(cfg.processed_path)

    def _tile_dirs(self):
        return sorted(
            (d for d in self.tiles_root.glob("tile *") if d.is_dir()),
            key=lambda d: int(d.name.removeprefix("tile ")),
        )

    def _load_meta(self, tile_dir: Path):

        meta_path = tile_dir / "metadata.json"
        
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            
            except Exception:
                return {}
        
        return {}

    def _save_meta(self, tile_dir: Path, meta: dict):
        (tile_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _read_named_raster(self, path: Path):

        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
            profile = src.profile
            names = [d if d else f"channel_{i + 1}" for i, d in enumerate(src.descriptions or [])]
        
        return arr, profile, names

    def _write_model_input(self, path: Path, arr: np.ndarray, profile: dict, channel_names: list[str]):
        
        tmp = path.with_suffix(".tmp.tif")
        out_profile = profile.copy()
        out_profile.update(count=arr.shape[0], dtype="float32")
        
        with rasterio.open(tmp, "w", **out_profile) as dst:
            dst.write(arr.astype(np.float32))
            for i, name in enumerate(channel_names, start=1):
                dst.set_band_description(i, str(name))

        tmp.replace(path)

    def _write_valid_mask(self, tile_dir: Path, arr: np.ndarray, profile: dict):

        valid = np.all(np.isfinite(arr), axis=0)
        qc_path = tile_dir / "QC.tif"

        if qc_path.exists():
        
            with rasterio.open(qc_path) as src:
                qc = src.read(1)
            valid &= (qc == 0)
        
        out_profile = profile.copy()
        out_profile.update(count=1, dtype="uint8")
        out = tile_dir / "valid_mask.tif"
        
        with rasterio.open(out, "w", **out_profile) as dst:
            dst.write(valid.astype(np.uint8), 1)
        
        return out, float(np.count_nonzero(valid) / valid.size)

    def _build_from_existing_model_input(self, tile_dir: Path, expected: list[str]):
        
        model_input = tile_dir / "model_input.tif"
        
        if not model_input.exists():
            return None

        arr, profile, names = self._read_named_raster(model_input)
        lookup = {str(name).upper(): i for i, name in enumerate(names)}
        keep = []
        missing = []

        for name in expected:
        
            idx = lookup.get(str(name).upper())
        
            if idx is None:
                missing.append(name)
            else:
                keep.append(idx)

        if missing:
            raise RuntimeError(
                f"{tile_dir.name}: existing model_input.tif is missing {missing}. Available: {names}"
            )

        return arr[keep], profile, [names[i] for i in keep]

    def run(self):

        expected = list(getattr(self.cfg, "input_channels", []) or [])

        if not expected:
            raise RuntimeError("Dataset has no derived model input channels to restack.")

        rebuilt = 0
        skipped = 0

        for tile_dir in self._tile_dirs():

            model_input = tile_dir / "model_input.tif"
            
            if not model_input.exists():
                skipped += 1
                continue

            arr, profile_raster, names = self._build_from_existing_model_input(tile_dir, expected)
            self._write_model_input(model_input, arr, profile_raster, expected)
            valid_path, valid_fraction = self._write_valid_mask(tile_dir, arr, profile_raster)

            meta = self._load_meta(tile_dir)
            finite = np.isfinite(arr)
            meta["model_input_path"] = str(model_input)
            meta["valid_mask_path"] = str(valid_path)
            meta["model_input_channels"] = list(expected)
            meta["model_input_channel_count"] = int(arr.shape[0])
            meta["model_input_nan_fraction"] = float(1.0 - (np.count_nonzero(finite) / finite.size))
            meta["valid_pixel_fraction"] = valid_fraction
            meta["profile_name"] = f"derived_{len(expected)}ch"
            self._save_meta(tile_dir, meta)
            rebuilt += 1

        metadata = ProfileMetadataBuilder(self.cfg).run()
        
        return {"rebuilt": rebuilt,
                "skipped": skipped,
                "manifest": str(metadata["manifest"]),
                "channel_stats": str(metadata["channel_stats"]),
                "input_channels": expected,
        }
