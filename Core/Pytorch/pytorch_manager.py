
import json
import torch
import rasterio
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset
from Core.Managers.path_manager import PathManager



# =========================================================
#    Multimodal dataset loader for Terra-AId.
#
#    Automatically detects available bands from:
#        - S2_stack.tif
#        - indices.tif (NDVI, BSI)
#        - DEM.tif
#        - SOIL.tif
#        - QC.tif
#
#    No per-band files required.
#    No fixed depth required.
# =========================================================
class MultimodalTileDataset(Dataset):

    def __init__(self, root_dir, cfg=None, bands=None, tile_size=256, transform=None):

        self.cfg = cfg

        pm = PathManager()
        self.root_dir = Path(pm.resolve_path(root_dir))

        self.tile_size = tile_size
        self.transform = transform

        # User-provided band list is optional now
        self.requested_bands = bands
        self.channel_stats = self._load_channel_stats()

        # -----------------------------------------------------
        # Tile directories: use cfg.tile_folder_pattern if present
        # -----------------------------------------------------
        pattern = "tile *"

        if self.cfg is not None and hasattr(self.cfg, "tile_folder_pattern"):
            pattern = self.cfg.tile_folder_pattern

        print(f"[MultimodalTileDataset] root={self.root_dir} pattern='{pattern}'")

        self.tile_dirs = sorted(
            d for d in self.root_dir.glob(pattern) if d.is_dir()
        )


    
    def __len__(self):
        return len(self.tile_dirs)

    
    def __getitem__(self, idx):
        tile_dir = self.tile_dirs[idx]

        meta = self._load_metadata(tile_dir)
        meta["tile_id"] = tile_dir.name

        x = self._build_tensor(tile_dir, meta)

        if self.transform:
            x = self.transform(x)

        return x, meta

    
    def _load_metadata(self, tile_dir):
        meta_path = tile_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ---------------------------------------------------------
    # Load training-set channel statistics when available.
    # ---------------------------------------------------------
    def _load_channel_stats(self):

        if self.cfg is None:
            return None

        stats_path = getattr(self.cfg, "channel_stats_path", None)

        if stats_path is None or not Path(stats_path).exists():
            return None

        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
            
        except Exception as exc:
            print(f"[WARN] Could not load channel stats: {exc}")
            return None

    
    def _load_raster(self, path):

        with rasterio.open(path) as src:
            arr = src.read()
        
        return torch.from_numpy(arr.astype(np.float32))

    
    def _resize(self, tensor):

        if tensor.shape[1:] != (self.tile_size, self.tile_size):
            tensor = F.interpolate(tensor.unsqueeze(0),
                                    size=(self.tile_size, self.tile_size),
                                    mode="bilinear",
                                    align_corners=False
            ).squeeze(0)

        return tensor

    # ---------------------------------------------------------
    # NEW: Case-insensitive file finder
    # ---------------------------------------------------------
    def _find_case_insensitive(self, tile_dir: Path, filename: str):

        target = filename.lower()
        
        for f in tile_dir.iterdir():
            if f.name.lower() == target:
                return f
        
        return None

    # ---------------------------------------------------------
    #    Automatically loads all available modalities in one stable order.
    #
    #    This function is now the single source of truth for tile depth:
    #
    #    S2_stack.tif bands -> indices.tif bands -> DEM.tif -> SOIL.tif bands -> QC.tif
    #
    #    The returned metadata includes the actual channel count and channel names,
    #    so training and prediction do not rely on stale YAML depth values.
    # ---------------------------------------------------------
    def _build_tensor(self, tile_dir: Path, meta: dict):

        tensors = []
        channel_names = []

        # ---------------------------------------------------------
        # Prefer profile-defined model_input.tif when present.
        # ---------------------------------------------------------
        model_input_path = self._find_case_insensitive(tile_dir, "model_input.tif")

        if model_input_path:

            with rasterio.open(model_input_path) as src:

                x = torch.from_numpy(src.read().astype(np.float32))
                descriptions = list(src.descriptions or [])

            # ----------------------------------------------------------------------------
            # Prefer real band descriptions from model_input.tif. If an older
            # file has blank descriptions, fall back to metadata/profile names
            # so prediction can align channels by B2/B3/... instead of channel_1.
            # ----------------------------------------------------------------------------

            profile_names = []

            if self.cfg is not None:
                profile_names = list(getattr(self.cfg, "input_channels", []) or [])
            
            meta_names = list(meta.get("model_input_channels", []) or [])

            channel_names = []

            for i in range(int(x.shape[0])):
            
                desc = descriptions[i] if i < len(descriptions) else None
                if desc:
                    channel_names.append(desc)
            
                elif i < len(meta_names):
                    channel_names.append(str(meta_names[i]))
            
                elif i < len(profile_names):
                    channel_names.append(str(profile_names[i]))
            
                else:
                    channel_names.append(f"channel_{i + 1}")

            # ----------------------------------------------------------------------------
            # If an old model_input.tif still contains mask/QC bands such as SCL,
            # filter it to the runtime-derived model input channels.
            # ----------------------------------------------------------------------------

            desired = list(getattr(self.cfg, "input_channels", []) or []) if self.cfg is not None else []
            if desired and channel_names:
                lookup = {str(name).upper(): i for i, name in enumerate(channel_names)}
                keep = [lookup[str(name).upper()] for name in desired if str(name).upper() in lookup]
                if keep and len(keep) != int(x.shape[0]):
                    x = x[keep]
                    channel_names = [channel_names[i] for i in keep]

            meta["channel_count"] = int(x.shape[0])
            meta["channel_names"] = channel_names
            meta["channel_source_order"] = ["model_input.tif"]

            return self._normalise(x)

        # ---------------------------------------------------------
        # Case-insensitive raster lookup fallback for old datasets.
        # ---------------------------------------------------------
        s2_path      = self._find_case_insensitive(tile_dir, "S2_stack.tif")
        indices_path = self._find_case_insensitive(tile_dir, "indices.tif")
        dem_path     = self._find_case_insensitive(tile_dir, "DEM.tif")
        soil_path    = self._find_case_insensitive(tile_dir, "SOIL.tif")
        qc_path      = self._find_case_insensitive(tile_dir, "QC.tif")

        # ---------------------------------------------------------
        # Load rasters if they exist
        # ---------------------------------------------------------
        s2_stack      = self._load_raster(s2_path) if s2_path else None
        indices_stack = self._load_raster(indices_path) if indices_path else None
        dem_tensor    = self._load_raster(dem_path) if dem_path else None
        soil_tensor   = self._load_raster(soil_path) if soil_path else None
        qc_tensor     = self._load_raster(qc_path) if qc_path else None

        def _append_stack(stack, prefix):

            if stack is None:
                return
            
            for i in range(stack.shape[0]):
                t = stack[i:i+1]
                t = self._resize(t)
                tensors.append(t)
                channel_names.append(f"{prefix}_{i + 1}")

        # ----------------------------------------------------------------------------
        # Sentinel-2 stack bands. For older datasets that do not have
        # model_input.tif, try to name channels from the dataset profile.
        # ----------------------------------------------------------------------------

        profile_channels = []

        if self.cfg is not None:
            profile_channels = list(getattr(self.cfg, "input_channels", []) or [])

        def _append_stack_named(stack, fallback_prefix, start_index=0):

            if stack is None:
                return 0
            
            for i in range(stack.shape[0]):
                t = stack[i:i+1]
                t = self._resize(t)
                tensors.append(t)
                profile_idx = start_index + i
                
                if profile_idx < len(profile_channels):
                    channel_names.append(str(profile_channels[profile_idx]))
                else:
                    channel_names.append(f"{fallback_prefix}_{i + 1}")

            return int(stack.shape[0])

        offset = 0
        offset += _append_stack_named(s2_stack, "S2", offset)

        # Derived indices, usually NDVI/BSI.
        offset += _append_stack_named(indices_stack, "INDEX", offset)

        # DEM
        if dem_tensor is not None:
            t = self._resize(dem_tensor[0:1])
            tensors.append(t)
            channel_names.append("DEM")

        # Soil stack
        _append_stack(soil_tensor, "SOIL")

        # QC mask
        if qc_tensor is not None:
            t = self._resize(qc_tensor[0:1])
            tensors.append(t)
            channel_names.append("QC")

        if not tensors:
            raise RuntimeError(f"No raster data found in tile folder: {tile_dir}")

        x = torch.cat(tensors, dim=0)

        # ----------------------------------------------------------------------------
        # Expose the real depth to callers. This is intentionally calculated
        # from the actual files, not guessed from the dataset config.
        # ----------------------------------------------------------------------------
        
        meta["channel_count"] = int(x.shape[0])
        meta["channel_names"] = channel_names
        meta["channel_source_order"] = ["S2_stack.tif", "indices.tif", "DEM.tif", "SOIL.tif", "QC.tif"]

        return self._normalise(x)

    
    def _normalise(self, x):

        x = x.float()
        method = None

        if self.cfg is not None:
            method = getattr(getattr(self.cfg, "normalisation", None), "method", None)

        if self.channel_stats and method in {"robust_zscore", "zscore"}:

            channels = self.channel_stats.get("channels", [])
            clip = getattr(getattr(self.cfg, "normalisation", None), "clip", [-5.0, 5.0])

            for c in range(min(x.shape[0], len(channels))):

                stats = channels[c]
                band = x[c]

                if method == "robust_zscore":
                    centre = stats.get("median")
                    scale = stats.get("iqr")
                else:
                    centre = stats.get("mean")
                    scale = stats.get("std")

                if centre is None or scale in (None, 0):
                    x[c] = torch.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
                    continue

                x[c] = (band - float(centre)) / (float(scale) + 1e-6)
                x[c] = torch.clamp(x[c], float(clip[0]), float(clip[1]))
                x[c] = torch.nan_to_num(x[c], nan=0.0, posinf=0.0, neginf=0.0)

            return x

        for c in range(x.shape[0]):

            band = torch.nan_to_num(x[c], nan=0.0, posinf=0.0, neginf=0.0)
            bmin = band.min()
            bmax = band.max()

            if (bmax - bmin) > 1e-6:
                x[c] = (band - bmin) / (bmax - bmin)
            else:
                x[c] = torch.zeros_like(band)
                
        return x
