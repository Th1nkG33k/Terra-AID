
import json
import numpy as np
import rasterio

from rasterio.features import rasterize
from rasterio.warp import transform_geom
from PIL import Image
from rasterio.merge import merge
from pathlib import Path


# =================================================================
#    Builds multimodal tiles (S2 + DEM + SOIL + indices + QC) from raw S2 tiles.#
#
#    NEW:
#        - Automatic DEM/SOIL quality detection
#        - Automatic RGB extraction per tile
#        - RGB mosaic stitching into Visuals/
#        - Manifest reporting
# =================================================================
class MultimodalProcessor:

    def __init__(self, cfg, path_manager, dem_processor, soil_processor, manifest_builder=None):

        self.cfg = cfg
        self.pm = path_manager
        self.dem_proc = dem_processor
        self.soil_proc = soil_processor
        self.manifest = manifest_builder

        # Directories from config. These are already resolved by DatasetConfig.
        self.raw_s2_root = Path(self.cfg.paths.raw_s2)
        self.raw_dem_root = Path(self.cfg.paths.raw_dem)
        self.raw_soil_root = Path(self.cfg.paths.raw_soil)
        self.processed_root = Path(self.cfg.processed_path)

        if not self.raw_s2_root.exists():
            raise FileNotFoundError(f"Raw S2 directory not found: {self.raw_s2_root}")

        # Visuals directory
        self.visuals_root = self.processed_root.parent / "Visuals"
        self.visuals_root.mkdir(exist_ok=True)

        # Auto-detection flags
        self.dem_valid = False
        self.soil_valid = False
        self.dem_reason = "DEM not evaluated"
        self.soil_reason = "SOIL not evaluated"

        # Perform DEM/SOIL analysis once per dataset
        self._analyse_dem()
        self._analyse_soil()

    # ---------------------------------------------------------
    # Raster quality analysis
    # Return (nonzero_ratio, std, range).
    # ---------------------------------------------------------
    def _analyse_raster(self, raster_path):

        with rasterio.open(raster_path) as src:
            arr = src.read().astype(np.float32)

        nonzero_ratio = np.count_nonzero(arr) / arr.size
        std = float(np.std(arr))
        rng = float(np.max(arr) - np.min(arr))

        return nonzero_ratio, std, rng

    def _analyse_dem(self):

        if not getattr(self.cfg.processing, "include_dem", False):
            self.dem_reason = "DEM disabled in config"
            return

        dem_raster = self._find_dem_raster()
        
        if dem_raster is None:
            self.dem_reason = "No DEM raster found"
            return

        nz, std, rng = self._analyse_raster(dem_raster)

        if nz < 0.01:
            self.dem_reason = f"DEM rejected: only {nz*100:.2f}% non-zero"
            return
        
        if std < 0.001:
            self.dem_reason = f"DEM rejected: too little variation (std={std:.6f})"
            return
        
        if rng < 0.1:
            self.dem_reason = f"DEM rejected: insufficient range (range={rng:.4f})"
            return

        self.dem_valid = True
        self.dem_reason = f"DEM accepted: nz={nz:.3f}, std={std:.3f}, range={rng:.3f}"

    def _analyse_soil(self):

        if not getattr(self.cfg.processing, "include_soil", False):
            self.soil_reason = "SOIL disabled in config"
            return

        soil_raster = self._find_soil_raster()

        if soil_raster is None:
            self.soil_reason = "No SOIL raster found"
            return

        nz, std, rng = self._analyse_raster(soil_raster)

        if nz < 0.01:
            self.soil_reason = f"SOIL rejected: only {nz*100:.2f}% non-zero"
            return
        
        if std < 0.001:
            self.soil_reason = f"SOIL rejected: too little variation (std={std:.6f})"
            return
        
        if rng < 0.1:
            self.soil_reason = f"SOIL rejected: insufficient range (range={rng:.4f})"
            return

        self.soil_valid = True
        self.soil_reason = f"SOIL accepted: nz={nz:.3f}, std={std:.3f}, range={rng:.3f}"

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _compute_qc_mask(self, s2_stack: np.ndarray, meta: dict) -> np.ndarray:

        band_names = meta["bands"]

        if "SCL" not in band_names:
            return np.zeros((meta["height"], meta["width"]), dtype=np.uint8)

        scl_idx = band_names.index("SCL")
        scl = s2_stack[scl_idx]

        cloud = np.isin(scl, [8, 9, 10])
        shadow = scl == 3
        snow = scl == 11
        saturation = scl == 1
        nodata = scl == 0

        qc = ((cloud.astype(np.uint8) << 0) |
              (shadow.astype(np.uint8) << 1) |
              (snow.astype(np.uint8) << 2) |
              (saturation.astype(np.uint8) << 3) |
              (nodata.astype(np.uint8) << 4)
        )

        return qc

    def _extract_indices(self, s2_stack: np.ndarray, meta: dict) -> np.ndarray:

        band_names = meta["bands"]

        if "NDVI" not in band_names or "BSI" not in band_names:
            return np.zeros((0, meta["height"], meta["width"]), dtype=np.float32)

        ndvi_idx = band_names.index("NDVI")
        bsi_idx = band_names.index("BSI")

        ndvi = s2_stack[ndvi_idx]
        bsi = s2_stack[bsi_idx]

        return np.stack([ndvi, bsi], axis=0)

    def _find_dem_raster(self) -> Path | None:

        tifs = sorted(self.raw_dem_root.glob("*.tif"))
        return tifs[0] if tifs else None

    def _find_soil_raster(self) -> Path | None:

        tifs = sorted(self.raw_soil_root.rglob("*.tif"))
        return tifs[0] if tifs else None

    # ---------------------------------------------------------
    # Profile-driven model input helpers
    # ---------------------------------------------------------
    def _safe_divide(self, numerator, denominator):

        return np.divide(numerator,
                         denominator,
                         out=np.zeros_like(numerator, dtype=np.float32),
                         where=np.abs(denominator) > 1e-6)


    # ---------------------------------------------------------
    # Build a lookup of available continuous channels for the tile.
    # ---------------------------------------------------------
    def _build_channel_lookup(self, s2_data: np.ndarray, band_names: list[str], dem_resampled=None, soil_resampled=None):

        lookup = {}

        for index, band_name in enumerate(band_names):
            lookup[str(band_name).upper()] = s2_data[index].astype(np.float32)

        if "NDVI" not in lookup and "B8" in lookup and "B4" in lookup:
            lookup["NDVI"] = self._safe_divide(lookup["B8"] - lookup["B4"], lookup["B8"] + lookup["B4"])

        if "BSI" not in lookup and {"B11", "B4", "B8", "B2"}.issubset(lookup.keys()):
            numerator = (lookup["B11"] + lookup["B4"]) - (lookup["B8"] + lookup["B2"])
            denominator = (lookup["B11"] + lookup["B4"]) + (lookup["B8"] + lookup["B2"])
            lookup["BSI"] = self._safe_divide(numerator, denominator)

        if "NDWI" not in lookup and "B3" in lookup and "B8" in lookup:
            lookup["NDWI"] = self._safe_divide(lookup["B3"] - lookup["B8"], lookup["B3"] + lookup["B8"])

        if dem_resampled is not None:
            lookup["DEM"] = dem_resampled.astype(np.float32)

        if soil_resampled is not None:
            for i in range(soil_resampled.shape[0]):
                lookup[f"SOIL_{i + 1}"] = soil_resampled[i].astype(np.float32)

        return lookup


    # ---------------------------------------------------------
    # Build the profile-defined model input stack.
    # ---------------------------------------------------------
    def _build_model_input_stack(self, s2_data: np.ndarray, band_names: list[str], dem_resampled=None, soil_resampled=None):

        profile = getattr(self.cfg, "profile", None)
        requested_channels = list(getattr(profile, "input_channels", []) or [])

        if not requested_channels:
            requested_channels = [b for b in band_names if str(b).upper() not in {"SCL", "QC"}]

        lookup = self._build_channel_lookup(s2_data, band_names, dem_resampled, soil_resampled)
        stacks = []
        used_channels = []
        missing_channels = []

        for channel in requested_channels:
            key = str(channel).upper()

            if key in lookup:
                stacks.append(lookup[key].astype(np.float32))
                used_channels.append(channel)
            else:
                missing_channels.append(channel)

        if missing_channels:
            raise RuntimeError(f"Missing required input channels for profile '{getattr(profile, 'name', None)}': {missing_channels}")

        if not stacks:
            raise RuntimeError("No profile input channels could be built for this tile.")

        return np.stack(stacks, axis=0).astype(np.float32), used_channels


    # ---------------------------------------------------------
    # Build valid pixel mask from finite inputs and QC if available.
    # ---------------------------------------------------------
    def _build_valid_mask(self, model_input: np.ndarray, qc_mask=None):

        finite_valid = np.all(np.isfinite(model_input), axis=0)

        if qc_mask is not None:
            qc_valid = qc_mask == 0
            finite_valid = finite_valid & qc_valid

        return finite_valid.astype(np.uint8)




    # =================================================================================
    # Ground-truth label helpers
    # =================================================================================

    # ---------------------------------------------------------
    # Load JSON text using common encodings used by exported GIS files.
    # ---------------------------------------------------------
    def _load_json_file(self, json_path: Path):

        json_path = Path(json_path)
        raw = json_path.read_bytes()

        if not raw:
            raise RuntimeError(f"Label file is empty: {json_path}")

        # Helpful diagnostics for files with the wrong extension.
        if raw[:2] == b"PK":
            raise RuntimeError(
                f"Label file appears to be a ZIP archive, not GeoJSON: {json_path}"
            )

        errors = []
        encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le",
                     "utf-16-be", "cp1252", "latin-1"]

        for encoding in encodings:
            try:
                text = raw.decode(encoding)
                text = text.lstrip("\ufeff").strip()

                if not text:
                    raise json.JSONDecodeError("empty decoded text", text, 0)

                return json.loads(text)

            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc}")

            except json.JSONDecodeError as exc:
                errors.append(f"{encoding}: {exc}")

        sample = raw[:32].hex(" ")

        raise RuntimeError(
                            "Could not read vector label file as GeoJSON. "
                            f"File: {json_path}\n"
                            "Tried encodings: utf-8-sig, utf-8, utf-16, utf-16-le, utf-16-be, cp1252, latin-1.\n"
                            "This usually means the file is not valid GeoJSON, or it was exported in another vector format.\n"
                            f"First 32 bytes: {sample}\n"
                            f"Details: {' | '.join(errors[:4])}"
        )


    # ---------------------------------------------------------
    # Load geometries through Fiona when a vector file is not plain JSON.
    # This supports GeoJSON files that GDAL can read even when Python json
    # decoding fails. Fiona is optional, so failures fall back to warnings.
    # ---------------------------------------------------------
    def _load_geometries_with_fiona(self, label_path: Path):

        try:
            import fiona
        except Exception as exc:
            raise RuntimeError(f"Fiona is not available for vector fallback: {exc}") from exc

        geometries = []

        with fiona.open(label_path) as src:

            for feature in src:
                geom = feature.get("geometry")

                if geom:
                    geometries.append(dict(geom))

        return geometries


    # ---------------------------------------------------------
    # Append supported GeoJSON geometries, including GeometryCollection.
    # ---------------------------------------------------------
    def _append_geojson_geometry(self, geometries: list, geom: dict):

        if not geom:
            return

        geom_type = geom.get("type")

        if geom_type == "GeometryCollection":

            for child in geom.get("geometries", []):
                self._append_geojson_geometry(geometries, child)
            return

        if geom_type in {"Polygon", "MultiPolygon", "LineString", "MultiLineString", "Point", "MultiPoint"}:
            geometries.append(geom)


    # ---------------------------------------------------------
    # Load GeoJSON geometries from a vector label file.
    # ---------------------------------------------------------
    def _load_geojson_geometries(self, label_path: Path):

        try:
            data = self._load_json_file(label_path)

            geometries = []

            if data.get("type") == "FeatureCollection":
                for feature in data.get("features", []):
                    self._append_geojson_geometry(geometries, feature.get("geometry"))

            elif data.get("type") == "Feature":
                self._append_geojson_geometry(geometries, data.get("geometry"))

            else:
                self._append_geojson_geometry(geometries, data)

            return geometries

        except Exception as json_exc:

            try:
                return self._load_geometries_with_fiona(label_path)

            except Exception as fiona_exc:
                raise RuntimeError(f"Could not load ground-truth vector file: {label_path}\n"
                                   f"JSON loader failed with: {json_exc}\n"
                                   f"Fiona fallback failed with: {fiona_exc}"
                ) from json_exc


    # ---------------------------------------------------------
    # Return candidate raw label files for this tile.
    #
    # IMPORTANT SCHEMA-RESET BEHAVIOUR
    # ---------------------------------------------------------
    # Dataset YAML no longer needs a verbose ``labels`` block for normal use.
    # If Raw/GroundTruth contains tile-level GeoJSON files, processing should
    # still rasterise them into Dataset/<tile>/ground_truth.tif.
    #
    # This keeps the config simple while making the processing stage responsible
    # for producing the raster mask required by prediction/evaluation.
    # ---------------------------------------------------------
    def _ground_truth_candidates(self, s2_path: Path, tile_dir: Path):

        labels = getattr(self.cfg, "labels", None)

        source_root = Path(getattr(self.cfg, "raw_ground_truth_path", self.cfg.paths.root / "Raw" / "GroundTruth"))
        if not source_root.exists():
            return []
        # ----------------------------------------------------------------------------
        # If a labels block exists, honour it.  If it does not exist, or if it
        # is the runtime default with type=None, fall back to the Terra-AID
        # convention used by labelled validation datasets:
        #   Raw/GroundTruth/<tile folder>/archaeology_selected.geojson
        # ----------------------------------------------------------------------------

        source = getattr(labels, "source", None) if labels is not None else None
        source_pattern = getattr(labels, "source_pattern", None) if labels is not None else None
        if not source_pattern:
            source_pattern = "{tile_folder}/archaeology_selected.geojson"

        candidates = []

        if source:
            src = Path(source)

            if not src.is_absolute():
                src = self.cfg.paths.root / src

            candidates.append(src)

        raw_tile_folder = s2_path.parent.parent.name

        # ----------------------------------------------------------------------------
        # Try to recover a useful tile index from the raw S2 filename, raw tile
        # folder, or processed folder.  This covers both ``tile 0`` and
        # ``tile_0_0`` style naming.
        # ----------------------------------------------------------------------------

        stem_tokens = s2_path.stem.replace("-", "_").replace(" ", "_").split("_")
        name_tokens = tile_dir.name.replace("-", "_").replace(" ", "_").split("_")
        raw_tokens = raw_tile_folder.replace("-", "_").replace(" ", "_").split("_")

        numeric_tokens = [t for t in stem_tokens + name_tokens + raw_tokens if str(t).isdigit()]
        tile_index = numeric_tokens[0] if numeric_tokens else s2_path.stem.split("_")[-1]

        tile_tokens = [raw_tile_folder,
                       raw_tile_folder.lower(),
                       raw_tile_folder.replace("Tile", "tile"),
                       raw_tile_folder.replace("tile", "Tile"),
                       raw_tile_folder.replace("_", " "),
                       raw_tile_folder.replace(" ", "_"),
                       f"tile {tile_index}",
                       f"Tile {tile_index}",
                       f"tile_{tile_index}",
                       f"Tile_{tile_index}",
                       tile_dir.name,
                       tile_dir.name.replace("_", " "),
                       tile_dir.name.replace(" ", "_")]

        filenames = ["archaeology_selected.geojson",
                     "archaeology_selected.json",
                     "ground_truth.geojson",
                     "ground_truth.json",
                     "labels.geojson",
                     "label.geojson"]

        for token in dict.fromkeys(tile_tokens):

            try:
                rel = source_pattern.format(tile_folder=token,
                                            tile_id=tile_index,
                                            processed_tile=tile_dir.name,
                                            raw_tile_folder=raw_tile_folder)
            except KeyError:
                rel = source_pattern.replace("{tile_folder}", token)

            path = Path(rel)

            if not path.is_absolute():
                path = source_root / path

            candidates.append(path)

            # Also try common vector filenames inside each candidate tile folder.
            token_dir = source_root / token
            for filename in filenames:
                candidates.append(token_dir / filename)

        # ----------------------------------------------------------------------------
        # Last-resort shallow search.  This is deliberately limited to one
        # folder level, so it does not become an expensive recursive scan.
        # ----------------------------------------------------------------------------

        for filename in filenames:
            candidates.extend(source_root.glob(f"*/{filename}"))

        # Return unique paths in order.
        return list(dict.fromkeys(candidates))


    # ---------------------------------------------------------
    # Rasterise a tile-level vector label into ground_truth.tif.
    # ---------------------------------------------------------
    def _rasterize_ground_truth(self, s2_path: Path, tile_dir: Path, raster_profile: dict):

        labels = getattr(self.cfg, "labels", None)

        # ----------------------------------------------------------------------------
        # Do not require a persisted labels block.  Ground-truth rasterisation is
        # triggered by the presence of a matching vector file under
        # Raw/GroundTruth.  If labels.type is explicitly set to a non-vector
        # value, skip; otherwise auto-detect.
        # ----------------------------------------------------------------------------

        label_type = getattr(labels, "type", None) if labels is not None else None
        if label_type not in (None, "", "vector"):
            return None

        candidates = self._ground_truth_candidates(s2_path, tile_dir)

        label_path = None
        for candidate in candidates:
            exists = candidate.exists()

            if exists and label_path is None:
                label_path = candidate

        if label_path is None:
            return None

        print(f"[GT] Rasterising ground truth for {tile_dir.name}: {label_path}")

        try:
            geometries = self._load_geojson_geometries(label_path)
        
        except Exception as exc:
            print(f"[WARN] Ground-truth label could not be loaded: {exc}")
            return None

        if not geometries:
            print(f"[WARN] Ground-truth label contains no supported geometries: {label_path}")
            return None
        
        # ----------------------------------------------------------------------------
        # GeoJSON labels are assumed to be EPSG:4326 unless a labels.crs
        # override is provided.  Reproject vector geometries to the processed
        # raster CRS before rasterisation so masks align with model_input.tif.
        # ----------------------------------------------------------------------------

        raster_crs = raster_profile.get("crs")
        label_crs = getattr(labels, "crs", None) if labels is not None else None
        label_crs = label_crs or "EPSG:4326"

        if raster_crs:
            try:
                geometries = [transform_geom(label_crs, raster_crs, geom) for geom in geometries]
            except Exception as exc:
                print(f"[WARN] Could not reproject ground-truth geometries from {label_crs} to {raster_crs}: {exc}")

        mask = rasterize([(geom, 1) for geom in geometries],
                         out_shape=(raster_profile["height"], raster_profile["width"]),
                         transform=raster_profile["transform"],
                         fill=0,
                         default_value=1,
                         all_touched=True,
                         dtype="uint8")

        gt_profile = raster_profile.copy()
        gt_profile.update(count=1, dtype="uint8", nodata=0)

        rasterized_name = getattr(labels, "rasterized_name", "ground_truth.tif")
        gt_out = tile_dir / rasterized_name

        with rasterio.open(gt_out, "w", **gt_profile) as dst:
            dst.write(mask, 1)

        return gt_out

    # ---------------------------------------------------------
    # RGB stitching
    # ---------------------------------------------------------
    def _stitch_rgb(self):

        rgb_files = sorted(self.processed_root.rglob("RGB.tif"))
        if not rgb_files:
            print("[WARN] No RGB tiles found — skipping RGB mosaic.")
            return None, None

        srcs = [rasterio.open(f) for f in rgb_files]
        mosaic, out_transform = merge(srcs)

        # --- optional GeoTIFF (internal use only) ---
        out_meta = srcs[0].meta.copy()
        out_meta.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            count=3,
            dtype="float32"
        )
        mosaic_path = self.visuals_root / f"{self.cfg.dataset_name}_RGB.tif"
        with rasterio.open(mosaic_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        # --- proper percentile normalisation for PNG AOI ---
        rgb = mosaic.transpose(1, 2, 0)
        finite = np.isfinite(rgb)
        rgb = np.where(finite, rgb, 0)

        # Ignore zeros when computing percentiles
        valid = rgb[rgb > 0]
        if valid.size == 0:
            vmin, vmax = 0, 1
        else:
            vmin = np.percentile(valid, 2)
            vmax = np.percentile(valid, 98)

        rgb_norm = np.clip((rgb - vmin) / (vmax - vmin + 1e-6), 0, 1)
        rgb8 = (rgb_norm * 255).astype(np.uint8)

        preview_path = self.visuals_root / f"{self.cfg.dataset_name}_RGB.png"
        Image.fromarray(rgb8).save(preview_path)

        for s in srcs:
            s.close()

        print(f"[OK] RGB mosaic saved → {mosaic_path}")
        print(f"[OK] RGB PNG AOI saved → {preview_path}")
        return mosaic_path, preview_path



    # ---------------------------------------------------------
    # Core tile builder
    # ---------------------------------------------------------
    def build_tile(self, s2_path: Path):
        
        print(f"[BUILD TILE] s2_path: {s2_path}")
        s2_path = Path(s2_path)

        # ----------------------------------------------------------------------------
        # Include the source batch folder in the tile id. Nile has repeated
        # nile_tile_0.tif ... nile_tile_10.tif inside each "Tile N/images"
        # folder, so using only the file stem overwrites previous batches.
        # ----------------------------------------------------------------------------
        
        batch_id = s2_path.parent.parent.name.replace(" ", "_")
        tile_id = f"{batch_id}_{s2_path.stem.split('_')[-1]}"
        tile_dir = self.processed_root / f"tile {tile_id}"
        tile_dir.mkdir(parents=True, exist_ok=True)

        # --- Load S2 stack ---
        with rasterio.open(s2_path) as s2_src:
            s2_data = s2_src.read()
            profile = s2_src.profile
            descriptions = list(s2_src.descriptions or [])

            if descriptions and all(descriptions):
                band_names = descriptions
            else:
                configured_bands = list(getattr(self.cfg.bands, "included", []) or [])
                if len(configured_bands) == s2_src.count:
                    band_names = configured_bands
                else:
                    band_names = [f"band_{i}" for i in range(1, s2_src.count + 1)]

        meta = {"tile_id": tile_id,
                "bands": band_names,
                "height": profile["height"],
                "width": profile["width"],
                "crs": str(profile["crs"]),
        }

        target_profile = {"crs": profile["crs"],
                          "transform": profile["transform"],
                          "height": profile["height"],
                          "width": profile["width"],
        }

        # -----------------------------------------------------
        # RGB extraction
        # -----------------------------------------------------
        try:

            b4 = band_names.index("B4")
            b3 = band_names.index("B3")
            b2 = band_names.index("B2")

            rgb = np.stack([s2_data[b4], s2_data[b3], s2_data[b2]], axis=0)

            rgb_profile = profile.copy()
            rgb_profile.update(count=3, dtype="float32")

            rgb_out = tile_dir / "RGB.tif"

            with rasterio.open(rgb_out, "w", **rgb_profile) as dst:
                dst.write(rgb)

        except ValueError:
            rgb_out = None

        # -----------------------------------------------------
        # DEM (only if valid)
        # -----------------------------------------------------
        if self.dem_valid:
            dem_raster = self._find_dem_raster()
            dem_resampled = self.dem_proc.load_and_resample(dem_raster, target_profile)

        else:
            dem_resampled = None

        # -----------------------------------------------------
        # Soil (only if valid)
        # -----------------------------------------------------
        if self.soil_valid:
            soil_raster = self._find_soil_raster()
            soil_resampled = self.soil_proc.load_and_resample(soil_raster, target_profile)

        else:
            soil_resampled = None

        # -----------------------------------------------------
        # QC mask
        # -----------------------------------------------------
        qc_mask = None

        if getattr(self.cfg.processing, "include_qc_mask", False):

            qc_mask = self._compute_qc_mask(s2_data, meta)
            qc_profile = profile.copy()
            qc_profile.update(count=1, dtype="uint8")
            qc_out = tile_dir / "QC.tif"
            
            with rasterio.open(qc_out, "w", **qc_profile) as dst:
                dst.write(qc_mask.astype(np.uint8), 1)
            
            meta["qc_path"] = str(qc_out)

        # -----------------------------------------------------
        # Indices (NDVI + BSI)
        # -----------------------------------------------------
        if getattr(self.cfg.processing, "include_indices", False):

            idx_stack = self._extract_indices(s2_data, meta)
            
            if idx_stack.shape[0] > 0:
                idx_profile = profile.copy()
                idx_profile.update(count=idx_stack.shape[0], dtype="float32")
                idx_out = tile_dir / "indices.tif"
            
                with rasterio.open(idx_out, "w", **idx_profile) as dst:
                    dst.write(idx_stack)
                    dst.set_band_description(1, "NDVI")
            
                    if idx_stack.shape[0] > 1:
                        dst.set_band_description(2, "BSI")
            
                meta["indices_path"] = str(idx_out)

        # -----------------------------------------------------
        # Profile-defined model input and valid mask
        # -----------------------------------------------------
        model_input, model_channels = self._build_model_input_stack(s2_data,
                                                                    band_names,
                                                                    dem_resampled=dem_resampled,
                                                                    soil_resampled=soil_resampled)

        model_profile = profile.copy()
        model_profile.update(count=model_input.shape[0], dtype="float32")

        model_input_out = tile_dir / "model_input.tif"

        with rasterio.open(model_input_out, "w", **model_profile) as dst:
            dst.write(model_input)
        
            for i, channel_name in enumerate(model_channels, start=1):
                dst.set_band_description(i, str(channel_name))

        valid_mask = self._build_valid_mask(model_input, qc_mask=qc_mask)
        valid_profile = profile.copy()
        valid_profile.update(count=1, dtype="uint8")

        valid_mask_out = tile_dir / "valid_mask.tif"

        with rasterio.open(valid_mask_out, "w", **valid_profile) as dst:
            dst.write(valid_mask, 1)

        finite = np.isfinite(model_input)
        meta["model_input_path"] = str(model_input_out)
        meta["valid_mask_path"] = str(valid_mask_out)
        meta["model_input_channels"] = list(model_channels)
        meta["model_input_channel_count"] = int(model_input.shape[0])
        meta["model_input_nan_fraction"] = float(1.0 - (np.count_nonzero(finite) / finite.size))
        meta["valid_pixel_fraction"] = float(np.count_nonzero(valid_mask) / valid_mask.size)
        meta["profile_name"] = getattr(getattr(self.cfg, "profile", None), "name", None)

        # -----------------------------------------------------
        # Save S2, DEM, Soil
        # -----------------------------------------------------
        s2_out = tile_dir / "S2_stack.tif"
        with rasterio.open(s2_out, "w", **profile) as dst:
            dst.write(s2_data)

        meta["s2_path"] = str(s2_out)

        if dem_resampled is not None:

            dem_profile = profile.copy()
            dem_profile.update(count=1, dtype="float32")
            dem_out = tile_dir / "DEM.tif"
            
            with rasterio.open(dem_out, "w", **dem_profile) as dst:
                dst.write(dem_resampled, 1)
            
            meta["dem_path"] = str(dem_out)

        if soil_resampled is not None:

            soil_profile = profile.copy()
            soil_profile.update(count=soil_resampled.shape[0], dtype="float32")
            soil_out = tile_dir / "SOIL.tif"
            
            with rasterio.open(soil_out, "w", **soil_profile) as dst:
                dst.write(soil_resampled)
            
            meta["soil_path"] = str(soil_out)


        # -----------------------------------------------------
        # Ground-truth rasterisation for validation tile sets
        # -----------------------------------------------------
        ground_truth_out = self._rasterize_ground_truth(s2_path, tile_dir, profile)

        if ground_truth_out is not None:
            meta["ground_truth_path"] = str(ground_truth_out)
            meta["has_ground_truth"] = True
        
        else:
            meta["has_ground_truth"] = False

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------
        with (tile_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] Built tile_{tile_id}")
        return tile_dir

    # ---------------------------------------------------------
    # Run over all raw S2 tiles
    # ---------------------------------------------------------
    def run(self):

        tile_dirs = sorted(self.raw_s2_root.glob("Tile*"))

        if not tile_dirs:
            raise FileNotFoundError(f"No Tile* directories found in raw S2 directory: {self.raw_s2_root}")

        built_tiles = 0

        for tdir in tile_dirs:

            images_dir = tdir / "images"
            print(f"RUN - Image_Dir: {images_dir}")
            
            if not images_dir.exists():
                continue

            for s2_file in sorted(images_dir.glob("*.tif")):

                print(f"Build Tile: {s2_file}")
                self.build_tile(s2_file)
                built_tiles += 1

        if built_tiles == 0:
            raise FileNotFoundError(f"No .tif image tiles found below: {self.raw_s2_root}")

        # -----------------------------------------------------
        # RGB Mosaic stitching
        # -----------------------------------------------------
        rgb_mosaic, rgb_preview = self._stitch_rgb()

        # -----------------------------------------------------
        # Manifest reporting
        # -----------------------------------------------------
        if self.manifest:
            self.manifest.add_entry({"dataset": self.cfg.dataset_name,
                                     "rgb_mosaic": str(rgb_mosaic) if rgb_mosaic else None,
                                     "rgb_preview": str(rgb_preview) if rgb_preview else None,
                                     "dem_included": self.dem_valid,
                                     "dem_reason": self.dem_reason,
                                     "soil_included": self.soil_valid,
                                     "soil_reason": self.soil_reason
            })

        print("[DONE] Dataset processing complete.")
