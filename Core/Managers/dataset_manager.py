
import yaml

from pathlib import Path
from Core.Managers.path_manager import PathManager
from Core.Managers.config_manager import ConfigManager, DatasetConfig
from Core.Processing.soil_processor import SoilProcessor
from Core.Processing.dem_processor import DEMProcessor
from Core.Processing.multimodal_prrocessor import MultimodalProcessor
from Core.Processing.profile_metadata_builder import ProfileMetadataBuilder
from Core.Utils.channel_policy import DEFAULT_BANDS, get_model_input_channels, get_mask_channels


# ============================================================
#    DATASET MANAGER
#
#    Dataset-facing service for the app.
#    ConfigManager is the single source of truth for loading YAML. DatasetManager
#    consumes DatasetConfig objects and exposes dataset operations to the UI and
#    processing tasks.
# ============================================================
class DatasetManager:

    # --------------------------------------------------------
    # Uses the shared ConfigManager when supplied; otherwise creates one.
    # --------------------------------------------------------

    def __init__(self, configs_dir: str | Path = None, config_manager: ConfigManager | None = None):

        self.config_manager = config_manager

        if self.config_manager is not None:
            self.pm = self.config_manager.pm
            self.configs_dir = self.pm.DATASET_CONFIGS
            self.datasets = self.config_manager.datasets
        else:
            self.pm = PathManager()
            self.configs_dir = Path(configs_dir) if configs_dir else self.pm.DATASET_CONFIGS
            self.config_manager = ConfigManager(self.pm)
            self.datasets = self.config_manager.datasets

        print(f"[DatasetManager] Using dataset configs from: {self.configs_dir}")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def list_datasets(self, role: str | None = None) -> list[str]:

        return [name for name, cfg in self.datasets.items()
                if self._role_allowed(getattr(cfg, "role", "mixed"), role)]


    # ---------------------------------------------------------
    # Check whether a dataset role is allowed for a selector mode.
    # ---------------------------------------------------------
    def _role_allowed(self, dataset_role: str, mode: str | None) -> bool:

        role = (dataset_role or "").lower()
        mode = (mode or "all").lower()
        allowed = {
            "all": {"training", "predictive", "evaluation"},
            "training": {"training"},
            "predictive": {"predictive"},
            "evaluation": {"evaluation"},
        }
        return role in allowed.get(mode, set())


    # ---------------------------------------------------------
    # Return UI-ready dataset summaries.
    # ---------------------------------------------------------
    def list_dataset_options(self, role: str | None = None) -> list[dict]:

        options = []

        for name, cfg in self.datasets.items():

            if not self._role_allowed(getattr(cfg, "role", "mixed"), role):
                continue

            options.append({"name": cfg.dataset_name,
                            "key": name,
                            "stage": cfg.stage,
                            "tile_count": cfg.tile_count,
                            "depth": getattr(cfg, "num_input_channels", getattr(cfg, "depth", None)),
                            "role": getattr(cfg, "role", "mixed"),
                            "structure": getattr(cfg, "structure", "aoi_grid"),
                            "profile": f"derived_{getattr(cfg, 'num_input_channels', 0)}ch",
                            "num_input_channels": getattr(cfg, "num_input_channels", getattr(cfg, "depth", None)),
                            "exists": cfg.exists(),
            })

        return options


    def get(self, name: str) -> DatasetConfig | None:
        return self.datasets.get(name)

    def reload(self):

        self.config_manager.reload()
        self.datasets = self.config_manager.datasets


    # ---------------------------------------------------------
    #    Split selected bands into model input channels and mask
    #    channels. SCL is stored as a mask layer by default because
    #    it is categorical scene-classification data, not reflectance.
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    #    Resolve band preset/custom checkbox state from the UI.
    # ---------------------------------------------------------
    def _selected_bands_from_values(self, values: dict) -> list[str]:

        default_bands = list(DEFAULT_BANDS)
        all_bands = list(getattr(self.pm, "ALL_BANDS", [])) or [
            "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12",
            "SCL", "VV", "VH", "NDVI", "BSI",
        ]

        mode = values.get("-CDC_BAND_MODE-", "Default bands")

        if mode == "Default bands":
            return default_bands

        if mode == "All bands":
            return all_bands

        selected = []
        for band in all_bands:
            if values.get(f"-CDC_BAND_{band}-", False):
                selected.append(band)

        if not selected:
            raise ValueError("Select at least one band, or choose Default bands / All bands.")

        return selected


    def _model_channels_from_bands(self, selected_bands: list[str], values: dict):

        # Dataset YAML stores selected/downloaded bands once. Model input and
        # mask channels are derived here and elsewhere using channel_policy.
        tmp = {
            "bands": {"included": selected_bands or []},
            "processing": {
                "allow_categorical_inputs": bool(values.get("-CDC_ALLOW_CATEGORICAL_INPUTS-", False))
            },
        }
        return get_model_input_channels(tmp), get_mask_channels(tmp)


    def _role_from_values(self, values: dict) -> str:
        label = str(values.get("-CDC_ROLE-", "Training") or "Training").strip().lower()
        mapping = {
            "training": "training",
            "evaluation": "predictive",
            "prediction": "evaluation",
        }
        if label not in mapping:
            raise ValueError(f"Unsupported dataset role: {label}")
        return mapping[label]


    # ---------------------------------------------------------
    #    Build the temporary downloaded-stage dataset config from
    #    the create-dataset UI values.
    # ---------------------------------------------------------
    def _build_dataset_config_from_values(self, values: dict, dataset_name: str) -> dict:

        def _value(key, default=None):
            value = values.get(key, default)
            return default if value in (None, "") else value

        def _float(key):
            value = _value(key)
            return float(value) if value not in (None, "") else None

        def _int(key, default=0):
            value = _value(key, default)
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        selected_bands = self._selected_bands_from_values(values)
        input_channels, mask_channels = self._model_channels_from_bands(selected_bands, values)

        cfg = {
            "dataset": {"name": dataset_name,
                        "version": "1.0",
                        "stage": "downloaded",
                        "tile_count": _int("-CDC_NUM_TILES-", 0),
                        "role": self._role_from_values(values),
                        "structure": "aoi_grid",
            },

            "paths": {"root": f"Data/Datasets/{dataset_name}",
                      "raw_s2": "Raw/S2",
                      "raw_dem": "Raw/DEM",
                      "raw_soil": "Raw/Soil",
                      "raw_ground_truth": "Raw/GroundTruth",
                      "processed_tiles": "Dataset",
                      "visuals_dir": "Visuals",
                      "tile_manifest": "tile_manifest.csv",
                      "channel_stats": "channel_stats.json",
            },

            "tile_structure": {"width": 512,
                               "height": 512,
                               "tile_format": "tif",
            },

            "AOI": {"min_lat": _float("-CDC_MIN_LAT-"),
                    "max_lat": _float("-CDC_MAX_LAT-"),
                    "min_lon": _float("-CDC_MIN_LON-"),
                    "max_lon": _float("-CDC_MAX_LON-"),
            },

            "date_range": {"start": _value("-CDC_DATE_START-"),
                           "end": _value("-CDC_DATE_END-"),
            },

            "bands": {"included": selected_bands,
                      "rgb_order": ["B4", "B3", "B2"],
            },

            "crs": {"epsg": 4326},

            "normalisation": {"method": "robust_zscore",
                              "stats_source": "channel_stats.json",
                              "clip": [-5.0, 5.0],
            },

            "processing": {"resolution": _value("-CDC_RESOLUTION-", "10m"),
                           "include_dem": bool(values.get("-CDC_DEM_Y-", False)),
                           "include_soil": bool(values.get("-CDC_SOIL_Y-", False)),
                           "include_indices": True,
                           "include_qc_mask": True,
            },

            "provenance": {"created_by": "Terra-AId",
                           "version": "2.0",
                           "note": "Temporary downloaded-stage dataset config. Download processing will be hooked in later.",
            },
        }

        return cfg

    # ---------------------------------------------------------
    #    Create the dataset project folder and save the dataset
    #    config in both the central registry and project folder.
    # ---------------------------------------------------------
    def create_dataset_from_values(self, values: dict) -> DatasetConfig:

        dataset_name = (values.get("-CDC_NAME-") or values.get("-CDC_DATASET_NAME-") or "").strip()

        if not dataset_name:
            raise ValueError("Dataset name is required.")

        cfg = self._build_dataset_config_from_values(values, dataset_name)

        dataset_root = self.pm.DATASETS_ROOT / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)

        for folder in ["Dataset", "Visuals", "Raw/S2", "Raw/DEM", "Raw/Soil", "Raw/GroundTruth"]:
            (dataset_root / folder).mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------
        # Ground-truth tile collections store vector labels below
        # Raw/GroundTruth/<tile folder>/archaeology_selected.geojson.
        # The processor will later rasterise those vectors into
        # Dataset/<tile folder>/ground_truth.tif.
        # --------------------------------------------------------

        project_config_path = dataset_root / f"{dataset_name}.yaml"
        central_config_path = self.pm.DATASET_CONFIGS / f"{dataset_name}.yaml"
        central_config_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_cfg = DatasetConfig(dataset_name, cfg, central_config_path, self.pm)
        clean_yaml = tmp_cfg.to_yaml()
        project_config_path.write_text(clean_yaml, encoding="utf-8")
        central_config_path.write_text(clean_yaml, encoding="utf-8")

        self.reload()

        created_cfg = self.get(dataset_name)
        if created_cfg is None:
            created_cfg = DatasetConfig(dataset_name, cfg, central_config_path, self.pm)
            self.datasets[dataset_name] = created_cfg

        print(f"[DatasetManager] Dataset '{dataset_name}' created at downloaded stage.")
        return created_cfg

    # ---------------------------------------------------------
    #    Save a copy of the central dataset config inside the
    #    dataset project folder.
    # ---------------------------------------------------------
    def _save_project_config_copy(self, cfg: DatasetConfig):

        project_config_path = cfg.paths.root / f"{cfg.dataset_name}.yaml"
        project_config_path.write_text(cfg.to_yaml(), encoding="utf-8")


    # ---------------------------------------------------------
    #    Runs the full dataset processing pipeline using MultimodalProcessor.
    #    Automatically updates the dataset stage and saves the config.
    # ---------------------------------------------------------
    def process_dataset(self, dataset_name):
        # --------------------------------------------------------
        # Reload configs before processing so manual YAML edits are picked
        # up without needing to restart the application.
        # --------------------------------------------------------
        self.reload()

        cfg = self.get(dataset_name)

        if cfg is None:
            raise KeyError(f"Dataset '{dataset_name}' not found in loaded configs")

        print(f"[DatasetManager] Processing config: {cfg.config_path}")
        print(f"[DatasetManager] Model input channels: {getattr(cfg, 'input_channels', [])} "
              f"({getattr(cfg, 'num_input_channels', None)} channels)")

        pm = self.pm

        # Build processors
        dem_proc = DEMProcessor(cfg, pm)
        soil_proc = SoilProcessor(cfg, pm)

        # Build multimodal processor
        processor = MultimodalProcessor(cfg, pm, dem_proc, soil_proc, manifest_builder=None)

        # Run processing pipeline
        processor.run()

        # Build profile-driven metadata after processing.
        metadata = ProfileMetadataBuilder(cfg).run()

        tile_count = len(list(Path(cfg.processed_path).glob("tile *")))
        # Update stage and persistent processing metadata. Channel count is derived
        # from bands.included at runtime and is no longer written to YAML.
        cfg.stage = "processed"
        cfg.tile_count = tile_count

        cfg.cfg.setdefault("dataset", {})["stage"] = "processed"
        cfg.cfg.setdefault("dataset", {})["tile_count"] = tile_count
        cfg.cfg.setdefault("paths", {})["tile_manifest"] = "tile_manifest.csv"
        cfg.cfg.setdefault("paths", {})["channel_stats"] = "channel_stats.json"
        cfg.cfg.setdefault("normalisation", {})["stats_source"] = "channel_stats.json"

        cfg.save()
        self._save_project_config_copy(cfg)
        self.reload()

        print(f"[DatasetManager] Dataset '{dataset_name}' processed successfully.")
        print(f"[DatasetManager] Manifest written: {metadata['manifest']}")
        print(f"[DatasetManager] Channel stats written: {metadata['channel_stats']}")
        return True

    # ---------------------------------------------------------
    #   Runs the StatisticsProcessor on a processed dataset.
    #   Saves all imagery to Visuals/ and all JSON/NPY to dataset root.
    # ---------------------------------------------------------
    def generate_statistics(self, dataset_name, worker=None):
        """
        Runs the StatisticsProcessor on a processed dataset.
        Saves visual/statistical outputs to the dataset Visuals folder.
        """
        from Core.Processing.statistics_processor import StatisticsProcessor

        # Reload before running so recent project/config edits and stage changes
        # are picked up without requiring an application restart.
        self.reload()

        cfg = self.get(dataset_name)
        if cfg is None:
            raise KeyError(f"Dataset '{dataset_name}' not found in loaded configs")

        stats_proc = StatisticsProcessor(cfg, worker=worker)
        stats_proc.run()

        cfg.stage = "statistics_generated"
        cfg.cfg.setdefault("dataset", {})["stage"] = "statistics_generated"
        cfg.save()
        self._save_project_config_copy(cfg)
        self.reload()

        print(f"[DatasetManager] Statistics generated for '{dataset_name}'.")
        return True
