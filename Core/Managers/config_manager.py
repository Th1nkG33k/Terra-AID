from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import yaml

from Core.Managers.path_manager import PathManager
from Core.Utils.channel_policy import (
    get_model_input_channels,
    get_mask_channels,
    make_runtime_profile,
)
from Models.builder import build_model as build_architecture_model


def _ns(data: dict | None, **defaults) -> SimpleNamespace:
    merged = dict(defaults)
    if isinstance(data, dict):
        merged.update(data)
    return SimpleNamespace(**merged)


def _list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_for_yaml(value):
    """Remove empty/derived clutter before writing YAML."""
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            # These are now runtime/system-derived; never persist them.
            if k in {"profile", "input_profile"}:
                continue
            if k == "device":
                continue
            if k in {"num_channels", "channel_names"}:
                continue
            if k in {"depth", "tiles_dir"}:
                continue
            if k == "labels" and isinstance(v, dict) and not v.get("type"):
                continue
            cv = _clean_for_yaml(v)
            if cv in (None, {}, []):
                continue
            cleaned[k] = cv
        return cleaned
    if isinstance(value, list):
        return [_clean_for_yaml(v) for v in value if _clean_for_yaml(v) not in (None, {}, [])]
    if value in ("", "YYYY-MM-DD"):
        return None
    return value


class BaseConfig:
    def __init__(self, name: str, cfg: dict, config_path: Path | None, pm: PathManager):
        self.name = name
        self.cfg = cfg or {}
        self.config_path = Path(config_path) if config_path else None
        self.pm = pm

    def cleaned_cfg(self) -> dict:
        return _clean_for_yaml(copy.deepcopy(self.cfg))

    def to_yaml(self) -> str:
        return yaml.dump(self.cleaned_cfg(), sort_keys=False)

    def save(self):
        if not self.config_path:
            raise RuntimeError(f"{self.__class__.__name__} has no config_path set; cannot save.")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(self.to_yaml(), encoding="utf-8")


class DatasetConfig(BaseConfig):
    """Dataset YAML reader.

    Dataset configs store physical dataset choices only. Model input channels are
    derived at runtime from bands.included using Core.Utils.channel_policy.
    """

    def __init__(self, name: str, cfg: dict, config_path: Path | None, pm: PathManager):
        super().__init__(name, cfg, config_path, pm)

        ds = self.cfg.get("dataset", {}) or {}
        self.dataset_name = ds.get("name", name)
        self.version = ds.get("version", "1.0")
        self.stage = ds.get("stage", "unknown")
        self.tile_count = ds.get("tile_count")
        self.role = ds.get("role", "mixed")
        self.structure = ds.get("structure", "aoi_grid")

        paths_cfg = self.cfg.get("paths", {}) or {}
        raw_root = paths_cfg.get("root")
        if raw_root and raw_root != ".":
            root = Path(raw_root)
            if not root.is_absolute():
                root = pm.PROJECT_ROOT / root
        else:
            root = pm.DATASETS_ROOT / self.dataset_name

        self.paths = SimpleNamespace()
        self.paths.root = root.resolve()

        def _resolve(subpath, default):
            p = Path(subpath) if subpath else Path(default)
            if not p.is_absolute():
                p = self.paths.root / p
            return p.resolve()

        self.paths.raw_s2 = _resolve(paths_cfg.get("raw_s2"), "Raw/S2")
        self.paths.raw_dem = _resolve(paths_cfg.get("raw_dem"), "Raw/DEM")
        self.paths.raw_soil = _resolve(paths_cfg.get("raw_soil"), "Raw/Soil")
        self.paths.raw_ground_truth = _resolve(paths_cfg.get("raw_ground_truth"), "Raw/GroundTruth")
        self.paths.processed_tiles = _resolve(paths_cfg.get("processed_tiles"), "Dataset")
        self.paths.visuals_dir = _resolve(paths_cfg.get("visuals_dir"), "Visuals")
        self.paths.tile_manifest = _resolve(paths_cfg.get("tile_manifest"), "tile_manifest.csv")
        self.paths.channel_stats = _resolve(paths_cfg.get("channel_stats"), "channel_stats.json")
        self.paths.tile_dir = self.paths.processed_tiles

        self.processed_path = self.paths.processed_tiles
        self.raw_s2_path = self.paths.raw_s2
        self.raw_dem_path = self.paths.raw_dem
        self.raw_soil_path = self.paths.raw_soil
        self.raw_ground_truth_path = self.paths.raw_ground_truth
        self.tile_dir_path = self.paths.tile_dir
        self.tile_manifest_path = self.paths.tile_manifest
        self.channel_stats_path = self.paths.channel_stats

        self.tile_structure = _ns(self.cfg.get("tile_structure", {}), width=None, height=None, tile_format="tif")

        aoi = self.cfg.get("AOI", {}) or {}
        self.min_lat = aoi.get("min_lat")
        self.max_lat = aoi.get("max_lat")
        self.min_lon = aoi.get("min_lon")
        self.max_lon = aoi.get("max_lon")
        self.AOI = _ns(aoi, min_lat=self.min_lat, max_lat=self.max_lat, min_lon=self.min_lon, max_lon=self.max_lon)

        bands = self.cfg.get("bands", {}) or {}
        self.bands = _ns(bands, included=[], rgb_order=[])
        self.bands.included = _list(getattr(self.bands, "included", []))
        self.bands.rgb_order = _list(getattr(self.bands, "rgb_order", []))

        proc = self.cfg.get("processing", {}) or {}
        self.processing = _ns(proc, resolution="10m", include_dem=False, include_soil=False,
                              include_indices=False, include_qc_mask=False, allow_categorical_inputs=False)

        # Runtime-derived channel information used by managers and models.
        # but not written back into YAML.
        self.input_channels = get_model_input_channels(self)
        self.mask_channels = get_mask_channels(self)
        self.num_input_channels = len(self.input_channels)
        self.depth = self.num_input_channels
        self.profile = make_runtime_profile(self)

        crs = self.cfg.get("crs", {}) or {}
        self.crs = _ns(crs, epsg=None)

        norm = self.cfg.get("normalisation", {}) or {}
        self.normalisation = _ns(norm, method=None, stats_source=None, clip=[-5.0, 5.0])

        splits = self.cfg.get("splits", {}) or {}
        self.splits = _ns(splits, train=None, val=None, test=None)

        labels = self.cfg.get("labels", {}) or {}
        self.labels = _ns(labels, type=None, source=None, source_root="Raw/GroundTruth",
                          source_pattern="{tile_folder}/archaeology_selected.geojson",
                          rasterized_name="ground_truth.tif", geometry_type=None, applies_to=None)

        self.date_range = _ns(self.cfg.get("date_range", {}), start=None, end=None)
        self.provenance = self.cfg.get("provenance", {}) or {}

    def exists(self) -> bool:
        return self.processed_path.exists() or self.raw_s2_path.exists()


class ModelConfig(BaseConfig):
    """Model YAML reader.

    Model configs store architecture/training choices only. Channel count/order is
    derived from the selected training dataset at runtime, not duplicated in YAML.
    """

    def __init__(self, name: str, cfg: dict, config_path: Path | None, pm: PathManager):
        super().__init__(name, cfg, config_path, pm)

        self.cfg.setdefault("model", {})
        self.cfg.setdefault("architecture", {})
        self.cfg.setdefault("optimizer", {})
        self.cfg.setdefault("scheduler", {})
        self.cfg.setdefault("training", {})
        self.cfg.setdefault("prediction", {})
        self.cfg.setdefault("paths", {})

        m = self.cfg.get("model", {}) or {}
        self.model_name = m.get("name", name)
        # Device is runtime-selected by the application startup.
        # Model YAML may contain an old device value, but it is intentionally ignored.
        self.device = pm.DEVICE
        self.yaml_device = None
        self.stage = m.get("stage", "created")

        paths_cfg = self.cfg.get("paths", {}) or {}
        raw_root = paths_cfg.get("root")
        if raw_root and raw_root != ".":
            root = Path(raw_root)
            if not root.is_absolute():
                root = pm.PROJECT_ROOT / root
        else:
            root = pm.MODELS_ROOT / self.model_name

        self.paths = SimpleNamespace()
        self.paths.root = root.resolve()

        def _resolve(subpath, default):
            p = Path(subpath) if subpath else Path(default)
            if not p.is_absolute():
                p = self.paths.root / p
            return p.resolve()

        self.paths.checkpoints = _resolve(paths_cfg.get("checkpoints"), "Checkpoints")
        self.paths.logs = _resolve(paths_cfg.get("logs"), "logs")
        self.paths.outputs = _resolve(paths_cfg.get("outputs"), "Visuals")
        self.checkpoints_path = self.paths.checkpoints
        self.logs_path = self.paths.logs
        self.outputs_path = self.paths.outputs

        arc = copy.deepcopy(self.cfg.get("architecture", {}) or {})
        # Ignore persisted channel mirrors from older configs. They are runtime-only now.
        arc.pop("num_channels", None)
        arc.pop("channel_names", None)
        self.architecture = _ns(arc, type="mae", base_channels=32, encoder_depth=5, decoder_depth=3,
                                embed_dim=128, decoder_dim=64, mask_ratio=0.75,
                                latent_channels=256, backbone="resnet50", pretrained=False,
                                freeze_encoder_epochs=0)
        self.architecture.num_channels = None
        self.architecture.channel_names = []

        self.optimizer = _ns(self.cfg.get("optimizer", {}), type="AdamW", lr=1e-4, weight_decay=1e-5)
        self.scheduler = _ns(self.cfg.get("scheduler", {}), type="None", warmup_epochs=0)
        self.training = _ns(self.cfg.get("training", {}), batch_size=1, num_workers=0, epochs=20,
                            early_stopping_patience=5, dataset=None)
        self.training_dataset = getattr(self.training, "dataset", None)
        self.prediction = _ns(self.cfg.get("prediction", {}), dataset=None, anomaly_score_mode="raw_all_channels",
                              prediction_preset="Balanced", threshold_mode="per_tile_percentile",
                              threshold_percentile=95.0, threshold_value=None,
                              threshold_metric="fp_penalised_f1", false_positive_penalty=0.2,
                              ignore_score_channels=["SCL", "QC"], min_component_pixels=0,
                              max_false_positive_rate=0.45, min_recall=0.20)
        self.prediction_dataset = getattr(self.prediction, "dataset", None)

        # Runtime only. Old UI can still read cfg.input_profile safely.
        self.input_profile = _ns({}, name=None, num_input_channels=None, input_channels=[], mask_channels=[])
        self.runtime_input_channels = []
        self.runtime_mask_channels = []
        self.provenance = self.cfg.get("provenance", {}) or {}

    def _sync_runtime_channels(self, input_channels: list[str], mask_channels: list[str] | None = None):
        input_channels = list(input_channels or [])
        mask_channels = list(mask_channels or [])
        self.runtime_input_channels = input_channels
        self.runtime_mask_channels = mask_channels
        self.architecture.num_channels = len(input_channels) if input_channels else None
        self.architecture.channel_names = input_channels
        self.input_profile = _ns({
            "name": f"derived_{len(input_channels)}ch" if input_channels else None,
            "num_input_channels": len(input_channels) if input_channels else None,
            "input_channels": input_channels,
            "mask_channels": mask_channels,
        })

    def set_runtime_input_from_dataset(self, dataset_cfg: DatasetConfig):
        self._sync_runtime_channels(dataset_cfg.input_channels, dataset_cfg.mask_channels)

    # Runtime name used by current managers/tasks. It no longer writes
    # channel lists to YAML.
    def set_input_profile_from_dataset(self, dataset_cfg: DatasetConfig):
        self.set_runtime_input_from_dataset(dataset_cfg)

    def save(self):
        # Do not write runtime channel mirrors.
        super().save()
        try:
            project_copy = self.paths.root / f"{self.model_name}.yaml"
            if self.config_path and project_copy.resolve() != self.config_path.resolve():
                project_copy.parent.mkdir(parents=True, exist_ok=True)
                project_copy.write_text(self.to_yaml(), encoding="utf-8")
        except Exception as exc:
            print(f"[ModelConfig] Warning: could not save project config copy: {exc}")

    def update_stage(self, new_stage: str):
        self.stage = new_stage
        self.cfg.setdefault("model", {})["stage"] = new_stage
        self.save()

    def set_training_dataset(self, dataset_name: str):
        self.training_dataset = dataset_name
        self.cfg.setdefault("training", {})["dataset"] = dataset_name
        self.save()

    def set_prediction_dataset(self, dataset_name: str):
        self.prediction_dataset = dataset_name
        self.cfg.setdefault("prediction", {})["dataset"] = dataset_name
        self.save()

    def build_model(self):
        if not getattr(self.architecture, "num_channels", None):
            raise RuntimeError(
                f"Model '{self.model_name}' has no runtime channel count. "
                "Select/process a training dataset or set cfg.architecture.num_channels before building."
            )
        return build_architecture_model(self)


class SearchConfig(BaseConfig):
    def __init__(self, name: str, cfg: dict, config_path: Path | None, pm: PathManager):
        super().__init__(name, cfg, config_path, pm)
        search = cfg.get("search", {}) or {}
        self.algorithm = search.get("algorithm")
        self.parameters = search.get("parameters", {})
        self.provenance = cfg.get("provenance", {}) or {}


class ConfigManager:
    def __init__(self, pm: PathManager | None = None):
        self.pm = pm or PathManager()
        self.datasets = {}
        self.models = {}
        self.searches = {}
        self.reload()

    def _load_yaml_files(self, cfg_dir: Path):
        if not cfg_dir or not cfg_dir.exists():
            return []
        return sorted(cfg_dir.glob("*.yaml"))

    def _load_datasets(self):
        for file in self._load_yaml_files(self.pm.DATASET_CONFIGS):
            cfg = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            self.datasets[file.stem] = DatasetConfig(file.stem, cfg, file, self.pm)

    def _load_models(self):
        for file in self._load_yaml_files(self.pm.MODEL_CONFIGS):
            cfg = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            self.models[file.stem] = ModelConfig(file.stem, cfg, file, self.pm)

    def _load_searches(self):
        cfg_dir = getattr(self.pm, "SEARCH_CONFIGS", None)
        for file in self._load_yaml_files(cfg_dir):
            cfg = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            self.searches[file.stem] = SearchConfig(file.stem, cfg, file, self.pm)

    def get_dataset(self, name):
        return self.datasets.get(name)

    def get_model(self, name):
        return self.models.get(name)

    def get_search(self, name):
        return self.searches.get(name)

    def reload(self):
        self.datasets.clear()
        self.models.clear()
        self.searches.clear()
        self._load_datasets()
        self._load_models()
        self._load_searches()
