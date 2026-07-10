import yaml
import torch

from pathlib import Path


# ============================================================
#    PATH MANAGER
#
    # Centralised path resolver for Terra-AId.
    # Only handles global application paths.
    # Dataset/model/search paths are resolved by their own configs.
# ============================================================
class PathManager:

    def __init__(self, config_path: str = "app_config.yaml"):

        # Terra-AId_v2/Core/Managers/path_manager.py → parents[2] = Terra-AId_v2/
        self.PROJECT_ROOT = Path(__file__).resolve().parents[2]

        # Absolute path to config file (Interface/app_config.yaml under project root)
        self.config_path = (self.PROJECT_ROOT / config_path).resolve()

        self._load_config()
        self._resolve_paths()

    # ------------------------------------------------------------
    # Load YAML
    # ------------------------------------------------------------
    def _load_config(self):

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        # ------------------------------------------------------------
        # Global band list (NEW)
        # ------------------------------------------------------------
        self.ALL_BANDS = (
            self.cfg.get("bands", {}).get("all", [])
        )

    # ------------------------------------------------------------
    # Resolve global paths
    # ------------------------------------------------------------
    def _resolve_paths(self):

        dirs = self.cfg["directories"]

        # Assets
        self.ASSETS_DIR = self.PROJECT_ROOT / dirs["assets"]

        # Top-level pillars
        self.DATASETS_DIR = self.PROJECT_ROOT / dirs["datasets"]

        # Root where each dataset folder lives
        self.DATASETS_ROOT = self.DATASETS_DIR

        # Optional: global visuals root (if you want shared visuals)
        self.VISUALS_ROOT = self.PROJECT_ROOT / dirs.get("visuals", "Visuals")

        self.MODELS_ROOT = self.PROJECT_ROOT / dirs["models"]
        self.SEARCHES_DIR = self.PROJECT_ROOT / dirs["searches"]

        # Central registries
        self.DATASET_CONFIGS = self.PROJECT_ROOT / dirs["dataset_configs"]
        self.MODEL_CONFIGS = self.PROJECT_ROOT / dirs["model_configs"]
        self.SEARCH_CONFIGS = self.PROJECT_ROOT / dirs["search_configs"]


        # ------------------------------------------------------------
        # DEVICE (global, shared across entire application)
        # ------------------------------------------------------------
        cfg_device = self.cfg.get("device", {}).get("default", "cuda")

        if cfg_device == "cuda" and torch.cuda.is_available():

            self.DEVICE = torch.device("cuda")
            self.IS_CUDA = True
            self.GPU_NAME = torch.cuda.get_device_name(0)
            self.GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory

        else:

            self.DEVICE = torch.device("cpu")
            self.IS_CUDA = False
            self.GPU_NAME = "CPU"
            self.GPU_MEMORY = None

    # ------------------------------------------------------------
    # Public path resolver
    # ------------------------------------------------------------
    def resolve_path(self, path):

        p = Path(path)
        
        if p.is_absolute():
            return str(p)
        
        return str((self.PROJECT_ROOT / p).resolve())

    # ------------------------------------------------------------
    # Asset helper
    # ------------------------------------------------------------
    def banner(self, name: str):

        print(str(self.ASSETS_DIR / name))
        return str(self.ASSETS_DIR / name)

    # ------------------------------------------------------------
    # Debug summary
    # ------------------------------------------------------------
    def summary(self):
        
        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"Assets: {self.ASSETS_DIR}")
        print(f"Datasets: {self.DATASETS_DIR}")
        print(f"Models: {self.MODELS_DIR}")
        print(f"Searches: {self.SEARCHES_DIR}")
        print(f"Dataset Configs: {self.DATASET_CONFIGS}")
        print(f"Model Configs: {self.MODEL_CONFIGS}")
        print(f"Search Configs: {self.SEARCH_CONFIGS}")
        print(f"Device: {self.DEVICE}")
        print(f"CUDA Available: {self.IS_CUDA}")
        print(f"GPU Name: {self.GPU_NAME}")
        print(f"GPU Memory: {self.GPU_MEMORY}")
        print(f"ALL_BANDS: {self.ALL_BANDS}")
