
import io
import numpy as np
import torch
import PySimpleGUI as sg
import rasterio

from pathlib import Path
from PIL import Image
from Interface.pages.view_base import ViewerBase
from ..theme import RText, RButton, COLORS, RHText, FONTS
from Core.Managers.path_manager import PathManager
from Core.Pytorch.pytorch_manager import MultimodalTileDataset
from Core.Utils.image_utility import ImageUtility


# ============================================================
# VIEW DATASET
#   
# 
#   
# ============================================================
class DatasetViewer(ViewerBase):
    key = "-PAGE_VIEWER_DATASET-"

    def __init__(self):
        
        super().__init__(entity=None, title="Dataset Viewer")

        self.cfg = None
        self.info_panel = None
        self.visuals_panel = None
        self.processing_panel = None
        self.band_col = None
        self.info_fields = {}

        # PyTorch + visuals
        self.torch_dataset = None
        self.sample_tensor = None
        self.sample_meta = None
        self.img_util = ImageUtility()
        self.pm = PathManager()

    # ------------------------------------------------------------
    # Information layout helpers
    # ------------------------------------------------------------
    def _info_value(self, key, width=36):
        text = sg.Text(
            "-",
            key=key,
            size=(width, 1),
            font=FONTS["body"],
            text_color=COLORS["text_primary"],
            background_color=COLORS["bg_dark"],
        )
        self.info_fields[key] = text
        return text

    def _info_row(self, label, key, width=36):
        return [
            sg.Text(
                label,
                size=(16, 1),
                font=FONTS["body"],
                text_color=COLORS["text_secondary"],
                background_color=COLORS["bg_dark"],
                justification="right",
            ),
            self._info_value(key, width=width),
        ]

    def _update_info(self, key, value):
        field = self.info_fields.get(key)
        if field:
            field.update("-" if value in (None, "") else str(value))

    def _display_role(self, role):
        role = str(role or "mixed").strip().lower()
        mapping = {
            "predictive": "evaluation",
            "ground_truth": "evaluation",
            "validation": "evaluation",
            "evaluation": "prediction",
            "discovery": "prediction",
            "survey": "prediction",
        }
        return mapping.get(role, role)

    # ------------------------------------------------------------
    # BUILD PAGE
    # ------------------------------------------------------------
    def build_views(self):
        
        # ------------------------------------------------------------
        # Information
        # ------------------------------------------------------------
        left_info = sg.Column(
            [
                self._info_row("Name", f"{self.key}_INFO_NAME"),
                self._info_row("Stage", f"{self.key}_INFO_STAGE"),
                self._info_row("Role", f"{self.key}_INFO_ROLE"),
                self._info_row("AOI", f"{self.key}_INFO_AOI", width=50),
                self._info_row("Tiles", f"{self.key}_INFO_TILES"),
            ],
            background_color=COLORS["bg_dark"],
            pad=((0, 25), (0, 0)),
            vertical_alignment="top",
        )

        right_info = sg.Column(
            [
                self._info_row("Structure", f"{self.key}_INFO_STRUCTURE"),
                self._info_row("Tile Size", f"{self.key}_INFO_TILE_SIZE"),
                self._info_row("Model Inputs", f"{self.key}_INFO_INPUTS", width=55),
                self._info_row("Mask / QC", f"{self.key}_INFO_MASKS", width=55),
                self._info_row("Processing", f"{self.key}_INFO_PROCESSING", width=55),
            ],
            background_color=COLORS["bg_dark"],
            pad=((25, 0), (0, 0)),
            vertical_alignment="top",
        )

        info_layout = [
            [RHText("Dataset Information")],
            [left_info, sg.VSeparator(color=COLORS["line_bright"]), right_info],
        ]

        view_info = sg.Column(info_layout,
                              key=f"{self.key}_INFO",
                              visible=False,
                              background_color=COLORS["bg_dark"],
        )
        self.add_view("info", view_info)

        # ------------------------------------------------------------
        # Visualisations
        # ------------------------------------------------------------
        self.btn_generate_stats = RButton("Generate Statistics", key=f"{self.key}_GEN_STATS")
        self.btn_rgb = RButton("RGB - Stitched", key=f"{self.key}_VIS_GLOBAL_RGB", visible=False)
        self.tile_selector = sg.Combo([], key=f"{self.key}_TILE_SELECT", readonly=True, size=(18, 1), visible=False)
        self.btn_tile_rgb = RButton("Tile RGB", key=f"{self.key}_VIS_TILE_RGB", visible=False)
        self.btn_tile_overlay = RButton("Tile Overlay", key=f"{self.key}_VIS_TILE_OVERLAY", visible=False)
        self.btn_umap_raw = RButton("UMAP (Raw)", key=f"{self.key}_UMAP_RAW", visible=False)
        self.btn_corr = RButton("Correlation Heatmap", key=f"{self.key}_CORR", visible=False)
        self.btn_hist = RButton("Band Histograms", key=f"{self.key}_HIST", visible=False)

        vis_layout = [
                        [RHText("Visualisations")],
                        [
                            self.btn_generate_stats,
                            self.btn_umap_raw,
                            self.btn_corr,
                            self.btn_hist,
                            self.btn_rgb,
                        ],
                        [
                            RText("Tile:"),
                            self.tile_selector,
                            self.btn_tile_rgb,
                            self.btn_tile_overlay,
                        ],
        ]

        view_visuals = sg.Column(vis_layout,
                                 key=f"{self.key}_VISUALS",
                                 visible=False,
                                 background_color=COLORS["bg_dark"],
        )
        self.add_view("visuals", view_visuals)


        # ------------------------------------------------------------
        # Processing
        # ------------------------------------------------------------
        process_layout = [
                            [RHText("Processing")],
                            [RButton("Process Dataset", key=f"{self.key}_PROCESS")],
        ]

        view_processing = sg.Column(process_layout,
                                    key=f"{self.key}_PROCESSING",
                                    visible=False,
                                    background_color=COLORS["bg_dark"],
        )
        self.add_view("processing", view_processing)


    # ------------------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------------------
    def load_dataset(self, cfg, window):
        
        self.cfg = cfg
        # ---------------------------------------------------------------------
        # Update concise, user-facing dataset information.
        # The old full band TRUE/FALSE list was useful during development,
        # but it made the top of the page too noisy for normal use.
        # ---------------------------------------------------------------------

        def _safe(value, default="-"):
            return default if value in (None, "") else value

        def _list_text(values):
            values = list(values or [])
            return ", ".join(map(str, values)) if values else "-"

        tile_width = _safe(getattr(getattr(cfg, "tile_structure", None), "width", None))
        tile_height = _safe(getattr(getattr(cfg, "tile_structure", None), "height", None))
        tile_format = _safe(getattr(getattr(cfg, "tile_structure", None), "tile_format", None))

        processing = getattr(cfg, "processing", None)
        normalisation = getattr(cfg, "normalisation", None)
        proc_parts = [
            f"Resolution: {_safe(getattr(processing, 'resolution', None))}",
            f"Normalisation: {_safe(getattr(normalisation, 'method', None))}",
            f"DEM: {_safe(getattr(processing, 'include_dem', None))}",
            f"Soil: {_safe(getattr(processing, 'include_soil', None))}",
            f"Indices: {_safe(getattr(processing, 'include_indices', None))}",
        ]

        self._update_info(f"{self.key}_INFO_NAME", cfg.dataset_name)
        self._update_info(f"{self.key}_INFO_STAGE", _safe(cfg.stage))
        self._update_info(f"{self.key}_INFO_ROLE", self._display_role(getattr(cfg, 'role', None)))
        self._update_info(f"{self.key}_INFO_AOI", f"Lat {cfg.min_lat}–{cfg.max_lat}  |  Lon {cfg.min_lon}–{cfg.max_lon}")
        self._update_info(f"{self.key}_INFO_TILES", _safe(cfg.tile_count))
        self._update_info(f"{self.key}_INFO_STRUCTURE", _safe(getattr(cfg, 'structure', None)))
        self._update_info(f"{self.key}_INFO_TILE_SIZE", f"{tile_width} x {tile_height}  |  {tile_format}")
        self._update_info(f"{self.key}_INFO_INPUTS", f"{_safe(getattr(cfg, 'num_input_channels', None))} channels  |  {_list_text(getattr(cfg, 'input_channels', []))}")
        self._update_info(f"{self.key}_INFO_MASKS", _list_text(getattr(cfg, 'mask_channels', [])))
        self._update_info(f"{self.key}_INFO_PROCESSING", "  |  ".join(proc_parts))

        # Apply stage logic
        self.apply_stage(cfg.stage)

        # Load PyTorch dataset if processed
        if cfg.stage in ("processed", "ready", "statistics_generated"):
            self._load_pytorch_dataset(cfg)

        # Populate tile selector for tile-level visualisation
        self._refresh_tile_selector()

    # ------------------------------------------------------------
    # STAGE LOGIC
    # ------------------------------------------------------------
    def apply_stage(self, stage):

        # Always show info
        self.views["info"].update(visible=True)

        # Default: hide everything else
        self.views["visuals"].update(visible=False)
        self.views["processing"].update(visible=False)

        # ------------------------------------------------------------
        # BUTTON VISIBILITY LOGIC (same as original)
        # ------------------------------------------------------------
        if stage == "statistics_generated":
            self.btn_generate_stats.update(visible=False)
            self.btn_umap_raw.update(visible=True)
            self.btn_corr.update(visible=True)
            self.btn_hist.update(visible=True)
        else:
            self.btn_generate_stats.update(visible=True)
            self.btn_umap_raw.update(visible=False)
            self.btn_corr.update(visible=False)
            self.btn_hist.update(visible=False)

        # ------------------------------------------------------------
        # PANEL VISIBILITY LOGIC (converted to model-viewer style)
        # ------------------------------------------------------------
        if stage in ("processed", "ready", "statistics_generated"):

            # ------------------------------------------------------------
            # Do not show the full stitched RGB by default. It can be too large
            # for PIL/Tkinter and is not needed for tile-level model work.
            # ------------------------------------------------------------
            
            self.btn_rgb.update(visible=False)
            self.tile_selector.update(visible=True)
            self.btn_tile_rgb.update(visible=True)
            self.btn_tile_overlay.update(visible=True)
            self.views["visuals"].update(visible=True)

        elif stage in ("raw", "downloaded", "processing"):
            self.btn_rgb.update(visible=False)
            self.tile_selector.update(visible=False)
            self.btn_tile_rgb.update(visible=False)
            self.btn_tile_overlay.update(visible=False)
            self.views["processing"].update(visible=True)

        # Save stage
        self.current_stage = stage


    # ------------------------------------------------------------
    # EVENT HANDLER
    # ------------------------------------------------------------
    def handle_event(self, event, values, window):

        if event == f"{self.key}_PROCESS":
            window.write_event_value("-TASK_PROCESS_DATASET-", self.cfg.dataset_name)

        elif event == f"{self.key}_RGB":
            self._show_visual("rgb")

        elif event == f"{self.key}_GBR":
            self._show_visual("gbr")

        elif event == f"{self.key}_ANOMALY":
            self._show_visual("anomaly")

        elif event == f"{self.key}_CLEANED":
            self._show_visual("cleaned")

        elif event == f"{self.key}_CLUSTER":
            self._show_visual("cluster")
        
        elif event == f"{self.key}_GEN_STATS":
            window.write_event_value("-TASK_GENERATE_STATISTICS-", self.cfg.dataset_name)

        elif event == f"{self.key}_UMAP_RAW":
            self._show_stat_image("umap_raw")

        elif event == f"{self.key}_CORR":
            self._show_stat_image("correlation_heatmap")

        elif event == f"{self.key}_HIST":
            self._show_stat_image("band_histograms")
        
        elif event == f"{self.key}_VIS_GLOBAL_RGB":
            rgb_path = self._visuals_dir() / f"{self.cfg.dataset_name}_RGB.png"
            self.img_util.show_image_window(rgb_path, title="Global RGB Mosaic")

        elif event == f"{self.key}_VIS_TILE_RGB":
            self._show_tile_rgb(values)

        elif event == f"{self.key}_VIS_TILE_OVERLAY":
            self._show_tile_overlay(values)


    # ------------------------------------------------------------
    # LOAD PYTORCH DATASET
    # ------------------------------------------------------------
    def _load_pytorch_dataset(self, cfg):

        self.torch_dataset = MultimodalTileDataset(root_dir=cfg.processed_path,
                                                   bands=cfg.bands.included,
                                                   tile_size=cfg.cfg.get("tile_size", 256),
                                                   transform=None,
                                                   cfg=cfg
        )

        # Cache first tile
        self.sample_tensor, self.sample_meta = self.torch_dataset[0]

    # ------------------------------------------------------------
    # BAND INDEX HELPER
    # ------------------------------------------------------------
    def _band_index(self, band_name: str) -> int:
        return self.cfg.bands.included.index(band_name)

    # ------------------------------------------------------------
    # VISUAL GENERATOR
    # ------------------------------------------------------------
    def _show_visual(self, mode: str):

        if self.sample_tensor is None:
            return

        x = self.sample_tensor  # (C,H,W)

        # RGB band indices
        r = x[self._band_index("S2:B4")]
        g = x[self._band_index("S2:B3")]
        b = x[self._band_index("S2:B2")]

        if mode == "rgb":
            arr = torch.stack([r, g, b], dim=0).cpu().numpy().transpose(1, 2, 0)

        elif mode == "gbr":
            arr = torch.stack([g, b, r], dim=0).cpu().numpy().transpose(1, 2, 0)

        elif mode == "anomaly":
            # For now: anomaly = original vs original
            self.img_util.visualize_anomaly(x, x)
            return

        elif mode == "cleaned":
            arr = torch.stack([r, g, b], dim=0).cpu().numpy().transpose(1, 2, 0)

        elif mode == "cluster":
            arr = r.cpu().numpy()

        # Convert to PIL
        arr = np.clip(arr, 0, 1)

        if arr.ndim == 2:
            img = Image.fromarray((arr * 255).astype("uint8"))
        else:
            img = Image.fromarray((arr * 255).astype("uint8"))

        # ------------------------------------------------------------
        # Mirror the model visualisation pattern: save the generated image
        # and open it through ImageUtility, which gives the user a Close button
        # instead of leaving an unhandled modal window on the event loop.
        # ------------------------------------------------------------
        out_dir = self._visuals_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.cfg.dataset_name}_{mode}.png"
        img.save(out_path, format="PNG")

        self.img_util.show_image_window(out_path, title=f"{mode.upper()} Preview")


    # ------------------------------------------------------------
    # TILE-LEVEL VISUALS
    # ------------------------------------------------------------
    def _processed_root(self) -> Path:
        return Path(self.pm.resolve_path(self.cfg.processed_path))


    def _tile_sort_key(self, path: Path):

        digits = "".join(ch for ch in path.name if ch.isdigit())
        return int(digits) if digits else path.name


    def _tile_dirs(self):

        root = self._processed_root()
        # Supports both historic "tile 0" and newer "tile_0" naming.
        dirs = [d for d in root.iterdir() if d.is_dir() and d.name.lower().startswith("tile")]
        
        return sorted(dirs, key=self._tile_sort_key)


    def _refresh_tile_selector(self):

        try:
        
            choices = [d.name for d in self._tile_dirs()]
            default = choices[0] if choices else ""
            self.tile_selector.update(values=choices, value=default)
        
        except Exception as e:
            print(f"[Tile selector error] {e}")


    def _selected_tile_dir(self, values):

        choices = self._tile_dirs()
        
        if not choices:
            raise FileNotFoundError("No tile folders found in processed dataset.")

        selected = values.get(f"{self.key}_TILE_SELECT") if values else None
        
        if selected:
            selected_path = self._processed_root() / selected
        
            if selected_path.exists():
                return selected_path

        return choices[0]


    def _normalise_rgb(self, rgb: np.ndarray) -> np.ndarray:
    
        rgb = rgb.astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.float32)
    
        for c in range(3):
    
            band = rgb[:, :, c]
            lo, hi = np.nanpercentile(band, (2, 98))
    
            if hi - lo > 1e-6:
                out[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1)
    
        return (out * 255).astype(np.uint8)


    def _read_tile_rgb(self, tile_dir: Path) -> np.ndarray:

        rgb_path = tile_dir / "RGB.tif"
        
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB.tif not found in {tile_dir}")

        with rasterio.open(rgb_path) as src:
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

        return self._normalise_rgb(rgb)


    def _find_overlay_mask(self, tile_dir: Path):
        # ------------------------------------------------------------
        # Prefer real ground-truth labels if present; fall back to QC.tif so the
        # button still shows something during development.
        # ------------------------------------------------------------
        candidates = [
            "ground_truth.tif", "GroundTruth.tif", "GROUND_TRUTH.tif",
            "labels.tif", "label.tif", "mask.tif", "GT.tif", "QC.tif",
        ]
        for name in candidates:
            p = tile_dir / name
            if p.exists():
                return p
        return None

    def _read_mask(self, mask_path: Path, target_shape) -> np.ndarray:

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        
        mask = np.nan_to_num(mask, nan=0.0)
        mask = mask > 0

        # Resize if needed to match the RGB tile.
        if mask.shape != target_shape:

            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_img = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            mask = np.array(mask_img) > 0

        return mask

    def _save_and_show_array(self, arr: np.ndarray, path: Path, title: str):
    
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path, format="PNG")
        self.img_util.show_image_window(path, title=title)

    def _show_tile_rgb(self, values):

        try:
        
            tile_dir = self._selected_tile_dir(values)
            rgb = self._read_tile_rgb(tile_dir)
            out_path = self._visuals_dir() / f"{self.cfg.dataset_name}_{tile_dir.name}_RGB.png"
            self._save_and_show_array(rgb, out_path, title=f"Tile RGB - {tile_dir.name}")
        
        except Exception as e:
            sg.popup_error(f"Failed to load tile RGB:\n{e}")

    def _show_tile_overlay(self, values):

        try:
        
            tile_dir = self._selected_tile_dir(values)
            rgb = self._read_tile_rgb(tile_dir)
            mask_path = self._find_overlay_mask(tile_dir)
        
            if mask_path is None:
                raise FileNotFoundError(
                    "No overlay mask found. Expected one of: ground_truth.tif, labels.tif, mask.tif, GT.tif, or QC.tif"
                )

            mask = self._read_mask(mask_path, rgb.shape[:2])
            overlay = rgb.copy().astype(np.float32)
            alpha = 0.45
            overlay[mask, 0] = (1 - alpha) * overlay[mask, 0] + alpha * 255
            overlay[mask, 1] = (1 - alpha) * overlay[mask, 1]
            overlay[mask, 2] = (1 - alpha) * overlay[mask, 2]
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            out_path = self._visuals_dir() / f"{self.cfg.dataset_name}_{tile_dir.name}_overlay.png"
            self._save_and_show_array(overlay, out_path, title=f"Tile Overlay - {tile_dir.name} ({mask_path.name})")
        
        except Exception as e:
            sg.popup_error(f"Failed to create tile overlay:\n{e}")


    # ------------------------------------------------------------
    #    Handles messages from worker threads during dataset processing.
    #    Mirrors the pattern used in PageLoadDataset.
    # ------------------------------------------------------------
    def on_worker_message(self, task_id, msg_type, data):

        match msg_type:

            case "status":
                print(f"[STATUS] {data}")

            case "progress":
                print(f"[PROGRESS] {data}%")

            case "result":
                    
                    match task_id:

                        case "generate_statistics":
                            # Update stage in memory
                            self.cfg.stage = "statistics_generated"

                            # Update UI visibility
                            self.apply_stage("statistics_generated")

                        case "process_dataset":
                            print("[RESULT] Dataset processed successfully")

            case "error":
                print(f"[ERROR] {data}")

            case "finished":

                # -----------------------------------------
                # NEW: Handle statistics completion
                # -----------------------------------------
                match task_id:

                    case "generate_statistics":
                        # Update stage in memory
                        self.cfg.stage = "statistics_generated"

                        # Update UI visibility
                        self.apply_stage("statistics_generated")
                    
                    case "process_dataset":

                        # Update stage in memory
                        self.cfg.stage = "processed"
                        
                        # Update UI visibility
                        self.apply_stage("processed")

                        print(f"[FINISHED] Task {task_id} complete")


    # ------------------------------------------------------------
    # Return the dataset visuals directory, tolerating older configs.
    # ------------------------------------------------------------
    def _visuals_dir(self) -> Path:
        
        visuals_dir = getattr(self.cfg.paths, "visuals_dir", None)
        if visuals_dir:
            return Path(visuals_dir)

        return Path(self.cfg.paths.root) / "Visuals"


    def _show_stat_image(self, name: str):
        img_path = self._visuals_dir() / f"{self.cfg.dataset_name}_{name}.png"

        if not img_path.exists():
            sg.popup_error(f"Statistic image not found:\n{img_path}")
            return

        self.img_util.show_image_window(
            img_path,
            title=name.replace("_", " ").title(),
        )
