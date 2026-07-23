import io
import numpy as np
import torch
import PySimpleGUI as sg
import rasterio

from pathlib import Path
from PIL import Image
from Interface.pages.view_base import ViewerBase
from ..theme import RText, RButton, COLORS, RHText, FONTS, BUTTON_COLORS
from Core.Managers.path_manager import PathManager
from Core.Pytorch.pytorch_manager import MultimodalTileDataset
from Core.Utils.image_utility import ImageUtility


# ============================================================
# VIEW DATASET
# ============================================================
class DatasetViewer(ViewerBase):
    key = "-PAGE_VIEWER_DATASET-"
    COLOUR_SHIFTS = ("RGB", "RBG", "GRB", "GBR", "BRG", "BGR")
    COLOUR_SHIFT_ORDERS = {
        "RGB": (0, 1, 2),
        "RBG": (0, 2, 1),
        "GRB": (1, 0, 2),
        "GBR": (1, 2, 0),
        "BRG": (2, 0, 1),
        "BGR": (2, 1, 0),
    }

    def __init__(self):
        super().__init__(entity=None, title="Dataset Viewer")

        self.cfg = None
        self.info_panel = None
        self.visuals_panel = None
        self.processing_panel = None
        self.band_col = None
        self.info_fields = {}

        # Tab/status elements
        self.txt_download_status = None
        self.txt_download_details = None
        self.txt_process_status = None
        self.txt_visuals_status = None
        self.process_actions_col = None
        self.visuals_actions_col = None

        # PyTorch + visuals
        self.torch_dataset = None
        self.sample_tensor = None
        self.sample_meta = None
        self.img_util = ImageUtility()
        self.pm = PathManager()
        self.colour_shift_buttons = {}

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

    def _safe(self, value, default="-"):
        return default if value in (None, "") else value

    def _list_text(self, values):
        values = list(values or [])
        return ", ".join(map(str, values)) if values else "-"

    def _display_stage(self, stage):
        return str(stage or "unknown").replace("_", " ").title()

    def _display_role(self, role):
        role = str(role or "mixed").strip().lower()
        mapping = {"training": "Training",
                   "predictive": "Evaluation",
                   "ground_truth": "Evaluation",
                   "validation": "Evaluation",
                   "evaluation": "Prediction",
                   "discovery": "Prediction",
                   "survey": "Prediction",
                   "prediction": "Prediction",
                   "mixed": "Mixed",
        }
        return mapping.get(role, role.replace("_", " ").title())

    def _set_text(self, element, value):
        if element is not None:
            element.update(str(self._safe(value)))

    def _asset_path(self, filename):
        if not filename:
            return None
        return Path(__file__).resolve().parents[2] / "assets" / filename

    # ---------------------------------------------------------------------
        
        # Dataset visualisation button. If an image with the supplied filename is
        # later added to /assets, it will be used automatically; otherwise a clean
        # text button is shown. This mirrors the View Model page pattern without
        # making the UI dependent on image assets being present now.
        
    # ---------------------------------------------------------------------
    def _visual_button(self, text, key, image_filename=None, visible=True):

        image_path = self._asset_path(image_filename)
        common = {"key": key,
                  "font": FONTS["body"],
                  "button_color": BUTTON_COLORS["secondary"],
                  "mouseover_colors": BUTTON_COLORS["secondary_hover"],
                  "border_width": 1,
                  "pad": (8, 8),
                  "visible": visible,
        }

        if image_path and image_path.exists():
            return sg.Button(
                "",
                image_filename=str(image_path),
                image_size=(165, 95),
                **common,
            )

        return sg.Button(text, size=(18, 3), **common)

    # ------------------------------------------------------------
    # BUILD PAGE
    # ------------------------------------------------------------
    def build_views(self):
        # ------------------------------------------------------------
        # Professional two-column information header
        # ------------------------------------------------------------
        left_info = sg.Column(
            [
                self._info_row("Name", f"{self.key}_INFO_NAME"),
                self._info_row("Stage", f"{self.key}_INFO_STAGE"),
                self._info_row("Role", f"{self.key}_INFO_ROLE"),
                self._info_row("AOI", f"{self.key}_INFO_AOI", width=52),
                self._info_row("Date Range", f"{self.key}_INFO_DATE_RANGE"),
                self._info_row("CRS", f"{self.key}_INFO_CRS"),
            ],
            background_color=COLORS["bg_dark"],
            pad=((0, 25), (0, 0)),
            vertical_alignment="top",
        )

        right_info = sg.Column(
            [
                self._info_row("Tiles", f"{self.key}_INFO_TILES"),
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
            [RHText("Information")],
            [left_info, sg.VSeparator(color=COLORS["line_bright"]), right_info],
        ]

        # ------------------------------------------------------------
        # Download tab
        # ------------------------------------------------------------
        self.txt_download_status = RText("Download Status: -")
        self.txt_download_details = RText("-", color=COLORS["text_secondary"])

        download_tab_layout = [
            [RText("Download")],
            [RText("Raw source data and AOI setup for this dataset.", color=COLORS["text_secondary"])],
            [self.txt_download_status],
            [self.txt_download_details],
        ]

        # ------------------------------------------------------------
        # Process tab
        # ------------------------------------------------------------
        self.txt_process_status = RText("Processing Status: -")
        self.btn_process = RButton("Process Dataset", key=f"{self.key}_PROCESS", w=0.20, visible=False)
        self.btn_umap_raw = self._visual_button("UMAP (Raw)", key=f"{self.key}_UMAP_RAW", image_filename="dataset_umap_raw.png", visible=False)
        self.btn_corr = self._visual_button("Correlation Heatmap", key=f"{self.key}_CORR", image_filename="dataset_correlation_heatmap.png", visible=False)
        self.btn_hist = self._visual_button("Band Histograms", key=f"{self.key}_HIST", image_filename="dataset_band_histograms.png", visible=False)
        self.btn_cluster_stitched = self._visual_button("Clustering - Stitched", key=f"{self.key}_CLUSTER_STITCHED", image_filename="dataset_clustering.png", visible=False)
        
        self.process_actions_col = sg.Column(
            [
                [RText("Process")],
                [RText("Convert the downloaded raw data into model-ready tiles and channel metadata.", color=COLORS["text_secondary"])],
                [self.txt_process_status],
                [self.btn_process],
                [
                    self.btn_umap_raw,
                    self.btn_corr,
                    self.btn_hist,
                    self.btn_cluster_stitched,
                ],
            ],
            key=f"{self.key}_PROCESS_ACTIONS_PANEL",
            background_color=COLORS["bg_dark"],
            visible=True,
            pad=(0, 10),
        )

        process_tab_layout = [[self.process_actions_col]]

        # ------------------------------------------------------------
        # Visuals tab
        # ------------------------------------------------------------
        self.txt_visuals_status = RText("Visual Status: -")
        self.btn_generate_stats = RButton("Generate Statistics", key=f"{self.key}_GEN_STATS", w=0.22, visible=False)

        self.colour_shift_buttons = {
            shift: self._visual_button(
                f"{shift} - Stitched",
                key=f"{self.key}_VIS_GLOBAL_{shift}",
                image_filename=f"dataset_global_{shift.lower()}.png",
                visible=False,
            )
            for shift in self.COLOUR_SHIFTS
        }
        # Retain the existing attribute for any external page code that refers
        # specifically to the canonical RGB stitched button.
        self.btn_rgb = self.colour_shift_buttons["RGB"]

        self.tile_selector = sg.Combo([], key=f"{self.key}_TILE_SELECT", readonly=True, size=(18, 1), visible=False)
        self.tile_label = RText("Tile:", key=f"{self.key}_TILE_LABEL", visible=False)
        self.btn_tile_rgb = self._visual_button("Tile RGB", key=f"{self.key}_VIS_TILE_RGB", image_filename="dataset_tile_rgb.png", visible=False)
        self.btn_tile_overlay = self._visual_button("Tile Overlay", key=f"{self.key}_VIS_TILE_OVERLAY", image_filename="dataset_tile_overlay.png", visible=False)
        self.btn_tile_cluster = self._visual_button("Tile Clustering", key=f"{self.key}_VIS_TILE_CLUSTER", image_filename="dataset_tile_cluster.png", visible=False)

        stitched_row_one = [self.colour_shift_buttons[name] for name in self.COLOUR_SHIFTS[:3]]
        stitched_row_two = [self.colour_shift_buttons[name] for name in self.COLOUR_SHIFTS[3:]]

        self.visuals_actions_col = sg.Column(
            [
                [RText("Visuals")],
                [RText("Review stitched colour composites and tile-level outputs.", color=COLORS["text_secondary"])],
                [self.txt_visuals_status],
                [self.btn_generate_stats],
                [RText("Stitched colour composites", color=COLORS["accent_highlight"])],
                stitched_row_one,
                stitched_row_two,
                [RText("Tile visuals", color=COLORS["accent_highlight"])],
                [
                    self.tile_label,
                    self.tile_selector,
                ],
                [
                    self.btn_tile_rgb,
                    self.btn_tile_overlay,
                    self.btn_tile_cluster,
                ],
            ],
            key=f"{self.key}_VISUALS_ACTIONS_PANEL",
            background_color=COLORS["bg_dark"],
            visible=True,
            pad=(0, 10),
        )

        visuals_tab_layout = [[self.visuals_actions_col]]

        tabs = sg.TabGroup(
            [[
                sg.Tab("Download", download_tab_layout, key=f"{self.key}_TAB_DOWNLOAD", background_color=COLORS["bg_dark"]),
                sg.Tab("Process", process_tab_layout, key=f"{self.key}_TAB_PROCESS", background_color=COLORS["bg_dark"]),
                sg.Tab("Visuals", visuals_tab_layout, key=f"{self.key}_TAB_VISUALS", background_color=COLORS["bg_dark"]),
            ]],
            key=f"{self.key}_TABGROUP",
            enable_events=True,
            background_color=COLORS["bg_dark"],
            tab_background_color=COLORS["bg_panel"],
            selected_background_color=COLORS["accent_primary"],
            selected_title_color=COLORS["text_on_accent"],
            title_color=COLORS["text_primary"],
            border_width=0,
            expand_x=True,
            expand_y=True,
            pad=((0, 0), (18, 0)),
        )

        main_layout = [*info_layout,
                       [tabs],
        ]

        main_view = sg.Column(main_layout,
                              key=f"{self.key}_MAIN",
                              visible=False,
                              background_color=COLORS["bg_dark"],
                              expand_x=True,
                              expand_y=True,
                              pad=(0, 0),
        )
        self.add_view("main", main_view)

    # ------------------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------------------
    def load_dataset(self, cfg, window):
        self.cfg = cfg

        tile_width = self._safe(getattr(getattr(cfg, "tile_structure", None), "width", None))
        tile_height = self._safe(getattr(getattr(cfg, "tile_structure", None), "height", None))
        tile_format = self._safe(getattr(getattr(cfg, "tile_structure", None), "tile_format", None))

        processing = getattr(cfg, "processing", None)
        normalisation = getattr(cfg, "normalisation", None)
        date_range = getattr(cfg, "date_range", None)
        crs = getattr(cfg, "crs", None)

        proc_parts = [
            f"Resolution {self._safe(getattr(processing, 'resolution', None))}",
            f"Normalisation {self._safe(getattr(normalisation, 'method', None))}",
            f"DEM {self._safe(getattr(processing, 'include_dem', None))}",
            f"Soil {self._safe(getattr(processing, 'include_soil', None))}",
            f"Indices {self._safe(getattr(processing, 'include_indices', None))}",
        ]

        self._update_info(f"{self.key}_INFO_NAME", cfg.dataset_name)
        self._update_info(f"{self.key}_INFO_STAGE", self._display_stage(cfg.stage))
        self._update_info(f"{self.key}_INFO_ROLE", self._display_role(getattr(cfg, "role", None)))
        self._update_info(f"{self.key}_INFO_AOI", f"Lat {cfg.min_lat}–{cfg.max_lat}  |  Lon {cfg.min_lon}–{cfg.max_lon}")
        self._update_info(f"{self.key}_INFO_DATE_RANGE", f"{self._safe(getattr(date_range, 'start', None))} to {self._safe(getattr(date_range, 'end', None))}")
        self._update_info(f"{self.key}_INFO_CRS", f"EPSG:{self._safe(getattr(crs, 'epsg', None))}")
        self._update_info(f"{self.key}_INFO_TILES", self._safe(cfg.tile_count))
        self._update_info(f"{self.key}_INFO_STRUCTURE", self._safe(getattr(cfg, "structure", None)))
        self._update_info(f"{self.key}_INFO_TILE_SIZE", f"{tile_width} x {tile_height}  |  {tile_format}")
        self._update_info(f"{self.key}_INFO_INPUTS", f"{self._safe(getattr(cfg, 'num_input_channels', None))} channels  |  {self._list_text(getattr(cfg, 'input_channels', []))}")
        self._update_info(f"{self.key}_INFO_MASKS", self._list_text(getattr(cfg, "mask_channels", [])))
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
        # ---------------------------------------------------------------------
        # One stable dataset page with a tab control. The stage controls which
        # actions are visible inside each tab rather than swapping entire pages.
        # ---------------------------------------------------------------------
        self.views["main"].update(visible=True)
        self.current_stage = stage

        stage_key = str(stage or "unknown").lower()
        is_processed = stage_key in ("processed", "ready", "statistics_generated")
        has_statistics = stage_key == "statistics_generated"
        can_process = stage_key in ("raw", "downloaded", "processing", "unknown")

        # ---------------------------------------------------------------------
        # Download tab status. There is currently no separate download task wired
        # from the viewer; dataset creation leaves the dataset at downloaded stage.
        # ---------------------------------------------------------------------
        if stage_key in ("raw", "downloaded"):
            self._set_text(self.txt_download_status, "Download Status: raw source data available")
            self._set_text(self.txt_download_details, "This dataset is ready to be processed into model-ready tiles.")
        elif is_processed:
            self._set_text(self.txt_download_status, "Download Status: complete")
            self._set_text(self.txt_download_details, "Raw source data has already been processed for this dataset.")
        elif stage_key == "processing":
            self._set_text(self.txt_download_status, "Download Status: complete")
            self._set_text(self.txt_download_details, "Dataset processing is currently running.")
        else:
            self._set_text(self.txt_download_status, "Download Status: unknown")
            self._set_text(self.txt_download_details, "Open the dataset configuration if the raw data stage needs to be checked.")

        # Process tab actions.
        if can_process:
            self._set_text(self.txt_process_status, "Processing Status: ready to process")
        elif is_processed:
            self._set_text(self.txt_process_status, "Processing Status: processed")
        else:
            self._set_text(self.txt_process_status, f"Processing Status: {self._display_stage(stage_key)}")

        self.btn_process.update(visible=can_process)

        # Visuals tab actions.
        if is_processed:
            self._set_text(self.txt_visuals_status, "Visual Status: processed tiles are available")
        else:
            self._set_text(self.txt_visuals_status, "Visual Status: process the dataset before creating visuals")

        self.btn_generate_stats.update(visible=is_processed and not has_statistics)
        self.btn_umap_raw.update(visible=has_statistics)
        self.btn_corr.update(visible=has_statistics)
        self.btn_hist.update(visible=has_statistics)
        self.btn_cluster_stitched.update(visible=has_statistics)

        # Stitched PNG previews are resized by ImageUtility when opened, so they
        # are safe to expose even when the underlying AOI mosaic is large.
        for button in self.colour_shift_buttons.values():
            button.update(visible=is_processed)

        tile_controls_visible = is_processed
        if hasattr(self, "tile_label"):
            self.tile_label.update(visible=tile_controls_visible)
        if hasattr(self, "tile_selector"):
            self.tile_selector.update(visible=tile_controls_visible)
        self.btn_tile_rgb.update(visible=tile_controls_visible)
        self.btn_tile_overlay.update(visible=tile_controls_visible)
        self.btn_tile_cluster.update(visible=has_statistics)

    # ------------------------------------------------------------
    # EVENT HANDLER
    # ------------------------------------------------------------
    def handle_event(self, event, values, window):
        handled = True

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

        elif event == f"{self.key}_CLUSTER_STITCHED":
            self._show_stat_image("clustering")

        elif event in {f"{self.key}_VIS_GLOBAL_{shift}" for shift in self.COLOUR_SHIFTS}:
            shift = event.rsplit("_", 1)[-1]
            self._show_colour_shift(shift)

        elif event == f"{self.key}_VIS_TILE_RGB":
            self._show_tile_rgb(values)

        elif event == f"{self.key}_VIS_TILE_OVERLAY":
            self._show_tile_overlay(values)

        elif event == f"{self.key}_VIS_TILE_CLUSTER":
            self._show_tile_cluster(values)

        elif event == f"{self.key}_TABGROUP":
            handled = True

        else:
            handled = False

        return handled

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
    # STITCHED COLOUR-SHIFT VISUALS
    # ------------------------------------------------------------
    def _colour_shift_path(self, shift: str) -> Path:
        shift = str(shift).upper()
        if shift not in self.COLOUR_SHIFT_ORDERS:
            raise ValueError(f"Unsupported colour shift: {shift}")
        return self._visuals_dir() / f"{self.cfg.dataset_name}_{shift}.png"

    def _ensure_colour_shift_preview(self, shift: str) -> Path:
        """Return an existing shift preview or derive it from RGB.png.

        Datasets processed before the additional composites were introduced may
        contain only the canonical stitched RGB preview.  A colour permutation
        does not require the raw tiles to be processed again: the display PNG can
        be produced losslessly by reordering the RGB channels.
        """
        shift = str(shift).upper()
        target_path = self._colour_shift_path(shift)

        if target_path.exists():
            return target_path

        rgb_path = self._colour_shift_path("RGB")
        if not rgb_path.exists():
            raise FileNotFoundError(
                "The stitched RGB preview is missing. Process the dataset before "
                "opening stitched colour composites.\n"
                f"Expected: {rgb_path}"
            )

        order = self.COLOUR_SHIFT_ORDERS[shift]
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

        shifted = rgb[:, :, list(order)]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(shifted, mode="RGB").save(target_path)
        print(f"[OK] Generated missing {shift} stitched preview → {target_path}")
        return target_path

    def _show_colour_shift(self, shift: str):
        try:
            preview_path = self._ensure_colour_shift_preview(shift)
            self.img_util.show_image_window(
                preview_path,
                title=f"{shift.upper()} Stitched Mosaic",
            )
        except Exception as e:
            sg.popup_error(f"Failed to load {shift.upper()} stitched image:\n{e}")

    # ------------------------------------------------------------
    # TILE-LEVEL VISUALS
    # ------------------------------------------------------------
    def _processed_root(self) -> Path:
        return Path(self.pm.resolve_path(self.cfg.processed_path))

    def _tile_dirs(self):
        root = self._processed_root()
        return sorted(
            (d for d in root.glob("tile *") if d.is_dir()),
            key=lambda d: int(d.name.removeprefix("tile ")),
        )

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

    def _show_tile_cluster(self, values):
        try:
            tile_dir = self._selected_tile_dir(values)
            cluster_path = tile_dir / "cluster_map.png"

            if not cluster_path.exists():
                raise FileNotFoundError(
                    f"Clustering image not found for {tile_dir.name}. Run Generate Statistics first."
                )

            self.img_util.show_image_window(
                cluster_path,
                title=f"Tile Clustering - {tile_dir.name}",
            )
        except Exception as e:
            sg.popup_error(f"Failed to load tile clustering:\n{e}")


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
