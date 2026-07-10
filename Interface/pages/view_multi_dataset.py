
import PySimpleGUI as sg

from pathlib import Path

from Interface.theme import COLORS, RText, RButton, RHText
from Core.Utils.image_utility import ImageUtility
from Core.Pytorch.pytorch_dataset_factory import PyTorchDatasetFactory
from Core.Managers.config_manager import ConfigManager
from Core.Managers.path_manager import PathManager


# ============================================================
# VIEW MULTI DATASET
#       Viewer for Multi‑Dataset Configs.
#
#    Shows:
#        • Multi‑info panel (top)
#        • Tabs for each dataset in the multi config
#        • Dataset info panel (reused conceptually from PageViewerDataset)
#        • Visualisation buttons (RGB, GBR, Anomaly, Cleaned, Clustering)
#      
# ============================================================
class PageViewerMultiDataset:

    key = "-PAGE_VIEWER_MULTI_DATASET-"

    def __init__(self):

        self.cfg = None                     # MultiDatasetConfig
        self.current_dataset_cfg = None     # DatasetConfig
        self.dataset_tabs = []

        # UI elements
        self.txt_multi_name = None
        self.txt_multi_tiles = None
        self.btn_view_aoi = None

        self.txt_name = None
        self.txt_latlon = None
        self.txt_depth_tiles = None
        self.band_col = None
        self.tabgroup = None

        self.img_util = ImageUtility()
        self.factory = PyTorchDatasetFactory()

        # Unified config loader (datasets/models/searches)
        self.cm = ConfigManager(PathManager())

    # ------------------------------------------------------------
    # BUILD PAGE
    # Build the static layout. Dynamic content is filled in load_multi_dataset().
    # ------------------------------------------------------------
    def build(self):
        # -------------------------
        # Multi‑info panel (top)
        # -------------------------
        self.txt_multi_name = RText("Name: -")
        self.txt_multi_tiles = RText("Total Tiles: -")
        self.btn_view_aoi = RButton("View AOI", key=f"{self.key}_VIEW_AOI")

        multi_info_panel = sg.Column(
            [
                [RHText("Multi‑Dataset Information")],
                [self.txt_multi_name, sg.Push(), self.btn_view_aoi],
                [self.txt_multi_tiles],
            ],
            key=f"{self.key}_MULTI_INFO",
            background_color=COLORS["bg_dark"],
            pad=(5, 5),
            expand_x=True,
        )

        # -------------------------
        # Tab group (empty until load)
        # -------------------------
        self.tabgroup = sg.TabGroup(
            [[sg.Push()]],
            key=f"{self.key}_TABGROUP",
            enable_events=True,
            background_color=COLORS["bg_dark"],
            tab_location='top',
            border_width=3,
            tab_border_width=3,
        )

        # -------------------------
        # Final layout
        # -------------------------
        layout = sg.Column(
            [
                [multi_info_panel],
                [self.tabgroup],
            ],
            key=self.key,
            background_color=COLORS["bg_panel"],
            visible=False,
            expand_x=True,
            expand_y=True,
            pad=(5, 5),
        )

        return layout



    # ------------------------------------------------------------
    # LOAD MULTI‑DATASET CONFIG
    # ------------------------------------------------------------
    def load_multi_dataset(self, cfg, window):
        """
        Called by mainW when user selects a multi‑dataset config.
        cfg is a MultiDatasetConfig instance.
        """
        self.cfg = cfg

        # -------------------------
        # Update multi‑info panel
        # -------------------------
        self.txt_multi_name.update(f"Name: {self.cfg.multi_name}")
        self.txt_multi_tiles.update(f"Total Tiles: {self.cfg.total_tiles}")

        # AOI button enabled only when stage == aoi_generated
        self.btn_view_aoi.update(disabled=(self.cfg.stage != "aoi_generated"))

        # -------------------------
        # Build dataset tabs
        # -------------------------
        tabs = []

        for entry in self.cfg.dataset_entries:
            ds_name = entry.get("name")
            if not ds_name:
                continue

            # Build dataset info panel for this tab
            ds_cfg = self.cm.get_dataset(ds_name)
            if not ds_cfg:
                continue

            info_panel = sg.Column([
                [RHText("Dataset Information")],
                [RText(f"Name: {ds_cfg.dataset_name}")],
                [RText(f"Lat: {ds_cfg.min_lat}–{ds_cfg.max_lat}  Lon: {ds_cfg.min_lon}–{ds_cfg.max_lon}")],
                [RText(f"Depth: {ds_cfg.depth}  Tiles: {ds_cfg.tile_count}")],
                [RText("Bands: " + ", ".join(getattr(ds_cfg.bands, "included", [])))],
            ],
            background_color=COLORS["bg_dark"],
            pad=(5, 5),
            expand_x=True)

            visuals_panel = sg.Column([
                [RHText("Visualisations")],
                [
                    RButton("RGB", key=f"{self.key}_RGB_{ds_name}"),
                    RButton("GBR", key=f"{self.key}_GBR_{ds_name}"),
                    RButton("Anomaly Map", key=f"{self.key}_ANOMALY_{ds_name}"),
                    RButton("Cleaned", key=f"{self.key}_CLEANED_{ds_name}"),
                    RButton("Clustering", key=f"{self.key}_CLUSTER_{ds_name}"),
                ],
            ],
            background_color=COLORS["bg_dark"],
            pad=(5, 5),
            expand_x=True)

            # Combine both panels into the tab layout
            tab_layout = [[info_panel], [visuals_panel]]
            
            tabs.append(sg.Tab(ds_name,
                               tab_layout,
                               key=f"{self.key}_TAB_{ds_name}",
                               background_color=COLORS["bg_dark"],
                        )
            )

        # -------------------------
        # Update the TabGroup correctly
        # -------------------------
        try:
            _ = self.tabgroup.Widget
        except Exception:
            window.finalize()

        if hasattr(self, 'previous_tab_keys'):

            for old_key in self.previous_tab_keys:

                if old_key in window.AllKeysDict:
                    window[old_key].update(visible=False)

        self.previous_tab_keys = [tab.Key for tab in tabs]

        for tab_element in tabs:
            self.tabgroup.add_tab(tab_element)

        window.refresh()


        # -------------------------
        # Auto‑select first dataset
        # -------------------------
        if self.cfg.dataset_entries:

            first_name = self.cfg.dataset_entries[0].get("name")

            if first_name:

                # Use PySimpleGUI native tab selection via its key
                first_tab_key = f"{self.key}_TAB_{first_name}"
                
                if first_tab_key in window.AllKeysDict:
                    window[first_tab_key].select()
                
                self._load_dataset_by_name(first_name, window)



    # ------------------------------------------------------------
    # LOAD INDIVIDUAL DATASET INTO INFO PANEL
    # ------------------------------------------------------------
    def _load_dataset_by_name(self, ds_name: str, window):
        """
        Resolve a DatasetConfig by name via ConfigManager and refresh the selected tab layout.
        """
        ds_cfg = self.cm.get_dataset(ds_name)
        
        if not ds_cfg:
            sg.popup_error(f"Dataset config '{ds_name}' not found.")
            return

        self.current_dataset_cfg = ds_cfg

        # Build new info panel for the selected dataset
        info_panel = sg.Column([
            [RHText("Dataset Information")],
            [RText(f"Name: {ds_cfg.dataset_name}")],
            [RText(f"Lat: {ds_cfg.min_lat}–{ds_cfg.max_lat}  Lon: {ds_cfg.min_lon}–{ds_cfg.max_lon}")],
            [RText(f"Depth: {ds_cfg.depth}  Tiles: {ds_cfg.tile_count}")],
            [RText("Bands: " + ", ".join(getattr(ds_cfg.bands, "included", [])))],
        ],
        background_color=COLORS["bg_dark"],
        pad=(5, 5),
        expand_x=True)

        visuals_panel = sg.Column([
            [RHText("Visualisations")],
            [
                RButton("RGB", key=f"{self.key}_RGB_{ds_name}"),
                RButton("GBR", key=f"{self.key}_GBR_{ds_name}"),
                RButton("Anomaly Map", key=f"{self.key}_ANOMALY_{ds_name}"),
                RButton("Cleaned", key=f"{self.key}_CLEANED_{ds_name}"),
                RButton("Clustering", key=f"{self.key}_CLUSTER_{ds_name}"),
            ],
        ],
        background_color=COLORS["bg_dark"],
        pad=(5, 5),
        expand_x=True)

        # Replace the tab’s layout
        tab_key = f"{self.key}_TAB_{ds_name}"
        if tab_key in window.AllKeysDict:
            window[tab_key].update([[info_panel], [visuals_panel]])
            window.refresh()

    # ------------------------------------------------------------
    # EVENT HANDLER
    # ------------------------------------------------------------
    def handle_event(self, event, values, window):

        # Tab changed
        if event == f"{self.key}_TABGROUP":
            selected = values[event]

            if isinstance(selected, str) and selected.startswith(f"{self.key}_TAB_"):
                ds_name = selected.replace(f"{self.key}_TAB_", "")
                self._load_dataset_by_name(ds_name, window)

        # AOI view
        elif event == f"{self.key}_VIEW_AOI":
            if self.cfg.stage != "aoi_generated":
                sg.popup("AOI not generated yet.")
                return

            aoi_path = self.cfg.paths.visuals_dir / f"{self.cfg.multi_name}_AOI.png"
            if aoi_path.exists():
                self.img_util.show_image_window(aoi_path)
            else:
                sg.popup_error("AOI image missing.")

        # Visualisations
        elif event == f"{self.key}_RGB":
            print("[MultiViewer] RGB clicked")

        elif event == f"{self.key}_GBR":
            print("[MultiViewer] GBR clicked")

        elif event == f"{self.key}_ANOMALY":
            print("[MultiViewer] Anomaly clicked")

        elif event == f"{self.key}_CLEANED":
            print("[MultiViewer] Cleaned clicked")

        elif event == f"{self.key}_CLUSTER":
            print("[MultiViewer] Clustering clicked")

        # Processing (if you decide to support per‑dataset processing from here)
        elif event == f"{self.key}_PROCESS":
            if self.current_dataset_cfg:
                window.write_event_value(
                    "-TASK_PROCESS_DATASET-",
                    self.current_dataset_cfg.dataset_name,
                )

    # ------------------------------------------------------------
    # WORKER MESSAGE HANDLER
    # ------------------------------------------------------------
    def on_worker_message(self, task_id, msg_type, data):
        """
        For future use (e.g., AOI generation tasks).
        """
        pass
