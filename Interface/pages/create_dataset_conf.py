import PySimpleGUI as sg

from Interface.theme import (
    RPanel,
    RButton,
    RText,
    COLORS,
    RCalendarButton,
    RHText,
)


# ============================================================
# Sentinel-2 / derived band presets
# ============================================================
# These are the bands Terra-AId currently handles in the dataset
# creation flow. The keys are the actual channel codes written to
# configs, not display names.
# ------------------------------------------------------------------

BAND_LABELS = {
    "B1": "B1 Coastal Aerosol",
    "B2": "B2 Blue",
    "B3": "B3 Green",
    "B4": "B4 Red",
    "B5": "B5 Red Edge 1",
    "B6": "B6 Red Edge 2",
    "B7": "B7 Red Edge 3",
    "B8": "B8 NIR",
    "B8A": "B8A Narrow NIR",
    "B9": "B9 Water Vapour",
    "B10": "B10 Cirrus",
    "B11": "B11 SWIR 1",
    "B12": "B12 SWIR 2",
    "SCL": "SCL Scene Class",
    "VV": "VV Sentinel-1",
    "VH": "VH Sentinel-1",
    "NDVI": "NDVI",
    "BSI": "BSI",
}

# ------------------------------------------------------------------
# The default stack is the one you have been standardising around for
# Nile/ground-truth compatibility.
# ------------------------------------------------------------------
DEFAULT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "SCL", "NDVI", "BSI"]


# Full selectable list. Keep this in the same order as app_config where possible.
ALL_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12",
    "SCL", "VV", "VH", "NDVI", "BSI",
]

BAND_MODES = ["Default bands", "All bands", "Custom"]


# ============================================================
# CREATE DATASET CONFIG
# ============================================================
class PageCreateDatasetConf:
    key = "-PAGE_CREATE_DATASET_CONF-"

    def __init__(self):
        self.default_bands = list(DEFAULT_BANDS)
        self.all_bands = list(ALL_BANDS)

    # ------------------------------------------------------------
    # Build page layout
    # ------------------------------------------------------------
    def build(self, window):

        # ============================================================
        # MAP / AOI
        # ============================================================
        map_panel = RPanel(
            key="-CDC_MAP_PANEL-",
            layout=[
                [RButton("Select AOI", key="-CDC_AOI_TASK-"), sg.Push()],
                [
                    RText("Min Lat:"),
                    sg.Input(
                        key="-CDC_MIN_LAT-",
                        size=(8, 1),
                        justification="center",
                        background_color=COLORS["bg_panel"],
                        text_color=COLORS["text_primary"],
                    ),
                    RText("Max Lat:"),
                    sg.Input(
                        key="-CDC_MAX_LAT-",
                        size=(8, 1),
                        justification="center",
                        background_color=COLORS["bg_panel"],
                        text_color=COLORS["text_primary"],
                    ),
                    RText("Min Lon:"),
                    sg.Input(
                        key="-CDC_MIN_LON-",
                        size=(8, 1),
                        justification="center",
                        background_color=COLORS["bg_panel"],
                        text_color=COLORS["text_primary"],
                    ),
                    RText("Max Lon:"),
                    sg.Input(
                        key="-CDC_MAX_LON-",
                        size=(8, 1),
                        justification="left",
                        background_color=COLORS["bg_panel"],
                        text_color=COLORS["text_primary"],
                    ),
                    sg.Push(),
                    RText("Num Tiles:"),
                    sg.Input(
                        key="-CDC_NUM_TILES-",
                        size=(8, 1),
                        justification="left",
                        background_color=COLORS["bg_panel"],
                        text_color=COLORS["text_primary"],
                    ),
                ],
            ],
        )

        # ============================================================
        # NAME
        # ============================================================
        name_panel = RPanel(
            key="-CDC_NAME_PANEL-",
            layout=[
                [
                    RText("Dataset Name"),
                    sg.Input(
                        key="-CDC_NAME-",
                        size=(30, 1),
                        background_color=COLORS["bg_panel"],
                        text_color=COLORS["text_primary"],
                    ),
                ],
            ],
        )


        # ============================================================
        # DATASET ROLE
        # ============================================================
        role_panel = RPanel(
            key="-CDC_ROLE_PANEL-",
            layout=[
                [
                    RText("Dataset Role"),
                    sg.Combo(
                        ["Training", "Evaluation", "Prediction"],
                        default_value="Training",
                        key="-CDC_ROLE-",
                        readonly=True,
                        size=(14, 1),
                    ),
                    RText("Training=train model | Evaluation=test/calibrate | Prediction=find anomalies",
                          key="-CDC_ROLE_HELP-"),
                ],
            ],
        )

        # ============================================================
        # SOIL + DEM OPTIONS
        # ============================================================
        # radio buttons because these are yes/no mutually-exclusive choices.
        soil_panel = RPanel(
            key="-CDC_SOIL_PANEL-",
            layout=[
                [
                    RText("SOIL Data"),
                    sg.Radio("Yes", "CDC_SOIL", key="-CDC_SOIL_Y-", default=True,
                             background_color=COLORS["bg_dark"], text_color=COLORS["text_primary"]),
                    sg.Radio("No", "CDC_SOIL", key="-CDC_SOIL_N-",
                             background_color=COLORS["bg_dark"], text_color=COLORS["text_primary"]),
                ],
            ],
        )

        dem_panel = RPanel(
            key="-CDC_DEM_PANEL-",
            layout=[
                [
                    RText("DEM Data"),
                    sg.Radio("Yes", "CDC_DEM", key="-CDC_DEM_Y-", default=True,
                             background_color=COLORS["bg_dark"], text_color=COLORS["text_primary"]),
                    sg.Radio("No", "CDC_DEM", key="-CDC_DEM_N-",
                             background_color=COLORS["bg_dark"], text_color=COLORS["text_primary"]),
                ],
            ],
        )

        tiles_panel = RPanel(key="-CDC_TILES_PANEL-", layout=[[RText("Select Tiles")]])

        # ============================================================
        # DATE RANGE
        # ============================================================
        min_date_panel = RPanel(
            key="-CDC_MIN_DATE_PANEL-",
            layout=[
                [
                    RText("Start Date"),
                    sg.Input(key="-CDC_DATE_START-", default_text="YYYY-MM-DD", size=(30, 1)),
                    RCalendarButton("Pick", target_key="-CDC_DATE_START-", format="%Y-%m-%d"),
                ],
            ],
        )

        max_date_panel = RPanel(
            key="-CDC_MAX_DATE_PANEL-",
            layout=[
                [
                    RText("End Date"),
                    sg.Input(key="-CDC_DATE_END-", default_text="YYYY-MM-DD", size=(30, 1)),
                    RCalendarButton("Pick", target_key="-CDC_DATE_END-", format="%Y-%m-%d"),
                ],
            ],
        )

        # ============================================================
        # RESOLUTION
        # ============================================================
        resolution_panel = RPanel(
            key="-CDC_RESOLUTION_PANEL-",
            layout=[
                [
                    RText("Resolution"),
                    sg.Combo(
                        ["10m", "20m", "60m"],
                        default_value="10m",
                        key="-CDC_RESOLUTION-",
                        readonly=True,
                        size=(10, 1),
                    ),
                ],
            ],
        )

        # ============================================================
        # BAND MODE + CUSTOM CHECKBOXES
        # ============================================================
        bands_mode_panel = RPanel(
            key="-CDC_BANDS_MODE_PANEL-",
            layout=[
                [
                    RText("Band Preset"),
                    sg.Combo(
                        BAND_MODES,
                        default_value="Default bands",
                        key="-CDC_BAND_MODE-",
                        readonly=True,
                        enable_events=True,
                        size=(16, 1),
                    ),
                    RText("Default: B2 B3 B4 B8 B11 B12 SCL NDVI BSI", key="-CDC_BAND_MODE_HELP-"),
                ],
            ],
        )

        custom_band_rows = self._build_band_checkbox_rows()
        custom_bands_panel = sg.Column(
            custom_band_rows,
            key="-CDC_CUSTOM_BANDS_PANEL-",
            visible=False,
            background_color=COLORS["bg_dark"],
            pad=(0, 0),
        )

        bands_panel = RPanel(
            key="-CDC_BANDS_PANEL-",
            layout=[
                [RText("Bands")],
                [bands_mode_panel],
                [custom_bands_panel],
            ],
        )

        # ============================================================
        # ACTION BUTTONS
        # ============================================================
        actions_panel = RPanel(
            key="-CDC_ACTIONS_PANEL-",
            layout=[
                [
                    RButton("Cancel", key="-PAGE_DATASETS-"),
                    RButton("Process", key="-TASK_CREATE_DATASET-"),
                ],
            ],
        )

        # ============================================================
        # PAGE LAYOUT
        # ============================================================
        layout = [
            [RHText("Dataset Configuration")],
            [sg.HorizontalSeparator(color=COLORS["line_bright"])],
            [map_panel],
            [name_panel, role_panel, sg.Push()],
            [tiles_panel, soil_panel, dem_panel, sg.Push()],
            [min_date_panel, max_date_panel, sg.Push()],
            [resolution_panel, bands_panel, sg.Push()],
            [sg.Push(), actions_panel],
        ]

        return sg.Column(
            layout,
            key=self.key,
            visible=False,
            expand_x=True,
            expand_y=True,
            background_color=COLORS["bg_dark"],
        )

    # ------------------------------------------------------------
    # Build custom checkbox rows
    # ------------------------------------------------------------
    def _build_band_checkbox_rows(self):
        rows = []
        row = []

        for idx, band in enumerate(self.all_bands, start=1):
            row.append(
                sg.Checkbox(
                    BAND_LABELS.get(band, band),
                    key=f"-CDC_BAND_{band}-",
                    default=band in self.default_bands,
                    background_color=COLORS["bg_dark"],
                    text_color=COLORS["text_primary"],
                    font=("Segoe UI", 10),
                    pad=(4, 2),
                )
            )

            if idx % 4 == 0:
                rows.append(row)
                row = []

        if row:
            rows.append(row)

        rows.append(
            [
                RButton("Use Default", key="-CDC_BANDS_SET_DEFAULT-"),
                RButton("Select All", key="-CDC_BANDS_SELECT_ALL-"),
                RButton("Clear", key="-CDC_BANDS_CLEAR-"),
            ]
        )

        return rows

    # ------------------------------------------------------------
    # PAGE EVENT HANDLER
    # ------------------------------------------------------------
    def handle_event(self, event, values, window):

        if event == "-CDC_AOI_TASK-":
            self.open_aoi_selector(values, window)
            return True

        if event == "-CDC_BAND_MODE-":
            self.apply_band_mode(values.get("-CDC_BAND_MODE-", "Default bands"), window)
            return True

        if event == "-CDC_BANDS_SET_DEFAULT-":
            window["-CDC_BAND_MODE-"].update("Custom")
            self._set_band_checks(window, self.default_bands)
            self.apply_band_mode("Custom", window)
            return True

        if event == "-CDC_BANDS_SELECT_ALL-":
            window["-CDC_BAND_MODE-"].update("Custom")
            self._set_band_checks(window, self.all_bands)
            self.apply_band_mode("Custom", window)
            return True

        if event == "-CDC_BANDS_CLEAR-":
            window["-CDC_BAND_MODE-"].update("Custom")
            self._set_band_checks(window, [])
            self.apply_band_mode("Custom", window)
            return True

        return False

    # ------------------------------------------------------------
    # Band mode UI behaviour
    # ------------------------------------------------------------
    def apply_band_mode(self, mode, window):
        mode = mode or "Default bands"

        if mode == "Default bands":
            self._set_band_checks(window, self.default_bands)
            window["-CDC_CUSTOM_BANDS_PANEL-"].update(visible=False)
            window["-CDC_BAND_MODE_HELP-"].update("Default: B2 B3 B4 B8 B11 B12 SCL NDVI BSI")

        elif mode == "All bands":
            self._set_band_checks(window, self.all_bands)
            window["-CDC_CUSTOM_BANDS_PANEL-"].update(visible=False)
            window["-CDC_BAND_MODE_HELP-"].update("All configured bands will be included")

        else:
            window["-CDC_CUSTOM_BANDS_PANEL-"].update(visible=True)
            window["-CDC_BAND_MODE_HELP-"].update("Select only the channels you want in this dataset")

        window.refresh()

    def _set_band_checks(self, window, selected_bands):
        selected = set(selected_bands or [])

        for band in self.all_bands:
            key = f"-CDC_BAND_{band}-"
            if key in window.AllKeysDict:
                window[key].update(value=band in selected)

    # ------------------------------------------------------------
    # Open AOI map selector and update the page fields.
    # ------------------------------------------------------------
    def open_aoi_selector(self, values, window):

        from Interface.pages.create_dataset_map import PageCreateDatasetMap

        selected_aoi = PageCreateDatasetMap().open(
            parent_window=window,
            initial_values=values,
        )

        if selected_aoi:
            west, south, east, north = selected_aoi["bbox"]
            window["-CDC_MIN_LAT-"].update(f"{south:.6f}")
            window["-CDC_MAX_LAT-"].update(f"{north:.6f}")
            window["-CDC_MIN_LON-"].update(f"{west:.6f}")
            window["-CDC_MAX_LON-"].update(f"{east:.6f}")
            window["-CDC_NUM_TILES-"].update("1")

    # ------------------------------------------------------------
    # WORKER MESSAGE HANDLER
    # ------------------------------------------------------------
    def on_worker_message(self, task_id, msg_type, data):
        print(f"[DatasetConf] {msg_type}: {data}")
