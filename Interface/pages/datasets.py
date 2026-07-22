import PySimpleGUI as sg

from Interface.theme import RText, RButton, COLORS, FONTS, BUTTON_COLORS
from Core.Managers.dataset_manager import DatasetManager


# ============================================================
# DATASETS PAGE
#
#   Dataset landing page with an embedded selector.
#   This replaces the old New/Load two-button page and avoids
#   opening the separate Select Dataset task popup from here.
# ============================================================
class PageDatasets:
    key = "-PAGE_DATASETS-"

    MAX_ROWS = 120

    filter_role_key = "-DATASETS_FILTER_ROLE-"
    filter_stage_key = "-DATASETS_FILTER_STAGE-"
    list_key = "-DATASETS_INLINE_LIST-"

    def __init__(self, dataset_manager=None):
        self.dm = dataset_manager or DatasetManager()
        self.dataset_items = []

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _safe(self, value, default="-"):
        return default if value in (None, "") else value

    def _dataset_item_key(self, idx):
        return f"-DATASETS_INLINE_SELECT_{idx}-"

    def _row_key(self, idx):
        return f"-DATASETS_INLINE_ROW_{idx}-"

    def _name_key(self, idx):
        return f"-DATASETS_INLINE_NAME_{idx}-"

    def _label_key(self, idx):
        return f"-DATASETS_INLINE_LABEL_{idx}-"

    def _normalise_filter(self, value, all_label):
        value = value or all_label
        return None if value == all_label else str(value).strip().lower()

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

    def _build_dataset_items(self):
        self.dm.reload()
        items = []

        for opt in self.dm.list_dataset_options(role="all"):
            name = opt.get("key") or opt.get("name")
            if not name:
                continue

            tile_count = self._safe(opt.get("tile_count"), "?")
            stage = self._safe(opt.get("stage"), "unknown")
            role = self._safe(opt.get("role"), "mixed")
            display_role = self._display_role(role)
            structure = self._safe(opt.get("structure"), "aoi_grid")
            channels = self._safe(opt.get("num_input_channels"), "?")

            label = f"{tile_count} tiles | {stage} | {display_role} | {structure} | {channels} model ch"
            items.append({
                          "name": name,
                          "label": label,
                          "stage": str(stage),
                          "role": str(display_role),
            })

        self.dataset_items = items
        return items

    def _filter_values(self):
        roles = sorted({item["role"] for item in self.dataset_items if item.get("role")})
        stages = sorted({item["stage"] for item in self.dataset_items if item.get("stage")})
        return ["All Roles", *roles], ["All Stages", *stages]

    def _matches_filters(self, item, role_filter, stage_filter):
        if role_filter and item.get("role", "").lower() != role_filter:
            return False
        if stage_filter and item.get("stage", "").lower() != stage_filter:
            return False
        return True

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
    def _build_filter_panel(self):
        combo_style = {
                        "font": FONTS["body"],
                        "background_color": COLORS["bg_panel"],
                        "text_color": COLORS["text_primary"],
                        "button_background_color": COLORS["accent_primary"],
                        "button_arrow_color": COLORS["text_on_accent"],
                        "readonly": True,
                        "enable_events": True,
                        "size": (24, 1),
        }

        return sg.Column(
                        [
                            [RText("Filter", font=FONTS["header"], justification="center")],
                            [
                                RText("Role", w=0.12),
                                sg.Combo(["All Roles"], default_value="All Roles", key=self.filter_role_key, **combo_style),
                            ],
                            [
                                RText("Stage", w=0.12),
                                sg.Combo(["All Stages"], default_value="All Stages", key=self.filter_stage_key, **combo_style),
                            ],
                        ],
                        background_color=COLORS["bg_dark"],
                        element_justification="center",
                        vertical_alignment="top",
                        pad=((40, 30), (35, 0)),
                        expand_y=True,
        )

    def _build_list_row(self, idx):
        info_col = sg.Column(
            [
                [sg.Text("", key=self._name_key(idx), font=FONTS["body"], text_color=COLORS["text_primary"], background_color=COLORS["bg_panel"], size=(28, 1))],
                [sg.Text("", key=self._label_key(idx), font=FONTS["body"], text_color=COLORS["text_primary"], background_color=COLORS["bg_panel"], size=(38, 1))],
            ],
            background_color=COLORS["bg_panel"],
            pad=(8, 6),
            expand_x=True,
        )

        button = sg.Button("Select",
                           key=self._dataset_item_key(idx),
                           font=FONTS["body"],
                           button_color=BUTTON_COLORS["primary"],
                           mouseover_colors=BUTTON_COLORS["primary_hover"],
                           border_width=0,
                           size=(8, 1),
                           pad=(8, 8),
        )

        return sg.pin(
            sg.Column([[info_col, button]],
                       key=self._row_key(idx),
                       background_color=COLORS["bg_panel"],
                       visible=False,
                       pad=(4, 4),
                       expand_x=True,
            )
        )

    def _build_selector_panel(self):
        rows = [[self._build_list_row(idx)] for idx in range(self.MAX_ROWS)]

        list_col = sg.Column(rows,
                             key=self.list_key,
                             background_color=COLORS["bg_panel"],
                             scrollable=True,
                             vertical_scroll_only=True,
                             size=(410, 500),
                             pad=(0, 0),
                             expand_y=True,
        )

        return sg.Column([
                                [RText("Load Dataset", color=COLORS["accent_highlight"], font=FONTS["header"], justification="center")],
                                [list_col],
                         ],
                         background_color=COLORS["bg_dark"],
                         element_justification="center",
                         vertical_alignment="top",
                         pad=((20, 0), (25, 0)),
                         expand_y=True,
        )

    def build(self, window):
        self._build_dataset_items()

        title_row = [RText("Datasets", key="-DATASETS_TITLE-", w=0.30)]
        separator = [sg.HorizontalSeparator(color=COLORS["line_bright"])]

        create_panel = sg.Column([
                                    [RButton("New Dataset", key="-PAGE_CREATE_DATASET_CONF-", w=0.16)],
                                 ],
                                 background_color=COLORS["bg_dark"],
                                 vertical_alignment="top",
                                 pad=((45, 40), (35, 0)),
        )

        content_row = [create_panel,
                       self._build_filter_panel(),
                       sg.VSeparator(color=COLORS["line_bright"]),
                       self._build_selector_panel(),
                       sg.Push(),
        ]

        layout = [title_row,
                  separator,
                  content_row,
                  [sg.HorizontalSeparator(color=COLORS["line_bright"])],
        ]

        return sg.Column(layout,
                         key=self.key,
                         expand_x=True,
                         expand_y=True,
                         background_color=COLORS["bg_dark"],
                         visible=False,
        )

    # ------------------------------------------------------------
    # Event helpers used by mainW.py
    # ------------------------------------------------------------
    def refresh(self, window):
        self._build_dataset_items()
        role_values, stage_values = self._filter_values()

        if window:
            try:
                window[self.filter_role_key].update(values=role_values)
                if window[self.filter_role_key].get() not in role_values:
                    window[self.filter_role_key].update(value="All Roles")

                window[self.filter_stage_key].update(values=stage_values)
                if window[self.filter_stage_key].get() not in stage_values:
                    window[self.filter_stage_key].update(value="All Stages")
            except Exception:
                pass

            self.apply_filters(window, {})

    def apply_filters(self, window, values):
        if not window:
            return

        role_value = values.get(self.filter_role_key)
        stage_value = values.get(self.filter_stage_key)

        try:
            role_value = role_value if role_value is not None else window[self.filter_role_key].get()
            stage_value = stage_value if stage_value is not None else window[self.filter_stage_key].get()

        except Exception:
            role_value = "All Roles"
            stage_value = "All Stages"

        role_filter = self._normalise_filter(role_value, "All Roles")
        stage_filter = self._normalise_filter(stage_value, "All Stages")

        for idx, item in enumerate(self.dataset_items[:self.MAX_ROWS]):
            is_visible = self._matches_filters(item, role_filter, stage_filter)

            try:
                window[self._name_key(idx)].update(item["name"])
                window[self._label_key(idx)].update(item["label"])
                window[self._row_key(idx)].update(visible=is_visible)

            except Exception:
                pass

        for idx in range(len(self.dataset_items), self.MAX_ROWS):

            try:
                window[self._row_key(idx)].update(visible=False)

            except Exception:
                pass

        try:
            window.refresh()
        except Exception:
            pass

    def is_filter_event(self, event):
        return event in {self.filter_role_key, self.filter_stage_key}

    def selected_dataset_from_event(self, event):
        if not isinstance(event, str):
            return None

        prefix = "-DATASETS_INLINE_SELECT_"
        if not event.startswith(prefix) or not event.endswith("-"):
            return None

        try:
            idx = int(event[len(prefix):-1])
        except ValueError:
            return None

        if idx < 0 or idx >= len(self.dataset_items):
            return None

        return self.dataset_items[idx]["name"]

    # ------------------------------------------------------------
    # WORKER MESSAGE HANDLER (not used yet)
    # ------------------------------------------------------------
    def on_worker_message(self, task_id, msg_type, data):
        pass
