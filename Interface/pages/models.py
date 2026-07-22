import PySimpleGUI as sg

from Interface.theme import RText, RButton, COLORS, FONTS, BUTTON_COLORS
from Core.Managers.model_manager import ModelManager


# ============================================================
# MODELS PAGE
#
#   Model landing page with an embedded selector.
#   This replaces the old two-button page and avoids opening the
#   separate Select Model task popup from the Models page.
# ============================================================
class PageModels:
    key = "-PAGE_MODELS-"

    MAX_ROWS = 120

    filter_arch_key = "-MODELS_FILTER_ARCH-"
    filter_stage_key = "-MODELS_FILTER_STAGE-"
    list_key = "-MODELS_INLINE_LIST-"

    def __init__(self, model_manager=None):
        self.mm = model_manager or ModelManager()
        self.model_items = []

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _safe(self, value, default="-"):
        return default if value in (None, "") else value

    def _model_item_key(self, idx):
        return f"-MODELS_INLINE_SELECT_{idx}-"

    def _row_key(self, idx):
        return f"-MODELS_INLINE_ROW_{idx}-"

    def _name_key(self, idx):
        return f"-MODELS_INLINE_NAME_{idx}-"

    def _label_key(self, idx):
        return f"-MODELS_INLINE_LABEL_{idx}-"

    def _normalise_filter(self, value, all_label):
        value = value or all_label
        return None if value == all_label else str(value).strip().lower()

    def _build_model_items(self):
        self.mm.reload()
        items = []

        for name in self.mm.list_models():
            cfg = self.mm.get(name)
            if cfg is None:
                continue

            architecture = self._safe(getattr(getattr(cfg, "architecture", None), "type", None), "unknown")
            stage = self._safe(getattr(cfg, "stage", None), "unknown")
            epochs = self._safe(getattr(getattr(cfg, "training", None), "epochs", None), "?")
            device = self._safe(getattr(cfg, "device", None), "?")
            training_dataset = getattr(cfg, "training_dataset", None)

            channels = "?"
            profile = "no dataset"
            if training_dataset and hasattr(self.mm, "config_manager"):
                ds = self.mm.config_manager.get_dataset(training_dataset)
                if ds is not None:
                    channels = getattr(ds, "num_input_channels", None) or "?"
                    profile = f"derived_{channels}ch" if channels != "?" else "no dataset"

            label = f"{epochs} epochs | {device} | {profile} | {channels} ch | {stage}"
            items.append({
                "name": name,
                "label": label,
                "architecture": str(architecture),
                "stage": str(stage),
            })

        self.model_items = items
        return items

    def _filter_values(self):
        architectures = sorted({item["architecture"] for item in self.model_items if item.get("architecture")})
        stages = sorted({item["stage"] for item in self.model_items if item.get("stage")})
        return ["All Architectures", *architectures], ["All Stages", *stages]

    def _matches_filters(self, item, arch_filter, stage_filter):
        if arch_filter and item.get("architecture", "").lower() != arch_filter:
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
                    RText("Architecture", w=0.12),
                    sg.Combo(["All Architectures"], default_value="All Architectures", key=self.filter_arch_key, **combo_style),
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
                [sg.Text("", key=self._label_key(idx), font=FONTS["body"], text_color=COLORS["text_primary"], background_color=COLORS["bg_panel"], size=(36, 1))],
            ],
            background_color=COLORS["bg_panel"],
            pad=(8, 6),
            expand_x=True,
        )

        button = sg.Button(
            "Select",
            key=self._model_item_key(idx),
            font=FONTS["body"],
            button_color=BUTTON_COLORS["primary"],
            mouseover_colors=BUTTON_COLORS["primary_hover"],
            border_width=0,
            size=(8, 1),
            pad=(8, 8),
        )

        return sg.pin(
            sg.Column(
                [[info_col, button]],
                key=self._row_key(idx),
                background_color=COLORS["bg_panel"],
                visible=False,
                pad=(4, 4),
                expand_x=True,
            )
        )

    def _build_selector_panel(self):
        rows = [[self._build_list_row(idx)] for idx in range(self.MAX_ROWS)]

        list_col = sg.Column(
            rows,
            key=self.list_key,
            background_color=COLORS["bg_panel"],
            scrollable=True,
            vertical_scroll_only=True,
            size=(390, 445),
            pad=(0, 0),
            expand_y=True,
        )

        return sg.Column(
            [
                [RText("Load Model", color=COLORS["accent_highlight"], font=FONTS["header"], justification="center")],
                [list_col],
            ],
            background_color=COLORS["bg_dark"],
            element_justification="center",
            vertical_alignment="top",
            pad=((20, 0), (25, 0)),
            expand_y=True,
        )

    def build(self, window):
        self._build_model_items()

        title_row = [RText("Models", key="-MODELS_TITLE-", w=0.30)]
        separator = [sg.HorizontalSeparator(color=COLORS["line_bright"])]

        create_panel = sg.Column(
            [
                [RButton("Create New Model", key="-PAGE_CREATE_MODEL_CONF-", w=0.16)],
            ],
            background_color=COLORS["bg_dark"],
            vertical_alignment="top",
            pad=((45, 40), (35, 0)),
        )

        content_row = [
            create_panel,
            self._build_filter_panel(),
            sg.VSeparator(color=COLORS["line_bright"]),
            self._build_selector_panel(),
        ]

        layout = [
            title_row,
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
        self._build_model_items()
        arch_values, stage_values = self._filter_values()

        if window:
            try:
                window[self.filter_arch_key].update(values=arch_values)
                if window[self.filter_arch_key].get() not in arch_values:
                    window[self.filter_arch_key].update(value="All Architectures")

                window[self.filter_stage_key].update(values=stage_values)
                if window[self.filter_stage_key].get() not in stage_values:
                    window[self.filter_stage_key].update(value="All Stages")
            except Exception:
                pass

            self.apply_filters(window, {})

    def apply_filters(self, window, values):
        if not window:
            return

        arch_value = values.get(self.filter_arch_key)
        stage_value = values.get(self.filter_stage_key)

        try:
            arch_value = arch_value if arch_value is not None else window[self.filter_arch_key].get()
            stage_value = stage_value if stage_value is not None else window[self.filter_stage_key].get()
        except Exception:
            arch_value = "All Architectures"
            stage_value = "All Stages"

        arch_filter = self._normalise_filter(arch_value, "All Architectures")
        stage_filter = self._normalise_filter(stage_value, "All Stages")

        visible_idx = 0
        for idx, item in enumerate(self.model_items[:self.MAX_ROWS]):
            is_visible = self._matches_filters(item, arch_filter, stage_filter)
            try:
                window[self._name_key(idx)].update(item["name"])
                window[self._label_key(idx)].update(item["label"])
                window[self._row_key(idx)].update(visible=is_visible)
            except Exception:
                pass
            if is_visible:
                visible_idx += 1

        for idx in range(len(self.model_items), self.MAX_ROWS):
            try:
                window[self._row_key(idx)].update(visible=False)
            except Exception:
                pass

        try:
            window.refresh()
        except Exception:
            pass

    def is_filter_event(self, event):
        return event in {self.filter_arch_key, self.filter_stage_key}

    def selected_model_from_event(self, event):
        if not isinstance(event, str):
            return None

        prefix = "-MODELS_INLINE_SELECT_"
        if not event.startswith(prefix) or not event.endswith("-"):
            return None

        try:
            idx = int(event[len(prefix):-1])
        except ValueError:
            return None

        if idx < 0 or idx >= len(self.model_items):
            return None

        return self.model_items[idx]["name"]

    # ------------------------------------------------------------
    # WORKER MESSAGE HANDLER
    # ------------------------------------------------------------
    def on_worker_message(self, task_id, msg_type, data):
        match msg_type:
            case "status":
                print(f"[Models] {data}")
            case "progress":
                print(f"[Models] Progress: {data}")
            case "result":
                print(f"[Models] Task result: {data}")
            case "error":
                sg.popup_error("Model Task Error", data)
            case "finished":
                print(f"Task {task_id} finished")
