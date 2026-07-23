import PySimpleGUI as sg

from Interface.theme import (
    RText,
    RButton,
    RScrollableSelector,
    COLORS,
    FONTS,
    BUTTON_COLORS,
)
from Core.Managers.model_manager import ModelManager


# ============================================================
# MODELS PAGE
#
#   Model landing page with an embedded selector. The filter is
#   stacked above the model list to give each result row enough
#   width for its details and Select action.
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

            architecture = self._safe(
                getattr(getattr(cfg, "architecture", None), "type", None),
                "unknown",
            )
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

            label = (
                f"{epochs} epochs  |  {device}  |  {profile}  |  "
                f"{channels} ch  |  {stage}"
            )
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
    def _combo_style(self):
        return {
            "font": FONTS["body"],
            "background_color": COLORS["bg_panel"],
            "text_color": COLORS["text_primary"],
            "button_background_color": COLORS["accent_primary"],
            "button_arrow_color": COLORS["text_on_accent"],
            "readonly": True,
            "enable_events": True,
            "size": (20, 1),
        }

    def _build_filter_panel(self):
        combo_style = self._combo_style()

        return sg.Column(
            [
                [
                    RText(
                        "Filter models",
                        font=FONTS["body_bold"],
                        color=COLORS["accent_highlight"],
                        bg=COLORS["bg_surface"],
                    )
                ],
                [
                    RText("Architecture", color=COLORS["text_secondary"], bg=COLORS["bg_surface"]),
                    sg.Combo(
                        ["All Architectures"],
                        default_value="All Architectures",
                        key=self.filter_arch_key,
                        **combo_style,
                    ),
                    RText("Stage", color=COLORS["text_secondary"], bg=COLORS["bg_surface"]),
                    sg.Combo(
                        ["All Stages"],
                        default_value="All Stages",
                        key=self.filter_stage_key,
                        **combo_style,
                    ),
                ],
            ],
            background_color=COLORS["bg_surface"],
            pad=((0, 0), (8, 10)),
            expand_x=True,
            element_justification="left",
        )

    def _build_list_row(self, idx):
        row_bg = COLORS["bg_surface"] if idx % 2 == 0 else COLORS["bg_panel"]

        info_col = sg.Column(
            [
                [
                    sg.Text(
                        "",
                        key=self._name_key(idx),
                        font=FONTS["body_bold"],
                        text_color=COLORS["text_primary"],
                        background_color=row_bg,
                        size=(34, 1),
                    )
                ],
                [
                    sg.Text(
                        "",
                        key=self._label_key(idx),
                        font=FONTS["body"],
                        text_color=COLORS["text_secondary"],
                        background_color=row_bg,
                        size=(50, 1),
                    )
                ],
            ],
            background_color=row_bg,
            pad=((12, 6), (7, 7)),
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
            pad=((8, 12), (8, 8)),
        )

        return sg.pin(
            sg.Column(
                [[info_col, button]],
                key=self._row_key(idx),
                background_color=row_bg,
                visible=False,
                pad=((4, 4), (3, 3)),
                expand_x=True,
            )
        )

    def _build_selector_panel(self):
        rows = [[self._build_list_row(idx)] for idx in range(self.MAX_ROWS)]
        list_col = RScrollableSelector(
            key=self.list_key,
            layout=rows,
            w=0.48,
            h=0.55,
            min_w=500,
            max_w=820,
            min_h=310,
            max_h=535,
            background_color=COLORS["bg_panel"],
        )

        return sg.Column(
            [
                [
                    RText(
                        "Load Model",
                        color=COLORS["accent_highlight"],
                        font=FONTS["header"],
                    )
                ],
                [
                    RText(
                        "Choose an existing model to review, train, evaluate or run predictions.",
                        color=COLORS["text_secondary"],
                    )
                ],
                [self._build_filter_panel()],
                [sg.HorizontalSeparator(color=COLORS["line_bright"], pad=((0, 0), (0, 8)))],
                [list_col],
            ],
            background_color=COLORS["bg_dark"],
            element_justification="left",
            vertical_alignment="top",
            pad=((28, 12), (24, 0)),
            expand_x=True,
            expand_y=True,
        )

    def _build_create_panel(self):
        return sg.Column(
            [
                [
                    RText(
                        "Create Model",
                        color=COLORS["accent_highlight"],
                        bg=COLORS["bg_surface"],
                        font=FONTS["header"],
                    )
                ],
                [
                    sg.Text(
                        "Configure a new architecture and prepare it for training against an existing dataset.",
                        font=FONTS["body"],
                        text_color=COLORS["text_secondary"],
                        background_color=COLORS["bg_surface"],
                        size=(34, 3),
                    )
                ],
                [RButton("Create New Model", key="-PAGE_CREATE_MODEL_CONF-", pad=((0, 0), (12, 4)))],
            ],
            background_color=COLORS["bg_surface"],
            vertical_alignment="top",
            pad=((38, 28), (24, 0)),
            size=(310, 180),
        )

    def build(self, window):
        self._build_model_items()

        title_row = [RText("Models", key="-MODELS_TITLE-", w=0.30)]
        separator = [sg.HorizontalSeparator(color=COLORS["line_bright"])]

        content_row = [
            self._build_create_panel(),
            sg.VSeparator(color=COLORS["line_bright"], pad=((12, 0), (24, 8))),
            self._build_selector_panel(),
        ]

        layout = [
            title_row,
            separator,
            content_row,
            [sg.HorizontalSeparator(color=COLORS["line_bright"])],
        ]

        return sg.Column(
            layout,
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

        for idx, item in enumerate(self.model_items[:self.MAX_ROWS]):
            is_visible = self._matches_filters(item, arch_filter, stage_filter)
            try:
                window[self._name_key(idx)].update(item["name"])
                window[self._label_key(idx)].update(item["label"])
                window[self._row_key(idx)].update(visible=is_visible)
            except Exception:
                pass

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
