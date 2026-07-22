

import PySimpleGUI as sg
import inspect

from Interface.controls.control_select_base import ControlSelectBase
from ..theme import RPanel, RText, COLORS, FONTS, RButton, RDSPanel, BUTTON_COLORS
from Core.Managers.dataset_manager import DatasetManager
from Core.Managers.path_manager import PathManager


class ControlSelectSearch(ControlSelectBase):

    key = "-CONTROL_SELECT_SEARCH-"

    def __init__(self):

        super().__init__()
        self.title = "Select Search"
        self.button_load_key = "-CTL_SEL_SH_LOAD-"
        self.button_cancel_key = "-CTL_SEL_SH_CANCEL-"

        # Load dataset configs
        pm = PathManager()


    # ------------------------------------------------------------
    # Build list of datasets from configs
    # ------------------------------------------------------------
    def _build_dataset_list(self):
        items = []

        for name in self.dm.list_searches():
            cfg = self.dm.get(name)

            tile_count = cfg.tile_count if cfg.tile_count else "?"
            stage = cfg.stage

            # Display format: (Name, "2700 tiles | processed")
            label = f"{tile_count} tiles | {stage}"
            print(f"build_dataset_list_{name}_{label}")
            items.append((name, label))
        
        return items


    # ------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------
    def build(self, window):

        return super().build(window, self.dataset_list)


    # ------------------------------------------------------------
    # Render a dataset item
    # ------------------------------------------------------------
    def build_item(self, data, idx):

        name, label = data

        if "multi" in name:
            # Left side: name + label
            dets_panel = RDSPanel(key=f"{self.key}_ITEM_{idx}_DETS_PANEL",
                                  layout=[
                                            [RText(name,  key=f"{self.key}_ITEM_{idx}_NAME")],
                                            [RText(label, key=f"{self.key}_ITEM_{idx}_LABEL")],
                                  ],
                                  w=0.50,
            )

            # Clickable button
            btn = RButton("Select",
                          key=f"{self.key}_ITEM_{idx}",
                          w=0.90,
                          pad=(5,5),
            )

            btn_panel = RDSPanel(key=f"{self.key}_ITEM_{idx}_BTN_PANEL",
                               layout=[[btn]],          
                               w=0.40,
            )


            # Parent panel returned to base class
            item_panel = RDSPanel(key=f"{self.key}_ITEM_{idx}_PANEL",
                                  layout=[[dets_panel, btn_panel]],   # ✔️ FIXED
                                  w=0.95,
            )
        else:
            # Left side: name + label
            dets_panel = RPanel(key=f"{self.key}_ITEM_{idx}_DETS_PANEL",
                                layout=[
                                          [RText(name,  key=f"{self.key}_ITEM_{idx}_NAME")],
                                          [RText(label, key=f"{self.key}_ITEM_{idx}_LABEL")],
                                ],
                                w=0.50,
            )

            # Clickable button
            btn = RButton("Select",
                        key=f"{self.key}_ITEM_{idx}",
                        w=0.90,
                        pad=(5,5),
            )

            btn_panel = RPanel(key=f"{self.key}_ITEM_{idx}_BTN_PANEL",
                            layout=[[btn]],          
                            w=0.40,
            )

            # Parent panel returned to base class
            item_panel = RPanel(key=f"{self.key}_ITEM_{idx}_PANEL",
                                layout=[[dets_panel, btn_panel]],   # ✔️ FIXED
                                w=0.95,
            )


        return item_panel


    # ------------------------------------------------------------
    # What to return when Load Selected is pressed
    # ------------------------------------------------------------
    def get_selected_search(self):
        """
        Returns the dataset name selected by the user.
        """
        if self.selected_index is None:
            return None

        name, _ = self.search_list[self.selected_index]
        
        return name

    # ------------------------------------------------------------------
    #   Opens the control as a modal popup window.
    #   Blocks until user selects an item or cancels.
    #   Returns the selected dataset name (or None).   
    # ------------------------------------------------------------------
    def show(self, parent_window):


        # self.build() already returns a Column or Frame layout
        control_layout = self.build(parent_window)

        win = sg.Window(self.title,
                        [[control_layout]],   
                        modal=True,
                        keep_on_top=True,
                        finalize=True,
                        background_color=COLORS["bg_dark"],
                        size=(350, 400),
        )

        selected_value = None
        while True:
            event, values = win.read()

            if event in (sg.WIN_CLOSED, self.button_cancel_key):
                break

            # --- SELECT button clicked ---
            if event.startswith(f"{self.key}_ITEM_") and event.count("_") == 4:
                idx = int(event.split("_")[-1])
                self.selected_index = idx

                # highlight
                for i in range(len(self.dataset_list)):
                    win[f"{self.key}_ITEM_{i}"].update(
                        button_color=BUTTON_COLORS["selection_inactive"]
                    )

                win[f"{self.key}_ITEM_{idx}"].update(
                    button_color=BUTTON_COLORS["selection"]
                )

                # CLOSE + RETURN
                selected_value = self.get_selected_search()
                print(selected_value)
                break

            # --- LOAD button ---
            if event == self.button_load_key:
                selected_value = self.get_selected_search()
                break

        win.close()
        return selected_value
