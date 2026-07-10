import PySimpleGUI as sg
from Interface.theme import (RClickPanel, RButton, RText, COLORS)


    # =======================================================
    #    Base class for all Terra‑AId selection popups.
    #    Handles:
    #    - item list building
    #    - click detection
    #    - selected item storage
    #    - load/cancel actions
    #    - consistent Terra‑AId styling
    # =======================================================
class ControlSelectBase:

    key = "-CONTROL_SELECT_BASE-"

    def __init__(self):
        
        self.selected = None
        self.items = []          # (idx, identifier)
        self.title = "Select Item"
        self.button_load_key = "-CTL_LOAD-"
        self.button_cancel_key = "-CTL_CANCEL-"

    # ------------------------------------------------------------
    #  Must be implemented by subclasses
    #  Subclasses must return an RPanel representing a single item.
    # ------------------------------------------------------------
    def build_item(self, data, idx):
        raise NotImplementedError

    # ------------------------------------------------------------
    #  Build the popup content
    # ------------------------------------------------------------
    def build(self, window, data_list):

        self.window = window

        list_layout = []
        self.items = []

        for idx, data in enumerate(data_list):

            item_panel = self.build_item(data, idx)
            list_layout.append([item_panel])
            self.items.append((idx, data))

        list_panel = RClickPanel(key=f"{self.key}_LIST",
                                 layout=list_layout,
                                 w=0.95,
        )


        actions = [
                    sg.Push(),
                    RButton("Cancel", key=self.button_cancel_key, w=0.18),
                    sg.Push(),
        ]

        layout = [
                    [RText(self.title, w=0.40)],
                    [sg.HorizontalSeparator(color=COLORS["line_bright"])],
                    [list_panel],
                    actions,
        ]

        return sg.Column(layout,
                         key=self.key,
                         visible=True,
                         expand_x=True,
                         expand_y=True,
                         background_color=COLORS["bg_dark"],
        )

    # ------------------------------------------------------------
    #  Event handler
    # ------------------------------------------------------------
    def handle_event(self, event):

        # Detect item click
        for idx, data in self.items:
            
            if event == f"{self.key}_ITEM_{idx}":
                self.selected = data
                return ("select", data)

        # Cancel button
        if event == self.button_cancel_key:
            return ("cancel", None)

        return (None, None)
