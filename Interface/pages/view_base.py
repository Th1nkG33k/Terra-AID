import PySimpleGUI as sg
from ..theme import COLORS, FONTS


# ============================================================
# HOME PAGE
#   Base class for all viewer pages.
#   Handles stacked views and stage switching.
#   
# ============================================================
class ViewerBase:

    def __init__(self, entity=None, title="Viewer"):
        self.entity = entity
        self.title = title
        self.views = {}          # name -> sg.Column
        self.current_stage = None

    # ------------------------------------------------------------
    # Build the viewer container
    # ------------------------------------------------------------
    def build(self):
        self.build_views()

        # Build stacked layout dynamically from ALL registered views
        stacked_layout = [[view] for view in self.views.values()]

        stack = sg.Column(
            stacked_layout,
            key=f"{self.key}_STACK",
            background_color=COLORS["bg_dark"],
            expand_x=True,
            expand_y=True,
            pad=(0, 0),
            scrollable=False,
            vertical_scroll_only=False
        )

        return sg.Column(
            [
                [sg.Text(self.title, font=FONTS["body"], background_color=COLORS["bg_dark"])],
                [stack]
            ],
            key=self.key,
            visible=False,
            expand_x=True,
            expand_y=True,
            background_color=COLORS["bg_dark"],
            scrollable=False,
            vertical_scroll_only=False
        )

    # ------------------------------------------------------------
    # Add a view panel
    # ------------------------------------------------------------
    def add_view(self, name: str, element):
        self.views[name] = element

    # ------------------------------------------------------------
    # Stage switching logic (optional override)
    #
    #   Default stage logic (dataset viewer).
    #   ModelViewer overrides this with apply_stage().
    # ------------------------------------------------------------
    def set_stage(self, stage: str):

        self.current_stage = stage

        # Always show INFO
        if "info" in self.views:
            self.views["info"].update(visible=True)

        # Hide others by default
        for name, view in self.views.items():
            if name != "info":
                view.update(visible=False)

        # Stage logic
        if stage in ("processed", "ready"):
            if "visuals" in self.views:
                self.views["visuals"].update(visible=True)

        elif stage in ("raw", "downloaded", "processing", "unknown"):
            if "processing" in self.views:
                self.views["processing"].update(visible=True)
