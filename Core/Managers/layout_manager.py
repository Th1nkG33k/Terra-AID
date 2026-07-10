
import PySimpleGUI as sg
from Interface.theme import COLORS, RBannerImage, RPanelFixed
from Core.Managers.path_manager import PathManager

# ============================================================
#    LAYOUT MANAGER
#
#   Builds the full Terra-AId window layout:
#       - Header banner
#       - Sidebar (fixed width)
#       - Content area (ALL pages stacked; PageManager controls visibility)
# ============================================================
class LayoutManager:

    def __init__(self, page_manager, sidebar_manager):

        self.page_manager = page_manager
        self.sidebar_manager = sidebar_manager
        self.paths = PathManager()

    # ------------------------------------------------------------
    # Header (Banner)
    # ------------------------------------------------------------
    def build_header(self):
        banner_path = self.paths.banner("Terra-AID-Header_2.png")
        return [RBannerImage(banner_path)]

    # ------------------------------------------------------------
    # Sidebar Wrapper
    # ------------------------------------------------------------
    def build_sidebar_wrapper(self):
        SIDEBAR_WIDTH = 200
        sidebar_layout = self.sidebar_manager.layout()

        sidebar_panel = RPanelFixed(key="-SIDEBAR-",
                                    layout=sidebar_layout,
                                    width=SIDEBAR_WIDTH,
        )

        return sg.Column([[sidebar_panel]],
                         key="-SIDEBAR_WRAPPER-",
                         size=(SIDEBAR_WIDTH, None),
                         pad=(0, 0),
                         background_color=COLORS["bg_dark"],
                         vertical_alignment="top",
        )

    # ------------------------------------------------------------
    # Content Wrapper (ALL pages stacked, NO update calls)
    #    Insert ALL registered pages into the layout.
    #    DO NOT call update() here — widgets do not exist yet.
    #    PageManager.show() will handle visibility AFTER layout is added.
    # ------------------------------------------------------------
    def build_content_wrapper(self):

        rows = []

        for key, page in self.page_manager.pages.items():
            
            rows.append([page])

        # Fallback if no pages exist
        if not rows:

            placeholder = sg.Text("No page loaded",
                                  background_color=COLORS["bg_panel"],
                                  text_color=COLORS["text_secondary"],
            )

            rows = [[placeholder]]

        content_area = sg.Column(rows,
                                 key="-CONTENT-",
                                 expand_x=True,
                                 expand_y=True,
                                 background_color=COLORS["bg_panel"],
                                 pad=(0, 0),
        )

        return sg.Column([[content_area]],
                         key="-CONTENT_WRAPPER-",
                         expand_x=True,
                         expand_y=True,
                         background_color=COLORS["bg_dark"],
                         pad=(0, 0),
        )

    # ------------------------------------------------------------
    # Full Layout
    # ------------------------------------------------------------
    def build_layout(self):

        return [
                self.build_header(),
                [
                    self.build_sidebar_wrapper(),
                    self.build_content_wrapper(),
                ],
        ]
