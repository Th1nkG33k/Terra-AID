
import PySimpleGUI as sg
from Interface.theme import RSideButton

# ============================================================
#   SIDEBAR MANAGER
#
    # Manages sidebar navigation buttons and their mapping to page keys.
    # Each button emits its page_key directly as the event.
# ============================================================
class SidebarManager:

    def __init__(self):

        self.buttons: list[sg.Button] = []
        self._map: dict[str, sg.Button] = {}  # page_key -> button
        self.active_page_key: str | None = None

    # ------------------------------------------------------------
    # Add a new sidebar button
    # Create a sidebar button that triggers `page_key` when clicked.
    # The button's event key IS the page_key.
    # ------------------------------------------------------------
    def add(self, label: str, page_key: str) -> sg.Button:

        btn = RSideButton(label, key=page_key)
        self.buttons.append(btn)
        self._map[page_key] = btn
        return btn

    # ------------------------------------------------------------
    # Layout for PySimpleGUI
    # Returns the sidebar layout as a vertical stack of buttons.
    # ------------------------------------------------------------
    def layout(self) -> list[list[sg.Button]]:
      
        return [[btn] for btn in self.buttons]

    # ------------------------------------------------------------
    # Highlight active page (future styling hook)
    # Marks a page as active. If you want visual highlighting,
    # update the RButton style here.
    # ------------------------------------------------------------
    def set_active(self, page_key: str):

        self.active_page_key = page_key

        for key, btn in self._map.items():

            btn.update(button_color=("white", "#0D47A1"))
            
            if key == self.active_page_key:
                btn.update(button_color=("white", "#1E88E5"))
