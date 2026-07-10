
# ============================================================
#    PAGE MANAGER
#
#    Tracks registered page layouts and optional page handler
#    objects. The UI controller can route events to the active
#    page without knowing which concrete page owns the event.
# ============================================================
class PageManager:

    def __init__(self):

        self.pages = {}
        self.page_handlers = {}
        self.active_page = None


    # ---------------------------------------------------------
    # Register a page layout and optional event handler object.
    # ---------------------------------------------------------
    def register(self, key, page_obj, handler=None):

        self.pages[key] = page_obj

        if handler is not None:
            self.page_handlers[key] = handler


    # ---------------------------------------------------------
    # Register or replace a handler for an existing page.
    # ---------------------------------------------------------
    def register_handler(self, key, handler):

        self.page_handlers[key] = handler


    # ---------------------------------------------------------
    # Build all pages once at startup.
    # ---------------------------------------------------------
    def build_all(self, window):

        for key, page in self.pages.items():
            self.pages[key] = page.build(window)


    # ---------------------------------------------------------
    # Show one page and hide all others.
    # ---------------------------------------------------------
    def show(self, key):

        for k, page in self.pages.items():
            page.update(visible=(k == key))

        self.active_page = key


    # ---------------------------------------------------------
    # Return the active page key.
    # ---------------------------------------------------------
    def get_active_page(self):
        return self.active_page


    # ---------------------------------------------------------
    # Return the active page handler if one was registered.
    # ---------------------------------------------------------
    def get_active_page_handler(self):

        if self.active_page is None:
            return None

        return self.page_handlers.get(self.active_page)
