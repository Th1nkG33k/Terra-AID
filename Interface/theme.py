import io

import PySimpleGUI as sg
from pathlib import Path
from PIL import Image, ImageOps

from Core.Utils.image_utility import ImageUtility
from Core.Managers.path_manager import PathManager


# ============================================================
#  Terra-AId Design System
#  Responsive PySimpleGUI helpers
# ============================================================

paths = PathManager()
img_util = ImageUtility()


# ============================================================
# COLOUR PALETTE
# ============================================================
COLORS = {
            "bg_dark": "#0F1115",
            "bg_panel": "#16191F",

            "accent_cyan": "#4CC9F0",
            "accent_teal": "#3A86FF",
            "accent_amber": "#FFBE0B",

            "line_dim": "#1E222A",
            "line_bright": "#2A303A",

            "text_primary": "#E6E6E6",
            "text_secondary": "#A8A8A8",
            "text_muted": "#6F6F6F",

            "success": "#80ED99",
            "warning": "#FFD166",
            "error": "#EF476F",
}


# ============================================================
# TYPOGRAPHY
# ============================================================
FONTS = {
            "title": ("Segoe UI Semibold", 18),
            "header": ("Segoe UI Semibold", 14),
            "body": ("Segoe UI", 11),
            "mono": ("Consolas", 10),
}


# ============================================================
# SPACING
# ============================================================
SPACING = {
            "pad_small": (5, 5),
            "pad_medium": (10, 10),
            "pad_large": (20, 20),
}


# ============================================================
# APPLY THEME
# ============================================================
def apply_terra_theme():
    sg.LOOK_AND_FEEL_TABLE["TerraAId"] = {
                                            "BACKGROUND": COLORS["bg_dark"],
                                            "TEXT": COLORS["text_primary"],
                                            "INPUT": COLORS["bg_panel"],
                                            "TEXT_INPUT": COLORS["text_primary"],
                                            "SCROLL": COLORS["line_bright"],
                                            "BUTTON": (COLORS["text_primary"], COLORS["accent_teal"]),
                                            "PROGRESS": COLORS["accent_cyan"],
                                            "BORDER": 1,
                                            "SLIDER_DEPTH": 0,
                                            "PROGRESS_DEPTH": 0,
    }
    sg.theme("TerraAId")


# ============================================================
# RESPONSIVE ENGINE
# ============================================================
RESPONSIVE_REGISTRY = []
RESPONSIVE_READY = False
RESPONSIVE_DEBUG = False


def set_responsive_debug(enabled=True):
    """Enable/disable resize debug output."""
    global RESPONSIVE_DEBUG
    RESPONSIVE_DEBUG = bool(enabled)


def set_responsive_ready(ready=True):
    """Convenience helper. You can also set theme.RESPONSIVE_READY = True directly."""
    global RESPONSIVE_READY
    RESPONSIVE_READY = bool(ready)


def clear_responsive_registry():
    """Use when rebuilding large parts of the UI to prevent stale widgets accumulating."""
    RESPONSIVE_REGISTRY.clear()


def register_responsive(element, resize_fn):
    """Register an element + resize function pair."""
    RESPONSIVE_REGISTRY.append((element, resize_fn))
    return element


def _widget_is_mapped(widget):
    """
    True if the widget or one of its parents is currently mapped.
    Hidden pages/columns can otherwise drift when resized.
    """
    try:

        current = widget
        while current is not None:

            if not current.winfo_ismapped():
                return False
            
            current = current.master

    except Exception:
        return False
    return True


def update_responsive_components(window):
    """
    Called from mainW.py on resize/page-change events.
    Safely updates registered responsive elements.
    """
    if not RESPONSIVE_READY:
        return

    try:

        win_w, win_h = window.size

    except Exception:
        return

    stale = []

    for idx, (element, resize_fn) in enumerate(RESPONSIVE_REGISTRY):

        if not hasattr(element, "Widget") or element.Widget is None:
            stale.append(idx)
            continue

        try:

            if not _widget_is_mapped(element.Widget):
                continue
        
        except Exception:
            continue

        try:

            resize_fn(win_w, win_h)
        
        except Exception as e:

            if RESPONSIVE_DEBUG:
                key = getattr(element, "Key", None)
                print(f"[RESPONSIVE ERROR] key={key!r} element={type(element).__name__}: {e}")
            
            continue

    # Remove stale widgets from the registry occasionally.
    # Reverse order keeps indexes valid.
    for idx in reversed(stale):
        try:

            RESPONSIVE_REGISTRY.pop(idx)
        
        except Exception:
            pass


# ============================================================
# UTILS
# ============================================================
def clamp(value, min_value=None, max_value=None):

    if min_value is not None:
        value = max(min_value, value)
    
    if max_value is not None:
        value = min(max_value, value)
    
    return value


def vw(win_w, percent, min_px=150, max_px=None):
    """Viewport width in pixels."""
    return clamp(int(win_w * percent), min_px, max_px)


def vh(win_h, percent, min_px=80, max_px=None):
    """Viewport height in pixels."""
    return clamp(int(win_h * percent), min_px, max_px)


def chars(win_w, percent, min_chars=8, max_chars=120):
    """
    Tkinter text/button/input widths are usually character units, not pixels.
    This converts a window percentage into an approximate character count.
    """
    return clamp(int((win_w * percent) / 9), min_chars, max_chars)


def rows(win_h, percent, min_rows=3, max_rows=40):
    """Approximate Tkinter text row count from window height."""
    return clamp(int((win_h * percent) / 22), min_rows, max_rows)


def _safe_configure(element, **kwargs):

    try:

        if hasattr(element, "Widget") and element.Widget is not None:
            element.Widget.configure(**kwargs)
    
    except Exception as e:
    
        if RESPONSIVE_DEBUG:
            key = getattr(element, "Key", None)
            print(f"[CONFIGURE ERROR] key={key!r}: {e}")


def _resolve_image_path(path):

    try:

        return paths.resolve_path(path)
    
    except Exception:
        return str(path)


# ============================================================
# PANELS
# ============================================================
def RPanel(key, layout, w=0.30, pad=SPACING["pad_small"], valign="top"):

    col = sg.Column(layout,
                    key=key,
                    pad=pad,
                    background_color=COLORS["bg_dark"],
                    expand_x=True,
                    expand_y=True,
                    vertical_alignment=valign,
    )

    def resize(win_w, win_h):
        _safe_configure(col, width=vw(win_w, w, min_px=30))

    register_responsive(col, resize)
    return col


def RClickPanel(key, layout, w=0.30, pad=SPACING["pad_small"], valign="top", scrollable=True):

    col = sg.Column(layout,
                    key=key,
                    pad=pad,
                    background_color=COLORS["bg_panel"],
                    expand_x=False,
                    expand_y=True,
                    vertical_alignment=valign,
                    scrollable=scrollable,
                    vertical_scroll_only=True,
    )

    def resize(win_w, win_h):
        _safe_configure(col, width=vw(win_w, w, min_px=180))

    register_responsive(col, resize)
    return col


def RPanelFixed(key, layout, width, pad=SPACING["pad_medium"]):

    return sg.Column(layout,
                     key=key,
                     pad=pad,
                     background_color=COLORS["bg_dark"],
                     size=(width, None),
                     expand_x=False,
                     expand_y=False,
    )

def RDSPanel(key, layout, w=0.30, pad=SPACING["pad_small"], valign="top"):

    col = sg.Column(layout,
                    key=key,
                    pad=pad,
                    background_color=COLORS["accent_amber"],
                    expand_x=True,
                    expand_y=True,
                    vertical_alignment=valign,
    )

    def resize(win_w, win_h):
        _safe_configure(col, width=vw(win_w, w, min_px=180))

    register_responsive(col, resize)
    return col

# ============================================================
# BUTTONS
# ============================================================
def RButton(text, key, w=None, pad=(5, 5), visible=True):

    btn = sg.Button(text,
                    key=key,
                    pad=pad,
                    button_color=(COLORS["text_primary"], COLORS["accent_teal"]),
                    mouseover_colors=(COLORS["text_primary"], COLORS["accent_cyan"]),
                    border_width=0,
                    font=FONTS["body"],
                    expand_x=True,
                    visible=visible,
    )

    def resize(win_w, win_h):

        if w:
            _safe_configure(btn, width=chars(win_w, w, min_chars=8, max_chars=55))

    register_responsive(btn, resize)
    return btn


def RButtonSmall(text, key, w=None, pad=(8, 6), visible=True):

    btn = sg.Button(text,
                    key=key,
                    pad=pad,
                    button_color=(COLORS["text_primary"], COLORS["accent_teal"]),
                    mouseover_colors=(COLORS["text_primary"], COLORS["accent_cyan"]),
                    border_width=0,
                    font=FONTS["body"],
                    expand_x=False,
                    visible=visible,
    )

    def resize(win_w, win_h):

        if w:
            _safe_configure(btn, width=chars(win_w, w, min_chars=6, max_chars=35))

    register_responsive(btn, resize)
    return btn


def RSideButton(text, key, w=None, pad=(5, 5), visible=True):

    initial_width = chars(1100, w, min_chars=14, max_chars=24) if w else 18

    btn = sg.Button(text,
                    key=key,
                    pad=pad,
                    button_color=(COLORS["text_primary"], COLORS["accent_teal"]),
                    mouseover_colors=(COLORS["text_primary"], COLORS["accent_cyan"]),
                    border_width=0,
                    font=FONTS["body"],
                    expand_x=False,
                    size=(initial_width, 1),
                    visible=visible,
    )

    def resize(win_w, win_h):

        if w:
            _safe_configure(btn, width=chars(win_w, w, min_chars=12, max_chars=24))

    register_responsive(btn, resize)
    return btn


# ============================================================
# MULTILINE
# ============================================================
def RMultiline(key, w=0.50, h=0.25, disabled=True, default_text=""):

    ml = sg.Multiline(default_text,
                      key=key,
                      disabled=disabled,
                      background_color=COLORS["bg_panel"],
                      text_color=COLORS["text_primary"],
                      border_width=1,
                      font=FONTS["mono"],
                      expand_x=True,
                      expand_y=True,
                      autoscroll=True,
    )

    def resize(win_w, win_h):

        _safe_configure(ml,
                        width=chars(win_w, w, min_chars=20, max_chars=140),
                        height=rows(win_h, h, min_rows=5, max_rows=45),
        )

    register_responsive(ml, resize)
    return ml


# ============================================================
# TEXT
# ============================================================
def RText(text, key=None, w=None, visible=True, color=None, bg=None, font=None, justification=None):

    t = sg.Text(text,
                key=key,
                font=font or FONTS["body"],
                text_color=color or COLORS["text_primary"],
                background_color=bg or COLORS["bg_dark"],
                visible=visible,
                justification=justification,
    )

    def resize(win_w, win_h):

        if w:
            _safe_configure(t, width=chars(win_w, w, min_chars=8, max_chars=140))

    register_responsive(t, resize)
    return t

def RHText(text, key=None, w=None, visible=True, color=None, bg=None, font=None, justification=None):

    t = sg.Text(text,
                key=key,
                font=font or FONTS["header"],
                text_color=color or COLORS["accent_amber"],
                background_color=bg or COLORS["bg_dark"],
                visible=visible,
                justification=justification,
    )

    def resize(win_w, win_h):

        if w:
            _safe_configure(t, width=chars(win_w, w, min_chars=8, max_chars=140))

    register_responsive(t, resize)
    return t



# ============================================================
# INPUT
# ============================================================
def RInput(key, w=None, default_text="", visible=True):

    inp = sg.Input(default_text,
                   key=key,
                   font=FONTS["body"],
                   text_color=COLORS["text_primary"],
                   background_color=COLORS["bg_panel"],
                   border_width=1,
                   expand_x=True,
                   visible=visible,
    )

    def resize(win_w, win_h):

        if w:
            _safe_configure(inp, width=chars(win_w, w, min_chars=5, max_chars=30))

    register_responsive(inp, resize)
    return inp


# ============================================================
# IMAGE
# ============================================================
def RImage(path, key, w=0.40, h_ratio=None, min_w=120):

    resolved = _resolve_image_path(path)
    img = sg.Image(resolved, key=key, background_color=COLORS["bg_dark"])

    def resize(win_w, win_h):

        try:

            width = vw(win_w, w, min_px=min_w, max_px=max(win_w - 20, min_w))

            if h_ratio:
                height = max(int(width * h_ratio), 30)
                data = img_util.load_and_resize_image(resolved, width=width, height=height)
                img.update(data=data, size=(width, height))

            else:
                # Non-resized image fallback. Tk image widgets accept pixel dimensions here.
                _safe_configure(img, width=width)

        except Exception as e:

            if RESPONSIVE_DEBUG:
                print(f"[RImage resize error] key={key!r}: {e}")

    register_responsive(img, resize)
    return img


# ============================================================
# IMAGE BUTTON
# ============================================================
def RImageButton(path, key, text="", size_ratio=0.15, pad=(10, 10)):

    resolved = _resolve_image_path(path)

    btn = sg.Button(image_filename=resolved,
                    key=key,
                    button_color=(COLORS["bg_panel"], COLORS["bg_panel"]),
                    border_width=0,
                    pad=(0, 5),
    )

    label = sg.Text(text,
                    font=FONTS["body"],
                    text_color=COLORS["text_primary"],
                    background_color=COLORS["bg_dark"],
                    justification="center",
    )

    col = sg.Column([[btn], [label]],
                    pad=pad,
                    background_color=COLORS["bg_dark"],
                    element_justification="center",
    )

    def resize(win_w, win_h):

        try:

            side = vw(win_w, size_ratio, min_px=80, max_px=180)
            data = img_util.load_and_resize_image(resolved, width=side, height=side)
            btn.update(image_data=data)

            _safe_configure(label, width=chars(win_w, size_ratio, min_chars=8, max_chars=25))
        
        except Exception as e:
        
            if RESPONSIVE_DEBUG:
                print(f"[RImageButton resize error] key={key!r}: {e}")

    register_responsive(col, resize)
    return col


# ============================================================
# BANNER IMAGE
# ============================================================
def RBannerImage(path, key="-HEADER_BANNER-", w=1.00, h_ratio=0.08, min_h=55, max_h=105):

    resolved = _resolve_image_path(path)

    img = sg.Image(key=key,
                   background_color=COLORS["bg_dark"],
                   expand_x=True,
                   expand_y=False,
                   visible=True,
                   filename=resolved,
                   size=(1000, 80),
    )

    def _fit_banner_bytes(width, height):
        with Image.open(resolved) as source:
            source = source.convert("RGBA")
            fitted = ImageOps.contain(source, (int(width), int(height)), Image.LANCZOS)
            canvas = Image.new("RGBA", (int(width), int(height)), COLORS["bg_dark"])
            x = (int(width) - fitted.width) // 2
            y = (int(height) - fitted.height) // 2
            canvas.alpha_composite(fitted, (x, y))

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()

    def resize(win_w, win_h):

        try:
            # Fixed-height banner box. The image is fitted inside it without
            # stretching, so the logo keeps its proportions on all window sizes.
            width = vw(win_w, w, min_px=300, max_px=max(win_w - 20, 300))
            height = clamp(int(win_h * h_ratio), min_h, max_h)
            data = _fit_banner_bytes(width, height)
            img.update(data=data, size=(width, height))

        except Exception as e:

            if RESPONSIVE_DEBUG:
                print(f"[RBannerImage resize error] key={key!r}: {e}")

    register_responsive(img, resize)
    return img


# ============================================================
# HEADER BAR
# ============================================================
def RHeaderBar(key="-HEADER_BAR-", h=0.08):

    col = sg.Column(
                    [
                        [
                            sg.Text("Terra-AId",
                                    font=FONTS["header"],
                                    text_color=COLORS["accent_cyan"],
                                    background_color=COLORS["bg_panel"],
                                    pad=SPACING["pad_medium"],
                            )
                        ],
                        [sg.HorizontalSeparator(color=COLORS["accent_amber"])],
                    ],
                    key=key,
                    background_color=COLORS["bg_panel"],
                    expand_x=True,
                    pad=(0, 0),
    )

    def resize(win_w, win_h):

        _safe_configure(col, height=vh(win_h, h, min_px=45, max_px=120))

    register_responsive(col, resize)
    return col


# ============================================================
# CALENDAR BUTTON
# ============================================================
def RCalendarButton(text, target_key, key=None, w=None, format="%Y-%m-%d"):

    btn = sg.CalendarButton(text,
                            key=key,
                            target=target_key,
                            format=format,
                            font=FONTS["body"],
                            button_color=(COLORS["text_primary"], COLORS["accent_teal"]),
                            border_width=0,
                            pad=(5, 5),
    )

    def resize(win_w, win_h):

        if w:
            _safe_configure(btn, width=chars(win_w, w, min_chars=8, max_chars=35))

    register_responsive(btn, resize)
    return btn


# ============================================================
# RADIO BUTTON
# ============================================================
def RRadio(text, group_id, key, default=False, w=None, pad=(0, 0), visible=True):

    radio = sg.Radio(text,
                     group_id,
                     key=key,
                     default=default,
                     font=FONTS["body"],
                     text_color=COLORS["text_primary"],
                     background_color=COLORS["bg_dark"],
                     pad=pad,
                     visible=visible,
    )

    def resize(win_w, win_h):

        if w:
            _safe_configure(radio, width=chars(win_w, w, min_chars=8, max_chars=70))

    register_responsive(radio, resize)
    return radio


# ============================================================
# FRAME
# ============================================================
def RFrame(title, layout, key, w=0.60, h=0.40):

    frame = sg.Frame(title,
                     layout,
                     key=key,
                     background_color=COLORS["bg_dark"],
                     title_color=COLORS["text_secondary"],
                     relief=sg.RELIEF_SUNKEN,
                     expand_x=True,
                     expand_y=True,
    )

    def resize(win_w, win_h):

        _safe_configure(frame,
                        width=vw(win_w, w, min_px=220),
                        height=vh(win_h, h, min_px=140),
        )

    register_responsive(frame, resize)
    return frame
