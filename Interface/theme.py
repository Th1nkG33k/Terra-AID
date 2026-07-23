import io
import colorsys

import numpy as np
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
            # Archaeological stratigraphy palette.
            # Keep colour names role-based so the visual style can be changed
            # centrally without editing individual pages or controls.
            "bg_dark": "#1D1915",          # Deep peat - application background
            "bg_panel": "#2A241E",         # Dark umber - inputs and panels
            "bg_surface": "#352D25",       # Raised earth - cards and raised surfaces

            "accent_primary": "#B97C45",   # Clay - primary actions and selection
            "accent_hover": "#C9945F",     # Sun-baked clay - hover and active states
            "accent_secondary": "#788168", # Sage - secondary emphasis
            "accent_highlight": "#D1AE69", # Sand - headings and separators

            "line_dim": "#4A3E34",
            "line_bright": "#665547",      # Stratigraphic boundary lines

            "text_primary": "#F2E9DC",     # Bone
            "text_secondary": "#C9B9A5",   # Weathered stone
            "text_muted": "#958572",       # Dust
            "text_on_accent": "#1D1915",   # Accessible dark text on clay/sand

            "success": "#8FA374",
            "warning": "#D1AE69",
            "error": "#B85D4C",
}

# Reusable component states. Pages should use these instead of embedding
# colour tuples so all interaction states remain controlled by theme.py.
BUTTON_COLORS = {
            "primary": (COLORS["text_on_accent"], COLORS["accent_primary"]),
            "primary_hover": (COLORS["text_on_accent"], COLORS["accent_hover"]),
            "secondary": (COLORS["text_primary"], COLORS["bg_surface"]),
            "secondary_hover": (COLORS["text_on_accent"], COLORS["accent_secondary"]),
            "sidebar": (COLORS["text_secondary"], COLORS["bg_panel"]),
            "sidebar_hover": (COLORS["text_on_accent"], COLORS["accent_hover"]),
            "sidebar_active": (COLORS["text_on_accent"], COLORS["accent_primary"]),
            "selection": (COLORS["text_on_accent"], COLORS["accent_primary"]),
            "selection_inactive": (COLORS["text_primary"], COLORS["bg_surface"]),
}

MAP_COLORS = {
            "aoi_outline": COLORS["accent_highlight"],
            "aoi_fill": COLORS["accent_primary"],
}

# Existing brand artwork contains blue/cyan illumination. The artwork is
# recoloured at runtime from the palette above, allowing the splash and header
# to follow theme.py without maintaining separate raster assets.
BRAND_IMAGE_STYLE = {
            "recolour_cool_tones": True,
            "cool_hue_min": 105,      # Pillow HSV scale: 0-255
            "cool_hue_max": 190,
            "minimum_saturation": 18,
}

_BRAND_IMAGE_CACHE = {}

# Temporary compatibility aliases for any external extensions that still import
# the former colour names. Terra-AID's own interface now uses semantic names.
COLORS.update({
            "accent_cyan": COLORS["accent_hover"],
            "accent_teal": COLORS["accent_primary"],
            "accent_amber": COLORS["accent_highlight"],
})


# ============================================================
# TYPOGRAPHY
# ============================================================
FONTS = {
            "title": ("Segoe UI Semibold", 18),
            "header": ("Segoe UI Semibold", 14),
            "body": ("Segoe UI", 11),
            "body_bold": ("Segoe UI Semibold", 11),
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
                                            "BUTTON": BUTTON_COLORS["primary"],
                                            "PROGRESS": COLORS["accent_primary"],
                                            "BORDER": 1,
                                            "SLIDER_DEPTH": 0,
                                            "PROGRESS_DEPTH": 0,
    }
    sg.theme("TerraAId")


# ============================================================
# THEMED BRAND IMAGES
# ============================================================
def _hex_to_hsv255(hex_colour):
    """Convert a #RRGGBB colour into Pillow's 0-255 HSV scale."""
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 255), int(s * 255), int(v * 255)


def load_themed_brand_image(path):
    """Load a brand image and remap its blue/cyan tones to the active palette."""
    resolved = str(_resolve_image_path(path))
    try:
        modified = Path(resolved).stat().st_mtime_ns
    except OSError:
        modified = 0

    cache_key = (
        resolved,
        modified,
        COLORS["accent_primary"],
        COLORS["accent_highlight"],
        tuple(sorted(BRAND_IMAGE_STYLE.items())),
    )

    cached = _BRAND_IMAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    with Image.open(resolved) as source:
        themed = source.convert("RGBA")

    if BRAND_IMAGE_STYLE["recolour_cool_tones"]:
        alpha = themed.getchannel("A")
        hsv = np.array(themed.convert("RGB").convert("HSV"), dtype=np.uint8)

        hue = hsv[:, :, 0].astype(np.int16)
        saturation = hsv[:, :, 1].astype(np.int16)
        value = hsv[:, :, 2].astype(np.int16)

        mask = (
            (hue >= BRAND_IMAGE_STYLE["cool_hue_min"])
            & (hue <= BRAND_IMAGE_STYLE["cool_hue_max"])
            & (saturation >= BRAND_IMAGE_STYLE["minimum_saturation"])
        )

        primary_h, primary_s, _ = _hex_to_hsv255(COLORS["accent_primary"])
        highlight_h, highlight_s, _ = _hex_to_hsv255(COLORS["accent_highlight"])
        brightness = value.astype(np.float32) / 255.0

        target_hue = (
            primary_h + ((highlight_h - primary_h) * brightness)
        ).clip(0, 255).astype(np.uint8)
        target_saturation = (
            (primary_s * (1.0 - brightness * 0.35))
            + (highlight_s * brightness * 0.35)
        ).clip(0, 220).astype(np.uint8)

        hsv[:, :, 0][mask] = target_hue[mask]
        hsv[:, :, 1][mask] = target_saturation[mask]

        themed = Image.fromarray(hsv, "HSV").convert("RGBA")
        themed.putalpha(alpha)

    # Keep a small cache; theme images are large and only the splash/header are used.
    _BRAND_IMAGE_CACHE.clear()
    _BRAND_IMAGE_CACHE[cache_key] = themed.copy()
    return themed


# ============================================================
# RESPONSIVE ENGINE
# ============================================================
RESPONSIVE_REGISTRY = []
RESPONSIVE_READY = False
RESPONSIVE_DEBUG = False


# ---------------------------------------------------------------------
# Enable/disable resize debug output.
# ---------------------------------------------------------------------
def set_responsive_debug(enabled=True):
    
    global RESPONSIVE_DEBUG
    RESPONSIVE_DEBUG = bool(enabled)


# ---------------------------------------------------------------------
# Convenience helper. You can also set theme.RESPONSIVE_READY = True directly.
# ---------------------------------------------------------------------
def set_responsive_ready(ready=True):
    
    global RESPONSIVE_READY
    RESPONSIVE_READY = bool(ready)


# ---------------------------------------------------------------------
# Use when rebuilding large parts of the UI to prevent stale widgets accumulating.
# ---------------------------------------------------------------------
def clear_responsive_registry():
    RESPONSIVE_REGISTRY.clear()


# ---------------------------------------------------------------------
# Register an element + resize function pair.
# ---------------------------------------------------------------------
def register_responsive(element, resize_fn):
    
    RESPONSIVE_REGISTRY.append((element, resize_fn))
    return element


# ---------------------------------------------------------------------
# True if the widget or one of its parents is currently mapped.
# Hidden pages/columns can otherwise drift when resized.
# ---------------------------------------------------------------------
def _widget_is_mapped(widget):

    try:

        current = widget
        while current is not None:

            if not current.winfo_ismapped():
                return False
            
            current = current.master

    except Exception:
        return False
    return True


# ---------------------------------------------------------------------
# Called from mainW.py on resize/page-change events.
# Safely updates registered responsive elements.
# ---------------------------------------------------------------------
def update_responsive_components(window):

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
    # Viewport width in pixels.
    return clamp(int(win_w * percent), min_px, max_px)


def vh(win_h, percent, min_px=80, max_px=None):
    # Viewport height in pixels.
    return clamp(int(win_h * percent), min_px, max_px)


def chars(win_w, percent, min_chars=8, max_chars=120):
    # ---------------------------------------------------------------------
    # Tkinter text/button/input widths are usually character units, not pixels.
    # This converts a window percentage into an approximate character count.
    # ---------------------------------------------------------------------
    return clamp(int((win_w * percent) / 9), min_chars, max_chars)


def rows(win_h, percent, min_rows=3, max_rows=40):
    # Approximate Tkinter text row count from window height.
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
                    background_color=COLORS["bg_surface"],
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

def RScrollableSelector(key, layout, w=0.46, h=0.55,
                        min_w=460, max_w=820, min_h=300, max_h=540,
                        background_color=None, pad=(0, 0)):
    """Scrollable list area used by the Dataset and Model landing pages.

    The list grows with the main form while retaining enough width for the
    fixed Select action at the right of every row. Keeping this behaviour in
    theme.py prevents the two landing pages drifting apart visually.
    """

    bg = background_color or COLORS["bg_panel"]
    initial_width = vw(1360, w, min_px=min_w, max_px=max_w)
    initial_height = vh(767, h, min_px=min_h, max_px=max_h)

    col = sg.Column(
        layout,
        key=key,
        background_color=bg,
        scrollable=True,
        vertical_scroll_only=True,
        size=(initial_width, initial_height),
        pad=pad,
        expand_x=True,
        expand_y=True,
    )

    def resize(win_w, win_h):
        _safe_configure(
            col,
            width=vw(win_w, w, min_px=min_w, max_px=max_w),
            height=vh(win_h, h, min_px=min_h, max_px=max_h),
        )

    register_responsive(col, resize)
    return col

def RDSPanel(key, layout, w=0.30, pad=SPACING["pad_small"], valign="top"):

    col = sg.Column(layout,
                    key=key,
                    pad=pad,
                    background_color=COLORS["accent_secondary"],
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
                    button_color=BUTTON_COLORS["primary"],
                    mouseover_colors=BUTTON_COLORS["primary_hover"],
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
                    button_color=BUTTON_COLORS["primary"],
                    mouseover_colors=BUTTON_COLORS["primary_hover"],
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
                    button_color=BUTTON_COLORS["sidebar"],
                    mouseover_colors=BUTTON_COLORS["sidebar_hover"],
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
                text_color=color or COLORS["accent_highlight"],
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
                    button_color=(COLORS["bg_surface"], COLORS["bg_surface"]),
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
# -------------------------------------------------------------------------------
    # Create a compact banner that resizes with the main form.

    # The complete source image is resampled into the available banner area on
    # every window-size change. This keeps the header compact, displays the whole
    # logo, and prevents Tk from showing a cropped section of the full-size file.
# -------------------------------------------------------------------------------
def RBannerImage(path, key="-HEADER_BANNER-", w=1.00, h_ratio=0.08,
                 min_h=55, max_h=105, min_w=420, max_w=860, horizontal_margin=24):


    resolved = _resolve_image_path(path)
    render_cache = {}
    last_size = [None, None]

    def _target_size(win_w, win_h):
        width = max(min_w, int(win_w * w) - horizontal_margin)
        height = clamp(int(win_h * h_ratio), min_h, max_h)
        return width, height

    def _banner_bytes(width, height):
        cache_key = (int(width), int(height))
        cached = render_cache.get(cache_key)
        if cached is not None:
            return cached

        with load_themed_brand_image(resolved) as source:
            source = source.convert("RGBA")
            fitted = source.resize((int(width), int(height)), Image.LANCZOS)

        buffer = io.BytesIO()
        fitted.save(buffer, format="PNG")
        data = buffer.getvalue()

        # Retain only the latest size so drag-resizing does not accumulate
        # large PNG byte arrays in memory.
        render_cache.clear()
        render_cache[cache_key] = data
        return data

    # Start with a correctly scaled copy. Supplying the full-size filename to
    # sg.Image caused the original 1536 x 220 artwork to be clipped into an
    # 80-pixel-high widget before the first responsive update.
    initial_width, initial_height = 1000, 80
    initial_data = _banner_bytes(initial_width, initial_height)

    img = sg.Image(
        data=initial_data,
        key=key,
        background_color=COLORS["bg_dark"],
        expand_x=True,
        expand_y=False,
        visible=True,
        size=(initial_width, initial_height),
        pad=(0, 0),
    )

    def resize(win_w, win_h):
        try:
            width, height = _target_size(win_w, win_h)
            if (width, height) == tuple(last_size):
                return

            data = _banner_bytes(width, height)
            img.update(data=data, size=(width, height))
            last_size[:] = [width, height]

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
                                    text_color=COLORS["accent_primary"],
                                    background_color=COLORS["bg_panel"],
                                    pad=SPACING["pad_medium"],
                            )
                        ],
                        [sg.HorizontalSeparator(color=COLORS["accent_highlight"])],
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
                            button_color=BUTTON_COLORS["primary"],
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
