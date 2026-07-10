import json
import PySimpleGUI as sg
import tkintermapview

from Interface.theme import (RPanel, RButtonSmall, RText, COLORS, RHText)
from Core.Managers.path_manager import PathManager


# ============================================================
# CREATE DATASET MAP
#
#   Popup window used by Create Dataset Configuration to select an AOI.
#   The tkintermapview widget must be created after the PySimpleGUI window has
#   been finalised.  For that reason this class owns a popup window and a small
#   event loop rather than being embedded in the main page column.
#   
# ============================================================
class PageCreateDatasetMap:

    key = "-PAGE_CREATE_DATASET_MAP-"

    def __init__(self):
        self.paths = PathManager()
        self.map_widget = None
        self.window = None
        self.marker = None
        self.aoi_polygon = None
        self.selected_lat = None
        self.selected_lon = None
        self.selected_bbox = None

    # ------------------------------------------------------------
    # WINDOW / MAP LIFECYCLE
    # Open the AOI selector as a popup and return the selected AOI.
    # Returns:
    #     dict | None: {
    #         "center": {"lat": float, "lon": float},
    #         "bbox": [west, south, east, north],
    #     }    
    # ------------------------------------------------------------
    def open(self, parent_window=None, initial_values=None):

        self._reset_state()

        layout = self.build()
        self.window = sg.Window("Select Area of Interest",
                                layout,
                                modal=True,
                                resizable=True,
                                finalize=True,
                                keep_on_top=False,
                                background_color=COLORS["bg_dark"],
                                size=(920, 560),
        )

        # ------------------------------------------------------------
        # Important: tkintermapview needs the popup to be fully drawn before
        # the map widget is created.  Creating it immediately after finalize can
        # leave a blank host frame on some Windows/Tk combinations.
        # ------------------------------------------------------------
        pending_initial_values = initial_values or {}
        map_initialised = False

        result = None

        while True:
            event, values = self.window.read(timeout=50)

            if not map_initialised:
                self.init_map(self.window)
                self._load_initial_values(pending_initial_values)
                map_initialised = True

            if event in (sg.WIN_CLOSED, "-CDM_CANCEL-"):
                break

            if event == "-CDM_CONTINUE-":
                result = self._build_result(values)
                if result is not None:
                    break
                continue

            self.handle_event(event, values, self.window)

        self.close()

        if parent_window is not None:
            parent_window.bring_to_front()

        return result

    def close(self):
        if self.window is not None:
            self.window.close()
        self.window = None
        self.map_widget = None
        self.marker = None
        self.aoi_polygon = None

    def _reset_state(self):
        self.map_widget = None
        self.window = None
        self.marker = None
        self.aoi_polygon = None
        self.selected_lat = None
        self.selected_lon = None
        self.selected_bbox = None

# ------------------------------------------------------------
# Create TkinterMapView once, after the popup window is visible.
# ------------------------------------------------------------
    def init_map(self, window):
        
        if self.map_widget is not None:
            return

        # ------------------------------------------------------------
        # The host element is a real Tk frame/canvas owned by the popup.
        # Force Tk to calculate its size before attaching tkintermapview.
        # ------------------------------------------------------------
        container = window["-MAP_CONTAINER-"].Widget
        window.refresh()
        container.update_idletasks()

        for child in container.winfo_children():
            child.destroy()

        width = max(container.winfo_width(), 600)
        height = max(container.winfo_height(), 420)

        self.map_widget = tkintermapview.TkinterMapView(container,
                                                        width=width,
                                                        height=height,
                                                        corner_radius=0,
        )
        self.map_widget.pack(fill="both", expand=True)

        # ------------------------------------------------------------
        # Use an explicit OpenStreetMap tile endpoint rather than relying on
        # tkintermapview defaults, which can vary by installed version.
        # ------------------------------------------------------------
        try:
            self.map_widget.set_tile_server(
                "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                max_zoom=19,
            )
        except TypeError:
            self.map_widget.set_tile_server(
                "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
            )

        # ------------------------------------------------------------
        # Set these after packing so the internal canvas/tile loader has a live
        # widget.  Without this order the map can appear as a blank frame.
        # ------------------------------------------------------------
        self.map_widget.set_position(51.5074, -0.1278)
        self.map_widget.set_zoom(10)
        self.map_widget.add_left_click_map_command(self._on_map_left_click)
        self.map_widget.max_zoom = 13
        self.map_widget.min_zoom = 10 

        self.map_widget.update_idletasks()
        window.refresh()

    def _load_initial_values(self, values):
        """Pre-populate the popup from the dataset configuration fields."""
        try:
            min_lat = float(values.get("-CDC_MIN_LAT-", ""))
            max_lat = float(values.get("-CDC_MAX_LAT-", ""))
            min_lon = float(values.get("-CDC_MIN_LON-", ""))
            max_lon = float(values.get("-CDC_MAX_LON-", ""))
        except (TypeError, ValueError):
            return

        lat = (min_lat + max_lat) / 2
        lon = (min_lon + max_lon) / 2
        self.selected_lat = lat
        self.selected_lon = lon
        self.selected_bbox = [min_lon, min_lat, max_lon, max_lat]

        self.window["-CDM_LAT-"].update(f"{lat:.6f}")
        self.window["-CDM_LON-"].update(f"{lon:.6f}")
        self._draw_bbox(self.selected_bbox)
        self.map_widget.set_position(lat, lon)

    # ------------------------------------------------------------
    # MAP ACTIONS
    # ------------------------------------------------------------
    def _on_map_left_click(self, coords):

        """Called by TkinterMapView when the user clicks the map."""
        lat, lon = float(coords[0]), float(coords[1])
        self.selected_lat = lat
        self.selected_lon = lon

        if self.marker is not None:
            self.marker.delete()

        self.marker = self.map_widget.set_marker(lat, lon, text="AOI centre")
        
        if self.window is not None:
            self.window.write_event_value("-MAP_POINT_SELECTED-", {"lat": lat, "lon": lon})
        
        bbox = self._bbox_from_center(lat=lat, lon=lon)


    def _bbox_from_center(self, lat, lon, half_size=0.01):
        """Return bbox as [west, south, east, north]."""
        west = lon - half_size
        south = lat - half_size
        east = lon + half_size
        north = lat + half_size
        return [west, south, east, north]

    def _draw_bbox(self, bbox):
        if self.map_widget is None:
            return

        west, south, east, north = bbox

        if self.aoi_polygon is not None:
            self.aoi_polygon.delete()

        polygon_points = [
            (north, west),
            (north, east),
            (south, east),
            (south, west),
        ]

        self.aoi_polygon = self.map_widget.set_polygon(
            polygon_points,
            outline_color="#4CC9F0",
            fill_color="#4CC9F0",
            border_width=2,
        )

    # ------------------------------------------------------------
    # Draw a square AOI around the selected or typed coordinate.
    # ------------------------------------------------------------
    def draw_aoi(self, window, values):
        
        try:
            lat = float(values.get("-CDM_LAT-") or self.selected_lat)
            lon = float(values.get("-CDM_LON-") or self.selected_lon)
        except (TypeError, ValueError):
            sg.popup_error("Click the map first, or enter valid latitude/longitude values.")
            return

        bbox = self._bbox_from_center(lat, lon)
        self.selected_lat = lat
        self.selected_lon = lon
        self.selected_bbox = bbox

        if self.marker is not None:
            self.marker.delete()
        self.marker = self.map_widget.set_marker(lat, lon, text="AOI centre")

        self._draw_bbox(bbox)
        self._update_fields(window, lat, lon, bbox)

    def clear_aoi(self, window):
        if self.marker is not None:
            self.marker.delete()
            self.marker = None

        if self.aoi_polygon is not None:
            self.aoi_polygon.delete()
            self.aoi_polygon = None

        self.selected_lat = None
        self.selected_lon = None
        self.selected_bbox = None

        window["-CDM_LAT-"].update("")
        window["-CDM_LON-"].update("")
        window["-CDM_BBOX-"].update("")

    # ------------------------------------------------------------
    # Load a simple GeoJSON polygon file and use its bounds as the AOI.
    # ------------------------------------------------------------
    def load_polygon(self, window):
    
        filename = sg.popup_get_file(
            "Select GeoJSON polygon",
            file_types=(("GeoJSON", "*.geojson *.json"), ("All files", "*.*")),
            no_window=True,
        )
        if not filename:
            return

        try:
            data = json.loads(open(filename, "r", encoding="utf-8").read())
            coords = self._extract_geojson_points(data)
            if not coords:
                raise ValueError("No polygon coordinates found.")

            lons = [point[0] for point in coords]
            lats = [point[1] for point in coords]
            bbox = [min(lons), min(lats), max(lons), max(lats)]
            lat = (bbox[1] + bbox[3]) / 2
            lon = (bbox[0] + bbox[2]) / 2

            self.selected_lat = lat
            self.selected_lon = lon
            self.selected_bbox = bbox

            self._draw_bbox(bbox)
            self._update_fields(window, lat, lon, bbox)
            self.map_widget.set_position(lat, lon)
        except Exception as exc:
            sg.popup_error(f"Could not load polygon: {exc}")

    # ------------------------------------------------------------
    # Extract lon/lat coordinate pairs from Polygon/MultiPolygon GeoJSON.
    # ------------------------------------------------------------
    def _extract_geojson_points(self, data):
        
        if data.get("type") == "FeatureCollection":
            points = []
            for feature in data.get("features", []):
                points.extend(self._extract_geojson_points(feature))
            return points

        if data.get("type") == "Feature":
            return self._extract_geojson_points(data.get("geometry", {}))

        geometry_type = data.get("type")
        coordinates = data.get("coordinates", [])

        if geometry_type == "Polygon":
            return [point[:2] for ring in coordinates for point in ring]

        if geometry_type == "MultiPolygon":
            return [point[:2] for polygon in coordinates for ring in polygon for point in ring]

        return []

    def _update_fields(self, window, lat, lon, bbox):
        window["-CDM_LAT-"].update(f"{lat:.6f}")
        window["-CDM_LON-"].update(f"{lon:.6f}")
        window["-CDM_BBOX-"].update(json.dumps({
            "center": {"lat": lat, "lon": lon},
            "bbox": bbox,
        }))

    def _build_result(self, values):
        if self.selected_bbox is None:
            self.draw_aoi(self.window, values)

        if self.selected_bbox is None:
            return None

        lat = self.selected_lat
        lon = self.selected_lon
        bbox = self.selected_bbox

        return {
            "center": {"lat": lat, "lon": lon},
            "bbox": bbox,
            "west": bbox[0],
            "south": bbox[1],
            "east": bbox[2],
            "north": bbox[3],
        }

    # ------------------------------------------------------------
    # EVENT HANDLING / LAYOUT
    # ------------------------------------------------------------
    def handle_event(self, event, values, window):
        if event == "-MAP_POINT_SELECTED-":
            payload = values[event]
            lat = payload["lat"]
            lon = payload["lon"]
            window["-CDM_LAT-"].update(f"{lat:.6f}")
            window["-CDM_LON-"].update(f"{lon:.6f}")

        elif event == "-CDM_DRAW_AOI-":
            self.draw_aoi(window, values)

        elif event == "-CDM_CLEAR_AOI-":
            self.clear_aoi(window)

        elif event == "-CDM_LOAD_POLY-":
            self.load_polygon(window)

    def build(self):
        coord_lat = sg.Input(
            key="-CDM_LAT-",
            size=(10, 1),
            justification="center",
            background_color=COLORS["bg_panel"],
            text_color=COLORS["text_primary"],
        )

        coord_lon = sg.Input(
            key="-CDM_LON-",
            size=(10, 1),
            justification="center",
            background_color=COLORS["bg_panel"],
            text_color=COLORS["text_primary"],
        )

        map_buttons = [
            [
                sg.Push(),
                RButtonSmall("Draw AOI", key="-CDM_DRAW_AOI-"),
                RButtonSmall("Clear AOI", key="-CDM_CLEAR_AOI-"),
                RButtonSmall("Load Polygon", key="-CDM_LOAD_POLY-"),
                sg.Push(),
            ]
        ]

        left_layout = [
            [RHText("Area of Interest")],
            [
                sg.Frame(
                    "Map",
                    [[sg.Column([[]],
                                key="-MAP_CONTAINER-",
                                size=(600, 420),
                                pad=(0, 0),
                                expand_x=True,
                                expand_y=True,
                                background_color=COLORS["bg_dark"],
                    )]],
                    key="-CD_MAP_FRAME-",
                    background_color=COLORS["bg_panel"],
                    title_color=COLORS["text_secondary"],
                    expand_x=True,
                    expand_y=True,
                    relief=sg.RELIEF_SUNKEN,
                    size=(600, 420),
                )
            ],
            *map_buttons,
        ]

        left_panel = RPanel(key="-CDM_LEFT_PANEL-", layout=left_layout, w=0.65)

        right_layout = [
            [RHText("Coordinates")],
            [RText("Lat"), coord_lat, RText("Lon"), coord_lon, sg.Push()],
            [sg.Input("", key="-CDM_BBOX-", visible=False)],
            [sg.HorizontalSeparator(color=COLORS["line_bright"])],
            [RText("1. Click the map or type a centre point.")],
            [RText("2. Draw AOI to create the bounding box.")],
            [RText("3. Continue to copy it back to the dataset config.")],
            [sg.Push(), RButtonSmall("Continue", key="-CDM_CONTINUE-"), RButtonSmall("Cancel", key="-CDM_CANCEL-"), sg.Push()],
        ]

        right_panel = RPanel(key="-CDM_RIGHT_PANEL-", layout=right_layout, w=0.35)

        return [[left_panel, right_panel]]
