from __future__ import annotations

from Core.Managers.path_manager import PathManager
from Core.Managers.config_manager import ConfigManager
from Core.Managers.dataset_manager import DatasetManager
from Core.Managers.model_manager import ModelManager
from Core.Managers.page_manager import PageManager
from Core.Managers.sidebar_manager import SidebarManager
from Core.Managers.thread_manager import TaskManager

# ============================================================
# AppContext
#    Shared application services for Terra-AID.
#
#    This object is created once at application startup and passed to the UI.
#    It prevents pages, popups, and task handlers from each creating their own
#    independent managers.
#
#   Design rule:
#     Configs describe. Managers decide. UI displays.
# ============================================================

class AppContext:

    def __init__(self):

        self.paths = PathManager()
        self.configs = ConfigManager(self.paths)

        # Managers consume ConfigManager instead of re-reading YAML directly.
        self.datasets = DatasetManager(config_manager=self.configs)
        self.models = ModelManager(config_manager=self.configs)

        # UI/application managers.
        self.pages = PageManager()
        self.sidebar = SidebarManager()
        self.tasks = TaskManager()

    # ---------------------------------------------
    # Reload config YAML and refresh manager views.
    # --------------------------------------------
    def reload_configs(self):
        
        self.configs.reload()
        self.datasets.reload()
        self.models.reload()
