import torch


# ============================================================
#    DEVICE MANAGER
#
#    Decides at start up which devices are available.
#    Config files do not override this decision.
# ============================================================
class DeviceManager:

    def __init__(self):
        self.device = self._detect_device()

    def _detect_device(self):

        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def get(self):
        return self.device

    def summary(self):
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(0)
        elif self.device.type == "mps":
            name = "Apple MPS"
        else:
            name = "CPU"

        print(f"Using device: {self.device} ({name})")
