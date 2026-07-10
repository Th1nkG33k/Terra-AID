import torch


# ============================================================
#    DEVICE MANAGER
#
#    Decides at start up which devices are available
# ============================================================
class DeviceManager:

    def __init__(self):
        self.device = self._detect_device()

    def _detect_device(self):

        if torch.cuda.is_available():
            return torch.device("cuda")
        
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        
        else:
            return torch.device("cpu")

    def get(self):
        return self.device

    def summary(self):
        print(f"Using device: {self.device}")
