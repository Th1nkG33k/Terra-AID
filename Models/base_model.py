from abc import ABC, abstractmethod

class ModelBase(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        pass

    def get_xai_target_layer(self):
        """
        Optional. Only needed for models such as ResNet50.
        """
        return None