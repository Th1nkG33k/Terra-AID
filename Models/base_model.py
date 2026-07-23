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
