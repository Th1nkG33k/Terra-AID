from abc import ABC, abstractmethod


class ExplainerBase(ABC):
    """Common interface for Terra-AId XAI explainers."""

    @abstractmethod
    def explain(self, model, x):
        """Return a dictionary of tensors/metadata explaining model output."""
        raise NotImplementedError
