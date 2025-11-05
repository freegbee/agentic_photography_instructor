from abc import ABC, abstractmethod


class AbstractHandler(ABC):

    @abstractmethod
    def _process_impl(self):
        raise NotImplementedError("Subclasses must implement this method")

    def process(self):
        return self._process_impl()