from abc import ABCMeta, abstractmethod

import attr


@attr.s(auto_attribs=True)
class AbstractNetwork(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, state: any) -> int:
        raise NotImplementedError()

    @abstractmethod
    def predict_random(self, state: any) -> int:
        raise NotImplementedError()

    @abstractmethod
    def fit_batch(self):
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        raise NotImplementedError()
