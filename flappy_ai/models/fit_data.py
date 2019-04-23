import attr
import time


@attr.s(auto_attribs=True)
class FitData:
    epsilon: float
    loss: float
    accuracy: float
    timestamp: float = attr.ib(default=attr.Factory(time.time))

