import attr
import time


@attr.s(auto_attribs=True)
class FitData:
    epsilon: float
    loss: float
    accuracy: float
    timestamp: float = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        self.timestamp = time.time()
