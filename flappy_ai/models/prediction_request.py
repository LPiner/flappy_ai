import attr
from typing import List


@attr.s(auto_attribs=True)
class PredictionRequest:
    data: any
