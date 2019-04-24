from typing import List

import attr


@attr.s(auto_attribs=True)
class PredictionRequest:
    data: any
