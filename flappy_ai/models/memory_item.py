from typing import List, Union

import attr
import numpy as np


# Roughly follows
# e = (s1, a1, r1+1, s1+1)
@attr.s(auto_attribs=True)
class MemoryItem:
    state: np.array  # The state that we acted on
    action: List[int]  # ? The action that we took on the state.
    next_state: np.array = attr.ib(default=None, init=False)
    reward: float = attr.ib(default=None, init=False)  # Reward for the action, captured by the next state.
    is_terminal: bool = attr.ib(default=None, init=False)  # did this end the game?
    # merged_state: np.array = attr.ib(default=None, init=False) # A merged set of states that came before
