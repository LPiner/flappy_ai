import random
from collections import deque
from typing import List

import attr

from flappy_ai.models.game_data import GameData
from flappy_ai.models.memory_item import MemoryItem


@attr.s(auto_attribs=True)
class GameHistory:
    size: int
    _buffer: deque = attr.ib(default=attr.Factory(deque), init=False)

    def __attrs_post_init__(self):
        self._buffer = deque(maxlen=self.size)

    def append(self, memory: MemoryItem):
        if len(self._buffer) > self.size:
            del self._buffer[0]
        self._buffer.append(memory)

    def __getitem__(self, idx):
        return self._buffer[idx]

    def __len__(self):
        return len(self._buffer)

    def get_sample_batch(self, batch_size=1) -> List[MemoryItem]:
        # Remember that this will return an entire game.
        return random.sample(self._buffer, batch_size)
