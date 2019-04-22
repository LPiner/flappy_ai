import attr
from flappy_ai.models.game_data import GameData
import random
from collections import deque
from typing import List


@attr.s(auto_attribs=True)
class GameHistory:
    size: int
    _buffer: List[GameData] = attr.ib(default=attr.Factory(list), init=False)

    def append(self, game_data: GameData):
        if len(self._buffer) > self.size:
            del self._buffer[0]
        self._buffer.append(game_data)

    def __getitem__(self, idx):
        return self._buffer[idx]

    def __len__(self):
        return len(self._buffer)

    def get_sample_batch(self, batch_size=1) -> List[GameData]:
        # Remember that this will return an entire game.
        return random.sample(self._buffer, batch_size)
