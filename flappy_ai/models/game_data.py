import attr
import numpy as np
from flappy_ai.models.memory_item import MemoryItem
from collections import deque

@attr.s(auto_attribs=True)
class GameData:

    # How many frames of history should we merge into an added frame.
    movement_frames: int = attr.ib(default=4)
    score: int = attr.ib(default=0)
    _memory: deque = attr.ib(default=attr.Factory(list), init=False)

    def total_frames(self) -> int:
        return len(self._memory)

    def append(self, memory_item: MemoryItem):
        # Get our shape.
        x, y = memory_item.state.shape
        # Merge the image history
        # Only works on graytscale atm due to the 1 shape.
        if len(self._memory) < 1:
            # If we dont have any history then just use the current frame
            merged_state = memory_item.state
        else:
            merged_state = np.mean(np.array([x.state for x in self._memory[self.movement_frames*-1:]]), axis=0)

        #import cv2
        #cv2.imwrite(f"tmp/{len(self)}.png", merged_state)
        memory_item.state = np.reshape(memory_item.state, (x, y, 1))
        memory_item.merged_state = np.reshape(merged_state, (x, y, 1))
        self._memory.append(memory_item)

    def __getitem__(self, idx) -> MemoryItem:
        return self._memory[idx]

    def __len__(self):
        return len(self._memory)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
