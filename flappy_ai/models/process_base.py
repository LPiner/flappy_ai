from multiprocessing import Pipe, Process
from multiprocessing.connection import PipeConnection
from flappy_ai.models.game import Game
from flappy_ai.models.game_data import GameData
from flappy_ai.models import TrainingRequest, PredictionRequest, MemoryItem, PredictionResult
import numpy as np
import attr
import time
import atexit


@attr.s(auto_attribs=True)
class ProcessBase:
    parent_pipe: PipeConnection = None # data into the process
    child_pipe: PipeConnection = None # data out from the process
    _child_process: int = None

    def __attrs_post_init__(self):
        self.parent_pipe, self.child_pipe = Pipe()

        # Ensure our child process is killed
        atexit.register(self.cleanup)

    @staticmethod
    def _process_execute(child_pipe: PipeConnection):
        raise NotImplementedError

    def start(self):
        self._child_process: Process = Process(target=self._process_execute, args=(self.child_pipe, ))
        self._child_process.start()

    def has_started(self) -> bool:
        return self._child_process is not None and self._child_process.is_alive()

    def is_alive(self) -> bool:
        return self._child_process is not None and self._child_process.is_alive()

    def is_completed(self) -> bool:
        return self._child_process is not None and not self._child_process.is_alive()

    def cleanup(self):
        if self._child_process is None:
            return
        if self.parent_pipe:
            # Processes upon receving None should quit.
            self.parent_pipe.send(None)
        timeout_sec = 5
        p_sec = 0
        for second in range(timeout_sec):
            if self._child_process.is_alive():
                time.sleep(1)
            p_sec += 1
            if p_sec >= timeout_sec:
                self._child_process.kill()