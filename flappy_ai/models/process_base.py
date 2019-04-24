import atexit
import time
from multiprocessing import Pipe, Process
from multiprocessing.connection import PipeConnection

import attr
import numpy as np

from flappy_ai.models import (EpisodeResult, MemoryItem, PredictionRequest,
                              PredictionResult)
from flappy_ai.models.game import Game
from flappy_ai.models.game_data import GameData


@attr.s(auto_attribs=True)
class ProcessBase:
    parent_pipe: PipeConnection = None  # data into the process
    child_pipe: PipeConnection = None  # data out from the process
    _child_process: int = None

    def __attrs_post_init__(self):
        self.parent_pipe, self.child_pipe = Pipe()

        # Ensure our child process is killed
        atexit.register(self.cleanup)

    @staticmethod
    def _process_execute(child_pipe: PipeConnection, *args, **kwargs):
        raise NotImplementedError

    def start(self, *args, **kwargs):
        """
        Please note that any changes you make in the _process_execute function will be used live.
        Every time a new process is opened the module is imported again. This allows you to
        make changes to the clients while the main process is still running.
        """
        self._child_process: Process = Process(target=self._process_execute, args=(self.child_pipe,), kwargs=kwargs)
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
        timeout_sec = 10
        p_sec = 0
        for second in range(timeout_sec):
            if self._child_process.is_alive():
                time.sleep(1)
            p_sec += 1
            if p_sec >= timeout_sec:
                self._child_process.kill()
