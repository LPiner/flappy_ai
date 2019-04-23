from multiprocessing import Pipe, Process
from multiprocessing.connection import PipeConnection
from typing import List
from flappy_ai.models.game import Game
from flappy_ai.models.game_data import GameData
from flappy_ai.models import TrainingRequest, PredictionRequest, MemoryItem, PredictionResult
from flappy_ai.models.process_base import ProcessBase
import numpy as np
import attr
import time
from structlog import get_logger


logger = get_logger(__name__)




@attr.s(auto_attribs=True)
class GameProcess(ProcessBase):

    @staticmethod
    def _process_execute(child_pipe: PipeConnection, *args, force_headless=True, **kwargs):
        game_data = GameData()

        with Game(headless=force_headless) as env:

            if child_pipe.poll() and child_pipe.recv() is None:
                # Shutdown request
                return

            loop_times: List[float] = []
            while not env.game_over():
                # A note for future games, it may be better to skip frames and repeat the last
                # action during that time.
                start_time = time.time()
                if not game_data.score:
                    state, reward, done = env.step(0)

                    item = MemoryItem(
                        state=state,
                        action=[1, 0],
                    )
                    game_data.append(item)

                child_pipe.send(
                    PredictionRequest(data=np.array(game_data[-1].state))
                )
                start_wait_time = time.time()
                action: PredictionResult = child_pipe.recv()
                wait_time = time.time() - start_wait_time
                if wait_time > .02:
                    logger.warn("[GameProcess] Took too long to receive action, tossing game!", wait_time=wait_time)
                    # If we take too long for an action then the states will not line up
                    # So we just toss the game.
                    return
                #action = agent.act(np.array(game_data[-1].state))
                next_state, reward, done = env.step(action.result)
                #cv2.imwrite(f"tmp/{game_data.total_frames()}.png", next_state)

                # The reward goes back one memory item since that is the action that created it.
                # same wth the terminal state.
                game_data[-1].reward = reward
                game_data[-1].is_terminal = done

                game_data.score += reward

                # One hot encoding.
                if action.result == 0:
                    taken_action = [1, 0]
                else:
                    taken_action = [0, 1]

                game_data.append(
                    MemoryItem(
                        state=next_state,
                        action=taken_action,
                    )
                )

                loop_time = time.time() - start_time
                if loop_time > .10:
                    logger.warn("[GameProcess] Took to long to complete loop, tossing game!", loop_time=loop_time)
                    return
                # Handy to know how long it takes to complete a game.
                loop_times.append(loop_time)

        # Send the session data up to the main process.
        # Do not exit until the data has been read.
        # Exiting before causes the data to be lost.
        child_pipe.send(TrainingRequest(game_data=game_data))
        logger.debug("[GameProcess] Completed.", average_loop_time=np.mean(loop_times))
        while child_pipe.poll():
            time.sleep(1)

