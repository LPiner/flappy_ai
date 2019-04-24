import time
from multiprocessing import Pipe, Process
from multiprocessing.connection import PipeConnection
from typing import List

import attr
import numpy as np
from structlog import get_logger

from flappy_ai.models import (EpisodeResult, MemoryItem, PredictionRequest,
                              PredictionResult)
from flappy_ai.models.game import Game
from flappy_ai.models.game_data import GameData
from flappy_ai.models.process_base import ProcessBase

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class GameProcess(ProcessBase):
    @staticmethod
    def _process_execute(child_pipe: PipeConnection, *args, force_headless=True, episode_number=None, **kwargs):
        game_data = GameData(episode_number=episode_number)

        with Game(headless=force_headless) as env:

            if child_pipe.poll() and child_pipe.recv() is None:
                # Shutdown request
                return

            loop_times: List[float] = []

            # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
            screen_history: List[np.array] = []
            while not env.game_over():

                # A note for future games, it may be better to skip frames and repeat the last
                # action during that time.
                # We cannot really skip frames here as its already slow to get them.
                start_time = time.time()
                while len(screen_history) < 4:
                    state, reward, done = env.step(0)
                    screen_history.append(state)

                # Each state is a array of the last 4 screens [screen1, 2, 3, 4])
                # This give the network understanding of movement.
                # In this case the network will be expecting an input shape of (4, 160, 120)
                # from there we can use stack to reshape the images into a single image (160, 120, 4)
                # I'm not sure if this works?
                # Maybe i need 4 seperate inputs to the network instead.
                state = np.stack(np.array(screen_history[-4:]), axis=2)

                child_pipe.send(PredictionRequest(data=state))
                start_wait_time = time.time()
                action: PredictionResult = child_pipe.recv()
                wait_time = time.time() - start_wait_time
                if wait_time > 0.02:
                    logger.warn("[GameProcess] Took too long to receive action, tossing game!", wait_time=wait_time)
                    # If we take too long for an action then the states will not line up
                    # So we just toss the game.
                    return
                next_state, reward, done = env.step(action.result)
                screen_history.append(next_state)

                next_state = np.stack(np.array(screen_history[-4:]), axis=2)
                # cv2.imwrite(f"tmp/{game_data.total_frames()}.png", next_state)

                # The reward goes back one memory item since that is the action that created it.
                # same wth the terminal state.
                if len(game_data) > 0:
                    game_data[-1].reward = reward
                    game_data[-1].is_terminal = done
                    game_data[-1].next_state = state

                if done:
                    break

                game_data.score += reward

                # One hot encoding.
                if action.result == 0:
                    taken_action = [1, 0]
                else:
                    taken_action = [0, 1]

                game_data.append(MemoryItem(state=next_state, action=taken_action))

                loop_time = time.time() - start_time
                # if loop_time > .10:
                #    logger.warn("[GameProcess] Took to long to complete loop, tossing game!", loop_time=loop_time)
                #    return
                # Handy to know how long it takes to complete a game.
                loop_times.append(loop_time)

        # Send the session data up to the main process.
        # Do not exit until the data has been read.
        # Exiting before causes the data to be lost.
        child_pipe.send(EpisodeResult(game_data=game_data))
        logger.debug("[GameProcess] Completed.", average_loop_time=np.mean(loop_times))
        while child_pipe.poll():
            time.sleep(1)
