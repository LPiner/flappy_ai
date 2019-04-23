from flappy_ai.models.game_process import GameProcess
from typing import List
import time
import multiprocessing
from flappy_ai.models import PredictionRequest, TrainingRequest
from flappy_ai.models.keras_process import KerasProcess
from structlog import get_logger

logger = get_logger(__name__)

MAX_CLIENTS = 5
CLIENTS: List[GameProcess] = []
KERAS_PROCESS = None

# https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
BATCH_SIZE = 32
EPISODES = 10000 # TODO, figure out a optimal number

if __name__ == "__main__":
    KERAS_PROCESS = KerasProcess()
    KERAS_PROCESS.start(batch_size=BATCH_SIZE)
    # Give the keras process time to spin up, load models, etc.
    time.sleep(5)
    COMPLETED_EPISODES= 0
    last_update = time.time()
    TRAINING_REQUESTS: List[TrainingRequest] = []

    while True:
        if not KERAS_PROCESS.is_alive():
            KERAS_PROCESS.start(batch_size=BATCH_SIZE)
            logger.warn("Keras process died....")

        # Calls join on completed processes but does not block. =)
        multiprocessing.active_children()

        for client in CLIENTS:
            if client.parent_pipe and client.parent_pipe.poll():
                # Queue is FIFO
                # we may need to toss data if we get too slow?
                request = client.parent_pipe.recv()
                KERAS_PROCESS.parent_pipe.send(request)
                if isinstance(request, PredictionRequest):
                    client.parent_pipe.send(KERAS_PROCESS.parent_pipe.recv())
                elif isinstance(request, TrainingRequest):
                    # The end result of the session
                    # Currently I consider set of GameData to be a batch size of one.
                    # This may be over training, idk
                    TRAINING_REQUESTS.append(request)
                    COMPLETED_EPISODES += 1

        # Prune off any completed clients
        CLIENTS = [x for x in CLIENTS if x.is_alive()]

        if (time.time() - last_update) / 60 > 5:
            last_update = time.time()
            # Only print updates and save every 5 minutes
            logger.debug("UPDATE", target_episodes=EPISODES, completed_episodes=COMPLETED_EPISODES)

        # Do the batch training after all the clients have completed
        # Maybe I need to abstract the training out to it's own process?
        if not CLIENTS:
            while TRAINING_REQUESTS:
                KERAS_PROCESS.parent_pipe.send(TRAINING_REQUESTS.pop())

        # If we are still below the targets interations, refill the clients and continue
        if COMPLETED_EPISODES >= EPISODES:
            if CLIENTS:
                continue
            else:
                break
        elif COMPLETED_EPISODES < EPISODES and not CLIENTS:
            while len(CLIENTS) < MAX_CLIENTS:
                c = GameProcess()
                CLIENTS.append(c)
                c.start()









