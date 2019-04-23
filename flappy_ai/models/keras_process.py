from multiprocessing.connection import PipeConnection
from flappy_ai.models.game import Game
from flappy_ai.models import TrainingRequest, PredictionRequest, PredictionResult
from flappy_ai.models.process_base import ProcessBase
import numpy as np
import attr
import time
from structlog import get_logger

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class KerasProcess(ProcessBase):

    @staticmethod
    def _process_execute(child_pipe: PipeConnection, *args, batch_size=32, **kwargs):
        from flappy_ai.models.dqn_agent import DQNAgent

        last_update = time.time()
        AGENT = DQNAgent(Game.actions())
        AGENT.load()

        while True:
            if not child_pipe.poll():
                continue

            request = child_pipe.recv()

            if isinstance(request, PredictionRequest):
                result = AGENT.predict(request.data)
                child_pipe.send(PredictionResult(result=result))
            elif isinstance(request, TrainingRequest):
                AGENT.memory.append(request.game_data)

                if len(AGENT.memory) > AGENT.observe_rate:
                    start_time = time.time()
                    AGENT.fit_batch(AGENT.memory.get_sample_batch(batch_size=batch_size))
                    logger.debug("[KerasProcess] Fit Batch Complete", runtime=time.time()-start_time, batch_size=batch_size)

                if AGENT.epsilon > AGENT.epsilon_min and len(AGENT.memory) > AGENT.observe_rate:
                    AGENT.epsilon -= (AGENT.start_epsilon - AGENT.epsilon_min) / AGENT.explore_rate
            elif request is None:
                AGENT.save()
                # Shutdown request
                if request is None:
                    return

            if (time.time() - last_update) / 60 > 5:
                # Only print updates and save every 5 minutes
                last_update = time.time()
                logger.debug("KERAS PROCESS UPDATE", epsilon=AGENT.epsilon, memory_len=len(AGENT.memory))
                logger.debug("Stats", loss=np.mean(AGENT.loss_history), acc=np.mean(AGENT.acc_history))
                AGENT.save()
