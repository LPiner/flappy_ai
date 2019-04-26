import time
from multiprocessing.connection import PipeConnection

import attr
import numpy as np
from structlog import get_logger

from flappy_ai.factories.network_factory import network_factory
from flappy_ai.models import EpisodeResult, PredictionRequest, PredictionResult
from flappy_ai.models.process_base import ProcessBase
from flappy_ai.types.network_types import NetworkTypes

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class KerasProcess(ProcessBase):
    @staticmethod
    def _process_execute(child_pipe: PipeConnection, *args, network_type: NetworkTypes = None, **kwargs):

        last_update = time.time()
        AGENT = network_factory(network_type=network_type)
        AGENT.load()

        while True:
            if not child_pipe.poll():
                time.sleep(0.01)
                continue

            request = child_pipe.recv()

            if isinstance(request, PredictionRequest):

                if np.random.rand() <= AGENT._session_epsilon or request.no_random:
                    result = AGENT.predict_random(request.data)
                else:
                    result = AGENT.predict(request.data)

                child_pipe.send(PredictionResult(result=result))

            elif isinstance(request, EpisodeResult):
                for item in request.game_data:
                    AGENT.memory.append(item)
                    if len(AGENT.memory) > AGENT.config.observe_frames_before_learning:
                        AGENT.fit_batch()
                        # logger.debug("[KerasProcess] Fit Batch Complete", runtime=time.time()-start_time, batch_size=batch_size)

            elif request is None:
                AGENT.save()
                # Shutdown request
                if request is None:
                    return

            if (time.time() - last_update) / 60 > 5:
                # Only print updates and save every 5 minutes
                last_update = time.time()
                logger.debug("KERAS PROCESS UPDATE", epsilon=AGENT._session_epsilon, memory_len=len(AGENT.memory))
                # logger.debug("Stats", loss=np.mean(AGENT.loss_history), acc=np.mean(AGENT.acc_history))
                AGENT.save()
