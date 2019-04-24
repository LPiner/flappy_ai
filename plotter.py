import json
import time
from typing import List

import matplotlib.pyplot as plt
from cattr import structure
from structlog import get_logger

from flappy_ai.models.fit_data import FitData

logger = get_logger(__name__)


# Plotting Stuff
plt.ion()

f, axarr = plt.subplots(3, 1)
f.subplots_adjust(hspace=0.3)
f.suptitle("Results Over Time")

axarr[0].set_title("Loss")
axarr[0].plot([])
axarr[1].set_title("Accuracy")
axarr[1].plot([])
axarr[2].set_title("Score Per Episode")
axarr[2].plot([])
plt.legend()


while True:

    try:
        with open("save/data.json", "r") as file:
            data = json.loads(file.read())
            fit_history: List[FitData] = [structure(x, FitData) for x in data.get("fit_history", [])]
            if fit_history:
                epsilon = fit_history[-1].epsilon

        with open("save/episode_results.json", "r") as file:
            data = json.loads(file.read())
            score_history: List[dict] = data.get("episode_results", [])
            score_history = sorted(score_history, key=lambda x: x["episode"])

        plt.cla()
        axarr[0].plot((range(0, len(fit_history))), [x.loss for x in fit_history])
        axarr[1].plot((range(0, len(fit_history))), [x.accuracy for x in fit_history])
        axarr[2].plot([x["episode"] for x in score_history], [x["score"] for x in score_history])

        axarr[0].relim()
        axarr[0].autoscale_view(True, True, True)
        axarr[1].relim()
        axarr[1].autoscale_view(True, True, True)
        axarr[2].relim()
        axarr[2].autoscale_view(True, True, True)
        plt.draw()
    except FileNotFoundError as e:
        logger.warn("Unable to load saved memory.")
    plt.pause(60)
