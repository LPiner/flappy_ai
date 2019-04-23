from flappy_ai.models.fit_data import FitData
import matplotlib.pyplot as plt
from typing import List
from cattr import structure
from structlog import get_logger
import time
import json

logger = get_logger(__name__)


# Plotting Stuff
plt.ion()

f, axarr = plt.subplots(2,1)
f.subplots_adjust(hspace=0.3)
f.suptitle('Results Over Time')

axarr[0].set_title("Loss")
axarr[0].plot([])
axarr[1].set_title("Accuracy")
axarr[1].plot([])
plt.legend()



while True:

    try:
        with open("save/data.json", "r") as file:
            data = json.loads(file.read())
            fit_history: List[FitData] = [structure(x, FitData) for x in data.get("fit_history", [])]
            if fit_history:
                epsilon = fit_history[-1].epsilon

        axarr[0].plot((range(0, len(fit_history))), [x.loss for x in fit_history])
        axarr[1].plot((range(0, len(fit_history))), [x.accuracy for x in fit_history])

        axarr[0].relim()
        axarr[0].autoscale_view(True, True, True)
        axarr[1].relim()
        axarr[1].autoscale_view(True, True, True)
        plt.draw()
    except FileNotFoundError as e:
        logger.warn("Unable to load saved memory.")
    plt.pause(60)

