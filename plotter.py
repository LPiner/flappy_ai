import matplotlib.pyplot as plt
from cattr import structure
from structlog import get_logger

from flappy_ai.models.sql_models.saved_episode_result import SavedEpisodeResult
from flappy_ai.models.sql_models.fit_data import FitData
from flappy_ai import Session

logger = get_logger(__name__)

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
    session = Session()
    score_history = session.query(SavedEpisodeResult).all()
    fit_data = session.query(FitData).all()
    score_history = sorted(score_history, key=lambda x: x.episode_number)


    plt.cla()
    axarr[0].plot((range(0, len(fit_data))), [x.loss for x in fit_data])
    axarr[1].plot((range(0, len(fit_data))), [x.accuracy for x in fit_data])
    axarr[2].plot([x.episode_number for x in score_history], [x.score for x in score_history])

    axarr[0].relim()
    axarr[0].autoscale_view(True, True, True)
    axarr[1].relim()
    axarr[1].autoscale_view(True, True, True)
    axarr[2].relim()
    axarr[2].autoscale_view(True, True, True)
    plt.draw()
    plt.pause(60)
