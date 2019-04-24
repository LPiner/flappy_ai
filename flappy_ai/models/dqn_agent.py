import json
import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cattr import structure, unstructure
from keras.callbacks import TensorBoard
from keras.layers import (BatchNormalization, Conv2D, Dense, Flatten, Input,
                          Lambda)
from keras.models import Sequential
from keras.optimizers import RMSprop
from structlog import get_logger

from flappy_ai.models.fit_data import FitData
from flappy_ai.models.game import Game
from flappy_ai.models.game_data import GameData
from flappy_ai.models.game_history import GameHistory
from flappy_ai.models.memory_item import MemoryItem

logger = get_logger(__name__)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

EPISODES = 100000


class DQNAgent:
    def __init__(self, action_size):
        self.data_shape = (160, 120, 4)
        self.action_size = action_size
        self.memory = GameHistory(size=100000)
        self.gamma = 0.99  # discount rate
        # self.epsilon = 1.0  # exploration rate
        """
        E is best ot start at 1 but we dont want the bird to flap too much.
        # see https://github.com/yenchenlin/DeepLearningFlappyBird
        """
        """
        In these experiments, we used the RMSProp algorithm with minibatches of size 32.  
        The behaviorpolicy during training was-greedy withannealed linearly from1to0.
        1over the first million frames, and fixed at0.1thereafter. 
         We trained for a total of10million frames and used a replay
         memory of one million most recent frames.
        """
        self.start_epsilon = 1
        self.epsilon = self.start_epsilon
        self.epsilon_min = 0.1
        self.explore_rate = 10000  # Games before e becomes e min.
        self.observe_rate = 500  # games before we start training
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.fit_history: List[FitData] = []

    def _build_model(self):

        # Deepmind paper on their atari breakout agent.
        # https://arxiv.org/pdf/1312.5602v1.pdf

        # With the functional API we need to define the inputs.
        frames_input = Input(self.data_shape, name="frames")

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = Lambda(lambda x: x / 255.0)(frames_input)

        model = Sequential()
        model.add(BatchNormalization(input_shape=self.data_shape))
        model.add(Conv2D(16, 8, strides=(4, 4), padding="valid", activation="relu"))
        model.add(Conv2D(32, 4, strides=(2, 2), padding="valid", activation="relu"))
        # model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_size))
        # Info on opts
        # http://ruder.io/optimizing-gradient-descent/

        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])

        return model

    def predict(self, state):
        """
        Note for later, predict expects and returns an array of items.
        so it wants an array of states, even if we only have 1 it still needs to be in a shape of (1, x, x, x)
        https://stackoverflow.com/questions/41563720/error-when-checking-model-input-expected-convolution2d-input-1-to-have-4-dimens
        state.shape
        (159, 81, 1)
        np.expand_dims(state, axis=0).shape
        (1, 159, 81, 1)
        """
        act_values = self.model.predict(np.expand_dims(state, axis=0))
        # act_values -> array([[ -3.0126321, -11.75323  ]], dtype=float32)

        random_action = False
        if np.random.rand() <= self.epsilon:
            random_action = True
            action = random.randrange(2)
        else:
            action = np.argmax(act_values[0])  # returns action

        # logger.debug("[act]", predicted_actions=act_values.tolist(), using_random_action=random_action, chosen_action=action)

        return action

    # def fit_batch(self, start_states, actions, rewards, next_states, is_terminal):
    def fit_batch(self, batch_items: List[MemoryItem]):
        """Do one deep Q learning iteration.

        Params:
        - model: The DQN
        - gamma: Discount factor (should be 0.99)
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal

        If yoy wish to see the images at this level you can use
        plt.imshow(start_states[0][:,:,0], cmap=plt.cm.binary)
        though the colors will be fucked
        """
        start_states = np.array([x.state for x in batch_items])
        actions = np.array([x.action for x in batch_items])
        rewards = np.array([x.reward for x in batch_items])
        next_states = np.array([x.next_state for x in batch_items])
        is_terminal = np.array([x.is_terminal for x in batch_items])
        # start_states = np.array([x.merged_state for x in game_data][:-1])

        # First, predict the Q values of the next states.
        next_Q_values = self.model.predict(next_states)
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        # tensorboard = TensorBoard(log_dir=f"logs/")
        history = self.model.fit(
            x=start_states,
            y=actions * Q_values[:, None],
            # epochs=1, batch_size=len(start_states), verbose=0, callbacks=[tensorboard]
            epochs=1,
            batch_size=len(start_states),
            verbose=0,
        )

        self.fit_history.append(
            FitData(epsilon=self.epsilon, loss=history.history["loss"][0], accuracy=history.history["acc"][0])
        )

    def load(self):
        try:
            self.model.load_weights("save/flappy.h5")
        except OSError as e:
            logger.warn("Unable to load saved weights.")

        try:
            with open("save/data.json", "r") as file:
                data = json.loads(file.read())
                self.fit_history: List[FitData] = []
                for item in data.get("fit_history", []):
                    self.fit_history.append(structure(item, FitData))

                if self.fit_history:
                    self.epsilon = self.fit_history[-1].epsilon
        except FileNotFoundError as e:
            logger.warn("Unable to load saved memory.")

    def save(self):
        self.model.save_weights("save/flappy.h5")
        with open("save/data.json", "w+") as file:
            file.write(json.dumps({"fit_history": unstructure(self.fit_history)}))
