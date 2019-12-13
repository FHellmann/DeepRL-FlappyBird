import os
import os.path
import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DQLearningAgent:

    def __init__(self, state_size, nr_actions):
        self.dense_dimension = 64
        self.weight_backup = "flappy_weight.h5"
        self.state_size = state_size
        self.nr_actions = nr_actions
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 0.10
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.nr_actions * self.dense_dimension * 2, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(self.nr_actions * self.dense_dimension, activation='relu'))
        model.add(Dense(self.nr_actions))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.epsilon = self.min_epsilon

        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def policy(self, state):
        return np.argmax(self.model.predict(self._expand_state(state), batch_size=1))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(self._expand_state(next_state))[0])
            target_f = self.model.predict(self._expand_state(state))
            target_f[0][action] = target
            self.model.fit(self._expand_state(state), target_f, epochs=1, verbose=0)

    @staticmethod
    def _expand_state(state):
        return np.expand_dims(state, axis=0)
