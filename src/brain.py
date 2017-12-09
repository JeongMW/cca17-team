import sys, random
import numpy as np
import keras.backend as K

from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Parameters about the game (CartPole)
        self.state_size = state_size
        self.action_size = action_size

        # Parameters for DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.1
        self.batch_size = 64

        # Replay memory
        self.memory = deque(maxlen=2000)

        # Make the main train model and the target model
        self.main_model = self.build_model()
        self.target_model = self.build_model()

        # Initialize target model
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        model.add(Dense(16, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='glorot_normal'))

        model.summary()  # Print information about the model

        opt = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def get_action(self, state, episode):
        return np.argmax(self.main_model.predict(state) + np.random.randn(1, self.action_size) / (episode / 5 + 1))

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        Q_values = self.main_model.predict(states)
        target_Q = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                Q_values[i][actions[i]] = rewards[i]
            else:
                Q_values[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_Q[i])

        self.main_model.fit(states, Q_values, batch_size=self.batch_size, epoch=1, verbose=0)

