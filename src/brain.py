import sys
import numpy as np
import keras.backend as K

from collections import deque
from keras.layers import Dense
from keras.models import Sequential

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

        # Make the train model and the target model
        self.main_model = self.build_model()
        self.target_model = self.build_model()

        # Initialize target model
        self.update_target_model()

    def build_model(self):
        # TODO
        print('Build a model for the learning agent')

    def update_target_model(self):
        # TODO
        print('Update the target model')

    def get_action(self, state):
        # TODO
        print('Choose an action by using our model')

    def append_sample(self, state, action, reward, next_state, done):
        # TODO
        print('Append samples which used for training our model')

    def train_model(self):
        # TODO
        print('Train our model by using samples')