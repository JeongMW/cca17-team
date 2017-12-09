import os, sys
import gym
import numpy as np

from brain import DQNAgent

class Game:
    def __init__(self):
        # TODO: implement additional features. (The code below can be modified.)
        self.env = gym.make("CartPole-v0")

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.agent = DQNAgent(state_size, action_size)

    def run_games(self, max_episodes):
        # TODO: Implement logics to make agent learn.

        for i in range(max_episodes):
            print("=========== EPISODE %d ==========" % (i + 1))

            state = self.env.reset()
            done = False

            while not done:
                self.env.render()
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)

if  __name__ == "__main__":
    game = Game()
    game.run_games(1)
