import os, sys
import numpy as np

from brain import DQNAgent
from keyboard_input import _Getch
getch = _Getch()

import gym
from gym.envs.registration import register

register(
    id = 'CartPole-v2',
    entry_point = 'gym.envs.classic_control:CartPoleEnv',
    tags = {'wrapper_config.TimeLimit.max_episode_steps': 10000},
    reward_threshold = 10000.0,
)


class Game:
    def __init__(self):
        self.env = gym.make("CartPole-v2")

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.agent = DQNAgent(state_size, action_size)

    def train_games(self, max_episodes):
        render = False

        for episode in range(max_episodes):

            state = self.env.reset()
            done = False
            step_count = 0

            while not done:
                if render:
                    self.env.render()

                action = self.agent.get_action(state, episode)
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -100

                # Append samples for replay
                self.agent.append_sample(state, action, reward, next_state, done)

                state = next_state
                step_count += 1

            # Game is done
            print("episode: {} / steps: {}".format(episode+1, step_count))
            if(len(self.agent.memory) >= self.agent.train_start_cutoff) and episode%10==9:
                render = (input("Want to render? y/[n]")=='y')
                for _ in range(50):
                    self.agent.train_model()
                self.agent.update_target_model()

    def teach_games(self, max_episodes):
        for episode in range(max_episodes):

            state = self.env.reset()
            done = False
            step_count = 0

            while not done and step_count < 10000:
                self.env.render()
                key_input = {
                        'j': 0,
                        'l': 1 }

                action = key_input[getch()]
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -100

                # Append samples for replay
                self.agent.append_sample(state, action, reward, next_state, done)

                state = next_state
                step_count += 1
            print("episode: {} / steps: {}".format(episode+1, step_count))

            # Teach is done then learn hard
        for i in range(500):
            for j in range(50):
                self.agent.train_model()
            self.agent.update_target_model()
            if i%50==0: print("learning")

    def run_games(self, max_episodes):
        for episode in range(max_episodes):
            state = self.env.reset()
            done = False

            while not done:
                self.env.render()
                action = self.agent.get_action(state, episode, False)  # Get action without random noise
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
            # Game is done
            print("episodes: {} / steps: {}".format(episode, step_count))

if  __name__ == "__main__":
    game = Game()
    game.train_games(1500)
