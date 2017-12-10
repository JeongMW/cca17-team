import os, sys
import gym
import numpy as np

from brain import DQNAgent

class Game:
    def __init__(self):
        # TODO: implement additional features. (The code below can be modified.)
        self.env = gym.make("CartPole-v2")

        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.agent = DQNAgent(state_size, action_size)

    def train_games(self, max_episodes):
        # TODO: Implement logics to make agent learn.

        for episode in range(max_episodes):
            print("=========== EPISODE %d ==========" % (episode + 1))

            state = self.env.reset()
            done = False
            step_count = 0

            while not done and step_count<10000:
                action = self.agent.get_action(state, episode)
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -100

                # Append samples for replay
                self.agent.append_sample(state, action, reward, next_state, done)

                next_state = state
                step_count += 1

            # Game is done
            print("episode: {} / steps: {}".format(episode+1, step_count))

            # Learn by every 10 episodes
            if(episode%10 == 1):
                self.agent.train_model()
                self.agent.update_target_model()

    def run_games(self, max_episodes):
        for i in range(max_episodes):
            state = self.env.reset()
            done = False

            while not done and steps<10000:
                self.env.render()
                action = self.agent.get_action(state, 1000000000) # We need get_action w/o random noise
                next_state, reward, done, _ = self.env,step(action)
                state = next_state
            # Game is done
            print("episodes: {} / steps: {}".format(episode, step_count))

if  __name__ == "__main__":
    game = Game()
    game.train_games(100)
