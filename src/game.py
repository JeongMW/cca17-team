import os, sys
import numpy as np
import gym

from brain import DQNAgent
from keyboard_input import _Getch
from gym.envs.registration import register

getch = _Getch()
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

        # Parameters for showing training
        self.render = False
        self._show_cutoff_list = [0, 100, 200, 400, 1600]
        self._show_cutoff_idx = 0
        self._show_cntdown = 2
        self._show_cnt = 3

    def learn_games(self, max_episodes, show_training=False):

        for episode in range(max_episodes):
            step_cnt = self._play_game(episode)

            # Game is done
            print("episode: {} / steps: {}".format(episode + 1, step_cnt))

            if show_training and step_cnt > self._show_cutoff_list[self._show_cutoff_idx]:
                self._show_game()

                if self._show_cutoff_idx == 5:
                    break

            if len(self.agent.memory) >= self.agent.train_start_cutoff and (episode + 1) % 10 == 0:
                self._train()

    def _play_game(self, episode):
        state = self.env.reset()
        done = False
        step_cnt = 0

        while not done:
            if self.render:
                self.env.render()

            action = self.agent.get_action(state, episode)
            next_state, reward, done, _ = self.env.step(action)

            if done:
                reward = -100

            # Append samples for replay
            self.agent.append_sample(state, action, reward, next_state, done)

            state = next_state
            step_cnt += 1

        return step_cnt

    def _show_game(self):
        show_cutoff = self._show_cutoff_list[self._show_cutoff_idx]

        if self.render:
            self._show_cnt -= 1

            if self._show_cnt == 0:
                self.render = False
                self._show_cutoff_idx += 1
                self._show_cnt = 3

        else:
            self._show_cntdown -= 1

            if self._show_cntdown == 0:
                self.render = True
                self._show_cntdown = 2

                print("============================================")
                print(" Steps passed over {}. We will self._show 3 times".format(show_cutoff))
                print(" Please press ENTER.")
                print("============================================")

                self.env.reset()
                self.env.render()

                input()

    def _train(self):
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


if __name__ == "__main__":
    game = Game()
    game.learn_games(10000, True)
