import gym

env = gym.make("MountainCarContinuous-v0")

def main():
    max_episodes = 50

    for episode in range(max_episodes):
        
        state = env.reset()
        done = False

        while not done:
            env.render()
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

if  __name__ == "__main__":
    main()
