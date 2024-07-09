import gym
from algorithms.ddpg_gym import DDPG

import matplotlib.pyplot as plt

# Create the MountainCar environment
env = gym.make('Pendulum-v1', render_mode="rgb_array")

# Create the DDPG agent
agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 1e-3)

train = True

if train: 
    # Initialize a list to store the rewards
    rewards = []

    # Train the agent
    print("Start training")
    for episode in range(100):
        cnt = 0
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done or terminate:
            cnt += 1
            # env.render()

            if cnt >= 10000:
                break

            action = agent.choose_action(state)
            next_state, reward, done, terminate, _ = env.step(action)
            agent.append_transition(state, action, reward, done, next_state)
            agent.learn()
            state = next_state
            total_reward += reward

        rewards.append(total_reward)  # Store the total reward for this episode
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    # Close the environment
    env.close()

    import os, pathlib
    # Get the current working directory
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "models")
    agent.save_model(model_path)

    # Plot the rewards
    # plt.plot(rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Change in Reward during Training')
    # plt.show()
else:
    import os, pathlib
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "models")
    agent.load_model(model_path)

    for _ in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        cnt = 0

        while not done or terminate:
            cnt += 1
            env.render()

            if cnt >= 1000:
                break

            action = agent.choose_action(state)
            next_state, reward, done, terminate, _ = env.step(action)
            state = next_state
            total_reward += reward

        print(f"Total Reward: {total_reward}")