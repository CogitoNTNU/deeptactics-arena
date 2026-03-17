import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
print(f"observation_0: {observation}")

episode_over = False
total_reward = 0.0

while not episode_over:
    # tilfeldig handling
    action = env.action_space.sample()
    # unpacker env state fra neste tidssteg etter action
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated
print(f"Episode ferdig, total reward {total_reward}")
env.close()
