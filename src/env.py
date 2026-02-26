import gymnasium as gym
from DQN import DQN 
import torch

env = gym.make("CartPole-v1")
state, _ = env.reset()
model = DQN()


for i in range(10000):
    state = torch.tensor(state)
    action = model.get_action(state)
    next_state, reward, done, trunctated, info = env.step(int(action))
    model.replay_buffer.append([state, next_state, reward, done, action])
    state = next_state

    if trunctated or done:
        env.reset()

    if i % 1000 == 0:
        model.update_target()

    model.train()