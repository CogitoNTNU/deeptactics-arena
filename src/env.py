import gymnasium as gym
import torch
from DQN import DQN

env = gym.make("CartPole-v1", render_mode="human")
model = DQN()

state, _ = env.reset()

for i in range(10000):
    state = torch.tensor(state)
    action = model.get_action(state)
    next_state, reward, done, trunctuated, info = env.step(int(action))
    model.replay_buffer.append([state, next_state, reward, done, action])
    state = next_state
    
    if trunctuated or done:
        env.reset()
    
    if i % 1000 == 0:
        model.update_target()
        
    env.render()
        
    model.train()