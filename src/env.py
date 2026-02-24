import gymnasium as gym
from DQN import DQN

env = gym.make("CartPole-v1")
model = DQN()

state, _ = env.reset()

for i in range(10000):
    action = model.get_action(state)
    next_state, reward, done, info = env.step(action)
    model.replay_buffer.append([state, next_state, reward, done, action])
    state = next_state
    
    if i % 1000:
        model.update_target()
        
    model.train()