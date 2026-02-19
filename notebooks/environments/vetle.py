from pettingzoo.classic import chess_v6 as kaz
import random

env = kaz.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(observation.keys())
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
        print([i for i in range(len(observation["action_mask"])) if observation["action_mask"][i] == 1])
        action = random.choice([i for i in range(len(observation["action_mask"])) if observation["action_mask"][i] == 1])
    
    env.step(action)
    print(info)

env.close()