import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import deque


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(4, 128)
        self.hidden_layer1 = nn.Linear(128, 128)
        self.hidden_layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 2)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer1(x)
        x = F.relu(x)
        x = self.hidden_layer2(x)
        x = F.relu(x)
        x = self.output_layer(
            x
        )  # output er Q-verdier, hver verdi er fremtidig expected reward
        # gitt at du i fremtiden tar den beste action at all steps
        return x  # 0 eller 1


class DQN:
    def __init__(self):
        self.policy_network = Network()
        self.target_network = Network()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.eps = 1.0
        self.gamma = 0.99
        self.replay_buffer = deque(maxlen=10000)
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=0.001)

    def get_action(self, x):
        if random.random() < self.eps:
            action = random.randint(0, 1)
        else:
            action = torch.argmax(self.policy_network.forward(x)).item()
        self.update_eps()
        return action

    def update_eps(self):
        self.eps = max(0.1, self.eps * 0.999)

    def get_experience(self, buffer_size=32):
        return random.sample(self.replay_buffer, buffer_size)

    def train(self):
        if len(self.replay_buffer) < 200:
            return

        samples = self.get_experience()
        target_predictions = []
        for state, next_state, reward, done, action in samples:
            if done:
                target_predictions.append(torch.tensor(reward))
            else:
                with torch.no_grad():
                    next_q_values = self.target_network(
                        torch.tensor(next_state)
                    )  # [22, 50]
                    q_value = torch.max(next_q_values)
                    target_predictions.append(reward + 0.99 * q_value)  # [15]

        policy_predictions = []
        for state, next_state, reward, done, action in samples:
            q_values = self.policy_network(torch.tensor(state))  # tensor[x, y]
            policy_predictions.append(q_values[int(action)])

        loss_func = nn.MSELoss()
        loss = loss_func(
            torch.stack(policy_predictions), torch.stack(target_predictions)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
