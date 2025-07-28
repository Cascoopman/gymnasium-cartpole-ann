import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim.adam
from gymnasium.core import Env
from torch import Tensor, nn
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 2, hidden_layer: int = 64):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return F.softmax(x, dim=0)

    def get_action(self, state):
        return torch.argmax(self.forward(state)).item()

    def backward(self, x_train: Tensor, target: Tensor, learning_rate):
        self.forward(x_train)

        loss = F.cross_entropy(input, target)
        loss = loss.sum()

        loss.backward()

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

        optim.step()


class Learner:
    def __init__(
        self,
        num_iterations: int = 200,
        memory_size: int = 2000,
        render: bool = False,
        e_rate: float = 0.80,
        e_decay: float = 0.90,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        self.env: Env = gym.make("CartPole-v1", render_mode="human" if render else None)
        self.render = render
        self.num_iterations = num_iterations
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.net = Net()
        self.e_rate = e_rate
        self.e_decay = e_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def learn(self) -> None:
        mini_batch = random.sample(self.memory, min(len(self.memory)), self.batch_size)

        raise NotImplementedError

        # training_data = []

        # for state, action, observation, done in mini_batch:
        #     break
        # test = DataLoader(self.memory, batch_size=self.batch_size)

        # self.net.backward(mini_batch, test, self.learning_rate)
        # self.e_rate = self.e_rate * self.e_decay
        # self.memory.clear()

    def run(self) -> None:
        for i in range(self.num_iterations):
            """
            1. cart position
            2. cart velocity
            3. pole angle
            4. pole velocity at tip
            """
            state, _ = self.env.reset()
            state = torch.from_numpy(state)

            done = False
            score = 0

            while not done:
                if random.uniform(0, 1) > self.e_rate:  # nosec: B311
                    action = self.net.get_action(state)
                else:
                    action = self.env.action_space.sample()

                obs, reward, terminated, _, _ = self.env.step(action)

                score += reward

                self.memory.append((state, action, obs, done))

                if self.render:
                    self.env.render()

                state = torch.from_numpy(obs)

                if score == 500:
                    print("Working model found - saving to disk.")
                    torch.save(self.net.state_dict(), "data")
                    done = True
                    break

                if terminated:
                    break

            if len(self.memory) == self.memory_size:
                self.learn()

            print(
                "Attempt ", i, " - Score ", score, " - Memory size ", len(self.memory)
            )


if __name__ == "__main__":
    learner = Learner(render=False)
    learner.run()
