"""Simple Q-learning implementation for the CartPole-v1 environment."""

import argparse
import logging
import random
from collections import deque
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from gymnasium.core import Env
from torch import nn

logger = logging.getLogger(__name__)


class Net(nn.Module):  # noqa: D101
    def __init__(self, input_dim: int = 4, output_dim: int = 2, hidden_layer: int = 256) -> None:  # noqa: D107
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        return self.fc3(x)

    def get_action(self, state: torch.Tensor) -> int:  # noqa: D102
        return torch.argmax(self.forward(state)).item()


class Learner:  # noqa: D101
    def __init__(  # noqa: D107, PLR0913
        self,
        num_iterations: int = 100000,
        memory_size: int = 10000,
        render: bool = False,
        e_rate: float = 1.0,
        e_decay: float = 0.995,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        target_update: int = 100,
        minimum_score: int = 1000,
    ) -> None:
        self.env: Env = gym.make("CartPole-v1", render_mode="human" if render else None)
        self.render = render
        self.num_iterations = num_iterations
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.net = Net()
        self.target_net = Net()  # Target network for stability
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.e_rate = e_rate
        self.e_decay = e_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.update_count = 0
        self.minimum_score = minimum_score

    def learn(self) -> None:  # noqa: D102
        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)  # nosec: B311

        x_train, y_train = [], []

        for state, action, reward, observation, done in mini_batch:
            # Get current Q-values
            current_q_values = self.net.forward(state)

            # Create target Q-values (copy current values)
            target_q_values = current_q_values.clone()

            if not done:
                # Get next state Q-values using target network
                next_q_values = self.target_net.forward(observation)
                # Update target for the action taken using Q-learning update rule
                target_q_values[action] = reward + 0.99 * torch.max(next_q_values)
            else:
                # If episode is done, set target to the actual reward
                target_q_values[action] = reward

            x_train.append(state)
            y_train.append(target_q_values)

        x_batch = torch.stack(x_train)
        y_batch = torch.stack(y_train)

        # Forward pass
        predictions = self.net.forward(x_batch)

        # Calculate MSE loss for Q-learning
        loss = F.mse_loss(predictions, y_batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def run(self) -> None:  # noqa: D102
        for i in range(self.num_iterations):
            """
            1. cart position
            2. cart velocity
            3. pole angle
            4. pole velocity at tip
            """
            state, _ = self.env.reset()
            state = torch.from_numpy(state).float()

            done = False
            score = 0

            while not done:
                if random.uniform(0, 1) > self.e_rate:  # nosec: B311
                    action = self.net.get_action(state)
                else:
                    action = self.env.action_space.sample()

                obs, reward, terminated, _, _ = self.env.step(action)
                observation = torch.from_numpy(obs).float()
                score += reward

                self.memory.append((state, action, reward, observation, terminated))

                if self.render:
                    self.env.render()

                state = observation

                if terminated:
                    break

            if score >= self.minimum_score:
                logger.info("Working model found - saving to disk.")
                torch.save(self.net.state_dict(), "data/model.pth")
                done = True
                break

            # Learn every episode if we have enough samples
            if len(self.memory) >= self.batch_size:
                self.learn()

            # Decay epsilon
            self.e_rate = max(0.01, self.e_rate * self.e_decay)

            logger.info("Attempt %s - Score %s - Memory size %s - Epsilon %s", i, score, len(self.memory), self.e_rate)


class Visualizer:  # noqa: D101
    def __init__(self, model_path: str) -> None:  # noqa: D107
        self.model = Net()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.env = gym.make("CartPole-v1", render_mode="human")

    def run(self) -> None:  # noqa: D102
        state, _ = self.env.reset()
        state = torch.from_numpy(state).float()

        done = False
        score = 0

        while not done:
            action = self.model.get_action(state)
            obs, reward, terminated, _, _ = self.env.step(action)
            observation = torch.from_numpy(obs).float()
            score += reward
            self.env.render()
            state = observation

            if terminated:
                break

        logger.info("Score: %s", score)

    def close(self) -> None:  # noqa: D102
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()
    if args.mode == "train":
        learner = Learner()
        learner.run()
    elif args.mode == "visualize":
        visualizer = Visualizer("data/model.pth")
        visualizer.run()
        visualizer.close()
    else:
        logger.error("Usage: python main.py --mode train")
        logger.error("Usage: python main.py --mode visualize")
