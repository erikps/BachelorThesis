from abc import ABC, abstractmethod

import gymnasium as gym
from tqdm import tqdm

from src.dataset import AttackInferenceDataset
from src.envs.environment import AttackInferenceEnvironment

class Agent(ABC):
    @abstractmethod
    def move(self, observation, environment: AttackInferenceEnvironment):
        pass

class RandomAgent(Agent):
    def move(self, observation, environment):
        return environment.action_space.sample()


class GnnAgent(Agent):
    def move(self, observation, environment):
        pass

def run_experiment(agent: Agent, iterations=500):
    dataset = AttackInferenceDataset()
    environment = AttackInferenceEnvironment(dataset, render_mode="human")
    observation, info = environment.reset()

    for _ in tqdm(range(iterations)):
        action = agent.move(observation, environment)
        observation, reward, terminated, truncated, info = environment.step(action)

        environment.render()

        if terminated:
            print("Found the result.")
            observation, info = environment.reset()

    environment.close()
        

if __name__ == "__main__":
    run_experiment(RandomAgent())
