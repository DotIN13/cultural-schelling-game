import numpy as np

from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from agents import CulturalAgent

def max_std(n_traits, trait_choices):
    """Calculate the maximum standard deviation for traits."""
    return np.std([0] * (n_traits // 2) + [trait_choices - 1] * (n_traits - (n_traits // 2)))

class CulturalModel(Model):
    def __init__(self, width=30, height=30, density=0.8, desired_similarity=0.5, n_traits=5, trait_choices=5, seed=None):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.density = density
        self.desired_similarity = desired_similarity
        self.n_traits = n_traits
        self.trait_choices = trait_choices
        self.max_std = max_std(n_traits, trait_choices) # For visualization purposes

        self.grid = SingleGrid(width, height, torus=True)
        self.satisfied = 0
        self.idle = 0

        for _, pos in self.grid.coord_iter():
            if self.random.random() < self.density:
                traits = [self.random.randint(0, trait_choices - 1) for _ in range(n_traits)]
                agent = CulturalAgent(self, pos, traits)
                self.grid.place_agent(agent, pos)

        self.datacollector = DataCollector(
            model_reporters={
                "share_satisfied": lambda m: (m.satisfied / len(m.agents)) * 100 if m.agents else 0,
                "diversity": lambda m: m.compute_diversity(),
            },
        )
        self.datacollector.collect(self)
        
    def compute_diversity(self):
        unique_traits = {tuple(agent.traits) for agent in self.agents}
        return len(unique_traits)

    def step(self):
        self.idle = 0
        self.satisfied = 0
        self.agents.shuffle_do("move")
        self.datacollector.collect(self)
        self.running = self.idle != len(self.agents)  # Continue if not all agents are idle
