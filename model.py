import numpy as np

from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from agents import CulturalAgent

def max_std(n_traits, trait_choices):
    """Calculate the maximum standard deviation possible for a trait vector of given length and value range.
    
    This is used to normalize trait-based visualizations (e.g., color mapping).
    Max std occurs when half traits are 0 and half are max value.
    """
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

        # Used for scaling or coloring agent traits in visualization (e.g., based on std dev of trait vector)
        self.max_std = max_std(n_traits, trait_choices)

        # Single-grid spatial layout, toroidal (agents wrap around edges)
        self.grid = SingleGrid(width, height, torus=True)
        
        self.satisfied = 0  # Count of agents who are satisfied (i.e., not moving)
        self.idle = 0       # Count of agents who neither moved nor changed traits

        # Populate grid with agents
        for _, pos in self.grid.coord_iter():
            if self.random.random() < self.density:
                traits = [self.random.randint(0, trait_choices - 1) for _ in range(n_traits)]
                agent = CulturalAgent(self, pos, traits)
                self.grid.place_agent(agent, pos)

        # Collectors to track model-level metrics over time
        self.datacollector = DataCollector(
            model_reporters={
                "share_satisfied": lambda m: (m.satisfied / len(m.agents)) * 100 if m.agents else 0,
                "diversity": lambda m: m.compute_diversity(),
            },
        )
        self.datacollector.collect(self)

    def compute_diversity(self):
        """Compute the number of unique trait combinations across all agents."""
        unique_traits = {tuple(agent.traits) for agent in self.agents}
        return len(unique_traits) / len(self.agents) if self.agents else 0

    def step(self):
        """Advance the model by one step: let agents decide to move or assimilate."""
        self.idle = 0       # Reset idle count
        self.satisfied = 0  # Reset satisfied count

        # Agents execute their `move()` behavior (which may include assimilation or relocation)
        self.agents.shuffle_do("move")

        # Collect updated statistics
        self.datacollector.collect(self)

        # Stop if all agents are idle (i.e., no further changes in location or traits)
        self.running = self.idle != len(self.agents)
