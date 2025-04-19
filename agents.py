from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np

class CulturalAgent(Agent):
    def __init__(self, model, pos, traits):
        super().__init__(model)
        self.traits = traits

    def move(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        # Compute similarity once
        avg_similarity = 0
        if neighbors:
            similarities = [sum(t1 == t2 for t1, t2 in zip(self.traits, n.traits)) / len(self.traits) for n in neighbors]
            avg_similarity = sum(similarities) / len(neighbors)            

        if avg_similarity < self.model.desired_similarity:
            self.model.grid.move_to_empty(self)
        else:
            self.assimilate(neighbors, similarities)
            self.model.satisfied += 1

    def assimilate(self, neighbors, similarities):
        if not neighbors:
            return

        # Randomly choose a neighbor index
        partner_index = self.random.choice(range(len(neighbors)))
        
        interactable = 0
        for i, n in enumerate(neighbors):
            sim = similarities[i]
            if sim == 0 or n.traits == self.traits:
                continue
            
            interactable += 1
            
            if i == partner_index:
                differing_indices = [i for i, (a, b) in enumerate(zip(self.traits, n.traits)) if a != b]
                if differing_indices:
                    chosen_index = self.random.choice(differing_indices)
                    self.traits[chosen_index] = n.traits[chosen_index]
        
        if interactable == 0:
            self.model.idle += 1
