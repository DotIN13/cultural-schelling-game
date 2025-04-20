from mesa import Agent

class CulturalAgent(Agent):
    def __init__(self, model, pos, traits):
        super().__init__(model)
        self.traits = traits  # Each agent has a vector of cultural traits (e.g., [2, 0, 4, 1])

    def move(self):
        """Decide whether to move or assimilate based on average similarity to neighbors."""
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        # Compute average similarity to neighbors (proportion of matching traits)
        avg_similarity = 0
        if neighbors:
            similarities = [
                sum(t1 == t2 for t1, t2 in zip(self.traits, n.traits)) / len(self.traits)
                for n in neighbors
            ]
            avg_similarity = sum(similarities) / len(neighbors)

        # If not similar enough to neighbors, move to a new location (as in Schelling)
        if avg_similarity < self.model.desired_similarity:
            self.model.grid.move_to_empty(self)
        else:
            # If similar enough, instead of moving, possibly assimilate (new mechanism)
            self.assimilate(neighbors, similarities)
            self.model.satisfied += 1  # Count as satisfied for this step

    def assimilate(self, neighbors, similarities):
        """Possibly change one of the agent's traits to match a neighbor's, if similar enough."""
        if not neighbors:
            return

        # Randomly select a neighbor to interact with
        partner_index = self.random.choice(range(len(neighbors)))

        interactable = 0  # Track how many neighbors are valid for interaction

        for i, n in enumerate(neighbors):
            sim = similarities[i]
            if sim == 0 or n.traits == self.traits:
                # Skip completely dissimilar or identical neighbors
                continue

            interactable += 1

            if i == partner_index:
                # From the chosen partner, select a differing trait and adopt it
                differing_indices = [
                    i for i, (a, b) in enumerate(zip(self.traits, n.traits)) if a != b
                ]
                if differing_indices:
                    chosen_index = self.random.choice(differing_indices)
                    self.traits[chosen_index] = n.traits[chosen_index]

        # If no neighbors are suitable for interaction, mark as idle
        if interactable == 0:
            self.model.idle += 1
