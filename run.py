from model import CulturalModel
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

# Parameter grid
densities = [0.1, 0.5, 0.8]
desired_similarities = [0.1, 0.2, 0.3, 0.5]
n_traits = [3, 5, 7]
trait_choices = [3, 5, 7, 10]

param_grid = list(product(densities, desired_similarities, n_traits, trait_choices))

# Simulation runner
def run_simulation(params):
    density, similarity, n_traits, trait_choices = params
    model = CulturalModel(
        density=density,
        desired_similarity=similarity,
        n_traits=n_traits,
        trait_choices=trait_choices
    )

    step_count = 0
    while model.running and step_count < 10000:
        model.step()
        step_count += 1

    df = model.datacollector.get_model_vars_dataframe()
    final_diversity = df["diversity"].iloc[-1] if not df.empty else 0
    final_satisfied = df["share_satisfied"].iloc[-1] if not df.empty else 0

    return {
        "density": density,
        "desired_similarity": similarity,
        "n_traits": n_traits,
        "trait_choices": trait_choices,
        "final_diversity": final_diversity,
        "final_satisfied": final_satisfied,
        "steps": step_count
    }

# Run with progress bar
if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(run_simulation, param_grid), total=len(param_grid)))

    df = pd.DataFrame(results)
    print(df.head())
    df.to_csv("grid_search_results.csv", index=False)
