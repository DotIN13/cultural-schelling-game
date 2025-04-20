from model import CulturalModel
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

# Parameter grid
densities = [0.1, 0.3, 0.5, 0.8]
desired_similarities = [0.1, 0.2, 0.3, 0.5, 0.9]
n_traits = [3, 5, 7, 10, 15]
trait_choices = [3, 5, 7, 10, 15]

param_grid = list(product(densities, desired_similarities, n_traits, trait_choices))
n_repeats = 5  # Number of runs per setup

# Expand grid for repeats with setup_id and run_id
expanded_grid = [
    (setup_id, run_id, params)
    for setup_id, params in enumerate(param_grid)
    for run_id in range(n_repeats)
]

# Simulation runner
def run_simulation(job):
    setup_id, run_id, (density, similarity, n_traits, trait_choices) = job
    seed = setup_id * 1000 + run_id  # Unique, repeatable seed

    model = CulturalModel(
        density=density,
        desired_similarity=similarity,
        n_traits=n_traits,
        trait_choices=trait_choices,
        seed=seed
    )

    step_count = 0
    while model.running and step_count < 10000:
        model.step()
        step_count += 1

    df = model.datacollector.get_model_vars_dataframe()
    final_diversity = df["diversity"].iloc[-1] if not df.empty else None
    final_satisfied = df["share_satisfied"].iloc[-1] if not df.empty else None

    return {
        "setup_id": setup_id,
        "run_id": run_id,
        "seed": seed,
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
        results = list(tqdm(pool.imap_unordered(run_simulation, expanded_grid), total=len(expanded_grid)))

    df = pd.DataFrame(results)
    print(df.head())
    df.to_csv("grid_search_results.csv", index=False)
