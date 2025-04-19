import colorsys
import numpy as np

import solara
from model import CulturalModel
from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component,
)

import altair as alt
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6, 5)

# Model parameters
model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "Random Seed"},
    "width": {"type": "SliderInt", "value": 30, "label": "Width", "min": 5, "max": 100, "step": 1},
    "height": {"type": "SliderInt", "value": 30, "label": "Height", "min": 5, "max": 100, "step": 1},
    "density": {"type": "SliderFloat", "value": 0.8, "label": "Population Density", "min": 0, "max": 1, "step": 0.01},
    "desired_similarity": {"type": "SliderFloat", "value": 0.5, "label": "Desired Similarity", "min": 0, "max": 1, "step": 0.01},
    "n_traits": {"type": "SliderInt", "value": 5, "label": "# of Traits", "min": 1, "max": 10, "step": 1},
    "trait_choices": {"type": "SliderInt", "value": 5, "label": "Trait Choices per Dimension", "min": 2, "max": 10, "step": 1},
}

def diversity_post_process(ax):
    line = ax.lines[0]
    x_data, y_data = line.get_xdata(), line.get_ydata()
    if x_data.size <= 1:
        return ax

    last_x, last_y = x_data[-1], y_data[-1]
    ax.text(last_x, last_y, f"{last_y}", fontsize=9,
            ha='center', va='bottom')
    
def mat_post_process(ax):
    for mappable in ax.collections + ax.images:
        mappable.set_clim(vmin=0, vmax=1.1)  # Adjust the color limits
        plt.colorbar(mappable, ax=ax)

def alt_post_process(plot):
    return plot.mark_circle(size=80, opacity=0.95).encode(
        color=alt.Color(
            "color:Q",
            scale=alt.Scale(
                domain=[0, 1],
                scheme="viridis",
            )
        ),
    ).properties(width=300, height=300)

model = CulturalModel()

def agent_portrayal(agent):
    """Portrayal function for CulturalAgent."""
    avg_trait = sum(agent.traits) / (agent.model.n_traits * agent.model.trait_choices)
    return {
        "color": avg_trait,
        "tooltip": f"Traits: {agent.traits}",
    }

HappyPlot = make_plot_component({"share_satisfied": "tab:blue"})
DiversityPlot = make_plot_component({"diversity": "tab:orange"}, post_process=diversity_post_process)
SpaceGraph = make_space_component(agent_portrayal, draw_grid=False, backend="matplotlib", post_process=mat_post_process)
SpaceGraphAltair = make_space_component(agent_portrayal, draw_grid=False, backend="altair", post_process=alt_post_process)

page = SolaraViz(
    model,
    components=[SpaceGraphAltair, HappyPlot, DiversityPlot],
    model_params=model_params,
    name="Cultural Segregation-Assimilation Model",
)

page
