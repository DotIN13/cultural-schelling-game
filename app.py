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

# Interactive model parameter sliders and inputs
model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "Random Seed"},
    "width": {"type": "SliderInt", "value": 30, "label": "Width", "min": 5, "max": 100, "step": 1},
    "height": {"type": "SliderInt", "value": 30, "label": "Height", "min": 5, "max": 100, "step": 1},
    "density": {"type": "SliderFloat", "value": 0.8, "label": "Population Density", "min": 0, "max": 1, "step": 0.01},
    "desired_similarity": {"type": "SliderFloat", "value": 0.5, "label": "Desired Similarity", "min": 0, "max": 1, "step": 0.01},
    "n_traits": {"type": "SliderInt", "value": 5, "label": "# of Traits", "min": 1, "max": 10, "step": 1},
    "trait_choices": {"type": "SliderInt", "value": 5, "label": "Trait Choices per Dimension", "min": 2, "max": 15, "step": 1},
}

# Annotate last value in diversity plot
def diversity_post_process(ax):
    line = ax.lines[0]
    x_data, y_data = line.get_xdata(), line.get_ydata()
    if x_data.size <= 1:
        return ax

    last_x, last_y = x_data[-1], y_data[-1]
    ax.text(last_x, last_y, f"{last_y:.2f}", fontsize=9, ha='center', va='bottom')

# Set color scale limits for matplotlib grid view
def mat_post_process(ax):
    for mappable in ax.collections + ax.images:
        mappable.set_clim(vmin=0, vmax=1.1)
        plt.colorbar(mappable, ax=ax)

# Style Altair plot (e.g., for grid coloring)
def alt_post_process(plot):
    return plot.mark_circle(size=80, opacity=0.95).encode(
        color=alt.Color(
            "color:Q",
            scale=alt.Scale(domain=[0, 1], scheme="viridis"),
        ),
    ).properties(width=350, height=350)

# Create default model instance
model = CulturalModel()

# Agent visualization logic (color based on average trait value)
def agent_portrayal(agent):
    avg_trait = sum(agent.traits) / (agent.model.n_traits * agent.model.trait_choices)
    return {
        "color": avg_trait,
        "tooltip": f"Traits: {agent.traits}",
    }

# Plot components
HappyPlot = make_plot_component({"share_satisfied": "tab:blue"})
DiversityPlot = make_plot_component({"diversity": "tab:orange"}, post_process=diversity_post_process)
SpaceGraph = make_space_component(agent_portrayal, draw_grid=False, backend="matplotlib", post_process=mat_post_process)
SpaceGraphAltair = make_space_component(agent_portrayal, draw_grid=False, backend="altair", post_process=alt_post_process)

# Compose Solara page
page = SolaraViz(
    model,
    components=[SpaceGraphAltair, HappyPlot, DiversityPlot],
    model_params=model_params,
    name="Cultural Segregation-Assimilation Model",
)

page
