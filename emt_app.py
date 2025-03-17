import mesa.datacollection
import solara
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)
import mesa
import random
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class Cell(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        # Cell properties
        self.cell_type = cell_type="E" if random.random() < 0.5 else "M"  # "E" (epithelial) or "M" (mesenchymal)
        
    def say_hi(self):
        print(f"Hi, I am a cell, my ID is {str(self.unique_id)} and I am of type {self.cell_type}")
    
    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        num_e_cells = sum(1 for n in neighbors if n.cell_type == "E")
        num_m_cells = sum(1 for n in neighbors if n.cell_type == "M")

        # EMT: Epithelial to Mesenchymal transition
        if self.cell_type == "E" and num_m_cells > 2:  # More than 2 mesenchymal neighbors
            self.cell_type = "M"

        # MET: Mesenchymal to Epithelial transition
        elif self.cell_type == "M" and num_e_cells > 2:  # More than 2 epithelial neighbors
            self.cell_type = "E"

        # Mesenchymal cells will move with some probability if unsatisfied (too many neighbours)
        elif self.cell_type == "M" and len(neighbors) > 3 and random.random() < 0.3:
            self.move()
    def move(self):
        # Get the adjacent cells
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        # Get the empty cells
        empty_cells = [cell for cell in neighbors if self.model.grid.is_cell_empty(cell)]
        if len(empty_cells) > 0:
            new_position = self.random.choice(empty_cells)
            self.model.grid.move_agent(self, new_position)

class EMTModel(mesa.Model):
    def __init__(self, num_cells, width, height, density=1.0, seed=None):
        super().__init__(seed=seed)
        self.density = density
        self.num_cells = num_cells * density

        # Initialize grid
        self.grid = mesa.space.MultiGrid(width, height, torus=True) # MultiGrid allows multiple agents per cell to start, but after simulation movement allow cells to spread out and tend to one agent per cell

        # Data collection
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "epithelial_count": lambda m: sum(1 for agent in m.agents if agent.cell_type == "E"),
                "mesenchymal_count": lambda m: sum(1 for agent in m.agents if agent.cell_type == "M"),
            },
            agent_reporters={"cell_type": "cell_type"},
        )

        # Create agents
        agents = Cell.create_agents(model=self, n=num_cells)

        # Create x and y positions for agents
        x = self.rng.integers(0, self.grid.width, size=(num_cells,))
        y = self.rng.integers(0, self.grid.height, size=(num_cells,))

        for a, i, j in zip(agents, x, y):
            # Add the agent to a random grid cell
            self.grid.place_agent(a, (i, j))

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


# Solara viz

def agent_portrayal(agent):
    if agent is None:
        return

    # Get agents in the same grid cell
    cellmates = agent.model.grid.get_cell_list_contents([agent.pos])

    # Count E and M cells in this grid cell
    num_e = sum(1 for a in cellmates if a.cell_type == "E")
    num_m = sum(1 for a in cellmates if a.cell_type == "M")
    total = num_e + num_m

    # Compute E/M ratio (0 = all M, 1 = all E)
    e_ratio = num_e / total if total > 0 else 0.5  # Default to neutral if empty

    # Get color from colormap based on ratio
    plasma_palette = sns.color_palette("plasma", as_cmap=True)
    rgba_color = plasma_palette(e_ratio)  
    hex_color = mcolors.to_hex(rgba_color)  # Convert to hex

    return {
        "color": hex_color,  # Assign gradient color
        "size": 25,
        "marker": "o",
    }

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "density": Slider("Cell density", 1.0, 0.1, 1.0, 0.1),
    "num_cells": 100,
    "width": 10,
    "height": 10,
}

model1 = EMTModel(num_cells=100, width=10, height=10)

lineplot_component = make_plot_component(
        {"epithelial_count": "tab:olive", "mesenchymal_count": "tab:purple"},
    )

def get_e_agents(model):
    """Display a text count of how many epithelial and mesenchymal."""
    return solara.Markdown(f"**Epithelial cells: {int(model.num_cells)}** \
                           <br /> \
                           Cells can transition between two states: epithelial (E) and mesenchymal (M). Cells can transition between these states based on the number of neighbors of the opposite type. If at each step a cell has more than 2 neighbors of the opposite type, it will transition to that type. Since mesenchymal cells can move, if a mesenchymal cell has more than 3 neighbors, it will move to an adjacent empty cell with a probability of 0.3.")


page = SolaraViz(
    model1,
    components=[
        make_space_component(agent_portrayal),
        lineplot_component,
        get_e_agents,
    ],
    model_params=model_params,
)
page