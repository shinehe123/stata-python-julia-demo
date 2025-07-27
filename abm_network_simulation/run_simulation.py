import networkx as nx
import numpy as np
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import seaborn as sns

# ---- network initialization ----

def initialize_network(n_nodes=30, m_edges=2, seed=None):
    """Generate a Barabasi-Albert scale-free network."""
    return nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)

# ---- agent and model definitions ----
class CityAgent(Agent):
    def __init__(self, model, culture):
        super().__init__(model)
        self.culture = culture

    def step(self):
        neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
        if not neighbors:
            return
        # connect to the neighbor with closest culture value
        best_neighbor = min(
            neighbors,
            key=lambda n: abs(self.culture - n.culture)
        )
        if np.random.random() < 0.3:
            self.model.G.add_edge(self.pos, best_neighbor.pos)

class InnovationNetwork(Model):
    def __init__(self, G):
        super().__init__()
        self.G = G
        self.grid = NetworkGrid(self.G)
        self.agent_list = []
        for node in self.G.nodes():
            culture = np.random.rand()
            agent = CityAgent(self, culture)
            self.agent_list.append(agent)
            self.grid.place_agent(agent, node)
        self.datacollector = DataCollector(
            model_reporters={
                "num_edges": lambda m: m.G.number_of_edges(),
                "clustering": lambda m: nx.average_clustering(m.G)
            }
        )

    def step(self):
        self.datacollector.collect(self)
        for agent in list(self.agent_list):
            agent.step()

# ---- simulation runner ----
def run(model, steps=50):
    for _ in range(steps):
        model.step()
    return model

if __name__ == '__main__':
    G = initialize_network(n_nodes=30, m_edges=2)
    model = InnovationNetwork(G)
    model = run(model, steps=50)
    results = model.datacollector.get_model_vars_dataframe()
    print(results.tail())
    sns.lineplot(data=results, x=results.index, y="num_edges", label="Edges")
    sns.lineplot(data=results, x=results.index, y="clustering", label="Clustering")
    plt.xlabel("Step")
    plt.title("Network Evolution")
    plt.tight_layout()
    plt.show()
