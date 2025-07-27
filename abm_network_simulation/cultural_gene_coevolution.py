import networkx as nx
import numpy as np

class City:
    def __init__(self, idx, traits):
        self.idx = idx
        self.traits = np.array(traits, dtype=float)


def cultural_distance(a, b):
    """Euclidean distance between two trait vectors."""
    diff = a - b
    return np.sqrt(np.sum(diff ** 2))


def link_probability(dist, lambd=0.8, alpha=2.0):
    """Probability of forming a link given cultural distance."""
    p = lambd * np.exp(-alpha * dist)
    return min(1.0, p)


def interact(city_a, city_b, rate=0.5):
    """Cities interact and converge on a random trait."""
    if np.random.rand() > rate:
        return
    diffs = np.where(city_a.traits != city_b.traits)[0]
    if len(diffs) == 0:
        return
    f = np.random.choice(diffs)
    # simple averaging toward each other
    val = (city_a.traits[f] + city_b.traits[f]) / 2.0
    city_a.traits[f] = val
    city_b.traits[f] = val


def mutate(city, rate=0.02):
    if np.random.rand() < rate:
        f = np.random.randint(len(city.traits))
        city.traits[f] = np.random.rand()


class InnovationNetwork:
    def __init__(self, n_cities=20, n_traits=5, seed=None):
        self.rng = np.random.default_rng(seed)
        self.graph = nx.Graph()
        self.cities = {}
        for i in range(n_cities):
            traits = self.rng.random(n_traits)
            city = City(i, traits)
            self.cities[i] = city
            self.graph.add_node(i)

    def step(self):
        nodes = list(self.graph.nodes)
        # link formation among unconnected pairs
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.graph.has_edge(nodes[i], nodes[j]):
                    continue
                d = cultural_distance(self.cities[nodes[i]].traits,
                                       self.cities[nodes[j]].traits)
                p = link_probability(d)
                if self.rng.random() < p:
                    self.graph.add_edge(nodes[i], nodes[j])

        # dissolve edges if cities diverged
        for (u, v) in list(self.graph.edges()):
            d = cultural_distance(self.cities[u].traits, self.cities[v].traits)
            if d > 1.5 and self.rng.random() < 0.3:
                self.graph.remove_edge(u, v)

        # cultural interaction on edges
        for (u, v) in self.graph.edges():
            interact(self.cities[u], self.cities[v])

        # mutation
        for c in self.cities.values():
            mutate(c)

    def metrics(self):
        return {
            'edges': self.graph.number_of_edges(),
            'clustering': nx.average_clustering(self.graph) if self.graph.number_of_edges() > 0 else 0
        }


def run_simulation(steps=10, seed=None):
    model = InnovationNetwork(seed=seed)
    history = []
    for _ in range(steps):
        model.step()
        history.append(model.metrics())
    return history


if __name__ == "__main__":
    hist = run_simulation(steps=20, seed=42)
    for t, m in enumerate(hist):
        print(f"Step {t}: edges={m['edges']} clustering={m['clustering']:.3f}")
