# Step 1: 环境搭建与工具库导入
# ---------------------------------
# 如需在本地安装依赖，可执行 `pip install networkx mesa matplotlib seaborn pandas`

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

# Step 2: 网络结构初始化（复杂网络建模函数）
# ---------------------------------------

def initialize_network_ba(n_nodes: int = 50, m_edges: int = 3, seed: int | None = None) -> nx.Graph:
    """创建 Barabási-Albert 无标度网络"""
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)
    return G


def initialize_network_ws(n_nodes: int = 50, k_neighbors: int = 4, p_rewire: float = 0.1, seed: int | None = None) -> nx.Graph:
    """创建 Watts-Strogatz 小世界网络"""
    G = nx.watts_strogatz_graph(n_nodes, k_neighbors, p_rewire, seed=seed)
    return G

# Step 3: ABM 主体类（Agent/Model 定义）
# ------------------------------------
class InnovationAgent(Agent):
    def __init__(self, unique_id: int, model: Model, culture_gene: float | str):
        super().__init__(unique_id, model)
        self.culture_gene = culture_gene

    def step(self) -> None:
        neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
        if not neighbors:
            return
        best_neighbor = max(neighbors, key=lambda a: 1 - abs(self.culture_gene - a.culture_gene))
        if np.random.rand() < 0.3:
            self.model.G.add_edge(self.pos, best_neighbor.pos)


class InnovationNetworkModel(Model):
    def __init__(self, G: nx.Graph, culture_gene_type: str = "continuous") -> None:
        self.G = G
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = NetworkGrid(self.G)
        for node in self.G.nodes():
            if culture_gene_type == "continuous":
                culture_gene = np.random.rand()
            else:
                culture_gene = np.random.choice(["A", "B", "C"])
            agent = InnovationAgent(node, self, culture_gene)
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)
        self.datacollector = DataCollector(
            model_reporters={
                "NumEdges": lambda m: m.G.number_of_edges(),
                "AverageClustering": lambda m: nx.average_clustering(m.G),
            }
        )

    def step(self) -> None:
        self.datacollector.collect(self)
        self.schedule.step()

# Step 4: 主仿真循环
# ------------------

def run_simulation(model: InnovationNetworkModel, n_steps: int = 100) -> InnovationNetworkModel:
    for _ in range(n_steps):
        model.step()
    return model

# Step 5: 结果分析与可视化
# ------------------------

def analyze_results(model: InnovationNetworkModel, results: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=results, x=results.index, y="NumEdges", label="Network Edges")
    sns.lineplot(data=results, x=results.index, y="AverageClustering", label="Clustering Coefficient")
    plt.xlabel("Simulation Step")
    plt.ylabel("Network Metrics")
    plt.title("Network Evolution Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    final_culture_gene = [model.grid.get_cell_list_contents([n])[0].culture_gene for n in model.G.nodes()]
    plt.figure(figsize=(8, 6))
    nx.draw(
        model.G,
        node_color=final_culture_gene,
        cmap=plt.cm.viridis,
        with_labels=True,
        node_size=250,
    )
    plt.title("Final Network Structure (Node Color: Culture Gene)")
    plt.show()
    sns.histplot(final_culture_gene, bins=10, kde=True)
    plt.xlabel("Culture Gene Value")
    plt.ylabel("Number of Agents")
    plt.title("Distribution of Culture Gene Among Agents (Final Step)")
    plt.tight_layout()
    plt.show()

    print("仿真结束后网络结构关键统计:")
    print(f"节点数：{model.G.number_of_nodes()}")
    print(f"边数：{model.G.number_of_edges()}")
    avg_degree = sum(dict(model.G.degree()).values()) / model.G.number_of_nodes()
    print(f"平均度数：{avg_degree:.2f}")
    print(f"平均聚类系数：{nx.average_clustering(model.G):.4f}")
    diameter = nx.diameter(model.G) if nx.is_connected(model.G) else "网络不连通"
    print(f"网络直径：{diameter}")

    if "InnovationPerformance" in results.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=results["AverageClustering"], y=results["InnovationPerformance"])
        plt.xlabel("Average Clustering Coefficient")
        plt.ylabel("Innovation Performance")
        plt.title("Innovation Performance vs Network Clustering")
        plt.tight_layout()
        plt.show()
        corr = results["AverageClustering"].corr(results["InnovationPerformance"])
        print(f"聚类系数与创新绩效的皮尔逊相关系数: {corr:.3f}")

    results.to_csv("abm_network_simulation_results.csv")
    print("分析与可视化已完成。")

# Step 6: main 函数
# ------------------

def main() -> None:
    G = initialize_network_ba(n_nodes=50, m_edges=3)
    model = InnovationNetworkModel(G, culture_gene_type="continuous")
    model = run_simulation(model, n_steps=100)
    results = model.datacollector.get_model_vars_dataframe()
    print(results.head())
    analyze_results(model, results)

if __name__ == "__main__":
    main()
