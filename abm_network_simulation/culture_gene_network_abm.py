# Step 1: 环境搭建与工具库导入

# 安装必要工具库（如果是Jupyter/Colab环境，直接运行此行即可。如果是在命令行，去掉 '!'）
!pip install networkx mesa matplotlib seaborn pandas

# 导入必要工具库
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
# -----------------------------------------------

# 示例1：无标度网络（Barabási–Albert模型）
def initialize_network_ba(n_nodes=50, m_edges=3, seed=None):
    """
    初始化一个无标度复杂网络
    n_nodes: 节点数
    m_edges: 每个新节点连接到已有节点的边数
    seed: 随机种子，便于复现实验
    """
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)
    return G

# 示例2：小世界网络（Watts-Strogatz模型）
def initialize_network_ws(n_nodes=50, k_neighbors=4, p_rewire=0.1, seed=None):
    """
    初始化一个小世界网络
    n_nodes: 节点数
    k_neighbors: 每个节点连接的最近邻居数
    p_rewire: 重新连边概率
    """
    G = nx.watts_strogatz_graph(n_nodes, k_neighbors, p_rewire, seed=seed)
    return G

# --------- 调用网络初始化函数举例 ---------
if __name__ == "__main__":
    # 创建一个50节点的无标度网络
    G = initialize_network_ba(n_nodes=50, m_edges=3)
    print(f"无标度网络：节点数={G.number_of_nodes()}，边数={G.number_of_edges()}")

    # 可视化网络结构
    plt.figure(figsize=(8, 6))
    nx.draw(G, node_color='skyblue', edge_color='gray', with_labels=True, node_size=200)
    plt.title("Barabási–Albert 无标度网络结构")
    plt.show()

# Step 3: ABM主体类（Agent/Model定义）
# -------------------------------------

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

import numpy as np

# 1. 定义创新主体 Agent
class InnovationAgent(Agent):
    def __init__(self, unique_id, model, culture_gene):
        super().__init__(unique_id, model)
        self.culture_gene = culture_gene  # 文化基因类型，可用float或类别变量

    def step(self):
        # 示例：Agent每一步与邻居互动，并根据文化适配度决定迁移/连接变化
        neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
        if not neighbors:
            return

        # 简单迁移或互动策略：与最相似文化基因的邻居互动
        best_neighbor = max(
            neighbors,
            key=lambda n: 1 - abs(self.culture_gene - self.model.schedule.agents[n].culture_gene)
        )

        # 示例：以一定概率创建新边
        if np.random.rand() < 0.3:
            self.model.G.add_edge(self.pos, best_neighbor)

        # 可扩展：更复杂的迁移/断边/创新行为

# 2. 定义主模型 Model
class InnovationNetworkModel(Model):
    def __init__(self, G, culture_gene_type='continuous'):
        """
        G: 初始复杂网络结构（networkx对象）
        culture_gene_type: 文化基因变量类型，可为 'continuous' 或 'categorical'
        """
        self.G = G
        self.schedule = RandomActivation(self)
        self.grid = NetworkGrid(self.G)

        # 创建Agent并绑定到网络节点
        for i, node in enumerate(self.G.nodes()):
            # 生成文化基因属性
            if culture_gene_type == 'continuous':
                culture_gene = np.random.rand()  # 连续型 [0,1]
            else:
                culture_gene = np.random.choice(['A', 'B', 'C'])  # 离散型类别
            agent = InnovationAgent(node, self, culture_gene)
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

        # 数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "NumEdges": lambda m: m.G.number_of_edges(),
                "AverageClustering": lambda m: nx.average_clustering(m.G),
            }
            # 可以根据需求添加更多指标
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Step 4: 主仿真循环与数据收集、结果可视化
# -------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 假设你已完成 G = initialize_network_ba() 并有 InnovationNetworkModel 类
# 例如：
# G = initialize_network_ba(n_nodes=50, m_edges=3)
# model = InnovationNetworkModel(G, culture_gene_type='continuous')

def run_simulation(model, n_steps=100):
    """运行主仿真循环"""
    for step in range(n_steps):
        model.step()
    return model

# 运行仿真
if __name__ == "__main__":
    # 初始化网络和模型
    G = initialize_network_ba(n_nodes=50, m_edges=3)
    model = InnovationNetworkModel(G, culture_gene_type='continuous')
    
    # 仿真100步
    model = run_simulation(model, n_steps=100)

    # 数据收集
    results = model.datacollector.get_model_vars_dataframe()
    print(results.head())

    # 结果可视化
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=results, x=results.index, y="NumEdges", label="Network Edges")
    sns.lineplot(data=results, x=results.index, y="AverageClustering", label="Clustering Coefficient")
    plt.xlabel("Simulation Step")
    plt.ylabel("Network Metrics")
    plt.title("Network Evolution Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Step 5: 结果分析与可视化
# --------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

# 假设results为DataCollector收集的指标数据，model为当前仿真模型

# 1. 网络结构演化趋势分析
plt.figure(figsize=(10, 5))
sns.lineplot(data=results, x=results.index, y="NumEdges", label="Network Edges")
sns.lineplot(data=results, x=results.index, y="AverageClustering", label="Clustering Coefficient")
plt.xlabel("Simulation Step")
plt.ylabel("Network Metrics")
plt.title("Network Evolution Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# 2. 最终网络结构可视化（节点颜色为文化基因）
final_culture_gene = [model.schedule.agents[n].culture_gene for n in model.G.nodes()]
plt.figure(figsize=(8, 6))
nx.draw(model.G, 
        node_color=final_culture_gene, 
        cmap=plt.cm.viridis, 
        with_labels=True, 
        node_size=250)
plt.title("Final Network Structure (Node Color: Culture Gene)")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Culture Gene')
plt.show()

# 3. 节点文化基因分布直方图
plt.figure(figsize=(7, 4))
sns.histplot(final_culture_gene, bins=10, kde=True)
plt.xlabel("Culture Gene Value")
plt.ylabel("Number of Agents")
plt.title("Distribution of Culture Gene Among Agents (Final Step)")
plt.tight_layout()
plt.show()

# 4. 统计网络关键指标
print("仿真结束后网络结构关键统计：")
print(f"节点数：{model.G.number_of_nodes()}")
print(f"边数：{model.G.number_of_edges()}")
print(f"平均度数：{sum(dict(model.G.degree()).values()) / model.G.number_of_nodes():.2f}")
print(f"平均聚类系数：{nx.average_clustering(model.G):.4f}")
print(f"网络直径（最大最短路径）：{nx.diameter(model.G) if nx.is_connected(model.G) else '网络不连通'}")
print()

# 5. （可选）创新绩效与网络结构相关性分析
# 如果你在DataCollector里存了创新绩效，可做相关性散点分析
# 例如：results['InnovationPerformance']，results['AverageClustering']

if "InnovationPerformance" in results.columns:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=results['AverageClustering'], y=results['InnovationPerformance'])
    plt.xlabel("Average Clustering Coefficient")
    plt.ylabel("Innovation Performance")
    plt.title("Innovation Performance vs Network Clustering")
    plt.tight_layout()
    plt.show()

    corr = results['AverageClustering'].corr(results['InnovationPerformance'])
    print(f"聚类系数与创新绩效的皮尔逊相关系数: {corr:.3f}")

# 6. （可选）不同场景/参数对比分析
# 可批量仿真不同参数，统计如平均聚类系数、平均路径长度、创新绩效等指标，并用barplot、boxplot等展示

# 7. 导出结果为CSV，便于后续论文或深度分析
results.to_csv("abm_network_simulation_results.csv")

print("分析与可视化已完成。")
