# stata-python-julia-demo
Codex环境测试用 -A cross-language research environment using Stata, Python, and Julia for Codex deployment.

## Running the ABM simulation
The agent-based model is provided in `abm_network_simulation/culture_gene_network_abm.ipynb`. To run it:

1. Install the required Python packages:
   ```bash
   pip install networkx mesa matplotlib seaborn pandas
   ```
2. Open the notebook in Jupyter and execute all cells. Alternatively you can run it as a script after removing the leading `!` from the `pip install` line.

Running the notebook writes simulation results to `abm_network_simulation_results.csv` and saves PNG figures. These files are ignored by Git using `.gitignore`.

