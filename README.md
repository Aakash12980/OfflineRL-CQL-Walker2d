# Compute-Constrained Offline RL: How Far Do CQL and BC Get with Limited Data and Training?

This project implements Conservative Q-Learning (CQL) for offline reinforcement learning on the `walker2d-medium-v2` dataset from D4RL and comparing the model performance on various settings.

# Abstract
Offline Reinforcement Learning (RL) has emerged as a powerful paradigm for leveraging fixed
datasets to train agents without online interactions. However, the performance of offline RL
methods is often constrained by limited compute budgets and restricted training data. This
project investigates the performance of two prominent approaches-Conservative Q-Learning
(CQL) and Behavior Cloning (BC)-under compute-constrained conditions. We systematically
explore different settings of conservative penalties and training epochs to evaluate stability, robustness, and overall performance. Empirical results reveal that while BC exhibits stable and
consistent learning, CQL demonstrates a complex trade-off between conservative regularization
strength and training stability. Our findings highlight that overly aggressive or weak conservative penalties can lead to performance collapse, while moderate settings yield robust outcomes.
This report provides a detailed analysis of parameter sensitivities, training patterns, and Q-value
distributions, offering insights into practical offline RL under constrained resources.

## Setup
1. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
2. Clone the repository: `git clone https://github.com/yourusername/OfflineRL-CQL-Walker2d.git`
3. Install dependencies: `poetry install`
4. Activate environment: `poetry shell`
5. Run training: `python src/trainer.py`

## Dependencies
- Python 3.8+
- PyTorch
- Gymnasium
- D4RL (via d3rlpy)
- MuJoCo

## Goals
- To systematically compare BC and CQL under equal compute budgets with 25% and 50%
dataset usage.
- To investigate the role of conservative regularization strength (weights 1, 5, 10) in shaping
stability and learning dynamics.
- To analyze Q-value distributions and training curves to diagnose collapse, instability, or
robustness.
- To identify practical guidelines for applying offline RL under real-world compute constraint

# Experiment Results
![Experiment Results](results.png)

# Conclusion
This project investigated the performance of BC and CQL under compute-constrained offline RL. Results
highlight the robustness of BC, the sensitivity of CQL to conservative penalty strength, and the critical
role of dataset fraction. While CQL can surpass BC in favorable settings, its instability under constrained
conditions makes BC a practical alternative when reliability is prioritized. Future work may explore
adaptive conservative penalties or hybrid BC-CQL approaches for robust performance across varying
compute budgets.


## License
MIT