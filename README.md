# OfflineRL-CQL-Walker2d

This project implements Conservative Q-Learning (CQL) for offline reinforcement learning on the `walker2d-medium-v2` dataset from D4RL. It serves as an exploratory project for a Master's thesis in Computer Science, focusing on offline RL for safe decision-making.

## Setup
1. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
2. Clone the repository: `git clone https://github.com/yourusername/OfflineRL-CQL-Walker2d.git`
3. Install dependencies: `poetry install`
4. Activate environment: `poetry shell`
5. Run training: `python src/train_cql.py`

## Dependencies
- Python 3.8+
- PyTorch
- Gym
- D4RL (via d3rlpy)
- MuJoCo (optional for evaluation)

## Goals
- Train a CQL agent on `walker2d-medium-v2`.
- Evaluate performance against baselines (e.g., Behavioral Cloning).
- Analyze challenges like distribution shift for thesis research.

## License
MIT