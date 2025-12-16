import os
import random
import numpy as np
import gymnasium as gym
from d3rlpy.algos import BCConfig, CQLConfig
from d3rlpy.datasets import get_minari
from torch.utils.tensorboard import SummaryWriter
from d3rlpy.models.q_functions import QRQFunctionFactory, MeanQFunctionFactory
from d3rlpy.models.encoders import VectorEncoderFactory
import torch
import glob
import d3rlpy
import minari
from d3rlpy.dataset import MDPDataset
import json
from datetime import datetime
from d3rlpy.metrics import EnvironmentEvaluator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


DATASET_NAME = "mujoco/walker2d/medium-v0"
ENV_NAME = "Walker2d-v4"

# Experiment configuration
FRACTIONS = [0.25, 0.5]
EPOCH_CONFIGS = [5, 7, 10]
STEPS_PER_EPOCH = 2000
EVAL_TRIALS = 5


BASE_LOG_DIR = "runs"
BASE_CKPT_DIR = "checkpoints"

# Experiment timestamp
EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")



def get_experiment_name(algorithm, fraction, epochs):
    """Generate a consistent experiment name

    Args:
        algorithm: The algorithm name (e.g., "CQL", "BC")
        fraction: The fraction of the dataset to use
        epochs: The number of training epochs

    Returns:
        A string representing the experiment name
    """
    return f"{algorithm}_frac{fraction:.2f}_epochs{epochs:02d}"


def get_experiment_paths(algorithm, fraction, epochs):
    """Get log and checkpoint paths for an experiment

    Args:
        algorithm: The algorithm name (e.g., "CQL", "BC")
        fraction: The fraction of the dataset to use
        epochs: The number of training epochs

    Returns:
        _type_: _description_
    """
    exp_name = get_experiment_name(algorithm, fraction, epochs)
    log_dir = os.path.join(BASE_LOG_DIR, EXPERIMENT_TIMESTAMP, exp_name)
    ckpt_dir = os.path.join(BASE_CKPT_DIR, EXPERIMENT_TIMESTAMP, exp_name)
    return log_dir, ckpt_dir


def subsample_dataset(dataset, fraction=0.25):
    """Subsample dataset by keeping only a fraction of entire trajectories,
    while reconstructing per-step terminals and timeouts from episode info.

    Args:
        dataset: The original MDPDataset
        fraction (float, optional): The fraction of the dataset to keep. Defaults to 0.25.
    """
    episodes = dataset.episodes
    n_trajs = len(episodes)
    k = max(1, int(n_trajs * fraction))

    rng = np.random.RandomState(SEED + int(fraction * 100))
    selected = rng.choice(n_trajs, k, replace=False)

    obs, acts, rews, terminals, timeouts = [], [], [], [], []
    total_returns = []

    for idx in selected:
        ep = episodes[idx]
        n_steps = len(ep.rewards)

        obs.append(ep.observations)
        acts.append(ep.actions)
        rews.append(ep.rewards)
        total_returns.append(np.sum(ep.rewards))

        done = np.zeros(n_steps, dtype=np.float32)
        timeout = np.zeros(n_steps, dtype=np.float32)

        if ep.terminated:
            done[-1] = 1.0
        if getattr(ep, "truncated", False) or getattr(ep, "timeout", False):
            timeout[-1] = 1.0

        terminals.append(done)
        timeouts.append(timeout)

    obs = np.concatenate(obs, axis=0)
    acts = np.concatenate(acts, axis=0)
    rews = np.concatenate(rews, axis=0)
    terminals = np.concatenate(terminals, axis=0)
    timeouts = np.concatenate(timeouts, axis=0)

    new_dataset = MDPDataset(obs, acts, rews, terminals, timeouts)

    # Print dataset statistics
    print(f"Using {k}/{n_trajs} trajectories ({fraction*100:.0f}%).")
    print(
        f"Dataset stats: {len(obs)} transitions, "
        f"return range: [{np.min(total_returns):.1f}, {np.max(total_returns):.1f}], "
        f"mean return: {np.mean(total_returns):.1f}"
    )

    return new_dataset


def load_latest_checkpoint(algo, algo_name, ckpt_dir):
    """Load the latest checkpoint for the given algorithm if available.

    Args:
        algo: Algorithm instance to load into
        algo_name: The name of the algorithm
        ckpt_dir: Directory to load checkpoints from

    Returns:
        The loaded algorithm and the epoch number
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = glob.glob(os.path.join(ckpt_dir, f"{algo_name}_epoch*"))
    if not ckpts:
        return algo, 0

    def extract_epoch(ckpt_path):
        try:
            return int(ckpt_path.split("epoch")[-1])
        except ValueError:
            return -1

    latest_ckpt = max(ckpts, key=extract_epoch)
    epoch_num = extract_epoch(latest_ckpt)
    algo = d3rlpy.load_learnable(latest_ckpt, device=DEVICE)
    print(f"🔄 Resuming {algo_name} from epoch {epoch_num}")
    return algo, epoch_num


def tb_logger(
    writer, algo_name, env, ckpt_dir, eval_episodes=5, milestones=None
):
    """TensorBoard callback with extra metrics and milestone checkpointing.

    Args:
        writer: A TensorBoard SummaryWriter instance
        algo_name: The name of the algorithm
        env: The environment to evaluate on
        ckpt_dir: Directory to save checkpoints
        eval_episodes (int, optional): Number of evaluation episodes. Defaults to 5.
        milestones: List of epochs to save checkpoints. Defaults to None.

    Returns:
        _type_: _description_
    """
    if milestones is None:
        milestones = [1, 5, 7, 10]

    def callback(algo, epoch, total_step):
        # Eval
        score = EnvironmentEvaluator(env, n_trials=eval_episodes)(algo, None)
        writer.add_scalar("score", score, epoch)
        writer.add_scalar("total_steps", total_step, epoch)

        print(f"Epoch {epoch}: Score = {score:.2f}")

        # Log training loss if available
        if hasattr(algo, "_loss"):
            writer.add_scalar("training_loss", algo._loss, epoch)

        # Only log Q-value distribution if algo supports it (CQL, not BC)
        if algo_name == "CQL" and hasattr(algo, "predict_value"):
            obs_batch = np.random.randn(
                64, *env.observation_space.shape
            ).astype(np.float32)
            act_batch = np.random.randn(64, *env.action_space.shape).astype(
                np.float32
            )
            try:
                q_vals = algo.predict_value(obs_batch, act_batch)
                writer.add_histogram("q_values", q_vals, epoch)
                writer.add_scalar("mean_q_value", np.mean(q_vals), epoch)
            except (NotImplementedError, AttributeError):
                pass

        # Save checkpoints only at milestones
        if epoch in milestones:
            ckpt_path = os.path.join(ckpt_dir, f"{algo_name}_epoch{epoch}")
            algo.save(ckpt_path)
            print(f"Saved {algo_name} checkpoint {ckpt_path}")

            algo.final_score = score

        return True

    return callback


def save_experiment_results(results, timestamp):
    """Save experiment results summary to JSON

    Args:
        results: The results dictionary
        timestamp: The experiment timestamp
    """
    results_dir = os.path.join("experiment_results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, "experiment_summary.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Experiment summary saved to {results_file}")


# Train BC model
def train_bc(dataset, env, fraction, max_epochs=10):
    """training behavior cloning model

    Args:
        dataset: dataset to train on
        env: environment for evaluation
        fraction: fraction of dataset used
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 10.

    """
    log_dir, ckpt_dir = get_experiment_paths("BC", fraction, max_epochs)
    exp_name = get_experiment_name("BC", fraction, max_epochs)

    print(f"\nStarting BC experiment: {exp_name}")
    writer = SummaryWriter(log_dir=log_dir)
    env_evaluator = EnvironmentEvaluator(env, n_trials=EVAL_TRIALS)

    # Improved BC configuration
    bc = BCConfig(
        learning_rate=1e-3,
        batch_size=256,
        encoder_factory=VectorEncoderFactory(hidden_units=[256, 256]),
    ).create(device=DEVICE)

    bc, start_epoch = load_latest_checkpoint(bc, "BC", ckpt_dir)

    bc.fit(
        dataset,
        n_steps=(max_epochs - start_epoch) * STEPS_PER_EPOCH,
        n_steps_per_epoch=STEPS_PER_EPOCH,
        callback=tb_logger(
            writer, "BC", env, ckpt_dir, milestones=EPOCH_CONFIGS
        ),
        evaluators={"environment": env_evaluator},
    )

    writer.close()

    return {
        "algorithm": "BC",
        "fraction": fraction,
        "epochs": max_epochs,
        "experiment_name": exp_name,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "final_score": getattr(bc, "final_score", None),
    }


# Train CQL model
def train_cql(dataset, env, fraction, max_epochs=10):
    """training conservative Q-learning model

    Args:
        dataset: dataset to train on
        env: environment for evaluation
        fraction: fraction of dataset used
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 10.

    """
    log_dir, ckpt_dir = get_experiment_paths("CQL", fraction, max_epochs)
    exp_name = get_experiment_name("CQL", fraction, max_epochs)

    print(f"\nStarting CQL experiment: {exp_name}")
    writer = SummaryWriter(log_dir=log_dir)
    env_evaluator = EnvironmentEvaluator(env, n_trials=EVAL_TRIALS)

    # CQL configuration
    cql = CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        alpha_learning_rate=1e-4,
        # Network architecture
        batch_size=256,
        actor_encoder_factory=VectorEncoderFactory(hidden_units=[256, 256]),
        critic_encoder_factory=VectorEncoderFactory(hidden_units=[256, 256]),
        q_func_factory=MeanQFunctionFactory(),

        conservative_weight=5.0,
        initial_alpha=1.0,
        alpha_threshold=10.0,
        n_action_samples=10,

        gamma=0.99,
        tau=0.005,
    ).create(device=DEVICE)

    cql, start_epoch = load_latest_checkpoint(cql, "CQL", ckpt_dir)

    total_steps = (max_epochs - start_epoch) * STEPS_PER_EPOCH

    print(
        f"Training CQL for {total_steps} steps ({max_epochs - start_epoch} epochs)"
    )

    cql.fit(
        dataset,
        n_steps=total_steps,
        n_steps_per_epoch=STEPS_PER_EPOCH,
        callback=tb_logger(
            writer, "CQL", env, ckpt_dir, milestones=EPOCH_CONFIGS
        ),
        evaluators={"environment": env_evaluator},
    )

    writer.close()

    return {
        "algorithm": "CQL",
        "fraction": fraction,
        "epochs": max_epochs,
        "experiment_name": exp_name,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "final_score": getattr(cql, "final_score", None),
    }


# Alternative CQL configurations
def train_cql_conservative(dataset, env, fraction, max_epochs=10):
    """More conservative CQL variant for challenging datasets

    Args:
        dataset: dataset to train on
        env: environment for evaluation
        fraction: fraction of dataset used
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 10.

    """
    log_dir, ckpt_dir = get_experiment_paths(
        "CQL_conservative", fraction, max_epochs
    )
    exp_name = get_experiment_name("CQL_conservative", fraction, max_epochs)

    print(f"\nStarting Conservative CQL experiment: {exp_name}")
    writer = SummaryWriter(log_dir=log_dir)
    env_evaluator = EnvironmentEvaluator(env, n_trials=EVAL_TRIALS)

    cql = CQLConfig(
        # Very conservative settings
        actor_learning_rate=3e-5,
        critic_learning_rate=1e-4,
        temp_learning_rate=3e-5,
        alpha_learning_rate=3e-5,
        batch_size=256,
        actor_encoder_factory=VectorEncoderFactory(
            hidden_units=[256, 256, 256]
        ),
        critic_encoder_factory=VectorEncoderFactory(
            hidden_units=[256, 256, 256]
        ),
        q_func_factory=MeanQFunctionFactory(),
        # Very conservative CQL parameters
        conservative_weight=10.0,
        initial_alpha=5.0,
        alpha_threshold=20.0,
        n_action_samples=5,
        gamma=0.99,
        tau=0.001,
    ).create(device=DEVICE)

    cql, start_epoch = load_latest_checkpoint(
        cql, "CQL_conservative", ckpt_dir
    )

    cql.fit(
        dataset,
        n_steps=(max_epochs - start_epoch) * STEPS_PER_EPOCH,
        n_steps_per_epoch=STEPS_PER_EPOCH,
        callback=tb_logger(
            writer, "CQL_conservative", env, ckpt_dir, milestones=EPOCH_CONFIGS
        ),
        evaluators={"environment": env_evaluator},
    )

    writer.close()
    return {
        "algorithm": "CQL_conservative",
        "fraction": fraction,
        "epochs": max_epochs,
        "experiment_name": exp_name,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "final_score": getattr(cql, "final_score", None),
    }


def train_cql_awac_style(dataset, env, fraction, max_epochs=10):
    """CQL with AWAC-style conservative weight scheduling

    Args:
        dataset: dataset to train on
        env: environment for evaluation
        fraction: fraction of dataset used
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 10.

    """
    log_dir, ckpt_dir = get_experiment_paths("CQL_awac", fraction, max_epochs)
    exp_name = get_experiment_name("CQL_awac", fraction, max_epochs)

    print(f"\nStarting AWAC-style CQL experiment: {exp_name}")
    writer = SummaryWriter(log_dir=log_dir)
    env_evaluator = EnvironmentEvaluator(env, n_trials=EVAL_TRIALS)

    cql = CQLConfig(
        # Moderate learning rates
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        alpha_learning_rate=1e-4,
        batch_size=256,
        actor_encoder_factory=VectorEncoderFactory(hidden_units=[256, 256]),
        critic_encoder_factory=VectorEncoderFactory(hidden_units=[256, 256]),
        q_func_factory=MeanQFunctionFactory(),
        # Moderate conservative parameters
        conservative_weight=1.0,
        initial_alpha=1.0,
        alpha_threshold=5.0,
        n_action_samples=10,
        gamma=0.99,
        tau=0.005,
    ).create(device=DEVICE)

    cql, start_epoch = load_latest_checkpoint(cql, "CQL_awac", ckpt_dir)

    cql.fit(
        dataset,
        n_steps=(max_epochs - start_epoch) * STEPS_PER_EPOCH,
        n_steps_per_epoch=STEPS_PER_EPOCH,
        callback=tb_logger(
            writer, "CQL_awac", env, ckpt_dir, milestones=EPOCH_CONFIGS
        ),
        evaluators={"environment": env_evaluator},
    )

    writer.close()
    return {
        "algorithm": "CQL_awac",
        "fraction": fraction,
        "epochs": max_epochs,
        "experiment_name": exp_name,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "final_score": getattr(cql, "final_score", None),
    }


# Main Experiment Runner
def run_all_experiments():
    print(f"Starting experiment suite: {EXPERIMENT_TIMESTAMP}")
    minari.download_dataset(DATASET_NAME)
    base_dataset, _ = get_minari(DATASET_NAME)
    env = gym.make(ENV_NAME)

    all_results = []

    for fraction in FRACTIONS:
        # Create dataset once per fraction
        dataset = subsample_dataset(base_dataset, fraction=fraction)

        # Train BC first (our baseline model)
        bc_result = train_bc(
            dataset, env, fraction, max_epochs=max(EPOCH_CONFIGS)
        )
        all_results.append(bc_result)

        # Train simple CQL model
        cql_result = train_cql(
            dataset, env, fraction, max_epochs=max(EPOCH_CONFIGS)
        )
        all_results.append(cql_result)

        # Train conservative CQL variant
        cql_conservative_result = train_cql_conservative(
            dataset, env, fraction, max_epochs=max(EPOCH_CONFIGS)
        )
        all_results.append(cql_conservative_result)

        # Train AWAC-style CQL variant
        cql_awac_result = train_cql_awac_style(
            dataset, env, fraction, max_epochs=max(EPOCH_CONFIGS)
        )
        all_results.append(cql_awac_result)

    save_experiment_results(
        {
            "timestamp": EXPERIMENT_TIMESTAMP,
            "results": all_results,
            "config": {
                "dataset": DATASET_NAME,
                "env": ENV_NAME,
                "fractions": FRACTIONS,
                "epochs": EPOCH_CONFIGS,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "eval_trials": EVAL_TRIALS,
                "seed": SEED,
            },
        },
        EXPERIMENT_TIMESTAMP,
    )

    return all_results


def run_single_experiment(algorithm="CQL", fraction=0.5, max_epochs=10):
    """Run a single experiment for debugging

    Args:
        algorithm (str, optional): algorithm name. Defaults to "CQL".
        fraction (float, optional): fraction of dataset. Defaults to 0.5.
        max_epochs (int, optional): number of epochs to train for. Defaults to 10.

    Raises:
        ValueError: if the algorithm is not recognized
    """
    print(f"Running single {algorithm} experiment")
    minari.download_dataset(DATASET_NAME)
    base_dataset, _ = get_minari(DATASET_NAME)
    env = gym.make(ENV_NAME)

    dataset = subsample_dataset(base_dataset, fraction=fraction)

    if algorithm == "BC":
        result = train_bc(dataset, env, fraction, max_epochs)
    elif algorithm == "CQL":
        result = train_cql(dataset, env, fraction, max_epochs)
    elif algorithm == "CQL_conservative":
        result = train_cql_conservative(dataset, env, fraction, max_epochs)
    elif algorithm == "CQL_awac":
        result = train_cql_awac_style(dataset, env, fraction, max_epochs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return result


if __name__ == "__main__":

    results = run_all_experiments()
    print("All experiments completed successfully!")
