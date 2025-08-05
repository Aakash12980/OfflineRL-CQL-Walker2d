import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import torch

from d3rlpy.datasets import get_minari
import gymnasium as gym
from d3rlpy.algos import CQLConfig, BCConfig
from d3rlpy.metrics import (
    TDErrorEvaluator,
    DiscountedSumOfAdvantageEvaluator,
    AverageValueEstimationEvaluator,
    InitialStateValueEstimationEvaluator,
    SoftOPCEvaluator,
    ContinuousActionDiffEvaluator,
    EnvironmentEvaluator,
)


class OptimizedRLTrainer:
    def __init__(
        self,
        dataset_name: str = "mujoco/walker2d/medium-v0",
        env_name: str = "Walker2d-v5",
        results_dir: str = "results",
        fast_mode: bool = True,
    ):
        """
        Initialize the optimized RL trainer

        Args:
            dataset_name: Minari dataset name
            env_name: Gym environment name
            results_dir: Directory to save results
            fast_mode: Enable optimizations for faster training
        """
        self.dataset_name = dataset_name
        self.env_name = env_name
        self.results_dir = results_dir
        self.fast_mode = fast_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Check and setup device
        self.device = self._setup_device()

        # Load dataset and environment
        print(f"Loading dataset: {dataset_name}")
        self.dataset, self.env_info = get_minari(dataset_name)

        print(f"Creating environment: {env_name}")
        self.env = gym.make(env_name)
        self.eval_env = gym.make(env_name)

        # Optimize dataset for faster training
        if self.fast_mode:
            self._optimize_dataset()

        # Initialize metrics storage
        self.training_metrics = {
            "cql": {"epochs": [], "metrics": []},
            "bc": {"epochs": [], "metrics": []},
        }

        self.evaluation_results = {"cql": [], "bc": []}

        # Best model tracking
        self.best_models = {
            "cql": {"score": -np.inf, "epoch": 0, "model_path": None},
            "bc": {"score": -np.inf, "epoch": 0, "model_path": None},
        }

    def _setup_device(self):
        """Setup optimal device for training"""
        if torch.cuda.is_available():
            device = "cuda:0"
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = "cpu"
            print("⚠ Using CPU (GPU not available)")

        return device

    def _optimize_dataset(self):
        """Optimize dataset for faster training"""
        print("Optimizing dataset for fast training...")

        # Limit dataset size for faster training
        max_episodes = 500  # Reduce from full dataset
        if len(self.dataset.episodes) > max_episodes:
            print(
                f"  Reducing dataset from {len(self.dataset.episodes)} to {max_episodes} episodes"
            )
            # Keep episodes with diverse returns
            returns = [np.sum(ep.rewards) for ep in self.dataset.episodes]
            indices = np.argsort(returns)
            # Take episodes from different performance levels
            selected_indices = []
            step = len(indices) // max_episodes
            for i in range(0, len(indices), max(1, step)):
                selected_indices.append(indices[i])
                if len(selected_indices) >= max_episodes:
                    break

            self.dataset.episodes = [
                self.dataset.episodes[i]
                for i in selected_indices[:max_episodes]
            ]

        print(f"  Final dataset size: {len(self.dataset.episodes)} episodes")
        print(f"  Total steps: {sum(len(ep) for ep in self.dataset.episodes)}")

    def setup_evaluators(self, lightweight: bool = False):
        """Setup evaluators with optimizations"""
        if lightweight:
            # Minimal evaluators for fast training
            return {
                "environment": EnvironmentEvaluator(
                    self.eval_env,
                    n_trials=3,
                    epsilon=0.0,
                )
            }

        # Standard evaluators but with reduced complexity
        env_evaluator = EnvironmentEvaluator(
            self.eval_env,
            n_trials=5,
            epsilon=0.0,
        )

        # Use smaller subset for other evaluators
        eval_episodes = self.dataset.episodes[:50]

        td_error_evaluator = TDErrorEvaluator(episodes=eval_episodes)
        avg_value_evaluator = AverageValueEstimationEvaluator(
            episodes=eval_episodes
        )
        initial_value_evaluator = InitialStateValueEstimationEvaluator(
            episodes=eval_episodes
        )
        advantage_evaluator = DiscountedSumOfAdvantageEvaluator(
            episodes=eval_episodes
        )
        opc_evaluator = SoftOPCEvaluator(
            return_threshold=self.calculate_return_threshold()
        )
        action_diff_evaluator = ContinuousActionDiffEvaluator(
            episodes=eval_episodes
        )

        return {
            "environment": env_evaluator,
            "td_error": td_error_evaluator,
            "avg_value": avg_value_evaluator,
            "initial_value": initial_value_evaluator,
            "advantage": advantage_evaluator,
            "opc": opc_evaluator,
            "action_diff": action_diff_evaluator,
        }

    def calculate_return_threshold(self):
        """Calculate return threshold for OPC evaluator"""
        returns = []
        for episode in self.dataset.episodes:
            episode_return = np.sum(episode.rewards)
            returns.append(episode_return)
        return np.percentile(returns, 75)

    def evaluate_policy(
        self, algo, algo_name: str, epoch: int, n_trials: int = None
    ):
        """Optimized policy evaluation"""
        if n_trials is None:
            n_trials = 3 if self.fast_mode else 10

        print(
            f"\nEvaluating {algo_name} at epoch {epoch} ({n_trials} trials)..."
        )

        eval_scores = []
        eval_lengths = []

        for trial in range(n_trials):
            obs, _ = self.eval_env.reset()
            total_reward = 0
            steps = 0
            done = False
            max_steps = 500 if self.fast_mode else 1000

            while not done and steps < max_steps:
                if obs.ndim == 1:
                    obs_batch = obs.reshape(1, -1)
                else:
                    obs_batch = obs

                action = algo.predict(obs_batch)[0]
                obs, reward, terminated, truncated, _ = self.eval_env.step(
                    action
                )
                total_reward += reward
                steps += 1
                done = terminated or truncated

            eval_scores.append(total_reward)
            eval_lengths.append(steps)

        eval_results = {
            "epoch": epoch,
            "mean_score": np.mean(eval_scores),
            "std_score": np.std(eval_scores),
            "min_score": np.min(eval_scores),
            "max_score": np.max(eval_scores),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths),
            "scores": eval_scores,
            "lengths": eval_lengths,
        }

        self.evaluation_results[algo_name].append(eval_results)

        print(f"{algo_name} Evaluation Results:")
        print(
            f"  Mean Score: {eval_results['mean_score']:.2f} ± {eval_results['std_score']:.2f}"
        )
        print(
            f"  Score Range: [{eval_results['min_score']:.2f}, {eval_results['max_score']:.2f}]"
        )
        print(
            f"  Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}"
        )

        return eval_results

    def save_best_model(self, algo, algo_name: str, score: float, epoch: int):
        """Save model if it's the best so far"""
        if score > self.best_models[algo_name]["score"]:
            model_path = os.path.join(
                self.results_dir, f"best_{algo_name}_model_{self.timestamp}.d3"
            )
            algo.save(model_path)

            self.best_models[algo_name].update(
                {"score": score, "epoch": epoch, "model_path": model_path}
            )

            print(
                f"New best {algo_name} model saved! Score: {score:.2f} at epoch {epoch}"
            )
            return True
        return False

    def train_cql(self, n_steps: int = None, n_steps_per_epoch: int = None):
        """Train CQL algorithm with optimizations"""
        print("\n" + "=" * 50)
        print("TRAINING CQL ALGORITHM")
        print("=" * 50)

        # training parameters
        if self.fast_mode:
            n_steps = n_steps or 30000
            n_steps_per_epoch = n_steps_per_epoch or 3000
            print("Fast mode enabled - using reduced training steps")
        else:
            n_steps = n_steps or 100000
            n_steps_per_epoch = n_steps_per_epoch or 10000

        # Optimized CQL configuration
        cql_config = CQLConfig(
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            temp_learning_rate=1e-3,
            alpha_learning_rate=1e-3,
            batch_size=(512 if self.device.startswith("cuda") else 256),
            gamma=0.99,
            tau=0.005,
            n_critics=2,
            initial_temperature=1.0,
            initial_alpha=2.0,
            alpha_threshold=5.0,
            conservative_weight=2.0,
            n_action_samples=5,
            soft_q_backup=False,
        )

        cql = cql_config.create(device=self.device)

        # Setup lightweight evaluators for fast training
        evaluators = self.setup_evaluators(lightweight=self.fast_mode)

        # Optimized training callback
        eval_frequency = 3 if self.fast_mode else 1

        def evaluation_callback(algo, epoch, total_step):
            if epoch % eval_frequency == 0:
                eval_results = self.evaluate_policy(algo, "cql", epoch)
                self.save_best_model(
                    algo, "cql", eval_results["mean_score"], epoch
                )

            # Save checkpoints less frequently
            if epoch % 20 == 0:
                checkpoint_path = os.path.join(
                    self.results_dir,
                    f"cql_checkpoint_epoch_{epoch}_{self.timestamp}.d3",
                )
                algo.save(checkpoint_path)

        print(
            f"Training CQL for {n_steps} steps ({n_steps//n_steps_per_epoch} epochs)..."
        )
        print(f"Device: {self.device}")

        epochs, metrics = cql.fit(
            self.dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            evaluators=evaluators,
            experiment_name=f"cql_walker2d_{self.timestamp}",
            with_timestamp=False,
            save_interval=10,
            callback=evaluation_callback,
        )

        self.training_metrics["cql"]["epochs"] = epochs
        self.training_metrics["cql"]["metrics"] = metrics

        # Final evaluation
        final_eval = self.evaluate_policy(cql, "cql", len(epochs))
        self.save_best_model(cql, "cql", final_eval["mean_score"], len(epochs))

        return cql, epochs, metrics

    def train_bc(self, n_steps: int = None, n_steps_per_epoch: int = None):
        """Train BC algorithm"""
        print("\n" + "=" * 50)
        print("TRAINING BC ALGORITHM")
        print("=" * 50)

        # training parameters
        if self.fast_mode:
            n_steps = n_steps or 10000
            n_steps_per_epoch = n_steps_per_epoch or 1000
        else:
            n_steps = n_steps or 20000
            n_steps_per_epoch = n_steps_per_epoch or 2000

        bc_config = BCConfig(
            learning_rate=3e-3,
            batch_size=512 if self.device.startswith("cuda") else 256,
            weight_decay=1e-4,
        )

        bc = bc_config.create(device=self.device)

        # Minimal evaluators for BC
        evaluators = {
            "environment": EnvironmentEvaluator(self.eval_env, n_trials=3),
        }

        def evaluation_callback(algo, epoch, total_step):
            if epoch % 2 == 0:
                eval_results = self.evaluate_policy(algo, "bc", epoch)
                self.save_best_model(
                    algo, "bc", eval_results["mean_score"], epoch
                )

        print(
            f"Training BC for {n_steps} steps ({n_steps//n_steps_per_epoch} epochs)..."
        )
        epochs, metrics = bc.fit(
            self.dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            evaluators=evaluators,
            experiment_name=f"bc_walker2d_{self.timestamp}",
            with_timestamp=False,
            callback=evaluation_callback,
        )

        self.training_metrics["bc"]["epochs"] = epochs
        self.training_metrics["bc"]["metrics"] = metrics

        # Final evaluation
        final_eval = self.evaluate_policy(bc, "bc", len(epochs))
        self.save_best_model(bc, "bc", final_eval["mean_score"], len(epochs))

        return bc, epochs, metrics

    def save_comprehensive_results(self):
        """Save all results to files"""
        print("\n" + "=" * 50)
        print("SAVING COMPREHENSIVE RESULTS")
        print("=" * 50)

        results = {
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "env_name": self.env_name,
            "fast_mode": self.fast_mode,
            "device": self.device,
            "training_metrics": self.training_metrics,
            "evaluation_results": self.evaluation_results,
            "best_models": self.best_models,
            "dataset_info": {
                "num_episodes": len(self.dataset.episodes),
                "total_steps": sum(len(ep) for ep in self.dataset.episodes),
                "action_space": str(self.env.action_space),
                "observation_space": str(self.env.observation_space),
            },
        }

        # Save JSON results
        json_path = os.path.join(
            self.results_dir, f"optimized_results_{self.timestamp}.json"
        )
        with open(json_path, "w") as f:
            json_results = self._convert_numpy_to_list(results)
            json.dump(json_results, f, indent=2)

        # Save pickle results
        pickle_path = os.path.join(
            self.results_dir, f"optimized_results_{self.timestamp}.pkl"
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

        self._create_evaluation_csv()
        self._create_training_csv()
        self._generate_plots()
        self._create_summary_report()

        print(f"✓ Results saved to: {json_path}")
        print(f"✓ Results saved to: {pickle_path}")

    def _convert_numpy_to_list(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._convert_numpy_to_list(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    def _create_evaluation_csv(self):
        """Create CSV file with evaluation results"""
        eval_data = []
        for algo_name in ["cql", "bc"]:
            for result in self.evaluation_results[algo_name]:
                eval_data.append(
                    {
                        "algorithm": algo_name,
                        "epoch": result["epoch"],
                        "mean_score": result["mean_score"],
                        "std_score": result["std_score"],
                        "min_score": result["min_score"],
                        "max_score": result["max_score"],
                        "mean_length": result["mean_length"],
                        "std_length": result["std_length"],
                    }
                )

        df = pd.DataFrame(eval_data)
        csv_path = os.path.join(
            self.results_dir, f"evaluation_results_{self.timestamp}.csv"
        )
        df.to_csv(csv_path, index=False)
        print(f"✓ Evaluation CSV saved to: {csv_path}")

    def _create_training_csv(self):
        """Create CSV file with training metrics"""
        training_data = []
        for algo_name in ["cql", "bc"]:
            if self.training_metrics[algo_name]["epochs"]:
                epochs = self.training_metrics[algo_name]["epochs"]
                metrics = self.training_metrics[algo_name]["metrics"]

                for epoch, metric_dict in zip(epochs, metrics):
                    row = {"algorithm": algo_name, "epoch": epoch}
                    if isinstance(metric_dict, dict):
                        for key, value in metric_dict.items():
                            row[key] = value
                    training_data.append(row)

        if training_data:
            df = pd.DataFrame(training_data)
            csv_path = os.path.join(
                self.results_dir, f"training_metrics_{self.timestamp}.csv"
            )
            df.to_csv(csv_path, index=False)
            print(f"✓ Training metrics CSV saved to: {csv_path}")

    def _generate_plots(self):
        """Generate optimized plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Training Results - {self.timestamp}", fontsize=14)

        # Evaluation scores
        ax1 = axes[0, 0]
        for algo_name in ["cql", "bc"]:
            if self.evaluation_results[algo_name]:
                epochs = [
                    r["epoch"] for r in self.evaluation_results[algo_name]
                ]
                scores = [
                    r["mean_score"] for r in self.evaluation_results[algo_name]
                ]
                ax1.plot(
                    epochs, scores, label=f"{algo_name.upper()}", marker="o"
                )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Score")
        ax1.set_title("Evaluation Scores")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Episode lengths
        ax2 = axes[0, 1]
        for algo_name in ["cql", "bc"]:
            if self.evaluation_results[algo_name]:
                epochs = [
                    r["epoch"] for r in self.evaluation_results[algo_name]
                ]
                lengths = [
                    r["mean_length"]
                    for r in self.evaluation_results[algo_name]
                ]
                ax2.plot(
                    epochs, lengths, label=f"{algo_name.upper()}", marker="s"
                )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean Episode Length")
        ax2.set_title("Episode Lengths")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Score distributions
        ax3 = axes[1, 0]
        for algo_name in ["cql", "bc"]:
            if self.evaluation_results[algo_name]:
                latest_scores = self.evaluation_results[algo_name][-1][
                    "scores"
                ]
                ax3.hist(
                    latest_scores,
                    alpha=0.7,
                    label=f"{algo_name.upper()}",
                    bins=8,
                )

        ax3.set_xlabel("Score")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Final Score Distributions")
        ax3.legend()

        # Best scores comparison
        ax4 = axes[1, 1]
        algorithms, best_scores = [], []
        for algo_name in ["cql", "bc"]:
            if self.best_models[algo_name]["score"] != -np.inf:
                algorithms.append(algo_name.upper())
                best_scores.append(self.best_models[algo_name]["score"])

        if algorithms:
            ax4.bar(algorithms, best_scores, color=["blue", "orange"])
            ax4.set_ylabel("Best Score")
            ax4.set_title("Best Scores Comparison")

        plt.tight_layout()
        plot_path = os.path.join(
            self.results_dir, f"training_plots_{self.timestamp}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to: {plot_path}")

    def _create_summary_report(self):
        """Create a summary report"""
        report_path = os.path.join(
            self.results_dir, f"summary_report_{self.timestamp}.txt"
        )

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("OFFLINE RL TRAINING REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Fast Mode: {self.fast_mode}\n")
            f.write(f"Device: {self.device}\n\n")

            # Dataset info
            f.write("DATASET INFO:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Episodes: {len(self.dataset.episodes)}\n")
            f.write(
                f"Steps: {sum(len(ep) for ep in self.dataset.episodes)}\n\n"
            )

            # Best models
            f.write("BEST RESULTS:\n")
            f.write("-" * 30 + "\n")
            for algo_name in ["cql", "bc"]:
                best = self.best_models[algo_name]
                if best["score"] != -np.inf:
                    f.write(
                        f"{algo_name.upper()}: {best['score']:.2f} (epoch {best['epoch']})\n"
                    )

        print(f"Summary report saved to: {report_path}")

    def run_complete_training(self):
        """Run optimized training pipeline"""
        print("🚀 Starting offline RL training pipeline...")
        print(f"Device: {self.device}")
        print(f"Fast mode: {self.fast_mode}")
        print(f"Results directory: {self.results_dir}")

        cql_model, cql_epochs, cql_metrics = self.train_cql()
        bc_model, bc_epochs, bc_metrics = self.train_bc()
        self.save_comprehensive_results()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Best CQL Score: {self.best_models['cql']['score']:.2f}")
        print(f"Best BC Score: {self.best_models['bc']['score']:.2f}")
        print(f"Device used: {self.device}")
        print(f"Results saved in: {self.results_dir}")
        print("=" * 60)

        return {
            "cql_model": cql_model,
            "bc_model": bc_model,
            "results_dir": self.results_dir,
            "timestamp": self.timestamp,
            "device": self.device,
        }


if __name__ == "__main__":
    trainer = OptimizedRLTrainer(
        dataset_name="mujoco/walker2d/medium-v0",
        env_name="Walker2d-v5",
        results_dir="walker2d_results",
        fast_mode=True,
    )

    results = trainer.run_complete_training()
    print(f"\n✅ Training completed! Check: {results['results_dir']}")
