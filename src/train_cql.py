import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

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
from d3rlpy.logging import FileAdapterFactory


class ComprehensiveRLTrainer:
    def __init__(
        self,
        dataset_name: str = "mujoco/walker2d/medium-v0",
        env_name: str = "Walker2d-v5",
        results_dir: str = "results",
    ):
        """
        Initialize the comprehensive RL trainer

        Args:
            dataset_name: Minari dataset name
            env_name: Gym environment name
            results_dir: Directory to save results
        """
        self.dataset_name = dataset_name
        self.env_name = env_name
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Load dataset and environment
        print(f"Loading dataset: {dataset_name}")
        self.dataset, self.env_info = get_minari(dataset_name)

        print(f"Creating environment: {env_name}")
        self.env = gym.make(env_name)
        self.eval_env = gym.make(env_name)

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

    def setup_evaluators(self):
        """Setup comprehensive evaluators for training and evaluation"""

        # Environment evaluator for policy evaluation
        env_evaluator = EnvironmentEvaluator(
            self.eval_env,
            n_trials=10,
            epsilon=0.0,
        )

        # TD Error evaluator
        td_error_evaluator = TDErrorEvaluator(
            episodes=self.dataset.episodes[:100]
        )

        # Value estimation evaluators
        avg_value_evaluator = AverageValueEstimationEvaluator(
            episodes=self.dataset.episodes[:100]
        )

        initial_value_evaluator = InitialStateValueEstimationEvaluator(
            episodes=self.dataset.episodes[:100]
        )

        # Advantage estimator
        advantage_evaluator = DiscountedSumOfAdvantageEvaluator(
            episodes=self.dataset.episodes[:100]
        )

        # OPC evaluator for offline policy evaluation
        opc_evaluator = SoftOPCEvaluator(
            return_threshold=self.calculate_return_threshold()
        )

        # Action difference evaluator
        action_diff_evaluator = ContinuousActionDiffEvaluator(
            episodes=self.dataset.episodes[:100]
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

    def evaluate_policy(self, algo, algo_name: str, epoch: int):
        """Comprehensive policy evaluation"""
        print(f"\nEvaluating {algo_name} at epoch {epoch}...")

        eval_scores = []
        eval_lengths = []

        for trial in range(10):
            obs, _ = self.eval_env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 1000:
                action = algo.predict([obs])[0]
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
                f"✓ New best {algo_name} model saved! Score: {score:.2f} at epoch {epoch}"
            )
            return True
        return False

    def train_cql(self, n_steps: int = 100000, n_steps_per_epoch: int = 10000):
        """Train CQL algorithm with comprehensive evaluation"""
        print("\n" + "=" * 50)
        print("TRAINING CQL ALGORITHM")
        print("=" * 50)

        # Setup CQL with optimized hyperparameters
        cql_config = CQLConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            alpha_learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            n_critics=2,
            initial_temperature=1.0,
            initial_alpha=5.0,
            alpha_threshold=10.0,
            conservative_weight=5.0,
            n_action_samples=10,
            soft_q_backup=False,
        )

        cql = cql_config.create(device="cpu")

        # Setup evaluators
        evaluators = self.setup_evaluators()

        # Setup logging
        log_adapter = FileAdapterFactory(
            root_dir=os.path.join(
                self.results_dir, f"cql_logs_{self.timestamp}"
            )
        )

        # Training with evaluation callbacks
        def evaluation_callback(algo, epoch):
            # Evaluate policy
            eval_results = self.evaluate_policy(algo, "cql", epoch)

            # Save best model
            self.save_best_model(
                algo, "cql", eval_results["mean_score"], epoch
            )

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(
                    self.results_dir,
                    f"cql_checkpoint_epoch_{epoch}_{self.timestamp}.d3",
                )
                algo.save(checkpoint_path)

        # Train CQL
        print(
            f"Training CQL for {n_steps} steps ({n_steps//n_steps_per_epoch} epochs)..."
        )
        epochs, metrics = cql.fit(
            self.dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            evaluators=evaluators,
            experiment_name=f"cql_walker2d_{self.timestamp}",
            with_timestamp=False,
            loggers=[log_adapter],
            save_interval=5,
            callback=evaluation_callback,
        )

        # Store training metrics
        self.training_metrics["cql"]["epochs"] = epochs
        self.training_metrics["cql"]["metrics"] = metrics

        # Final evaluation
        final_eval = self.evaluate_policy(cql, "cql", len(epochs))
        self.save_best_model(cql, "cql", final_eval["mean_score"], len(epochs))

        return cql, epochs, metrics

    def train_bc(self, n_steps: int = 20000, n_steps_per_epoch: int = 2000):
        """Train BC algorithm with comprehensive evaluation"""
        print("\n" + "=" * 50)
        print("TRAINING BC ALGORITHM (Baseline)")
        print("=" * 50)

        # Setup BC with optimized hyperparameters
        bc_config = BCConfig(
            learning_rate=1e-3, batch_size=256, weight_decay=1e-4
        )

        bc = bc_config.create(device="cpu")

        # Setup evaluators (subset for BC as it's simpler)
        evaluators = {
            "environment": EnvironmentEvaluator(self.eval_env, n_trials=5),
            "action_diff": ContinuousActionDiffEvaluator(
                episodes=self.dataset.episodes[:50]
            ),
        }

        # Setup logging
        log_adapter = FileAdapterFactory(
            root_dir=os.path.join(
                self.results_dir, f"bc_logs_{self.timestamp}"
            )
        )

        # Training with evaluation callbacks
        def evaluation_callback(algo, epoch):
            eval_results = self.evaluate_policy(algo, "bc", epoch)
            self.save_best_model(algo, "bc", eval_results["mean_score"], epoch)

        # Train BC
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
            loggers=[log_adapter],
            callback=evaluation_callback,
        )

        # Store training metrics
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

        # Create comprehensive results dictionary
        results = {
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "env_name": self.env_name,
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
            self.results_dir, f"comprehensive_results_{self.timestamp}.json"
        )
        with open(json_path, "w") as f:
            json_results = self._convert_numpy_to_list(results)
            json.dump(json_results, f, indent=2)

        # Save pickle results
        pickle_path = os.path.join(
            self.results_dir, f"comprehensive_results_{self.timestamp}.pkl"
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

        # Create evaluation summary CSV
        self._create_evaluation_csv()

        # Create training metrics CSV
        self._create_training_csv()

        # Generate plots
        self._generate_plots()

        # Create summary report
        self._create_summary_report()

        print(f"✓ Results saved to: {json_path}")
        print(f"✓ Results saved to: {pickle_path}")
        print(f"✓ CSV files and plots generated in: {self.results_dir}")

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

    def _generate_plots(self):
        """Generate comprehensive plots"""
        plt.style.use(
            "seaborn-v0_8"
            if "seaborn-v0_8" in plt.style.available
            else "default"
        )

        # Plot 1: Training curves comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Training Results Comparison - {self.timestamp}", fontsize=16
        )

        # Evaluation scores over time
        ax1 = axes[0, 0]
        for algo_name in ["cql", "bc"]:
            if self.evaluation_results[algo_name]:
                epochs = [
                    r["epoch"] for r in self.evaluation_results[algo_name]
                ]
                scores = [
                    r["mean_score"] for r in self.evaluation_results[algo_name]
                ]
                stds = [
                    r["std_score"] for r in self.evaluation_results[algo_name]
                ]

                ax1.plot(
                    epochs, scores, label=f"{algo_name.upper()}", marker="o"
                )
                ax1.fill_between(
                    epochs,
                    np.array(scores) - np.array(stds),
                    np.array(scores) + np.array(stds),
                    alpha=0.3,
                )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Evaluation Score")
        ax1.set_title("Evaluation Scores Over Time")
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
        ax2.set_title("Episode Lengths Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Score distributions (latest evaluation)
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
                    bins=10,
                )

        ax3.set_xlabel("Evaluation Score")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Final Score Distributions")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Best scores comparison
        ax4 = axes[1, 1]
        algorithms = []
        best_scores = []

        for algo_name in ["cql", "bc"]:
            if self.best_models[algo_name]["score"] != -np.inf:
                algorithms.append(algo_name.upper())
                best_scores.append(self.best_models[algo_name]["score"])

        if algorithms:
            bars = ax4.bar(algorithms, best_scores, color=["blue", "orange"])
            ax4.set_ylabel("Best Score")
            ax4.set_title("Best Scores Comparison")
            ax4.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, score in zip(bars, best_scores):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(best_scores) * 0.01,
                    f"{score:.1f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plot_path = os.path.join(
            self.results_dir, f"training_plots_{self.timestamp}.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Plots saved to: {plot_path}")

    def _create_summary_report(self):
        """Create a summary report in text format"""
        report_path = os.path.join(
            self.results_dir, f"summary_report_{self.timestamp}.txt"
        )

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE OFFLINE RL TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Environment: {self.env_name}\n\n")

            # Dataset statistics
            f.write("DATASET STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of episodes: {len(self.dataset.episodes)}\n")
            f.write(
                f"Total steps: {sum(len(ep) for ep in self.dataset.episodes)}\n"
            )
            f.write(f"Action space: {self.env.action_space}\n")
            f.write(f"Observation space: {self.env.observation_space}\n\n")

            # Best models summary
            f.write("BEST MODELS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            for algo_name in ["cql", "bc"]:
                best = self.best_models[algo_name]
                if best["score"] != -np.inf:
                    f.write(f"{algo_name.upper()}:\n")
                    f.write(f"  Best Score: {best['score']:.2f}\n")
                    f.write(f"  Best Epoch: {best['epoch']}\n")
                    f.write(f"  Model Path: {best['model_path']}\n\n")

            # Final evaluation results
            f.write("FINAL EVALUATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            for algo_name in ["cql", "bc"]:
                if self.evaluation_results[algo_name]:
                    final_result = self.evaluation_results[algo_name][-1]
                    f.write(f"{algo_name.upper()}:\n")
                    f.write(
                        f"  Mean Score: {final_result['mean_score']:.2f} ± {final_result['std_score']:.2f}\n"
                    )
                    f.write(
                        f"  Score Range: [{final_result['min_score']:.2f}, {final_result['max_score']:.2f}]\n"
                    )
                    f.write(
                        f"  Mean Episode Length: {final_result['mean_length']:.1f} ± {final_result['std_length']:.1f}\n\n"
                    )

            # Comparison
            if (
                len(self.evaluation_results["cql"]) > 0
                and len(self.evaluation_results["bc"]) > 0
            ):
                cql_score = self.evaluation_results["cql"][-1]["mean_score"]
                bc_score = self.evaluation_results["bc"][-1]["mean_score"]
                improvement = ((cql_score - bc_score) / abs(bc_score)) * 100

                f.write("PERFORMANCE COMPARISON:\n")
                f.write("-" * 40 + "\n")
                f.write(f"CQL vs BC improvement: {improvement:.1f}%\n")
                f.write(
                    f"Winner: {'CQL' if cql_score > bc_score else 'BC'}\n\n"
                )

        print(f"Summary report saved to: {report_path}")

    def run_complete_training(self):
        """Run complete training pipeline"""
        print("Starting comprehensive offline RL training pipeline...")
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Timestamp: {self.timestamp}")

        cql_model, cql_epochs, cql_metrics = self.train_cql()
        bc_model, bc_epochs, bc_metrics = self.train_bc()

        self.save_comprehensive_results()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Best CQL Score: {self.best_models['cql']['score']:.2f}")
        print(f"Best BC Score: {self.best_models['bc']['score']:.2f}")
        print(f"Results saved in: {self.results_dir}")
        print("=" * 80)

        return {
            "cql_model": cql_model,
            "bc_model": bc_model,
            "results_dir": self.results_dir,
            "timestamp": self.timestamp,
        }


if __name__ == "__main__":
    trainer = ComprehensiveRLTrainer(
        dataset_name="mujoco/walker2d/medium-v0",
        env_name="Walker2d-v5",
        results_dir="walker2d_cql_results",
    )

    results = trainer.run_complete_training()

    print("\nTraining pipeline completed!")
    print(f"Check the results directory: {results['results_dir']}")
