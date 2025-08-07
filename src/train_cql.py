import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import torch
import glob

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


class ImprovedRLTrainer:
    def __init__(
        self,
        dataset_name: str = "mujoco/walker2d/medium-v0",
        env_name: str = "Walker2d-v5",
        results_dir: str = "results",
        fast_mode: bool = False,
        resume_from_checkpoint: bool = False,
        checkpoint_dir: str = None,
    ):
        """
        Initialize the RL trainer

        Args:
            resume_from_checkpoint: Whether to resume from existing checkpoint
            checkpoint_dir: Directory to look for checkpoints (if None, uses results_dir)
        """
        self.dataset_name = dataset_name
        self.env_name = env_name
        self.results_dir = results_dir
        self.fast_mode = fast_mode
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_dir = checkpoint_dir or results_dir

        # If resuming, try to load existing timestamp, otherwise create new one
        if resume_from_checkpoint:
            self.timestamp = (
                self._find_latest_timestamp()
                or datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        else:
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

        # Calculate D4RL normalization scores
        self.min_score, self.max_score = self._calculate_d4rl_scores()
        print(
            f"D4RL normalization: min={self.min_score:.2f}, max={self.max_score:.2f}"
        )

        # Initialize metrics storage
        self.training_metrics = {
            "cql": {"epochs": [], "metrics": []},
            "bc": {"epochs": [], "metrics": []},
        }

        self.evaluation_results = {"cql": [], "bc": []}

        # Best model tracking
        self.best_models = {
            "cql": {
                "score": -np.inf,
                "normalized_score": -np.inf,
                "epoch": 0,
                "model_path": None,
            },
            "bc": {
                "score": -np.inf,
                "normalized_score": -np.inf,
                "epoch": 0,
                "model_path": None,
            },
        }

        # Load previous results if resuming
        if resume_from_checkpoint:
            self._load_previous_results()

    def _find_latest_timestamp(self):
        """Find the latest timestamp from existing results"""
        pattern = os.path.join(self.checkpoint_dir, "improved_results_*.json")
        result_files = glob.glob(pattern)

        if result_files:
            # Extract timestamps from filenames and find the latest
            timestamps = []
            for file in result_files:
                filename = os.path.basename(file)
                # Extract timestamp from filename like "improved_results_20240107_143022.json"
                if filename.startswith(
                    "improved_results_"
                ) and filename.endswith(".json"):
                    timestamp_part = filename.replace(
                        "improved_results_", ""
                    ).replace(".json", "")
                    timestamps.append(timestamp_part)

            if timestamps:
                latest_timestamp = max(timestamps)
                print(
                    f"Found existing results with timestamp: {latest_timestamp}"
                )
                return latest_timestamp

        return None

    def _load_previous_results(self):
        """Load previous training results and metrics"""
        results_file = os.path.join(
            self.checkpoint_dir, f"improved_results_{self.timestamp}.json"
        )

        if os.path.exists(results_file):
            print(f"Loading previous results from: {results_file}")

            try:
                with open(results_file, "r") as f:
                    previous_results = json.load(f)

                # Restore training metrics
                if "training_metrics" in previous_results:
                    self.training_metrics = previous_results[
                        "training_metrics"
                    ]

                # Restore evaluation results
                if "evaluation_results" in previous_results:
                    self.evaluation_results = previous_results[
                        "evaluation_results"
                    ]

                # Restore best models
                if "best_models" in previous_results:
                    self.best_models = previous_results["best_models"]

                print("✅ Previous results loaded successfully!")
                print(
                    f"CQL: {len(self.evaluation_results.get('cql', []))} evaluations"
                )
                print(
                    f"BC: {len(self.evaluation_results.get('bc', []))} evaluations"
                )

            except Exception as e:
                print(f"⚠️ Warning: Could not load previous results: {e}")
        else:
            print(f"No previous results found at: {results_file}")

    def _find_latest_checkpoint(self, algo_name: str):
        """Find the latest checkpoint for a given algorithm"""
        pattern = os.path.join(
            self.checkpoint_dir,
            f"{algo_name}_checkpoint_epoch_*_{self.timestamp}.d3",
        )
        checkpoint_files = glob.glob(pattern)

        if not checkpoint_files:
            # Also try to find checkpoints with any timestamp
            pattern = os.path.join(
                self.checkpoint_dir, f"{algo_name}_checkpoint_epoch_*.d3"
            )
            checkpoint_files = glob.glob(pattern)

        if checkpoint_files:
            # Extract epoch numbers and find the latest
            latest_checkpoint = None
            latest_epoch = -1

            for checkpoint_file in checkpoint_files:
                try:
                    # Extract epoch number from filename
                    filename = os.path.basename(checkpoint_file)
                    # Filename format: algo_checkpoint_epoch_EPOCH_timestamp.d3
                    parts = filename.split("_")
                    epoch_idx = parts.index("epoch") + 1
                    epoch_num = int(parts[epoch_idx])

                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_checkpoint = checkpoint_file

                except (ValueError, IndexError):
                    continue

            if latest_checkpoint:
                print(
                    f"Found {algo_name.upper()} checkpoint at epoch {latest_epoch}: {latest_checkpoint}"
                )
                return latest_checkpoint, latest_epoch

        return None, 0

    def _load_checkpoint(self, algo, checkpoint_path):
        """Load a checkpoint into the algorithm"""
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            algo.load(checkpoint_path)
            print("✅ Checkpoint loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False

    def _setup_device(self):
        """Setup optimal device for training"""
        if torch.cuda.is_available():
            device = "cuda:0"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = "cpu"
            print("Using CPU (GPU not available)")

        return device

    def _calculate_d4rl_scores(self):
        """Calculate min/max scores for D4RL normalization"""
        # For Walker2d-medium, these are approximate D4RL normalization values
        # Based on literature values for proper comparison
        min_score = 1.629  # Random policy score
        max_score = 4592.3  # Expert policy score
        return min_score, max_score

    def normalize_d4rl_score(self, raw_score):
        """Convert raw score to D4RL normalized score (0-100+)"""
        return (
            100
            * (raw_score - self.min_score)
            / (self.max_score - self.min_score)
        )

    def train_cql(self, n_steps: int = None, n_steps_per_epoch: int = None):
        """Train CQL model"""
        print("\n" + "=" * 50)
        print("TRAINING CQL ALGORITHM")
        print("=" * 50)

        # Improved training parameters
        if self.fast_mode:
            n_steps = n_steps or 100000
            n_steps_per_epoch = n_steps_per_epoch or 5000
            print("Fast mode enabled - using moderate training steps")
        else:
            n_steps = n_steps or 1000000
            n_steps_per_epoch = n_steps_per_epoch or 10000

        # Create CQL configuration
        cql_config = CQLConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            alpha_learning_rate=3e-4,
            batch_size=512 if self.device.startswith("cuda") else 256,
            gamma=0.99,
            tau=0.005,
            n_critics=2,
            initial_temperature=1.0,
            initial_alpha=5.0,
            alpha_threshold=10.0,
            conservative_weight=5.0,
            n_action_samples=10,
            soft_q_backup=True,
        )

        cql = cql_config.create(device=self.device)

        # Check for checkpoint resumption
        start_epoch = 0
        if self.resume_from_checkpoint:
            checkpoint_path, resume_epoch = self._find_latest_checkpoint("cql")
            if checkpoint_path and self._load_checkpoint(cql, checkpoint_path):
                start_epoch = resume_epoch
                print(f"🔄 Resuming CQL training from epoch {start_epoch}")

                # Adjust remaining steps
                completed_steps = start_epoch * n_steps_per_epoch
                remaining_steps = max(0, n_steps - completed_steps)

                if remaining_steps <= 0:
                    print("✅ CQL training already completed!")
                    return (
                        cql,
                        self.training_metrics["cql"]["epochs"],
                        self.training_metrics["cql"]["metrics"],
                    )

                n_steps = remaining_steps
                print(f"Remaining steps: {remaining_steps}")

        # Setup evaluators
        evaluators = self.setup_evaluators(lightweight=False)

        # Evaluation callback with checkpoint saving
        eval_frequency = 5 if self.fast_mode else 2

        def evaluation_callback(algo, epoch, total_step):
            actual_epoch = start_epoch + epoch

            if epoch % eval_frequency == 0:
                eval_results = self.evaluate_policy(
                    algo, "cql", actual_epoch, n_trials=10
                )
                normalized_score = self.normalize_d4rl_score(
                    eval_results["mean_score"]
                )
                eval_results["normalized_score"] = normalized_score

                print(
                    f"CQL Epoch {actual_epoch}: Raw Score = {eval_results['mean_score']:.2f}, "
                    f"Normalized Score = {normalized_score:.2f}"
                )

                self.save_best_model(
                    algo,
                    "cql",
                    eval_results["mean_score"],
                    normalized_score,
                    actual_epoch,
                )

            # Save checkpoints more frequently
            if epoch % 20 == 0 and epoch > 0:
                checkpoint_path = os.path.join(
                    self.results_dir,
                    f"cql_checkpoint_epoch_{actual_epoch}_{self.timestamp}.d3",
                )
                algo.save(checkpoint_path)
                print(f"💾 CQL checkpoint saved at epoch {actual_epoch}")

                # Also save current results
                self.save_comprehensive_results()

        print(f"Training CQL for {n_steps} steps...")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")

        res = cql.fit(
            self.dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            evaluators=evaluators,
            experiment_name=f"improved_cql_walker2d_{self.timestamp}",
            with_timestamp=False,
            save_interval=20,
            callback=evaluation_callback,
        )

        # Adjust epoch numbers for resumed training
        epochs, metrics = zip(*res)
        adjusted_epochs = [start_epoch + epoch for epoch in epochs]

        # Update training metrics
        if start_epoch > 0:
            # Append to existing metrics
            self.training_metrics["cql"]["epochs"].extend(adjusted_epochs)
            self.training_metrics["cql"]["metrics"].extend(list(metrics))
        else:
            # Replace metrics
            self.training_metrics["cql"]["epochs"] = adjusted_epochs
            self.training_metrics["cql"]["metrics"] = list(metrics)

        # Final evaluation
        final_epoch = adjusted_epochs[-1] if adjusted_epochs else start_epoch
        final_eval = self.evaluate_policy(cql, "cql", final_epoch, n_trials=20)
        final_normalized = self.normalize_d4rl_score(final_eval["mean_score"])
        final_eval["normalized_score"] = final_normalized
        self.save_best_model(
            cql, "cql", final_eval["mean_score"], final_normalized, final_epoch
        )

        return cql, adjusted_epochs, list(metrics)

    def train_bc(self, n_steps: int = None, n_steps_per_epoch: int = None):
        """Train BC model"""
        print("\n" + "=" * 50)
        print("TRAINING BC ALGORITHM")
        print("=" * 50)

        # Training parameters
        if self.fast_mode:
            n_steps = n_steps or 50000
            n_steps_per_epoch = n_steps_per_epoch or 2500
        else:
            n_steps = n_steps or 200000
            n_steps_per_epoch = n_steps_per_epoch or 5000

        # Create BC configuration
        bc_config = BCConfig(
            learning_rate=1e-3,
            batch_size=512 if self.device.startswith("cuda") else 256,
        )

        bc = bc_config.create(device=self.device)

        # Check for checkpoint resumption
        start_epoch = 0
        if self.resume_from_checkpoint:
            checkpoint_path, resume_epoch = self._find_latest_checkpoint("bc")
            if checkpoint_path and self._load_checkpoint(bc, checkpoint_path):
                start_epoch = resume_epoch
                print(f"🔄 Resuming BC training from epoch {start_epoch}")

                # Adjust remaining steps
                completed_steps = start_epoch * n_steps_per_epoch
                remaining_steps = max(0, n_steps - completed_steps)

                if remaining_steps <= 0:
                    print("✅ BC training already completed!")
                    return (
                        bc,
                        self.training_metrics["bc"]["epochs"],
                        self.training_metrics["bc"]["metrics"],
                    )

                n_steps = remaining_steps
                print(f"Remaining steps: {remaining_steps}")

        # Setup evaluators
        evaluators = self.setup_evaluators(lightweight=False)

        def evaluation_callback(algo, epoch, total_step):
            actual_epoch = start_epoch + epoch

            if epoch % 5 == 0:
                eval_results = self.evaluate_policy(
                    algo, "bc", actual_epoch, n_trials=10
                )
                normalized_score = self.normalize_d4rl_score(
                    eval_results["mean_score"]
                )
                eval_results["normalized_score"] = normalized_score

                print(
                    f"BC Epoch {actual_epoch}: Raw Score = {eval_results['mean_score']:.2f}, "
                    f"Normalized Score = {normalized_score:.2f}"
                )

                self.save_best_model(
                    algo,
                    "bc",
                    eval_results["mean_score"],
                    normalized_score,
                    actual_epoch,
                )

            # Save checkpoints
            if epoch % 10 == 0 and epoch > 0:  # Every 10 epochs for BC
                checkpoint_path = os.path.join(
                    self.results_dir,
                    f"bc_checkpoint_epoch_{actual_epoch}_{self.timestamp}.d3",
                )
                algo.save(checkpoint_path)
                print(f"💾 BC checkpoint saved at epoch {actual_epoch}")

                # Also save current results
                self.save_comprehensive_results()

        print(f"Training BC for {n_steps} steps...")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")

        res = bc.fit(
            self.dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            evaluators=evaluators,
            experiment_name=f"improved_bc_walker2d_{self.timestamp}",
            with_timestamp=False,
            callback=evaluation_callback,
        )

        # Adjust epoch numbers for resumed training
        epochs, metrics = zip(*res)
        adjusted_epochs = [start_epoch + epoch for epoch in epochs]

        # Update training metrics
        if start_epoch > 0:
            # Append to existing metrics
            self.training_metrics["bc"]["epochs"].extend(adjusted_epochs)
            self.training_metrics["bc"]["metrics"].extend(list(metrics))
        else:
            # Replace metrics
            self.training_metrics["bc"]["epochs"] = adjusted_epochs
            self.training_metrics["bc"]["metrics"] = list(metrics)

        # Final evaluation
        final_epoch = adjusted_epochs[-1] if adjusted_epochs else start_epoch
        final_eval = self.evaluate_policy(bc, "bc", final_epoch, n_trials=20)
        final_normalized = self.normalize_d4rl_score(final_eval["mean_score"])
        final_eval["normalized_score"] = final_normalized
        self.save_best_model(
            bc, "bc", final_eval["mean_score"], final_normalized, final_epoch
        )

        return bc, adjusted_epochs, list(metrics)

    def setup_evaluators(self, lightweight: bool = False):
        """Setup evaluators with optimizations"""
        if lightweight:
            return {
                "environment": EnvironmentEvaluator(
                    self.eval_env,
                    n_trials=5,
                    epsilon=0.0,
                )
            }

        # Comprehensive evaluators
        env_evaluator = EnvironmentEvaluator(
            self.eval_env,
            n_trials=10,
            epsilon=0.0,
        )

        eval_episodes = self.dataset.episodes[:100]

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
        """Enhanced policy evaluation with D4RL normalization"""
        if n_trials is None:
            n_trials = 5 if self.fast_mode else 10

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
            max_steps = 1000

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

        # Add normalized score
        normalized_score = self.normalize_d4rl_score(
            eval_results["mean_score"]
        )
        eval_results["normalized_score"] = normalized_score

        self.evaluation_results[algo_name].append(eval_results)

        print(f"{algo_name} Evaluation Results:")
        print(
            f"  Raw Score: {eval_results['mean_score']:.2f} ± {eval_results['std_score']:.2f}"
        )
        print(f"  D4RL Normalized Score: {normalized_score:.2f}")
        print(
            f"  Score Range: [{eval_results['min_score']:.2f}, {eval_results['max_score']:.2f}]"
        )
        print(
            f"  Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}"
        )

        return eval_results

    def save_best_model(
        self,
        algo,
        algo_name: str,
        score: float,
        normalized_score: float,
        epoch: int,
    ):
        """Save model if it's the best so far"""
        if normalized_score > self.best_models[algo_name]["normalized_score"]:
            model_path = os.path.join(
                self.results_dir, f"best_{algo_name}_model_{self.timestamp}.d3"
            )
            algo.save(model_path)

            self.best_models[algo_name].update(
                {
                    "score": score,
                    "normalized_score": normalized_score,
                    "epoch": epoch,
                    "model_path": model_path,
                }
            )

            print(f"🎉 NEW BEST {algo_name.upper()} MODEL!")
            print(
                f"   Raw Score: {score:.2f} | Normalized Score: {normalized_score:.2f} | Epoch: {epoch}"
            )
            return True
        return False

    def run_complete_training(self):
        """Run training pipeline with checkpoint resumption"""
        print(
            "🚀 Starting IMPROVED offline RL training pipeline with checkpoint resumption..."
        )
        print(f"Device: {self.device}")
        print(f"Fast mode: {self.fast_mode}")
        print(f"Resume from checkpoint: {self.resume_from_checkpoint}")
        print(f"Results directory: {self.results_dir}")

        # Train with checkpoint resumption
        cql_model, cql_epochs, cql_metrics = self.train_cql()
        bc_model, bc_epochs, bc_metrics = self.train_bc()

        # Save final results
        self.save_comprehensive_results()

        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETED WITH CHECKPOINT SUPPORT!")
        print("=" * 70)
        print(f"Best CQL Raw Score: {self.best_models['cql']['score']:.2f}")
        print(
            f"Best CQL D4RL Score: {self.best_models['cql']['normalized_score']:.2f}"
        )
        print(f"Best BC Raw Score: {self.best_models['bc']['score']:.2f}")
        print(
            f"Best BC D4RL Score: {self.best_models['bc']['normalized_score']:.2f}"
        )
        print(f"Results saved in: {self.results_dir}")
        print("=" * 70)

        return {
            "cql_model": cql_model,
            "bc_model": bc_model,
            "results_dir": self.results_dir,
            "timestamp": self.timestamp,
            "device": self.device,
            "best_scores": self.best_models,
        }

    def save_comprehensive_results(self):
        """Enhanced results saving with D4RL normalization"""
        results = {
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "env_name": self.env_name,
            "fast_mode": self.fast_mode,
            "device": self.device,
            "d4rl_normalization": {
                "min_score": self.min_score,
                "max_score": self.max_score,
            },
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
            self.results_dir, f"improved_results_{self.timestamp}.json"
        )
        with open(json_path, "w") as f:
            json_results = self._convert_numpy_to_list(results)
            json.dump(json_results, f, indent=2)

        # Create summary report
        self._create_enhanced_summary_report()

        print(f"✅ Enhanced results saved to: {json_path}")

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

    def _create_enhanced_summary_report(self):
        """Create enhanced summary report with checkpoint information"""
        report_path = os.path.join(
            self.results_dir, f"enhanced_summary_{self.timestamp}.txt"
        )

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(
                "ENHANCED OFFLINE RL TRAINING REPORT WITH CHECKPOINT SUPPORT\n"
            )
            f.write("=" * 70 + "\n\n")

            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Fast Mode: {self.fast_mode}\n")
            f.write(f"Resume from Checkpoint: {self.resume_from_checkpoint}\n")
            f.write(f"Device: {self.device}\n\n")

            # D4RL Normalization Info
            f.write("D4RL NORMALIZATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Min Score (Random): {self.min_score:.2f}\n")
            f.write(f"Max Score (Expert): {self.max_score:.2f}\n\n")

            # Training Progress
            f.write("TRAINING PROGRESS:\n")
            f.write("-" * 30 + "\n")
            if self.training_metrics["cql"]["epochs"]:
                f.write(
                    f"CQL Epochs Completed: {max(self.training_metrics['cql']['epochs'])}\n"
                )
            if self.training_metrics["bc"]["epochs"]:
                f.write(
                    f"BC Epochs Completed: {max(self.training_metrics['bc']['epochs'])}\n"
                )
            f.write("\n")

            # Final results
            f.write("FINAL RESULTS:\n")
            f.write("-" * 30 + "\n")
            for algo_name in ["cql", "bc"]:
                best = self.best_models[algo_name]
                if best["score"] != -np.inf:
                    f.write(f"{algo_name.upper()}:\n")
                    f.write(f"  Raw Score: {best['score']:.2f}\n")
                    f.write(
                        f"  D4RL Normalized Score: {best['normalized_score']:.2f}\n"
                    )
                    f.write(f"  Best Epoch: {best['epoch']}\n")
                    f.write(f"  Model Path: {best['model_path']}\n\n")

            # Performance Assessment
            f.write("PERFORMANCE ASSESSMENT:\n")
            f.write("-" * 30 + "\n")
            for algo_name in ["cql", "bc"]:
                normalized_score = self.best_models[algo_name][
                    "normalized_score"
                ]
                if normalized_score > 80:
                    assessment = "EXCELLENT"
                elif normalized_score > 60:
                    assessment = "GOOD"
                elif normalized_score > 40:
                    assessment = "FAIR"
                elif normalized_score > 20:
                    assessment = "POOR"
                else:
                    assessment = "VERY POOR"

                f.write(
                    f"{algo_name.upper()}: {assessment} ({normalized_score:.2f})\n"
                )

            f.write("\n")

            # Dataset Statistics
            f.write("DATASET STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of Episodes: {len(self.dataset.episodes)}\n")
            f.write(
                f"Total Steps: {sum(len(ep) for ep in self.dataset.episodes)}\n"
            )
            f.write(f"Action Space: {self.env.action_space}\n")
            f.write(f"Observation Space: {self.env.observation_space}\n\n")

            # Training Configuration
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Device Used: {self.device}\n")
            f.write(f"Fast Mode: {self.fast_mode}\n")
            f.write(f"Checkpoint Resumption: {self.resume_from_checkpoint}\n")
            f.write(f"Results Directory: {self.results_dir}\n\n")

            # Evaluation Summary
            f.write("EVALUATION SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for algo_name in ["cql", "bc"]:
                evaluations = self.evaluation_results[algo_name]
                if evaluations:
                    f.write(
                        f"{algo_name.upper()} - {len(evaluations)} evaluations performed\n"
                    )
                    best_eval = max(
                        evaluations, key=lambda x: x["normalized_score"]
                    )
                    f.write(
                        f"  Best Performance: {best_eval['normalized_score']:.2f} at epoch {best_eval['epoch']}\n"
                    )
            f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            cql_score = self.best_models["cql"]["normalized_score"]
            bc_score = self.best_models["bc"]["normalized_score"]

            if cql_score > bc_score:
                f.write(
                    "• CQL outperformed BC - Consider using CQL for deployment\n"
                )
            else:
                f.write(
                    "• BC outperformed CQL - Consider using BC for deployment\n"
                )

            if max(cql_score, bc_score) < 40:
                f.write("• Low performance detected - Consider:\n")
                f.write("  - Increasing training steps\n")
                f.write("  - Tuning hyperparameters\n")
                f.write("  - Using different dataset or algorithm\n")
            elif max(cql_score, bc_score) < 60:
                f.write(
                    "• Moderate performance - Consider hyperparameter tuning\n"
                )
            else:
                f.write("• Good performance achieved!\n")

        print(f"✅ Enhanced summary report saved to: {report_path}")

    def create_visualization_plots(self):
        """Create comprehensive visualization plots"""
        plt.style.use(
            "seaborn-v0_8"
            if "seaborn-v0_8" in plt.style.available
            else "default"
        )

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Offline RL Training Results - {self.timestamp}",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Training Progress (Normalized Scores)
        ax1 = axes[0, 0]
        for algo_name in ["cql", "bc"]:
            evaluations = self.evaluation_results[algo_name]
            if evaluations:
                epochs = [eval_result["epoch"] for eval_result in evaluations]
                scores = [
                    eval_result["normalized_score"]
                    for eval_result in evaluations
                ]
                ax1.plot(
                    epochs,
                    scores,
                    marker="o",
                    label=f"{algo_name.upper()}",
                    linewidth=2,
                    markersize=4,
                )

        ax1.set_xlabel("Training Epoch")
        ax1.set_ylabel("D4RL Normalized Score")
        ax1.set_title("Training Progress (D4RL Normalized)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Raw Score Distribution
        ax2 = axes[0, 1]
        for algo_name in ["cql", "bc"]:
            evaluations = self.evaluation_results[algo_name]
            if evaluations:
                # Get all individual scores from all evaluations
                all_scores = []
                for eval_result in evaluations:
                    all_scores.extend(eval_result["scores"])
                ax2.hist(
                    all_scores,
                    alpha=0.7,
                    label=f"{algo_name.upper()}",
                    bins=20,
                    density=True,
                )

        ax2.set_xlabel("Raw Score")
        ax2.set_ylabel("Density")
        ax2.set_title("Score Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Episode Length Analysis
        ax3 = axes[1, 0]
        for algo_name in ["cql", "bc"]:
            evaluations = self.evaluation_results[algo_name]
            if evaluations:
                epochs = [eval_result["epoch"] for eval_result in evaluations]
                lengths = [
                    eval_result["mean_length"] for eval_result in evaluations
                ]
                ax3.plot(
                    epochs,
                    lengths,
                    marker="s",
                    label=f"{algo_name.upper()}",
                    linewidth=2,
                    markersize=4,
                )

        ax3.set_xlabel("Training Epoch")
        ax3.set_ylabel("Mean Episode Length")
        ax3.set_title("Episode Length Progress")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Performance Comparison
        ax4 = axes[1, 1]
        algos = []
        raw_scores = []
        normalized_scores = []

        for algo_name in ["cql", "bc"]:
            if self.best_models[algo_name]["score"] != -np.inf:
                algos.append(algo_name.upper())
                raw_scores.append(self.best_models[algo_name]["score"])
                normalized_scores.append(
                    self.best_models[algo_name]["normalized_score"]
                )

        if algos:
            x_pos = np.arange(len(algos))
            width = 0.35

            ax4.bar(
                x_pos - width / 2,
                raw_scores,
                width,
                label="Raw Score",
                alpha=0.8,
            )
            ax4_twin = ax4.twinx()
            ax4_twin.bar(
                x_pos + width / 2,
                normalized_scores,
                width,
                label="D4RL Normalized",
                alpha=0.8,
                color="orange",
            )

            ax4.set_xlabel("Algorithm")
            ax4.set_ylabel("Raw Score", color="blue")
            ax4_twin.set_ylabel("D4RL Normalized Score", color="orange")
            ax4.set_title("Best Performance Comparison")
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(algos)

            # Add value labels on bars
            for i, (raw, norm) in enumerate(
                zip(raw_scores, normalized_scores)
            ):
                ax4.text(
                    i - width / 2,
                    raw + max(raw_scores) * 0.01,
                    f"{raw:.1f}",
                    ha="center",
                    va="bottom",
                )
                ax4_twin.text(
                    i + width / 2,
                    norm + max(normalized_scores) * 0.01,
                    f"{norm:.1f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(
            self.results_dir, f"training_analysis_{self.timestamp}.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Visualization plots saved to: {plot_path}")
        return plot_path

    def generate_training_report(self):
        """Generate a comprehensive training report with all results"""
        report_data = {
            "training_summary": {
                "timestamp": self.timestamp,
                "dataset": self.dataset_name,
                "environment": self.env_name,
                "device": self.device,
                "fast_mode": self.fast_mode,
                "resume_checkpoint": self.resume_from_checkpoint,
            },
            "best_performances": self.best_models,
            "training_progress": {
                "cql_evaluations": len(self.evaluation_results["cql"]),
                "bc_evaluations": len(self.evaluation_results["bc"]),
                "total_epochs": {
                    "cql": (
                        max(self.training_metrics["cql"]["epochs"])
                        if self.training_metrics["cql"]["epochs"]
                        else 0
                    ),
                    "bc": (
                        max(self.training_metrics["bc"]["epochs"])
                        if self.training_metrics["bc"]["epochs"]
                        else 0
                    ),
                },
            },
            "dataset_info": {
                "num_episodes": len(self.dataset.episodes),
                "total_steps": sum(len(ep) for ep in self.dataset.episodes),
                "d4rl_normalization": {
                    "min_score": self.min_score,
                    "max_score": self.max_score,
                },
            },
        }

        return report_data


def main():
    """Main function to run the RL trainer"""

    print("=" * 60)
    print("Training started...")
    print("=" * 60)

    trainer = ImprovedRLTrainer(
        dataset_name="mujoco/walker2d/medium-v0",
        env_name="Walker2d-v5",
        results_dir="improved_results",
        fast_mode=True,
        resume_from_checkpoint=True,
    )

    results_resumed = trainer.run_complete_training()

    return results_resumed


if __name__ == "__main__":
    # Run the training pipeline
    fresh_results, resumed_results = main()
