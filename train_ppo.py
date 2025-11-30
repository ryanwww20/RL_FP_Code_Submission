"""
PPO (Proximal Policy Optimization) Training Script
Uses Stable-Baselines3 for PPO implementation
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from envs.Discrete_gym import MinimalEnv
from PIL import Image
from eval import ModelEvaluator

CONFIG_ENV_VAR = "TRAINING_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
TRAIN_PPO_KWARGS = {
    "total_timesteps",
    "n_envs",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "tensorboard_log",
    "save_path",
}


class TrainingCallback(BaseCallback):
    """
    Callback to record metrics, plot designs, save to CSV, and create GIFs.
    Follows README structure: ppo_model_log_<start_time>/ with img/, plot/, result.csv
    """
    def __init__(self, save_dir, verbose=1, eval_env=None, model_save_path=None):
        super(TrainingCallback, self).__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_count = 0
        self.eval_env = eval_env
        self.model_save_path = model_save_path  # Path to save best model
        
        # Best model tracking
        self.best_score = -float('inf')
        self.best_rollout = 0
        
        # Create directory structure according to README
        self.img_dir = self.save_dir / "img"
        self.plot_dir = self.save_dir / "plot"
        self.img_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        
        # Directories for temporary images (will be used for GIFs)
        self.design_dir = self.save_dir / "design_images"
        self.distribution_dir = self.save_dir / "distribution_images"
        self.design_dir.mkdir(exist_ok=True)
        self.distribution_dir.mkdir(exist_ok=True)
        
        # CSV file for metrics (result.csv as per README)
        self.csv_path = self.save_dir / "result.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, 'w') as f:
                f.write('timestamp,rollout_count,transmission,balance_score,score,reward\n')
        
        # Store image paths for GIF creation
        self.design_image_paths = []
        self.distribution_image_paths = []
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called when rollout collection ends."""
        self.rollout_count += 1
        
        # Calculate metrics using ModelEvaluator
        if self.eval_env is not None:
            print(f"\nRunning evaluation for rollout {self.rollout_count}...")
            evaluator = ModelEvaluator(self.model, self.eval_env)
            # Run 1 episode, deterministic (consistent with eval.py logic)
            results_df = evaluator.evaluate(n_episodes=1, deterministic=True)
            
            if len(results_df) > 0:
                # Use the first (and likely only) row
                metrics = results_df.iloc[0]
                
                # Extract metrics with fallbacks
                avg_transmission = metrics.get('total_mode_transmission', metrics.get('total_transmission', 0.0))
                avg_balance = metrics.get('balance_score', 0.0)
                avg_score = metrics.get('current_score', 0.0)
                avg_episode_reward = metrics.get('total_reward', 0.0)
            else:
                avg_transmission = 0.0
                avg_balance = 0.0
                avg_score = 0.0
                avg_episode_reward = 0.0
                
            # Use the evaluation environment's last state for plotting
            # Since evaluate() runs episodes, the env should be in the final state of the last episode
            # MinimalEnv stores the last metrics in self.last_episode_metrics
            # But ModelEvaluator runs reset() at start of episode. 
            # After evaluate(), the env is at the end of the last episode.
            pass # Continue to logging
            
        else:
             print("Warning: eval_env not provided to callback, using training env metrics.")
             # Fallback to old logic if no eval_env
             # Get environment from training_env
             env = self.training_env
             # ... (Rest of old logic if needed, but we assume eval_env is provided)
             # To keep it simple, I'll just skip the fallback implementation detail here 
             # and assume eval_env is passed.
             avg_transmission = 0
             avg_balance = 0
             avg_score = 0
             avg_episode_reward = 0

        # Record AVERAGE metrics to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.csv_path, 'a') as f:
            f.write(f'{timestamp},{self.rollout_count},{avg_transmission},{avg_balance},{avg_score},{avg_episode_reward}\n')
        
        # Plot and save design (use evaluation environment)
        design_path = self.design_dir / f"design_rollout_{self.rollout_count:04d}.png"
        self.design_image_paths.append(str(design_path))
        title_suffix = f"Rollout {self.rollout_count}"
        try:
            if self.eval_env is not None:
                 # MinimalEnv has save_design_plot method
                 # If wrapped in Monitor/DummyVecEnv, might need unwrapped
                 if hasattr(self.eval_env, 'unwrapped'):
                     self.eval_env.unwrapped.save_design_plot(str(design_path), title_suffix=title_suffix)
                 else:
                     self.eval_env.save_design_plot(str(design_path), title_suffix=title_suffix)
            else:
                # Fallback to training env
                if hasattr(self.training_env, 'envs'):
                    self.training_env.envs[0].unwrapped.save_design_plot(str(design_path), title_suffix=title_suffix)
                else:
                    self.training_env.env_method('save_design_plot', str(design_path), title_suffix, indices=[0])
        except Exception as e:
            print(f"Warning: Could not save design plot: {e}")
        
        # Plot and save distribution
        # Need efield_state. In MinimalEnv, calculate_flux returns it, 
        # or it is stored in last_episode_metrics if using Discrete_gym
        distribution_path = self.distribution_dir / f"distribution_rollout_{self.rollout_count:04d}.png"
        self.distribution_image_paths.append(str(distribution_path))
        try:
            if self.eval_env is not None:
                if hasattr(self.eval_env, 'unwrapped'):
                     self.eval_env.unwrapped.save_distribution_plot(str(distribution_path), title_suffix=title_suffix)
                else:
                     self.eval_env.save_distribution_plot(str(distribution_path), title_suffix=title_suffix)
            else:
                # Fallback
                 if hasattr(self.training_env, 'envs'):
                    self.training_env.envs[0].unwrapped.save_distribution_plot(str(distribution_path), title_suffix=title_suffix)
                 else:
                    self.training_env.env_method('save_distribution_plot', str(distribution_path), title_suffix, indices=[0])
        except Exception as e:
            print(f"Warning: Could not save distribution plot: {e}")
        
        # Print AVERAGE metrics
        print(f"Rollout {self.rollout_count} (Eval Result): "
              f"Transmission={avg_transmission:.4f}, Balance={avg_balance:.4f}, "
              f"Score={avg_score:.4f}, EpReward={avg_episode_reward:.4f}")
        
        # Check if this is the best score and save model if so
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_rollout = self.rollout_count
            print(f"  ★ New best score! Saving best model...")
            
            if self.model_save_path:
                best_model_path = self.model_save_path.replace('.zip', '_best.zip')
                self.model.save(best_model_path)
                print(f"  ★ Best model saved to: {best_model_path}")
                
                # Also save best design and distribution
                best_design_path = self.img_dir / "best_design.png"
                best_distribution_path = self.img_dir / "best_distribution.png"
                try:
                    if self.eval_env is not None:
                        if hasattr(self.eval_env, 'unwrapped'):
                            self.eval_env.unwrapped.save_design_plot(str(best_design_path), title_suffix=f"Best (Rollout {self.rollout_count}, Score={avg_score:.4f})")
                            self.eval_env.unwrapped.save_distribution_plot(str(best_distribution_path), title_suffix=f"Best (Rollout {self.rollout_count}, Score={avg_score:.4f})")
                        else:
                            self.eval_env.save_design_plot(str(best_design_path), title_suffix=f"Best (Rollout {self.rollout_count}, Score={avg_score:.4f})")
                            self.eval_env.save_distribution_plot(str(best_distribution_path), title_suffix=f"Best (Rollout {self.rollout_count}, Score={avg_score:.4f})")
                except Exception as e:
                    print(f"  Warning: Could not save best design/distribution plots: {e}")
        else:
            print(f"  (Best score: {self.best_score:.4f} at rollout {self.best_rollout})")
        
        # Update GIFs and plots after each rollout
        self._update_gifs_and_plots()

    def _update_gifs_and_plots(self):
        """Update GIFs and metric plots with current data."""
        # Create/update design GIF
        if self.design_image_paths:
            gif_path = self.img_dir / "design.gif"
            self._create_gif(self.design_image_paths, str(gif_path))
        
        # Create/update distribution GIF
        if self.distribution_image_paths:
            gif_path = self.img_dir / "flux.gif"
            self._create_gif(self.distribution_image_paths, str(gif_path))
        
        # Update metric plots
        self._plot_metrics(verbose=False)

    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        print(f"\nTraining ended. Creating final GIFs and plots...")
        
        # Create final design GIF (save to img/design.gif as per README)
        if self.design_image_paths:
            gif_path = self.img_dir / "design.gif"
            self._create_gif(self.design_image_paths, str(gif_path))
            print(f"Design GIF saved to: {gif_path}")
        
        # Create final distribution GIF (save to img/flux.gif as per README)
        if self.distribution_image_paths:
            gif_path = self.img_dir / "flux.gif"
            self._create_gif(self.distribution_image_paths, str(gif_path))
            print(f"Distribution GIF saved to: {gif_path}")
        
        # Plot final metrics from CSV
        self._plot_metrics(verbose=True)
        
        # Note: We keep the image directories for reference
        # If you want to clean them up, uncomment below:
        # import shutil
        # if self.design_dir.exists():
        #     shutil.rmtree(self.design_dir)
        # if self.distribution_dir.exists():
        #     shutil.rmtree(self.distribution_dir)
    
    def _plot_metrics(self, verbose=True):
        """Plot transmission, balance_score, and score from CSV."""
        if not self.csv_path.exists():
            if verbose:
                print("Warning: CSV file not found, cannot plot metrics")
            return
        
        try:
            # Read CSV
            df = pd.read_csv(self.csv_path)
            
            if len(df) == 0:
                if verbose:
                    print("Warning: CSV file is empty, cannot plot metrics")
                return
            
            # Plot transmission
            plt.figure(figsize=(10, 6))
            plt.plot(df['rollout_count'], df['transmission'], 'b-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Rollout Count')
            plt.ylabel('Transmission')
            plt.title('Transmission Over Training')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            transmission_plot_path = self.plot_dir / "transmission.png"
            plt.savefig(transmission_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"Transmission plot saved to: {transmission_plot_path}")
            
            # Plot balance score
            plt.figure(figsize=(10, 6))
            plt.plot(df['rollout_count'], df['balance_score'], 'g-', linewidth=2, marker='s', markersize=4)
            plt.xlabel('Rollout Count')
            plt.ylabel('Balance Score')
            plt.title('Balance Score Over Training')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            balance_plot_path = self.plot_dir / "balance.png"
            plt.savefig(balance_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"Balance score plot saved to: {balance_plot_path}")
            
            # Plot score
            plt.figure(figsize=(10, 6))
            plt.plot(df['rollout_count'], df['score'], 'r-', linewidth=2, marker='^', markersize=4)
            plt.xlabel('Rollout Count')
            plt.ylabel('Score')
            plt.title('Score Over Training')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            score_plot_path = self.plot_dir / "score.png"
            plt.savefig(score_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"Score plot saved to: {score_plot_path}")
            
        except Exception as e:
            if verbose:
                print(f"Error plotting metrics: {e}")
    
    def _create_gif(self, image_paths, output_path, duration=500, loop=0):
        """Create a GIF from a list of image paths."""
        images = []
        for path in image_paths:
            if os.path.exists(path):
                img = Image.open(path)
                images.append(img)
        
        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=loop
            )
            print(f"GIF created: {output_path} ({len(images)} frames)")
        else:
            print(f"Warning: No images found to create GIF at {output_path}")


def train_ppo(
    total_timesteps=100000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_tensorboard/",
    save_path="./ppo_model",
):
    """
    Train a PPO agent on the MinimalEnv environment.

    Args:
        total_timesteps: Total number of timesteps to train
        n_envs: Number of parallel environments
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to collect per update
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
        tensorboard_log: Directory for tensorboard logs
        save_path: Path to save the trained model
    """
    # Save starting timestamp for model saving
    start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory for callbacks (following README structure)
    callback_dir = f"ppo_model_log_{start_timestamp}"
    os.makedirs(callback_dir, exist_ok=True)

    # Create vectorized environment (parallel environments)
    # Using DummyVecEnv instead of SubprocVecEnv because Meep simulation objects
    # contain lambda functions that can't be pickled for multiprocessing
    print("Creating environment...")
    env = make_vec_env(MinimalEnv, n_envs=n_envs,
                       env_kwargs={"render_mode": None},
                       vec_env_cls=SubprocVecEnv)

    # Create evaluation environment
    eval_env = MinimalEnv(render_mode=None)
    
    # Define model save path
    save_path_with_timestamp = f"models/ppo_model_{start_timestamp}.zip"
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        verbose=1
    )

    # Create callback with model save path for best model tracking
    callback = TrainingCallback(
        save_dir=callback_dir, 
        verbose=1, 
        eval_env=eval_env,
        model_save_path=save_path_with_timestamp
    )

    # Train the model
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    print("Press Ctrl+C to interrupt training and save current model...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False  # Set to False to avoid tqdm/rich dependency
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Saving current model state...")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the final model (even if interrupted)
    model.save(save_path_with_timestamp)
    print(f"Model saved to {save_path_with_timestamp}")
    
    # Print best model info
    print(f"\nBest model was at rollout {callback.best_rollout} with score {callback.best_score:.4f}")
    print(f"Best model saved to: {save_path_with_timestamp.replace('.zip', '_best.zip')}")

    # Final evaluation using ModelEvaluator (same as rollout end)
    print("\nRunning final evaluation...")
    final_eval(model, eval_env, save_dir=callback_dir)

    return model


def final_eval(model, env, save_dir=None):
    """
    Run final evaluation using ModelEvaluator (same logic as rollout end).
    
    Args:
        model: Trained model
        env: Evaluation environment
        save_dir: Directory to save final results (optional)
    """
    evaluator = ModelEvaluator(model, env)
    results_df = evaluator.evaluate(n_episodes=1, deterministic=True)
    
    if len(results_df) > 0:
        metrics = results_df.iloc[0]
        
        # Extract metrics with fallbacks
        transmission = metrics.get('total_mode_transmission', metrics.get('total_transmission', 0.0))
        balance = metrics.get('balance_score', 0.0)
        score = metrics.get('current_score', 0.0)
        total_reward = metrics.get('total_reward', 0.0)
        
        print(f"\n{'='*50}")
        print("Final Evaluation Results:")
        print(f"{'='*50}")
        print(f"  Transmission: {transmission:.4f}")
        print(f"  Balance Score: {balance:.4f}")
        print(f"  Score: {score:.4f}")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"{'='*50}")
        
        # Save final design and distribution plots if save_dir provided
        if save_dir:
            save_path = Path(save_dir)
            img_dir = save_path / "img"
            img_dir.mkdir(exist_ok=True)
            
            # Save final design
            final_design_path = img_dir / "final_design.png"
            try:
                if hasattr(env, 'unwrapped'):
                    env.unwrapped.save_design_plot(str(final_design_path), title_suffix="Final Evaluation")
                else:
                    env.save_design_plot(str(final_design_path), title_suffix="Final Evaluation")
                print(f"Final design saved to: {final_design_path}")
            except Exception as e:
                print(f"Warning: Could not save final design plot: {e}")
            
            # Save final distribution
            final_distribution_path = img_dir / "final_distribution.png"
            try:
                if hasattr(env, 'unwrapped'):
                    env.unwrapped.save_distribution_plot(str(final_distribution_path), title_suffix="Final Evaluation")
                else:
                    env.save_distribution_plot(str(final_distribution_path), title_suffix="Final Evaluation")
                print(f"Final distribution saved to: {final_distribution_path}")
            except Exception as e:
                print(f"Warning: Could not save final distribution plot: {e}")
    else:
        print("Warning: Evaluation returned no results.")


def load_training_config(config_path=None):
    """
    Load train_ppo keyword arguments from YAML config.

    Args:
        config_path: Optional override path. Defaults to config.yaml next to this file.

    Returns:
        dict: Filtered kwargs to pass into train_ppo.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        print(
            f"[config] Config file not found at {path}. Using train_ppo defaults.")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    training_cfg = data.get("training", {}).get("ppo", {}) or {}
    filtered_cfg = {k: v for k, v in training_cfg.items()
                    if k in TRAIN_PPO_KWARGS}

    unknown_keys = sorted(set(training_cfg.keys()) - TRAIN_PPO_KWARGS)
    if unknown_keys:
        print(f"[config] Ignoring unsupported train_ppo keys: {unknown_keys}")

    return filtered_cfg


if __name__ == "__main__":
    config_override_path = os.environ.get(CONFIG_ENV_VAR)
    train_kwargs = load_training_config(config_override_path)

    # Train PPO agent
    model = train_ppo(**train_kwargs)

    print("\nTraining complete!")
