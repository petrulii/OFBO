import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import haiku as hk
import jax
import math

from agent import Agent
from utils import ReplayBuffer

import wandb

import os
import imageio
from PIL import Image

class Trainer:
    """
    Trainer class for model-based reinforcement learning with bilevel optimization.
    Handles the full training procedure for OMD, MLE, and VEP agent types.
    """
    
    def __init__(self, config, logger=None):
        """
        Parameters:
        - config: Configuration object containing training settings
        - logger: Optional logger object (not used if tracking with wandb)
        """
        self.config = config
        self.logger = logger
        self.build_trainer()
        # Initialize wandb run with name which includes config.agent_type+non_stationary+end_transition_clear_buffer+model_hidden_dim where some are boolean and model_hidden_dim is an integer, also buffer capacity, inner_lr and tau two real values
        custom_name = f"{self.config.agent_type}_{'n_st' if self.config.non_stationary else 'st'}_{self.config.model_hidden_dim}_{self.config.inner_lr}_{self.config.tau}_{self.config.alpha}_{self.config.grad_buffer_size}_{self.config.beta}"
        wandb.init(project="cartpole_nips", job_type="train", name=custom_name, config=self.config)
    
    def build_trainer(self):
        """Initialize environment, agent, and replay buffer."""

        # Create environments and wrap to handle cart position wrapping
        self.env = gym.make(self.config.env_name)
        self.eval_env = gym.make(self.config.env_name)
        
        # Set seeds for reproducibility
        for e in [self.eval_env, self.env]:
            e.reset(seed=self.config.seed)
            e.action_space.seed(self.config.seed)
            e.observation_space.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Initialize random number generator
        self.rngs = hk.PRNGSequence(self.config.seed)
        
        # Create agent and replay buffer
        self.agent = Agent(self.config, self.env.observation_space, self.env.action_space)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, self.config.buffer_capacity)
        
        # Initialize training variables
        self.step = 0
        self.episode = 0
        self.episode_return = 0
        self.episode_step = 0

        # Create directory for GIFs
        if self.config.video_frequency > 0:
            self.video_dir = os.path.join(wandb.run.dir, "videos")
            os.makedirs(self.video_dir, exist_ok=True)
        
        # Initialize optimal interval for non-stationary reward
        # The pole angle can be observed between (-.418, .418) radians (or ±24°)
        # but the episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°)
        # We want 3 optimal intervals inside the range (-.2095, .2095)
        #self.opt_intervals = [(-0.2095, -0.01), (0.01, 0.2095)]
        self.interval_indices = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        #self.opt_intervals = [(-0.06, 0.06), (-0.2095, -0.06), (0.06, 0.2095)]
        self.opt_intervals = [(-0.2095, 0.06), (-0.06, 0.2095)]
        #self.interval_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        idx = 0
        self.opt_interval = self.opt_intervals[idx]  # Initialize with the first interval
    
    def train(self):
        """
        Main training loop.
        
        Runs the full training procedure:
        1. Collects experience through environment interaction
        2. Updates the agent through a selected optimization method
        3. Periodically evaluates performance and logs metrics
        """
        
        # Initialize environment
        obs, _ = self.env.reset()
        
        # Timing variables
        start_time = time.time()
        last_eval_time = start_time
        done = False
        
        # Main training loop
        while self.step < self.config.num_train_steps:

            self.step += 1

            # If the environment is non-stationary, update opt_interval
            self.opt_interval = self.update_interval()
            #if self.config.non_stationary and (self.step % self.config.opt_interval_update_freq == 0):
                # Periodically pick one of the three intervals
                #self.opt_interval = self.opt_intervals[self.interval_indices[idx]]
                #idx = idx + 1
                #if idx >= len(self.interval_indices):
                #    idx = 0

            # Periodically evaluate agent
            if self.step % self.config.eval_frequency == 0:
                # Passing random generator to ensure reproducibility
                eval_return = evaluate(self.agent, self.eval_env, next(self.rngs), self.opt_interval)
                eval_time = time.time()
                
                wandb.log({
                    'eval/avg_episode_reward': eval_return,
                    'eval/time': eval_time - last_eval_time,
                    'buffer_size': len(self.replay_buffer)
                }, step=self.step)
                
                last_eval_time = eval_time

            # Periodically render and save GIF
            if self.config.video_frequency > 0 and self.step % self.config.video_frequency == 0:
                # Create a separate environment for rendering
                render_env = gym.make(self.config.env_name, render_mode="rgb_array")
                
                # Generate filename with step number
                gif_filename = os.path.join(self.video_dir, f"step_{self.step:07d}.gif")
                
                # Create GIF and log to wandb
                _ = create_gif(self.agent, render_env, gif_filename, self.opt_interval)
                
                # Log to wandb
                wandb.log({"video": wandb.Video(gif_filename, fps=30, format="gif")}, step=self.step)
                
                # Close the rendering environment
                render_env.close()
            
            # Select action (with epsilon-greedy exploration)
            if np.random.rand() < self.config.eps or self.step < self.config.init_steps:
                action = self.env.action_space.sample()  # Random action
            else:
                action = action.item()  # Greedy action
            
            # Execute action in environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Handle episode termination
            done_float = float(done)
            # Allow infinite bootstrap if termination is due to time limit
            done_no_max = 0 if truncated else done_float

            # Get pole angle from observation
            pole_angle = next_obs[2]
            
            # Scale reward if pole is in optimal interval
            if not (self.opt_interval[0] <= pole_angle <= self.opt_interval[1]):
                reward = 0

            # Track episode return
            self.episode_return += reward

            # Store transition in replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done_float, done_no_max)
            
            # Update current state and counters
            obs = next_obs
            self.episode_step += 1
            
            # Handle end of episode
            if (done or truncated):
                # Log episode results
                wandb.log({
                    'train/avg_episode_reward': self.episode_return,
                    'train/time': time.time() - start_time,
                    'train/opt_interval_a': self.opt_interval[0],
                    'train/opt_interval_b': self.opt_interval[1],
                    'train/pole_angle': pole_angle
                }, step=self.step)
                
                # Reset environment and episode tracking
                obs, _ = self.env.reset()
                start_time = time.time()
                last_eval_time = start_time
                done = False
                self.episode_return = 0
                self.episode_step = 0
                self.episode += 1
            
            # Select next action
            action = self.agent.act(self.agent.params_Q, obs, next(self.rngs))
            # Start training after init_steps and when not in warmup period
            if self.step >= self.config.init_steps and (not hasattr(self, 'need_warmup') or not self.need_warmup):
                # Update agent
                model_grad_norm, Q_grad_norm, model_loss, Q_loss = self.agent.update(self.replay_buffer)
                # Log training metrics
                wandb.log({
                    'train/model_grad_norm': model_grad_norm,
                    'train/Q_grad_norm': Q_grad_norm,
                    'train/model_loss': model_loss,
                    'train/Q_loss': Q_loss
                }, step=self.step)
        
        # Final evaluation with more episodes
        final_eval_return = evaluate(self.agent, self.eval_env, next(self.rngs), self.opt_interval, num_eval_episodes=20)
        
        wandb.log({'eval/final_episode_return': final_eval_return}, step=self.step)
        
        wandb.finish()

        return final_eval_return

    def update_interval(self):
        """
        Updates the optimal interval for reward scaling.

        For non-stationary environments, linearly progresses from interval 0 to interval 1
        over the course of the training period (no oscillation).

        Returns:
            tuple: The current optimal interval (min, max)
        """
        if not self.config.non_stationary:
            return self.opt_intervals[0]

        # Linear progress from 0 to 1
        progress = min(1.0, self.step / self.config.num_train_steps)

        interval_0 = self.opt_intervals[0]
        interval_1 = self.opt_intervals[1]

        min_val = (1 - progress) * interval_0[0] + progress * interval_1[0]
        max_val = (1 - progress) * interval_0[1] + progress * interval_1[1]

        return (min_val, max_val)


def evaluate(agent, eval_env, rng, opt_interval, num_eval_episodes=10):
    """Evaluate the agent's performance in the environment."""
    average_episode_reward = 0
    for episode in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        done, truncated = False, False
        episode_reward = 0
        while not (done or truncated):
            rng, _ = jax.random.split(rng)
            action = agent.act(agent.params_Q, obs, rng).item()
            obs, reward, done, truncated, info = eval_env.step(action)
            # Get pole angle from observation
            pole_angle = obs[2]
            
            # Scale reward if pole is in optimal interval
            if not (opt_interval[0] <= pole_angle <= opt_interval[1]):
                reward = 0
            
            episode_reward += reward
        average_episode_reward += episode_reward
    average_episode_reward /= num_eval_episodes
    return average_episode_reward


# Video rendering function
def create_gif(agent, env, filename, opt_interval, max_steps=500):
    """
    Renders a single episode of the agent's performance and saves it as a GIF.
    
    Parameters:
    - agent: The RL agent
    - env: The environment
    - filename: Where to save the GIF
    - opt_interval: Optional interval for reward scaling
    - max_steps: Maximum number of steps to render
    
    Returns:
    - episode_reward: Total reward accumulated during the episode
    """
    # Reset environment
    obs, _ = env.reset()
    done, truncated = False, False
    episode_reward = 0
    
    # Create directory for frames if it doesn't exist
    frames_dir = os.path.join(os.path.dirname(filename), "temp_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Lists to store frames and rewards
    frames = []
    rewards = []
    
    # Render initial frame
    frame = env.render()
    frames.append(frame)
    
    step = 0
    
    # Loop until episode ends or max steps reached
    while not (done or truncated) and step < max_steps:
        # Select action
        action = agent.act(agent.params_Q, obs, jax.random.PRNGKey(0)).item()
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        # Get pole angle from observation
        pole_angle = obs[2]
        
        # Scale reward if pole is in optimal interval
        if not (opt_interval[0] <= pole_angle <= opt_interval[1]):
            reward = 0
        
        episode_reward += reward
        rewards.append(reward)
        
        # Render frame and append to list
        frame = env.render()
        frames.append(frame)
        
        step += 1
    
    # Save frames as GIF
    imageio.mimsave(filename, frames, fps=30)
    
    # Clean up temporary files
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        try:
            os.remove(frame_path)
        except:
            pass
    
    try:
        os.rmdir(frames_dir)
    except:
        pass
    
    return episode_reward