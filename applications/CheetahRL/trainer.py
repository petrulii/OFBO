import os
import gym
import numpy as np
import time
import haiku as hk
import jax
import math

from agent import Agent
from utils import ReplayBuffer, evaluate

import wandb
import sys

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
        custom_name = f"{self.config.agent_type}_{'non_stationary' if self.config.non_stationary else 'stationary'}_{self.config.model_hidden_dim}_{self.config.buffer_capacity}_{self.config.inner_lr}_{self.config.tau}"
        wandb.init(project="cheetah", job_type="train", name=custom_name)
    
    def build_trainer(self):
        """Initialize environment, agent, and replay buffer."""
        # Create environments
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
        action_shape = self.env.action_space.shape
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, action_shape, self.config.buffer_capacity)
        
        # Initialize training variables
        self.step = 0
        self.episode = 0
        self.episode_return = 0
        self.episode_step = 0
        self.new_mass = 0
    
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

            # If the environment is non-stationary, update mass
            if self.config.non_stationary:
                self.new_mass = self.update_mass()
            
            # Periodically evaluate agent
            if self.step % self.config.eval_frequency == 0:
                # Passing random generator to ensure reproducibility
                eval_return = evaluate(self.agent, self.eval_env, next(self.rngs), mass_ratio=self.new_mass)
                eval_time = time.time()
                
                wandb.log({
                    'eval/avg_episode_reward': eval_return,
                    'eval/time': eval_time - last_eval_time,
                    'buffer_size': len(self.replay_buffer)
                }, step=self.step)
                
                last_eval_time = eval_time
            

            # Select action (with epsilon-greedy exploration for continuous action spaces)
            if np.random.rand() < self.config.eps or self.step < self.config.init_steps:
                raw_action = self.env.action_space.sample()  # Random action
            else:
                # Get action from agent - no need to call .item() for continuous actions
                raw_action = self.agent.act(self.agent.params_Q, obs, next(self.rngs))

            # Convert JAX array to numpy if needed
            if hasattr(raw_action, 'shape') and not isinstance(raw_action, np.ndarray):
                action = np.array(raw_action)
            else:
                action = raw_action

            # Remove the batch dimension (squeeze the first dimension if it's 1)
            if len(action.shape) > 1 and action.shape[0] == 1:
                action = action.squeeze(0)  # This converts (1, 17) to (17,)

            # Now use the properly formatted action
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Execute action in environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Handle episode termination
            done_float = float(done)
            # Allow infinite bootstrap if termination is due to time limit
            done_no_max = 0 if truncated else done_float
            
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
                    'train/cart_mass': self.new_mass
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
                self.agent.update(self.replay_buffer)
        
        # Final evaluation with more episodes
        final_eval_return = evaluate(self.agent, self.eval_env, next(self.rngs), num_eval_episodes=20, mass_ratio=self.new_mass)
        
        wandb.log({
            'eval/final_episode_return': final_eval_return
        }, step=self.step)
        
        wandb.finish()

        return final_eval_return

    def update_mass(self):
        """Update humanoid mass to transition smoothly between discrete plateaus."""
        # Initialize on first call
        if not hasattr(self, 'plateau_values'):
            # Example: different plateau values at 0.8 and 1.2 times base mass
            self.plateau_values = [1.2, 0.8, 1.2, 0.8, 1.2]
            
            # Start at the first plateau
            self.current_plateau_idx = 0
            self.next_plateau_idx = 1
            
            # Get base humanoid mass
            # The mass is stored in the xml model parameters
            # For Humanoid-v4 using MuJoCo, we need to access the model parameters
            self.base_humanoid_mass = self.env.model.body_mass[1]  # Index 1 is the torso
            
            # Initial mass setting
            self.current_mass = self.base_humanoid_mass * self.plateau_values[self.current_plateau_idx]
            self.target_mass = self.base_humanoid_mass * self.plateau_values[self.next_plateau_idx]
            
            # Transition parameters remain the same
            self.transition_steps = self.config.transition_duration
            self.steps_in_transition = 0
            self.steps_on_plateau = self.step
            self.plateau_duration = self.config.plateau_duration
            
            # Set initial mass for the torso and all connected body parts
            for i in range(1, self.env.model.nbody):  # Skip worldbody (index 0)
                self.env.model.body_mass[i] = self.env.model.body_mass[i] * self.plateau_values[self.current_plateau_idx]
        
        # Rest of the update_mass function remains similar, but update all body masses
        
        # When changing mass, update all body masses
        if self.steps_in_transition > 0:
            # Calculate smooth transition using sigmoid function
            progress = self.steps_in_transition / self.transition_steps
            smoothed_progress = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))
            
            # Calculate new mass ratio
            mass_ratio = self.current_mass / self.base_humanoid_mass + smoothed_progress * (self.target_mass / self.base_humanoid_mass - self.current_mass / self.base_humanoid_mass)
            
            # Update all body masses proportionally
            for i in range(1, self.env.model.nbody):  # Skip worldbody (index 0)
                original_mass = self.env.model.body_mass[i] / (self.current_mass / self.base_humanoid_mass)
                self.env.model.body_mass[i] = original_mass * mass_ratio
            
            self.steps_in_transition += 1
        
        # Return torso mass as reference
        return self.env.model.body_mass[1]
