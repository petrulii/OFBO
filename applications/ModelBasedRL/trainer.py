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
        wandb.init(project="cartpole_omd", job_type="train", name=custom_name)
    
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
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, self.config.buffer_capacity)
        
        # Initialize training variables
        self.step = 0
        self.episode = 0
        self.episode_return = 0
        self.episode_step = 0

        # Non-stationary environment settings
        self.base_masscart = self.env.unwrapped.masscart # Store the original mass values
        self.base_masspole = self.env.unwrapped.masspole
        self.masscart = self.base_masscart
    
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
                self.masscart = self.update_mass()
            
            # Periodically evaluate agent
            if self.step % self.config.eval_frequency == 0:
                # Passing random generator to ensure reproducibility
                eval_return = evaluate(self.agent, self.eval_env, next(self.rngs), masscart=self.masscart)
                eval_time = time.time()
                
                wandb.log({
                    'eval/avg_episode_reward': eval_return,
                    'eval/time': eval_time - last_eval_time,
                    'buffer_size': len(self.replay_buffer)
                }, step=self.step)
                
                last_eval_time = eval_time
            
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
                    'train/cart_mass': self.env.unwrapped.masscart
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
        final_eval_return = evaluate(self.agent, self.eval_env, next(self.rngs), num_eval_episodes=20, masscart=self.masscart)
        
        wandb.log({
            'eval/final_episode_return': final_eval_return
        }, step=self.step)
        
        wandb.finish()

        return final_eval_return

    def update_mass(self):
        """Update cart mass to transition smoothly between discrete plateaus."""
        # Initialize on first call
        if not hasattr(self, 'plateau_values'):
            # Example: 2 different plateau values at 0.5 and 1.5 times base mass
            self.plateau_values = [1.5, 0.5, 1.5, 0.5, 1.5, 0.5]
            
            # Start at the first plateau
            self.current_plateau_idx = 0
            
            # Initial mass setting
            self.current_mass = self.base_masscart * self.plateau_values[self.current_plateau_idx]
            
            # Transition parameters
            self.transition_steps = self.config.transition_duration
            self.steps_in_transition = 0
            self.steps_on_plateau = 0
            self.plateau_duration = self.config.plateau_duration

            # Warmup tracking
            self.need_warmup = False
            self.warmup_steps = 0
            self.warmup_size = self.config.batch_size * 2  # Number of samples to collect before resuming training
            
            # Set initial mass
            self.env.unwrapped.masscart = self.current_mass
            return self.env.unwrapped.masscart
        
        # Check if we need to start a new transition
        if self.steps_in_transition == 0 and self.steps_on_plateau >= self.plateau_duration:
            # Reset plateau counter
            self.steps_on_plateau = 0
            
            # Store current mass as starting point and calculate target
            self.current_mass = self.env.unwrapped.masscart  # Use actual current mass
            self.target_mass = self.base_masscart * self.plateau_values[self.current_plateau_idx+1]
            
            self.steps_in_transition = 1  # Start transition

        elif self.steps_in_transition > 0:
            # We're in a transition between plateaus
            if self.steps_in_transition <= self.transition_steps:
                # Calculate smooth transition using sigmoid function
                progress = self.steps_in_transition / self.transition_steps
                # Sigmoid gives a smooth S-curve transition
                smoothed_progress = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))
                
                # Interpolate between current and target mass
                new_mass = self.current_mass + smoothed_progress * (self.target_mass - self.current_mass)

                self.env.unwrapped.masscart = new_mass
                self.steps_in_transition += 1
                
                # If transition completed, reset counter
                if self.steps_in_transition > self.transition_steps:
                    self.steps_in_transition = 0
                    self.steps_on_plateau = 0

                    # Update the plateau index
                    self.current_plateau_idx = (self.current_plateau_idx + 1) % len(self.plateau_values)
                
                    if self.config.end_transition_clear_buffer:
                        # Clear the replay buffer when reaching a new plateau by creating a new one
                        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, self.config.buffer_capacity)
                    
                    # Set warmup flag to avoid training until buffer has enough samples
                    self.need_warmup = True
                    self.warmup_steps = 0
        else:
            # We're on a plateau - mass stays constant
            self.steps_on_plateau += 1
        
        # Update total mass
        self.env.unwrapped.total_mass = self.env.unwrapped.masscart + self.env.unwrapped.masspole
        
        # If we need warmup after buffer clear, increment counter
        if self.need_warmup:
            self.warmup_steps += 1
            if self.warmup_steps >= self.warmup_size:
                self.need_warmup = False
    
        return self.env.unwrapped.masscart
