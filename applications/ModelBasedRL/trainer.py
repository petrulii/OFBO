import os
import gym
import numpy as np
import time
import haiku as hk
import jax

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
        wandb.init(project="cartpole_omd", job_type="train", name=config.agent_type)
        self.config = config
        self.logger = logger
        self.build_trainer()
    
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
        action = self.agent.act(self.agent.params_Q, obs, next(self.rngs))
        
        # Timing variables
        start_time = time.time()
        last_eval_time = start_time
        done = False
        
        # Main training loop
        while self.step < self.config.num_train_steps:
            # Periodically evaluate agent
            if self.step % self.config.eval_frequency == 0:
                # Passing random generator to ensure reproducibility
                eval_return = evaluate(self.agent, self.eval_env, next(self.rngs))
                eval_time = time.time()
                
                wandb.log({
                    'eval/avg_episode_reward': eval_return,
                    'eval/time': eval_time - last_eval_time
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
            done_no_max = 0 if self.episode_step + 1 == self.env._max_episode_steps else done_float
            
            # Track episode return
            self.episode_return += reward
            
            # Store transition in replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done_float, done_no_max)
            
            # Update current state and counters
            obs = next_obs
            self.episode_step += 1
            self.step += 1
            
            # Handle end of episode
            if done:
                # Log episode results
                wandb.log({
                    'train/avg_episode_reward': self.episode_return,
                    'train/time': time.time() - start_time
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
            
            # Start training after init_steps
            if self.step >= self.config.init_steps:
                # Update agent
                agent_metrics = self.agent.update(self.replay_buffer)
        
        # Final evaluation with more episodes
        final_eval_return = evaluate(self.agent, self.eval_env, next(self.rngs), num_eval_episodes=20)
        
        wandb.log({
            'eval/final_episode_return': final_eval_return
        }, step=self.step)
        
        wandb.finish()
        return final_eval_return