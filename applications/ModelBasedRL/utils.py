import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
import numpy as np
import pickle


# Replay buffer class for storing transitions

class ReplayBuffer:
  """Buffer to store environment transitions. Discrete actions only"""
  def __init__(self, obs_shape, capacity):
    self.capacity = capacity

    self.obses = jnp.empty((capacity, *obs_shape), dtype=jnp.float32)
    self.next_obses = jnp.empty((capacity, *obs_shape), dtype=jnp.float32)
    self.actions = jnp.empty((capacity, 1), dtype=jnp.float32)
    self.rewards = jnp.empty((capacity, 1), dtype=jnp.float32)
    self.not_dones = jnp.empty((capacity, 1), dtype=jnp.float32)
    self.not_dones_no_max = jnp.empty((capacity, 1), dtype=jnp.float32)

    self.idx = 0
    self.full = False

  def __len__(self):
    return self.capacity if self.full else self.idx

  def add(self, obs, action, reward, next_obs, done, done_no_max):
    
    self.obses = self.obses.at[self.idx].set(obs)
    self.actions = self.actions.at[self.idx].set(action)
    self.rewards = self.rewards.at[self.idx].set(reward)
    self.next_obses = self.next_obses.at[self.idx].set(next_obs)
    self.not_dones = self.not_dones.at[self.idx].set(not done)
    self.not_dones_no_max = self.not_dones_no_max.at[self.idx].set(not done_no_max)

    self.idx = (self.idx + 1) % self.capacity
    self.full = self.full or self.idx == 0

  def sample(self, batch_size, replace=False):
    idxs = np.random.choice(len(self), size=batch_size, replace=replace)

    obses = self.obses[idxs]
    actions = self.actions[idxs]
    rewards = self.rewards[idxs]
    next_obses = self.next_obses[idxs]
    not_dones = self.not_dones[idxs]
    not_dones_no_max = self.not_dones_no_max[idxs]

    return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

  def save(self, data_path):
    all_data = [
      self.obses,
      self.actions,
      self.rewards,
      self.next_obses,
      self.not_dones,
      self.not_dones_no_max
    ]
    pickle.dump(all_data, open(data_path, 'wb'))
  
  def load(self, data_path):
    self.obses, self.actions, self.rewards, self.next_obses, \
      self.not_dones, self.not_dones_no_max = pickle.load(open(data_path, "rb"))
    self.capacity = len(self.obses)
    self.full = True

# Evaluation function used in the main training loop

def evaluate(agent, eval_env, rng, num_eval_episodes=10):
    """Evaluate the agent's performance in the environment."""
    average_episode_reward = 0
    for episode in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            rng, _ = jax.random.split(rng)
            action = agent.act(agent.params_Q, obs, rng).item()
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
        average_episode_reward += episode_reward
    average_episode_reward /= num_eval_episodes
    return average_episode_reward

# Custom VJP implementations for bilevel optimization

@partial(jax.custom_vjp, nondiff_argnums=(0, 5))
def root_solve(param_func, init_xs, params, replay, rng, solvers):
    """
    Root-finding approach to bilevel optimization.
    
    Args:
        param_func: Function that computes gradients of inner objective
        init_xs: Initial parameters for inner optimization
        params: Outer parameters (e.g., world model parameters)
        replay: Batch of transitions
        rng: Random key
        solvers: Tuple containing solver functions
        
    Returns:
        Solution of inner optimization
    """
    # To mimic two_phase_solve API
    fwd_solver = solvers[0]
    return fwd_solver(init_xs, params, replay, rng)

def root_solve_fwd(param_func, init_xs, params, replay, rng, solvers):
    """Forward pass for root_solve."""
    sol = root_solve(param_func, init_xs, params, replay, rng, solvers)
    # Stop gradient on target parameters
    tpQ = jax.lax.stop_gradient(sol.target_params_Q)
    return sol, (sol.params_Q, params, replay, rng, tpQ)

def root_solve_bwd(param_func, solvers, res, g):
    """Backward pass for root_solve."""
    pQ, params, replay, rng, tpQ = res
    # Compute vector-Jacobian product
    _, vdp_fun = jax.vjp(lambda y: param_func(y, pQ, replay, rng, tpQ), params)
    g_main = g[0] if isinstance(g, tuple) else g
    vdp = vdp_fun(g_main)[0]
    # Zero gradients for parameters that don't need them
    z_sol, z_replay, z_rng = jax.tree_map(jnp.zeros_like, (pQ, replay, rng))
    return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng

# Register custom VJP rules
root_solve.defvjp(root_solve_fwd, root_solve_bwd)


# For functional bilevel optimization

# Include params_dual_Q and opt_state_dual_Q to match original implementation
InnnerSol = namedtuple('InnerSol', 'val_Q val_target_Q loss_Q vals_Q grad_norm_Q entropy_Q params_Q target_params_Q opt_state_Q next_obs_nll')

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 7))
def inner_solution(inner_loss, inner_model, init_xs, params, replay, rng, params_dual_Q, solvers):
    """
    Functional approach to bilevel optimization.
    
    Args:
        inner_loss: Function that computes inner loss gradients
        inner_model: Inner model (Q-network)
        init_xs: Initial parameters for inner optimization
        params: Outer parameters (e.g., world model parameters)
        replay: Batch of transitions
        rng: Random key
        params_dual_Q: Parameters of dual Q-network
        solvers: Tuple containing solver functions
        
    Returns:
        Solution of inner optimization with Q-values
    """
    # Forward solver for inner optimization
    fwd_solver = solvers[0]
    sol = fwd_solver(init_xs, params, replay, rng)
    
    # Backward solver for dual optimization
    bwd_solver = solvers[1]
    
    # Get Q-values from optimized parameters
    obs, action, reward, next_obs, not_done, not_done_no_max = replay
    val_Q = inner_model.apply(jax.lax.stop_gradient(sol.params_Q), obs)
    val_target_Q = inner_model.apply(jax.lax.stop_gradient(sol.target_params_Q), next_obs)

    return InnnerSol(val_Q, val_target_Q, sol.loss_Q, sol.vals_Q,
                sol.grad_norm_Q, sol.entropy_Q, sol.params_Q, sol.target_params_Q, 
                sol.opt_state_Q, sol.next_obs_nll)

def argmin_fwd(inner_loss, inner_model, init_xs, params, replay, rng, params_dual_Q, solvers):
    """Forward pass for inner_solution."""
    sol = inner_solution(inner_loss, inner_model, init_xs, params, replay, rng, params_dual_Q, solvers)
    # Stop gradients on various components
    tpQ = jax.lax.stop_gradient(sol.target_params_Q)
    val_target_Q = jax.lax.stop_gradient(sol.val_target_Q)
    val_Q = jax.lax.stop_gradient(sol.val_Q)
    return sol, (val_Q, init_xs, params, replay, rng, val_target_Q, params_dual_Q)

def argmin_bwd(inner_loss, inner_model, solvers, res, g):
    """Backward pass for inner_solution using backward solver."""
    pQ, init_xs, params, replay, rng, tpQ, params_dual_Q = res
    g_main = g[0] if isinstance(g, tuple) else g
    
    # Use backward solver to approximate gradients
    bwd_solver = solvers[1]
    sol = bwd_solver(params_dual_Q, replay, rng, g_main)

    # Compute vector-Jacobian product
    _, vdp_fun = jax.vjp(lambda y: inner_loss(y, pQ, replay, rng, tpQ), params)
    vdp = vdp_fun(sol.val_dual_Q)[0]
    
    # Zero gradients for parameters that don't need them
    z_sol, z_replay, z_rng, z_dual = jax.tree_map(jnp.zeros_like, (init_xs, replay, rng, params_dual_Q))
    return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng, z_dual

def argmin_bwd_exact(inner_loss, inner_model, solvers, res, g):
    """Exact backward pass for inner_solution (when not using dual network)."""
    pQ, init_xs, params, replay, rng, tpQ, params_dual_Q = res
    g_main = g[0] if isinstance(g, tuple) else g

    # Compute vector-Jacobian product directly
    _, vdp_fun = jax.vjp(lambda y: inner_loss(y, pQ, replay, rng, tpQ), params)
    vdp = vdp_fun(g_main)[0]

    # Zero gradients for parameters that don't need them
    z_sol, z_replay, z_rng, z_dual = jax.tree_map(jnp.zeros_like, (init_xs, replay, rng, params_dual_Q))
    return z_sol, jax.tree_map(lambda x: -x, vdp), z_replay, z_rng, z_dual

# Register custom VJP rules - use either argmin_bwd or argmin_bwd_exact
inner_solution.defvjp(argmin_fwd, argmin_bwd_exact)