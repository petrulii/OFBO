# utils.py
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from functools import partial
from collections import namedtuple
import numpy as np
import pickle
import time


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
        # Add observation times array
        self.obs_times = jnp.empty((capacity, 1), dtype=jnp.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        """
        Add a transition to the buffer.
        """
        self.obses = self.obses.at[self.idx].set(obs)
        self.actions = self.actions.at[self.idx].set(action)
        self.rewards = self.rewards.at[self.idx].set(reward)
        self.next_obses = self.next_obses.at[self.idx].set(next_obs)
        self.not_dones = self.not_dones.at[self.idx].set(not done)
        self.not_dones_no_max = self.not_dones_no_max.at[self.idx].set(not done_no_max)
        # Add current timestamp
        current_time = time.time()
        self.obs_times = self.obs_times.at[self.idx].set(current_time)

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

    def sample_time_weighted(self, batch_size, decay_factor=0.05, replace=True):
        """
        Sample transitions with probability proportional to recency.
        
        Args:
            batch_size: Number of transitions to sample
            decay_factor: Controls how quickly importance decays with time
                        Higher values prioritize recent samples more strongly
            replace: Whether to sample with replacement
        """
        buffer_size = len(self)
        
        # Calculate time-based sampling weights
        current_time = time.time()
        times = self.obs_times[:buffer_size].flatten()
        relative_times = current_time - times
        
        # Compute probabilities using exponential decay
        # More recent observations get higher probabilities
        probs = jnp.exp(-decay_factor * relative_times)
        
        # Convert to numpy for np.random.choice and normalize
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        # Sample based on time weights
        idxs = np.random.choice(buffer_size, size=batch_size, replace=replace, p=probs)
        
        # Retrieve sampled transitions
        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        not_dones = self.not_dones[idxs]
        not_dones_no_max = self.not_dones_no_max[idxs]
        obs_times = self.obs_times[idxs]
        
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def save(self, data_path):
        all_data = [
        self.obses,
        self.actions,
        self.rewards,
        self.next_obses,
        self.not_dones,
        self.not_dones_no_max,
        self.obs_times  # Save observation times
        ]
        pickle.dump(all_data, open(data_path, 'wb'))
        
    def load(self, data_path):
        loaded_data = pickle.load(open(data_path, "rb"))
        # Handle both old format (6 elements) and new format (7 elements)
        if len(loaded_data) == 7:
            self.obses, self.actions, self.rewards, self.next_obses, \
            self.not_dones, self.not_dones_no_max, self.obs_times = loaded_data
        else:
            # For backwards compatibility with old saved buffers
            self.obses, self.actions, self.rewards, self.next_obses, \
            self.not_dones, self.not_dones_no_max = loaded_data
            # Initialize observation times for old data
            self.obs_times = jnp.zeros((len(self.obses), 1), dtype=jnp.float32)
            
        self.capacity = len(self.obses)
        self.full = True

    def clear(self):
        """
        Empty the replay buffer and reset its state.
        After calling this method, the buffer will be empty and ready to be filled from scratch.
        """
        # Reset all arrays to empty (maintain original shapes and types)
        obs_shape = self.obses.shape[1:]
        self.obses = jnp.empty((self.capacity, *obs_shape), dtype=jnp.float32)
        self.next_obses = jnp.empty((self.capacity, *obs_shape), dtype=jnp.float32)
        self.actions = jnp.empty((self.capacity, 1), dtype=jnp.float32)
        self.rewards = jnp.empty((self.capacity, 1), dtype=jnp.float32)
        self.not_dones = jnp.empty((self.capacity, 1), dtype=jnp.float32)
        self.not_dones_no_max = jnp.empty((self.capacity, 1), dtype=jnp.float32)
        self.obs_times = jnp.empty((self.capacity, 1), dtype=jnp.float32)
        
        # Reset index and full flag
        self.idx = 0
        self.full = False

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
    #g_main = g[0] if isinstance(g, tuple) else g
    g_main = g[1] if isinstance(g, tuple) else g  # use ∂ outer loss / ∂ val_target_Q (outer)
    
    # Use backward solver to approximate gradients
    bwd_solver = solvers[1]
    #sol = bwd_solver(params_dual_Q, replay, rng, g_main)
    sol = bwd_solver(params_dual_Q, replay, rng, g_main, params) # pass the current outer params (λ) explicitly to the adjoint solver

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
inner_solution.defvjp(argmin_fwd, argmin_bwd)
    
class GradientBufferManager:
    def __init__(self, buffer_size, grad_example):
        """Initialize the gradient buffer manager."""
        self.buffer_size = buffer_size
        self.buffer_count = 0

        # Create buffer of zeros with shape (buffer_size, ...) for each grad leaf
        self.grad_buffer = tree_map(
            lambda g: jnp.zeros((buffer_size,) + g.shape, dtype=g.dtype),
            grad_example
        )

        # Initialize average gradient to zeros
        self.avg_grad = tree_map(jnp.zeros_like, grad_example)

    def update(self, grads):
        """Update the buffer with new gradients and compute the new average."""
        if self.buffer_count < self.buffer_size:
            # Add to buffer and update average
            idx = self.buffer_count
            self.grad_buffer = tree_map(
                lambda buf, g: buf.at[idx].set(g),
                self.grad_buffer, grads
            )
            self.buffer_count += 1

            self.avg_grad = tree_map(
                lambda avg, g: (avg * (self.buffer_count - 1) + g) / self.buffer_count,
                self.avg_grad, grads
            )
        else:
            # Replace oldest gradient (FIFO) and update average
            old_grad = tree_map(lambda buf: buf[0], self.grad_buffer)

            self.grad_buffer = tree_map(
                lambda buf, g: jnp.concatenate([buf[1:], jnp.expand_dims(g, 0)], axis=0),
                self.grad_buffer, grads
            )

            self.avg_grad = tree_map(
                lambda avg, g_new, g_old: avg + (g_new - g_old) / self.buffer_size,
                self.avg_grad, grads, old_grad
            )

        return self.grad_buffer, self.buffer_count, self.avg_grad