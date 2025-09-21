# utils.py
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves
from functools import partial
from collections import namedtuple
from typing import Callable, Optional, Tuple

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

# Adding helpers for baselines
def tree_add(a, b):
    return tree_map(lambda x, y: x + y, a, b)

def tree_axpy(alpha, x, y):
    """alpha * x + y on pytrees."""
    return tree_map(lambda u, v: alpha * u + v, x, y)

def tree_zeros_like(x):
    return tree_map(jnp.zeros_like, x)

def tree_dot(a, b):
    """Sum of vdot over matching leaves of two pytrees."""
    #chex.assert_trees_all_equal_structs(a, b)
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return sum(jnp.vdot(x, y) for x, y in zip(leaves_a, leaves_b))

def conjugate_gradient(
    matvec: Callable,               # v -> A v  (same pytree structure as b)
    b,                              # rhs pytree
    x0=None,                        # optional initial guess (same pytree as b)
    max_iter: int = 50,
    tol: float = 1e-6,
    M: Optional[Callable] = None    # optional preconditioner: r -> M^{-1} r
):
    """
    Solve A x = b by (preconditioned) Conjugate Gradient in pytree space.
    matvec: function that returns A v (pytree) for a pytree v
    M: optional preconditioner; if None, identity is used
    """
    if x0 is None:
        x = jax.tree_map(jnp.zeros_like, b)
    else:
        x = x0

    Ax = matvec(x)
    r  = jax.tree_map(lambda bi, Axi: bi - Axi, b, Ax)   # r = b - A x
    if M is None:
        z = r
    else:
        z = M(r)

    p  = z
    rz = tree_dot(r, z)                                   # rr (preconditioned)
    b_norm = jnp.sqrt(tree_dot(b, b))
    tol_abs2 = (tol * (1.0 + b_norm))**2                  # absolute+relative safeguard

    def cond_fun(carry):
        k, x, r, z, p, rz = carry
        return jnp.logical_and(k < max_iter, rz > tol_abs2)

    def body_fun(carry):
        k, x, r, z, p, rz = carry
        Ap   = matvec(p)
        pAp  = tree_dot(p, Ap) + 1e-12                    # numeric guard
        alpha = rz / pAp

        x_new = jax.tree_map(lambda xi, pi: xi + alpha * pi, x, p)
        r_new = jax.tree_map(lambda ri, Api: ri - alpha * Api, r, Ap)

        if M is None:
            z_new = r_new
        else:
            z_new = M(r_new)

        rz_new = tree_dot(r_new, z_new)
        beta   = rz_new / (rz + 1e-12)
        p_new  = jax.tree_map(lambda zi, pi: zi + beta * pi, z_new, p)

        return (k + 1, x_new, r_new, z_new, p_new, rz_new)

    # Initialize carry with rz already computed, so rr/rz is in scope for the first cond check.
    carry0 = (0, x, r, z, p, rz)
    k, x, r, z, p, rz = jax.lax.while_loop(cond_fun, body_fun, carry0)
    return x

def hessian_vector_product(f_scalar, params, v):
    # H·v = JVP(∇f, v)
    return jax.jvp(jax.grad(f_scalar), (params,), (v,))[1]