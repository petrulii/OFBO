# trainer.py
import os
import sys
import numpy as np
import time
import jax
import jax.numpy as jnp
import optax
import chex
import haiku as hk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from jax import value_and_grad, jit, grad
from jax.lax import stop_gradient
from functools import partial
from collections import namedtuple

# Import bilevel optimization tools
from utils import inner_solution, root_solve, GradientBufferManager

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError as e:
    WANDB_AVAILABLE = False
    print(e, file=sys.stderr)
    print("wandb not available, logging will be disabled", file=sys.stderr)

# Data structures for tracking state
AuxParams = namedtuple('AuxParams', 'inner_params target_inner_params dual_params opt_state_inner opt_state_dual rng lambda_reg')
InnerSol = namedtuple('InnerSol', 'params loss grad_norm opt_state')

class BilevelRegressionAgent:
    """Agent for solving non-stationary bilevel regression problems."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize random number generator
        self.rngs = hk.PRNGSequence(config.seed)
        
        # Demo inputs for network initialization
        demo_input = jnp.ones((1, config.input_dim))
        
        # Initialize inner model (regression network)
        self.inner_net = hk.without_apply_rng(hk.transform(partial(self._network_fn, config.hidden_dim, config.output_dim)))
        self.inner_params = self.inner_net.init(next(self.rngs), demo_input)
        self.target_inner_params = self.inner_params
        
        # Initialize optimizers
        self.inner_opt = optax.adam(config.inner_lr)
        self.outer_opt = optax.adam(config.outer_lr)
        self.opt_state_inner = self.inner_opt.init(self.inner_params)
        
        # For funcBO: initialize dual network
        if config.agent_type == 'funcBO':
            self.dual_net = hk.without_apply_rng(hk.transform(
                partial(self._network_fn, config.hidden_dim, config.output_dim)
            ))
            self.dual_params = self.dual_net.init(next(self.rngs), demo_input)
            self.dual_opt = optax.adam(config.inner_lr)
            self.opt_state_dual = self.dual_opt.init(self.dual_params)
        
        # Outer parameter (regularization vector)
        init_val = float(config.lambda_reg)
        self.lambda_reg = jnp.full((config.window_size,), init_val, dtype=jnp.float32)
        self.opt_state_outer = self.outer_opt.init(self.lambda_reg)

        # For gradient averaging
        if config.average_hypergradients:
            self.grad_buffer_manager = GradientBufferManager(
                buffer_size = config.grad_buffer_size,
                grad_example = jnp.full((config.window_size,), init_val, dtype=jnp.float32)
            )
    
    def _network_fn(self, hidden_dim, output_dim, x):
        """Define network architecture."""
        layers = [
            hk.Linear(hidden_dim, w_init=hk.initializers.Orthogonal()),
            jax.nn.gelu,
            hk.Linear(hidden_dim, w_init=hk.initializers.Orthogonal()),
            jax.nn.gelu,
            hk.Linear(output_dim, w_init=hk.initializers.Orthogonal(scale=1e-1))
        ]
        mlp = hk.Sequential(layers)
        return mlp(x)
    
    #@partial(jax.jit, static_argnums=(0,))
    #def inner_loss(self, lambda_reg, inner_params, X, Y):
    #    """Inner objective: MSE + output-norm regularization (per-output coordinate)."""
    #    predictions = self.inner_net.apply(inner_params, X)
    #    data_loss = jnp.mean((Y - predictions) ** 2)
    #    reg_loss = jnp.mean(jnp.sum(lambda_reg * (predictions ** 2), axis=-1))
    #    return data_loss + reg_loss

    @partial(jax.jit, static_argnums=(0,))
    def inner_loss_window_weighted(self, params_lambda, params_h, X_in, Y_in, deltas, block_idx):
        """
        sum_s λ_s * (1/B) * Σ_i ||Y_{s,i} - h(X_i)||^2 on stacked window blocks,
        implemented without dynamic reshapes/ints (JAX-safe).
        """
        preds    = self.inner_net.apply(params_h, X_in)             # (N_tot, d_out)
        per_samp = jnp.sum((Y_in - preds) ** 2, axis=-1)            # (N_tot,)
        segs     = block_idx.reshape(-1).astype(jnp.int32)          # (N_tot,)
        deltas_1 = deltas.reshape(-1)                               # (N_tot,)

        num_segments = int(self.config.window_size)  # static Python int

        # Sum and counts per block id in [0, num_segments)
        sum_per_block   = jnp.zeros((num_segments,), per_samp.dtype).at[segs].add(per_samp)
        count_per_block = jnp.zeros((num_segments,), per_samp.dtype).at[segs].add(1.0)
        mean_loss_block = sum_per_block / jnp.maximum(count_per_block, 1.0)  # (num_segments,)

        # Per-block Δ (all samples in a block share the same Δ, so mean == that Δ)
        sum_delta_block = jnp.zeros((num_segments,), deltas_1.dtype).at[segs].add(deltas_1)
        delta_block     = sum_delta_block / jnp.maximum(count_per_block, 1.0)  # (num_segments,)

        # Map Δ -> window slot index in [0, window_size-1]
        slot_idx = jnp.clip(delta_block.astype(jnp.int32) - 1, 0, self.config.window_size - 1)

        # Pick λ for each (existing) block; segments with count 0 contribute 0 anyway
        lam_vec = params_lambda[slot_idx]  # (num_segments,)

        # Exact objective: sum_s λ_s * (1/B) Σ_i loss_{s,i}
        return jnp.sum(lam_vec * mean_loss_block)

    #@partial(jax.jit, static_argnums=(0,))
    #def grad_inner_loss(self, lambda_reg, val_h, replay, rng_unused, _unused):
    #    """
    #    Functional inner gradient: ∂/∂v [ mean(||Y_in - v||^2) + mean(⟨lambda_reg, v^2⟩) ].
    #    Here 'val_h' must be h(X_inner).
    #    We stash Y_inner at replay[2] to reuse RL's inner_solution interface.
    #    """
    #    Y_in = replay[2]  # we will pack Y_inner here
    #    def loss_wrt_values(v):
    #        data = jnp.mean((Y_in - v)**2)
    #        reg  = jnp.mean(jnp.sum(lambda_reg * (v**2), axis=-1))
    #        return data + reg
    #    return jax.grad(loss_wrt_values)(val_h)
    
    @partial(jax.jit, static_argnums=(0,))
    def grad_inner_loss(self, lambda_reg, val_h, replay, rng_unused, _unused):
        """
        For sample i in block s: grad_i = (2/B_s) * λ_{slot(s)} * (h_i - y_i),
        where B_s is the size of block s. We compute 1/B_s via per-block counts.
        """
        Y_in      = replay[2]                                  # (N_tot, d_out)
        deltas    = replay[1].reshape(-1)                      # (N_tot,)
        block_idx = replay[5].reshape(-1).astype(jnp.int32)    # (N_tot,)

        residual = (val_h - Y_in)                              # (N_tot, d_out)

        num_segments = int(self.config.window_size)
        # counts per block
        ones  = jnp.ones(block_idx.shape[0], dtype=val_h.dtype)
        counts = jnp.zeros((num_segments,), val_h.dtype).at[block_idx].add(1.0)  # (num_segments,)
        inv_B_per_sample = 1.0 / jnp.maximum(counts[block_idx], 1.0)             # (N_tot,)

        # λ per sample via Δ -> slot mapping
        slot_ids = jnp.clip(deltas.astype(jnp.int32) - 1, 0, self.config.window_size - 1)  # (N_tot,)
        lam_per_sample = lambda_reg[slot_ids]                                              # (N_tot,)

        grad_vals = 2.0 * inv_B_per_sample[:, None] * lam_per_sample[:, None] * residual   # (N_tot, d_out)
        return grad_vals
    
    @partial(jax.jit, static_argnums=(0,))
    def outer_loss_simple(self, inner_params, X, Y):
        """Outer objective: MSE."""
        predictions = self.inner_net.apply(inner_params, X)
        return jnp.mean((Y - predictions) ** 2)
    
    @partial(jax.jit, static_argnums=(0,))
    def dual_loss(self, dual_params, X, outer_grad_at_predictions):
        """Loss for dual network in funcBO."""
        dual_predictions = self.dual_net.apply(dual_params, X)
        return jnp.mean((dual_predictions - outer_grad_at_predictions) ** 2)
    
    def forward_solver(self, lambda_reg, inner_params, opt_state_inner, X, Y):
        """Solve inner problem via gradient descent."""
        params = inner_params
        opt_state = opt_state_inner
        
        for _ in range(self.config.num_inner_steps):
            # Create a function that evaluates both fun and the gradient of fun, argnums specifies wrt which arg to differentiate
            loss_val, grads = value_and_grad(self.inner_loss, argnums=1)(lambda_reg, params, X, Y)
            updates, opt_state = self.inner_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
    
        grad_norm = jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(grads)))
        return InnerSol(params, loss_val, grad_norm, opt_state)

    def backward_solver(self, dual_params, opt_state_dual, X, outer_grad):
        """Solve dual problem for funcBO."""
        params = dual_params
        opt_state = opt_state_dual
        
        for _ in range(self.config.num_inner_steps):
            # Create a function that evaluates both fun and the gradient of fun, argnums specifies wrt which arg to differentiate
            loss_val, grads = value_and_grad(self.dual_loss)(params, X, outer_grad)
            updates, opt_state = self.dual_opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        
        # Return dual network predictions
        dual_predictions = self.dual_net.apply(params, X)
        return dual_predictions, opt_state
    
    def loss_funcBO(self, lambda_reg, aux_params, X_inner, Y_inner, X_outer, Y_outer, deltas_inner=None, block_index=None):
        """
        Assumptions this follows:
        - `inner_solution` orchestrates the bilevel step via a custom VJP and expects:
            sol = inner_solution(
                grad_inner_loss_fn,         # computes grads wrt inner params
                inner_network,              # the prediction network or its module
                params_inner,               # parameters of the inner network
                params_outer,               # outer params (here: lambda_reg)
                data_blob,                  # anything needed for loss evals
                rng,                        # PRNGKey (can be None if unused)
                params_dual,                # current dual params
                (fwd_solver, bwd_solver),   # tuple of solver callables
            )
        - Your forward solver updates inner params given lambda_reg on (X_inner, Y_inner).
        - Your backward solver trains the dual to match the outer-gradient signal.
        - `aux_params` carries initial params / opt states / rng.
        """

        # Reuse existing trained states if available; otherwise fall back to what's provided.
        #inner_params = getattr(aux_params, "inner_params", None)   # inner network weights (MLP)
        #dual_params  = getattr(aux_params, "dual_params",  None)   # dual network weights
    
        # Pseudo-replay shaped like RL
        n = X_inner.shape[0]
        dummy_nd = jnp.ones((n, 1), dtype=jnp.float32)

        # defaults
        if deltas_inner is None:
            deltas_inner = jnp.ones((n, 1), dtype=jnp.float32)
        if block_index is None:
            block_index = jnp.zeros((n, 1), dtype=jnp.int32)

        # Pack replay as (obs, actions, reward, next_obs, not_dones, not_dones_no_max)
        # Here: actions := deltas (Δ), and we also need block ids
        # We'll carry block ids in the "not_dones_no_max" slot (any free slot works consistently)
        replay = (X_inner, deltas_inner, Y_inner, X_outer, dummy_nd, block_index.astype(jnp.int32))

        # Forward solver: update inner params on (X_inner, Y_inner) with λ
        def fwd_solver(params_h, params_lambda, rpl, rng_unused):
            X_in, deltas, Y_in = rpl[0], rpl[1].reshape(-1), rpl[2]
            block_idx = rpl[5].reshape(-1).astype(jnp.int32)

            def inner_loss_weighted(p):
                return self.inner_loss_window_weighted(params_lambda, p, X_in, Y_in, deltas, block_idx)

            params = aux_params.inner_params
            opt_state = aux_params.opt_state_inner
            for _ in range(self.config.num_inner_steps):
                loss_val, grads = value_and_grad(inner_loss_weighted)(params)
                updates, opt_state = self.inner_opt.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

            grad_norm = jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(grads)))
            Sol = namedtuple('Sol', 'params_Q loss_Q vals_Q grad_norm_Q entropy_Q '
                                    'target_params_Q opt_state_Q next_obs_nll')
            target = params
            return Sol(params, loss_val, None, grad_norm, None, target, opt_state, None)

        # Backward solver
        def bwd_solver(params_dual, rpl, rng_unused, outer_grad_on_outer, lambda_for_adj):
            X_in, deltas, Y_in, X_out = rpl[0], rpl[1].reshape(-1), rpl[2], rpl[3]
            block_idx = rpl[5].reshape(-1).astype(jnp.int32)

            a_in  = self.dual_net.apply(params_dual, X_in)    # (N_tot, d_out)
            a_out = self.dual_net.apply(params_dual, X_out)   # (B, d_out)

            a2        = jnp.sum(a_in ** 2, axis=-1)                  # (N_tot,)
            segs      = block_idx                                    # (N_tot,)
            num_segs  = int(self.config.window_size)

            sum_a     = jnp.zeros((num_segs,), a2.dtype).at[segs].add(a2)
            cnts      = jnp.zeros((num_segs,), a2.dtype).at[segs].add(1.0)
            per_block_a = sum_a / jnp.maximum(cnts, 1.0)             # (num_segs,)

            deltas_1    = deltas                                     # (N_tot,)
            sum_delta   = jnp.zeros((num_segs,), deltas_1.dtype).at[segs].add(deltas_1)
            delta_block = sum_delta / jnp.maximum(cnts, 1.0)         # (num_segs,)

            slot_idx = jnp.clip(delta_block.astype(jnp.int32) - 1, 0, self.config.window_size - 1)
            lam_vec  = jax.lax.stop_gradient(lambda_for_adj)[slot_idx]  # (num_segs,)

            # 0.5 a^T H a with H = blockdiag((2/B_s) λ_s I) ⇒ λ_s * mean_i ||a||^2 per block
            hess_term = jnp.sum(lam_vec * per_block_a)

            # linear outer term unchanged
            lin_term  = jnp.mean(jnp.sum(a_out * outer_grad_on_outer, axis=-1))
            loss = hess_term + lin_term

            def loss_wrt_params(p):
                a_in_p  = self.dual_net.apply(p, X_in)
                a_out_p = self.dual_net.apply(p, X_out)
                a2_p    = jnp.sum(a_in_p ** 2, axis=-1)
                sum_a_p = jnp.zeros((num_segs,), a2_p.dtype).at[segs].add(a2_p)
                per_block_a_p = sum_a_p / jnp.maximum(cnts, 1.0)
                hess_p = jnp.sum(lam_vec * per_block_a_p)
                lin_p  = jnp.mean(jnp.sum(a_out_p * outer_grad_on_outer, axis=-1))
                return hess_p + lin_p

            grads = jax.grad(loss_wrt_params)(params_dual)
            updates, new_opt = self.dual_opt.update(grads, aux_params.opt_state_dual)
            new_params = optax.apply_updates(params_dual, updates)

            DualSol = namedtuple('DualSol','params_dual_Q val_dual_Q loss_dual_Q opt_state_dual_Q grad_norm_dual_Q')
            gn = jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(grads)))
            a_in_new = self.dual_net.apply(new_params, X_in)
            return DualSol(new_params, a_in_new, loss, new_opt, gn)

        #def bwd_solver(params_dual, rpl, rng_unused, outer_grad_on_outer):
        #    X_out = rpl[3]  # X_outer
        #    # Simple MSE training of the dual to match the adjoint signal
        #    preds = self.dual_net.apply(params_dual, X_out)
        #    loss  = jnp.mean((preds - outer_grad_on_outer)**2)
        #    grads = jax.grad(lambda p: jnp.mean((self.dual_net.apply(p, X_out)
        #                                        - outer_grad_on_outer)**2))(params_dual)
        #    updates, new_opt = self.dual_opt.update(grads, aux_params.opt_state_dual)
        #    new_params = optax.apply_updates(params_dual, updates)
        #    DualSol = namedtuple('DualSol',
        #                        'params_dual_Q val_dual_Q loss_dual_Q opt_state_dual_Q grad_norm_dual_Q')
        #    gn = jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(grads)))
        #    return DualSol(new_params, self.dual_net.apply(new_params, X_out), loss, new_opt, gn)

        # Call the custom-VJP driver
        sol = inner_solution(
            self.grad_inner_loss,   # functional inner-gradient (values→grads) on X_inner
            self.inner_net,         # Haiku-transformed model with .apply
            aux_params.inner_params,
            lambda_reg,
            replay,
            getattr(aux_params, "rng", None),
            aux_params.dual_params,     # unused unless you enable dual-VJP (see note)
            (fwd_solver, bwd_solver)
        )

        # IMPORTANT: outer loss uses the VALUES from inner_solution to hit the custom VJP.
        outer_mse = jnp.mean((sol.val_target_Q - Y_outer)**2)

        return outer_mse, sol

    def loss_omd(self, lambda_reg_array, aux_params, X_inner, Y_inner, X_outer, Y_outer):
        """Loss function for OMD agent."""
        lambda_reg = lambda_reg_array[0]
        
        # Define constraint function: gradient of inner loss should be zero
        def constraint_func(lambda_reg, inner_params, X, Y):
            _, grads = value_and_grad(self.inner_loss)(inner_params, X, Y)
            return grads
        
        def fwd_solver(inner_params, lambda_unused, X, Y):
            return self.forward_solver(inner_params, aux_params.opt_state_inner, X, Y)
        
        # Solve inner problem on (X_inner, Y_inner)
        sol = fwd_solver(aux_params.inner_params, lambda_reg, X_inner, Y_inner)
        # Compute outer loss on (X_outer, Y_outer)
        outer_loss = self.outer_loss_simple(lambda_reg, sol.params, X_outer, Y_outer)
        
        return outer_loss, sol

    def loss_mle(self, lambda_reg_array, aux_params, X_inner, Y_inner, X_outer, Y_outer):
        """MLE baseline: solve inner on inner batch, evaluate outer on outer batch (no reg)."""
        lambda_reg = lambda_reg_array[0]

        # Solve inner on inner batch
        sol = self.forward_solver(aux_params.inner_params, aux_params.opt_state_inner, X_inner, Y_inner)

        # outer objective = inner loss evaluated on outer batch (consistent holdout)
        preds = self.inner_net.apply(sol.params, X_outer)
        outer_loss = jnp.mean((Y_outer - preds) ** 2)

        # Return outer loss and inner solution
        return outer_loss, sol
    

    @partial(jax.jit, static_argnums=(0, 8))
    def update_step_outer(self, lambda_reg_array, aux_params, opt_state_outer, X_inner, Y_inner, X_outer, Y_outer, loss_fn):
        (value, aux_out), grads = value_and_grad(loss_fn, has_aux=True)(lambda_reg_array, aux_params, X_inner, Y_inner, X_outer, Y_outer)
        updates, opt_state_outer = self.outer_opt.update(grads, opt_state_outer)
        new_lambda = optax.apply_updates(lambda_reg_array, updates)
        new_lambda = jnp.maximum(new_lambda, 0.)
        return value, new_lambda, opt_state_outer, aux_out
    
    def update(self, X_inner, Y_inner, X_outer=None, Y_outer=None,deltas_inner=None, block_index=None):
        """Update the outer parameter."""
        lambda_reg_array = self.lambda_reg
        
        if self.config.agent_type == 'funcBO':
            aux_params = AuxParams(
                self.inner_params, self.target_inner_params, self.dual_params,
                self.opt_state_inner, self.opt_state_dual, next(self.rngs), self.lambda_reg
            )
            loss_fn = self.loss_funcBO
        
        elif self.config.agent_type == 'omd':
            aux_params = AuxParams(
                self.inner_params, self.target_inner_params, None,
                self.opt_state_inner, None, next(self.rngs), self.lambda_reg
            )
            loss_fn = self.loss_omd
        
        else:  # mle
            aux_params = AuxParams(
                self.inner_params, self.target_inner_params, None,
                self.opt_state_inner, None, next(self.rngs), self.lambda_reg
            )
            loss_fn = self.loss_mle
        
        # Update outer parameters
        if self.config.average_hypergradients and hasattr(self, 'grad_buffer_manager'):
            # Compute gradients without applying them, pass both inner and outer batches
            (value, aux_out), grads = value_and_grad(loss_fn, has_aux=True)(
                lambda_reg_array, aux_params, X_inner, Y_inner, X_outer, Y_outer,
                deltas_inner, block_index)

            _, _, avg_grad = self.grad_buffer_manager.update(grads)

            final_grad = jax.tree_map(
                lambda g, avg: self.config.beta * g + (1 - self.config.beta) * avg,
                grads, avg_grad
            )

            updates, self.opt_state_outer = self.outer_opt.update(final_grad, self.opt_state_outer)
            new_lambda_array = optax.apply_updates(lambda_reg_array, updates)
            new_lambda_array = jnp.maximum(new_lambda_array, 0.)
        else:
            wrapped_loss = (
                lambda lr, ap, Xi, Yi, Xo, Yo:
                    loss_fn(lr, ap, Xi, Yi, Xo, Yo, deltas_inner, block_index)
            )
            value, new_lambda_array, self.opt_state_outer, aux_out = self.update_step_outer(
                lambda_reg_array, aux_params, self.opt_state_outer,
                X_inner, Y_inner, X_outer, Y_outer, wrapped_loss
            )
        
        # Update agent parameters
        self.lambda_reg = new_lambda_array
        self.inner_params = aux_out.params_Q
        self.opt_state_inner = aux_out.opt_state_Q
        self.target_inner_params = aux_out.target_params_Q
        
        if self.config.agent_type == 'funcBO':
            # Update dual parameters would go here if we tracked them
            pass
        
        # Soft update of target parameters
        if self.config.warm_start:
            tau = self.config.tau
            self.target_inner_params = jax.tree_map(
                lambda p, tp: tau * p + (1 - tau) * tp,
                self.inner_params, self.target_inner_params
            )
        
        return {
            'outer_loss': float(value),
            'inner_loss': float(aux_out.loss_Q),
            'grad_norm': float(aux_out.grad_norm_Q),
            'lambda_reg_0': float(self.lambda_reg[0]),  # log first coordinate
            'lambda_reg_1': float(self.lambda_reg[1]) if self.lambda_reg.shape[0] > 1 else None,
            'lambda_reg_2': float(self.lambda_reg[2]) if self.lambda_reg.shape[0] > 2 else None,
            'lambda_reg_18': float(self.lambda_reg[18]) if self.lambda_reg.shape[0] > 18 else None,
            'lambda_reg_19': float(self.lambda_reg[19]) if self.lambda_reg.shape[0] > 19 else None,
            'lambda_reg_99': float(self.lambda_reg[99]) if self.lambda_reg.shape[0] > 99 else None,
        }


class BilevelRegressionTrainer:
    """Trainer for non-stationary bilevel regression problems."""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        
        # Initialize wandb if available and no logger provided
        if WANDB_AVAILABLE and logger is None:
            wandb.init(
                project="bilevel_regression",
                name=f"{config.agent_type}_{config.shift_type}",
                config=config.__dict__ if hasattr(config, '__dict__') else config
            )
        
        # Set random seed
        np.random.seed(config.seed)
        
        # Initialize agent
        self.agent = BilevelRegressionAgent(config)
        
        # Generate fixed input data
        self.X = self._generate_input_data()
        
        # Disjoint inner/outer split
        n = self.X.shape[0]
        perm = np.random.permutation(n)

        outer_frac = getattr(self.config, "split_outer", 0.5)  # default 20% outer
        n_out = max(1, int(outer_frac * n))

        self.outer_idx = perm[:n_out]
        self.inner_idx = perm[n_out:]

        # Cache the split inputs to avoid accidental mixing later
        self.X_in  = self.X[self.inner_idx]
        self.X_out = self.X[self.outer_idx]

        # Initialize true function
        self.true_net = hk.without_apply_rng(hk.transform(
            partial(self._true_network_fn, config.hidden_dim, config.output_dim)
        ))
        rng = jax.random.PRNGKey(config.seed)
        self.true_params = self.true_net.init(rng, self.X[:1])
        self.Y_true = self.true_net.apply(self.true_params, self.X)
        
        # Generate shift direction
        self.shift_direction = jax.random.normal(rng, (config.output_dim,))
        self.shift_direction = self.shift_direction / jnp.linalg.norm(self.shift_direction)
    
    def _generate_input_data(self):
        """Generate fixed input data."""
        rng = jax.random.PRNGKey(self.config.seed)
        return jax.random.normal(rng, (self.config.n_samples, self.config.input_dim))
    
    def _true_network_fn(self, hidden_dim, output_dim, x):
        """Define the true function network."""
        layers = [
            hk.Linear(hidden_dim, w_init=hk.initializers.TruncatedNormal(mean=0.1)),
            jax.nn.gelu,
            hk.Linear(output_dim, w_init=hk.initializers.TruncatedNormal(mean=0.1)),
        ]
        mlp = hk.Sequential(layers)
        return mlp(x)

    def generate_target_shift(self, t):
        """Generate time-varying target shift."""
        if self.config.shift_type == 'linear':
            return self.config.alpha * t
        elif self.config.shift_type == 'sinusoidal':
            return self.config.beta * jnp.sin(self.config.omega * t)
        else:
            return 0.0

    def get_target_at_time(self, t, key):
        """Get target Y_t for time step t."""
        shift_magnitude = self.generate_target_shift(t)
        shift = shift_magnitude * self.shift_direction
        noise = self.config.noise_level * jax.random.normal(key, self.Y_true.shape)
        Y_t = self.Y_true + shift[None, :] + noise

        # Debug prints for a few timesteps
        if t in [1, 9, 99, 999, 9999, 99999]:
            print(f"\n--- Debug info at t={t} ---", file=sys.stderr)
            print(f"Shift magnitude: {shift_magnitude}", file=sys.stderr)
            print(f"Shift direction (first 3): {np.array(self.shift_direction[:3])}", file=sys.stderr)
            print(f"First 3 X: \n{np.array(self.X[:3])}", file=sys.stderr)
            print(f"First 3 noise: \n{np.array(noise[:3])}", file=sys.stderr)
            print(f"First 3 Y_true: \n{np.array(self.Y_true[:3])}", file=sys.stderr)
            print(f"First 3 Y_t (shifted + noise): \n{np.array(Y_t[:3])}", file=sys.stderr)

        return Y_t
    
    def compute_param_distance(self):
        """Compute L2 distance between true and learned parameters."""
        # Flatten both parameter trees
        true_flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(self.true_params)])
        learned_flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(self.agent.inner_params)])
        
        # Check if parameter counts match
        if true_flat.shape[0] != learned_flat.shape[0]:
            print(f"Warning: Parameter count mismatch - True: {true_flat.shape[0]}, Learned: {learned_flat.shape[0]}", file=sys.stderr)
            return float('nan'), float('nan')
        
        # Compute L2 distance
        l2_distance = jnp.linalg.norm(true_flat - learned_flat)
        
        # Compute relative distance (normalized by true parameter norm)
        true_norm = jnp.linalg.norm(true_flat)
        relative_distance = l2_distance / (true_norm + 1e-8)
        
        return float(l2_distance), float(relative_distance)
    
    def evaluate(self, t):
        """Evaluate current agent performance."""
        step_key_seq = hk.PRNGSequence(self.config.seed)
        key = next(step_key_seq)
        Y_t = self.get_target_at_time(t, key)
        
        # Get predictions
        predictions = self.agent.inner_net.apply(self.agent.inner_params, self.X)
        
        # Compute losses
        mse_loss = float(jnp.mean((Y_t - predictions) ** 2))
        
        # Compute parameter distance
        #l2_dist, rel_dist = self.compute_param_distance()
        
        return {
            'eval_mse_loss': mse_loss,
            'eval_lambda_0': float(self.agent.lambda_reg[0])
        }
    
    def plot_approximation_over_time(self, save_path="fig.png"):
        """Plot how the network approximates the data over time."""
        # Create time steps for visualization
        time_steps = [1, self.config.T_max // 4, self.config.T_max // 2, 
                    3 * self.config.T_max // 4, self.config.T_max]
        
        # For 1D output, create line plots. For higher dimensions, use first dimension
        if self.config.input_dim <= 2 and self.config.output_dim == 1:
            # 1D or 2D input, 1D output - create line/surface plots
            self._plot_1d_approximation(time_steps, save_path)
        else:
            # Higher dimensional - create prediction vs target scatter plots
            self._plot_prediction_scatter(time_steps, save_path)

    def _plot_1d_approximation(self, time_steps, save_path):
        """Plot 1D approximation over time."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Sort data by first input dimension for line plots
        if self.config.input_dim == 1:
            sort_idx = jnp.argsort(self.X[:, 0])
            X_sorted = self.X[sort_idx]
        else:
            # For 2D input, create a grid for visualization
            x_range = jnp.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 50)
            y_range = jnp.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 50)
            X_grid, Y_grid = jnp.meshgrid(x_range, y_range)
            X_sorted = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
            sort_idx = jnp.arange(len(X_sorted))
        
        step_key_seq = hk.PRNGSequence(self.config.seed)

        for i, t in enumerate(time_steps):
            ax = axes[i]
            
            # Get true target at time t
            key = next(step_key_seq)
            Y_true_t = self.get_target_at_time(t, key)
            if self.config.input_dim == 1:
                Y_true_sorted = Y_true_t[sort_idx]
            else:
                # For 2D input, evaluate true function on grid
                Y_true_sorted = self.true_net.apply(self.true_params, X_sorted)
                shift_magnitude = self.generate_target_shift(t)
                shift = shift_magnitude * self.shift_direction
                Y_true_sorted = Y_true_sorted + shift[None, :]
            
            # Get network prediction at time t (use current parameters)
            Y_pred_sorted = self.agent.inner_net.apply(self.agent.inner_params, X_sorted)
            
            if self.config.input_dim == 1:
                # Line plot for 1D input
                ax.plot(X_sorted[:, 0], Y_true_sorted[:, 0], 'b-', label='True', alpha=0.7)
                ax.plot(X_sorted[:, 0], Y_pred_sorted[:, 0], 'r--', label='Predicted', alpha=0.7)
                ax.set_xlabel('Input')
                ax.set_ylabel('Output')
            else:
                # Contour plot for 2D input
                Z_true = Y_true_sorted[:, 0].reshape(50, 50)
                Z_pred = Y_pred_sorted[:, 0].reshape(50, 50)
                
                c1 = ax.contour(X_grid, Y_grid, Z_true, colors='blue', alpha=0.6, linestyles='-')
                c2 = ax.contour(X_grid, Y_grid, Z_pred, colors='red', alpha=0.6, linestyles='--')
                ax.set_xlabel('Input 1')
                ax.set_ylabel('Input 2')
            
            # Compute MSE for this time step
            mse = jnp.mean((Y_true_sorted - Y_pred_sorted) ** 2)
            
            ax.set_title(f'Time {t}, λ={self.agent.lambda_reg[0]:.3f}\nMSE={mse:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(time_steps) < 6:
            fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Approximation plot saved to {save_path}", file=sys.stderr)

    def _plot_prediction_scatter(self, time_steps, save_path):
        """Plot prediction vs target scatter plots for high-dimensional cases."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = plt.cm.viridis(jnp.linspace(0, 1, len(time_steps)))

        step_key_seq = hk.PRNGSequence(self.config.seed)
        
        for i, t in enumerate(time_steps):
            ax = axes[i]
            
            # Get true target and prediction at time t
            key = next(step_key_seq)
            Y_true_t = self.get_target_at_time(t, key)
            Y_pred_t = self.agent.inner_net.apply(self.agent.inner_params, self.X)
            
            # Use first output dimension for visualization
            y_true_dim0 = Y_true_t[:, 0] if self.config.output_dim > 1 else Y_true_t.flatten()
            y_pred_dim0 = Y_pred_t[:, 0] if self.config.output_dim > 1 else Y_pred_t.flatten()
            
            # Scatter plot
            ax.scatter(y_true_dim0, y_pred_dim0, c=[colors[i]], alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(y_true_dim0.min(), y_pred_dim0.min())
            max_val = max(y_true_dim0.max(), y_pred_dim0.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
            
            # Compute correlation
            corr = jnp.corrcoef(y_true_dim0, y_pred_dim0)[0, 1]
            mse = jnp.mean((y_true_dim0 - y_pred_dim0) ** 2)
            
            ax.set_xlabel('True Output')
            ax.set_ylabel('Predicted Output')
            ax.set_title(f'Time {t}, λ={self.agent.lambda_reg[0]:.3f}\nMSE={mse:.4f}, Corr={corr:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Remove empty subplot
        if len(time_steps) < 6:
            fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Approximation plot saved to {save_path}", file=sys.stderr)

    def train(self):
        """Main training loop."""
        print(f"Starting training with agent_type={self.config.agent_type}, shift_type={self.config.shift_type}", file=sys.stderr)
        
        start_time = time.time()
        step_key_seq = hk.PRNGSequence(self.config.seed)  # step-scoped RNG

        for t in range(1, self.config.T_max + 1):
            key_inner = next(step_key_seq)
            key_outer = next(step_key_seq)

            # Build independent targets, then SLICE them on the fixed split
            Y_inner_full = self.get_target_at_time(t, key_inner)
            Y_outer_full = self.get_target_at_time(t, key_outer)

            Y_inner_pool = Y_inner_full[self.inner_idx]   # shape: (N_in, d_out)
            Y_outer_pool = Y_outer_full[self.outer_idx]   # shape: (N_out, d_out)

            N_in  = self.X_in.shape[0]
            N_out = self.X_out.shape[0]

            # Decide one common minibatch size B
            B_cfg = getattr(self.config, "batch_size", None)
            if B_cfg is None:
                # fall back to the smallest pool size if not specified
                B = min(N_in, N_out)
            else:
                B = min(B_cfg, N_in, N_out)

            # Sample *independently* from each pool (keeps inner/outer disjoint by construction)
            idx_in  = jax.random.choice(key_inner, N_in,  (B,), replace=False)
            idx_out = jax.random.choice(key_outer, N_out, (B,), replace=False)

            X_inner = self.X_in[idx_in]
            Y_inner = Y_inner_pool[idx_in]

            X_outer = self.X_out[idx_out]
            Y_outer = Y_outer_pool[idx_out]

            # Sanity: equal batch sizes, still disjoint pools
            assert X_inner.shape[0] == X_outer.shape[0] == Y_inner.shape[0] == Y_outer.shape[0] == B

            # Build windowed inner batch with BLOCKS (no normalization)
            w = int(getattr(self.config, "window_size", 1))
            starts = list(range(max(1, t - w), t))  # s ∈ {t-w,...,t-1} clipped at 1
            if len(starts) == 0:
                X_in_concat = X_inner
                Y_in_concat = Y_inner
                deltas_inner = jnp.ones((X_inner.shape[0], 1))        # Δ = 1 placeholder
                block_index  = jnp.zeros((X_inner.shape[0], 1))       # block id 0
            else:
                # reuse the SAME idx_in for alignment across all s
                B = X_inner.shape[0]
                X_in_concat = jnp.concatenate([X_inner for _ in starts], axis=0)

                # build Ys at each s and stack in the same order as 'starts'
                Y_blocks = []
                deltas_list = []
                block_ids = []
                for k, s in enumerate(starts):
                    key_s = next(step_key_seq)
                    Y_s_full = self.get_target_at_time(s, key_s)
                    Y_s_pool = Y_s_full[self.inner_idx]
                    Y_blocks.append(Y_s_pool[idx_in])

                    Δ = float(t - s)  # 1..w
                    deltas_list.append(jnp.full((B, 1), Δ))
                    block_ids.append(jnp.full((B, 1), k))  # block id k

                Y_in_concat   = jnp.concatenate(Y_blocks, axis=0)
                deltas_inner  = jnp.concatenate(deltas_list, axis=0)  # (w*B, 1)
                block_index   = jnp.concatenate(block_ids, axis=0)    # (w*B, 1), ints

            # Pass minibatches to the agent
            #metrics = self.agent.update(X_inner, Y_inner, X_outer=X_outer, Y_outer=Y_outer)
            metrics = self.agent.update(X_in_concat, Y_in_concat, X_outer=X_outer, Y_outer=Y_outer, deltas_inner=deltas_inner, block_index=block_index)


            # Add time step to metrics
            metrics['time_step'] = t
            metrics['shift_magnitude'] = float(self.generate_target_shift(t))
            
            # Periodic evaluation
            if t % self.config.eval_frequency == 0:
                eval_metrics = self.evaluate(t)
                metrics.update(eval_metrics)
                
                elapsed_time = time.time() - start_time
                metrics['elapsed_time'] = elapsed_time

                print(f"Step {t}: outer_loss={metrics['outer_loss']:.4f}, "
                      f"eval_mse_loss={metrics['eval_mse_loss']:.4f}, "
                      f"lambda_0={metrics['lambda_reg_0']:.4f}", file=sys.stderr)
            
            # Log metrics
            if self.config.log_frequency > 0 and t % self.config.log_frequency == 0:
                if WANDB_AVAILABLE:
                    wandb.log(metrics, step=t)
                elif self.logger is not None:
                    self.logger.log_metrics(metrics, step=t)
        
        # Final evaluation
        final_metrics = self.evaluate(self.config.T_max)
        final_loss = final_metrics['eval_mse_loss']
        
        print(f"Training completed. Final total loss: {final_loss:.4f}")
        
        self.plot_approximation_over_time("fig.png")

        if WANDB_AVAILABLE:
            wandb.log({'final_loss': final_loss}, step=self.config.T_max)
            wandb.finish()
        
        return final_loss