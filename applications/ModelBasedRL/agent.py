import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.scipy import stats
from jax.scipy.special import logsumexp
from jax.lax import stop_gradient
import optax
import chex
import haiku as hk
from functools import partial
from collections import namedtuple

# Import custom VJP implementations
from utils import root_solve, inner_solution, GradientBufferManager

# Data structures for tracking state
AuxP = namedtuple('AuxP', 'params_Q target_params_Q opt_state_Q rng')
AuxOut = namedtuple('AuxOut', 'vals_Q entropy_Q next_obs_nll')
AuxPdual = namedtuple('AuxPdual', 'params_Q target_params_Q params_dual_Q opt_state_Q opt_state_dual_Q rng')

# Loss functions
def nll_loss(x, m, ls):
    """Negative log-likelihood loss for probabilistic predictions."""
    return -stats.norm.logpdf(x, m, jnp.exp(ls)).sum(-1).mean()

def mse_loss(x, xp):
    """Mean squared error loss for deterministic predictions."""
    return ((x - xp) ** 2).sum(-1).mean()

class Agent:
    """Model-based reinforcement learning agent with bilevel optimization."""
    
    def __init__(self, args, obs_space, action_space):
        self.args = args
        self.obs_dim = obs_space.shape[0]
        self.action_dim = action_space.n
        self.obs_range = (obs_space.low, obs_space.high)
        
        # Example observations for network initialization
        demo_obs = jnp.ones((1, self.obs_dim))
        demo_obs_action = jnp.ones((1, self.obs_dim + self.action_dim))
        
        # Initialize random number generator
        self.rngs = hk.PRNGSequence(self.args.seed)
        
        # Initialize networks based on agent type
        if self.args.agent_type == 'vep':
            self.V = self._init_network('V', (self.obs_dim, self.action_dim, self.args.hidden_dim))
            self.params_V = self.V.net.init(next(self.rngs), demo_obs)

        # World model (T) for transition dynamics
        self.T = self._init_network('T', (self.obs_dim, self.action_dim, self.args.model_hidden_dim))
        self.params_T = self.T.net.init(next(self.rngs), demo_obs_action)
        self.opt_state_T = self.T.opt.init(self.params_T)

        # Q-function for action values
        self.Q = self._init_network('Q', (self.obs_dim, self.action_dim, self.args.hidden_dim))
        self.params_Q = self.target_params_Q = self.Q.net.init(next(self.rngs), demo_obs)
        self.opt_state_Q = self.Q.opt.init(self.params_Q)

        # For functional bilevel optimization agent, add dual Q-network
        if self.args.agent_type == 'funcBO':
            self.dual_Q = self._init_network('dual_Q', (self.obs_dim, self.action_dim, self.args.hidden_dim))
            self.params_dual_Q = self.dual_Q.net.init(next(self.rngs), demo_obs)
            self.opt_state_dual_Q = self.dual_Q.opt.init(self.params_dual_Q)

        # For gradient averaging
        if self.args.average_hypergradients:
            # Initialize gradient buffer manager
            self.grad_buffer_manager = GradientBufferManager(
                buffer_size=self.args.grad_buffer_size,
                grad_example=self.params_T
            )

    def _init_network(self, net_type, dims):
        """Initialize a network and its optimizer."""
        net = hk.without_apply_rng(hk.transform(
            partial(self._network_fn, net_type, dims)
        ))
        
        # Choose optimizer based on network type
        if net_type == 'Q':
            opt = optax.adam(self.args.inner_lr)
        elif net_type == 'dual_Q':
            opt = optax.adam(self.args.inner_lr)
        else:
            opt = optax.adam(self.args.lr)
            
        Network = namedtuple(net_type, 'net opt')
        return Network(net, opt)
    
    def _network_fn(self, net_type, dims, x):
        """Define network architecture based on type."""
        obs_dim, action_dim, hidden_dim = dims
        activation = jax.nn.relu
        init = hk.initializers.Orthogonal(scale=jnp.sqrt(2.0))
        
        # Common hidden layers
        layers = [
            hk.Linear(hidden_dim, w_init=init), activation,
            hk.Linear(hidden_dim, w_init=init), activation,
        ]
        
        final_init = hk.initializers.Orthogonal(scale=1e-2)
        # Special case for VEP ensemble
        if self.args.agent_type == 'vep' and net_type == 'V':
            ensemble = []
            for i in range(self.args.num_ensemble_vep):
                vf_layers = [
                    hk.Linear(hidden_dim, w_init=init), activation,
                    hk.Linear(hidden_dim, w_init=init), activation,
                    hk.Linear(1, w_init=final_init)
                ]
                mlp = hk.Sequential(vf_layers)
                ensemble.append(mlp(x))
            return ensemble
            
        # Output layer based on network type
        if net_type == 'V':
            layers += [hk.Linear(1, w_init=final_init)]
        elif net_type == 'Q' or net_type == 'dual_Q':
            layers += [hk.Linear(action_dim, w_init=final_init)]
        elif net_type == 'T':
            layers += [hk.Linear(obs_dim, w_init=final_init)]
        
        # Special cases for double networks
        if (net_type == 'Q' or net_type == 'dual_Q') and not self.args.no_double:
            # Double Q-learning with two networks
            layers2 = [
                hk.Linear(hidden_dim, w_init=init), activation,
                hk.Linear(hidden_dim, w_init=init), activation,
                hk.Linear(action_dim, w_init=final_init)
            ]
            mlp1, mlp2 = hk.Sequential(layers), hk.Sequential(layers2)
            return mlp1(x), mlp2(x)
        elif net_type == 'T' and not self.args.no_learn_reward:
            # Separate reward prediction for world model
            layers2 = [
                hk.Linear(hidden_dim, w_init=init), activation,
                hk.Linear(hidden_dim, w_init=init), activation,
                hk.Linear(1, w_init=final_init)
            ]
            mlp1, mlp2 = hk.Sequential(layers), hk.Sequential(layers2)
            return mlp1(x), mlp2(x)
        else:
            # Single network
            mlp = hk.Sequential(layers)
            return mlp(x)
    
    # ----- AGENT ACTION SELECTION -----
    
    @partial(jax.jit, static_argnums=(0,))
    def act(self, params_Q, obs, rng):
        """Select action based on current Q-function."""
        obs = jnp.array(obs[0]) if isinstance(obs, tuple) else obs[None, ...]
        
        current_Q = self.Q.net.apply(params_Q, obs[None, ...])
        if not self.args.no_double:
            # Average Q-values for double Q-networks
            current_Q = 0.5 * (current_Q[0] + current_Q[1])
            
        # Either take greedy action or sample from softmax distribution
        if self.args.hard:
            action = jnp.argmax(current_Q, axis=-1)
        else:
            action = jax.random.categorical(rng, current_Q / self.args.alpha)
            
        return action if isinstance(obs, list) else action[0]
    
    # ----- WORLD MODEL METHODS -----
    
    @partial(jax.jit, static_argnums=(0,))
    def model_pred(self, params_T, obs, action, rng):
        """Predict next state and reward using world model."""
        # Convert action to one-hot representation
        if not isinstance(action, int):
            action = action[:, 0]
        a = jax.nn.one_hot(action, self.action_dim)
        
        # Combine state and action
        x = jnp.concatenate((obs, a), axis=-1)
        
        # Deterministic model
        if self.args.no_learn_reward:
            next_obs_pred = self.T.net.apply(params_T, x)
            reward_pred = None
        else:
            next_obs_pred, reward_pred = self.T.net.apply(params_T, x)
            
        return next_obs_pred, next_obs_pred, None, reward_pred
    
    def batch_real_to_model(self, params_T, batch, rng):
        """Generate predicted transitions from real transitions."""
        obs, action, reward, next_obs, not_done, not_done_no_max = batch
        
        # Get predictions
        next_obs_pred, means, logstds, reward_pred = self.model_pred(
            params_T, obs, action, rng
        )
        
        # Use real rewards if not learning rewards
        if self.args.no_learn_reward:
            reward_pred = reward
            
        # Create new batch with model predictions
        batch_model = obs, action, reward_pred, next_obs_pred, not_done, not_done_no_max
        
        # Calculate prediction error
        nll = mse_loss(next_obs_pred, next_obs)
            
        return batch_model, nll
    
    # ----- Q-FUNCTION METHODS -----
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_Q(self, params_Q, target_params_Q, batch):
        """Compute loss for Q-function (inner optimization)."""
        obs, action, reward, next_obs, not_done, not_done_no_max = batch
        
        # Get Q-values
        target_Q = self.Q.net.apply(stop_gradient(target_params_Q), next_obs)
        Q_s = self.Q.net.apply(params_Q, obs)
        
        return self.func_loss_Q(Q_s, target_Q, batch)
    
    @partial(jax.jit, static_argnums=(0,))
    def func_loss_Q(self, Q_s, target_Q, batch):
        """Functional version of Q-learning loss that works with Q-values directly."""
        obs, action, reward, next_obs, not_done, not_done_no_max = batch
        
        # Calculate target value based on agent configuration
        if self.args.hard:
            # Hard max (standard Q-learning)
            if self.args.no_double:
                target_V = jnp.max(target_Q, axis=-1, keepdims=True)
            else:
                target_Q = jnp.minimum(target_Q[0], target_Q[1])
                target_V = jnp.max(target_Q, axis=-1, keepdims=True)
        else:
            # Soft max (entropy-regularized)
            if self.args.no_double:
                target_V = self.args.alpha * logsumexp(target_Q / self.args.alpha, 
                                                axis=-1, keepdims=True)
            else:
                target_Q = jnp.minimum(target_Q[0], target_Q[1])
                target_V = self.args.alpha * logsumexp(target_Q / self.args.alpha, 
                                                axis=-1, keepdims=True)
        
        # Bellman target: r + γV(s')
        target_Q = (reward + (not_done_no_max * self.args.discount * target_V))[:, 0]
        
        # Compute losses
        if self.args.no_double:
            current_Q = Q_s[jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            vals_Q = current_Q.mean()
            entropy_Q = (-jax.nn.log_softmax(Q_s) * jax.nn.softmax(Q_s)).sum(-1).mean()
            mse_Q = jnp.mean((current_Q - target_Q)**2)
        else:
            current_Q1 = Q_s[0][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            current_Q2 = Q_s[1][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            
            entropy_Q = (-jax.nn.log_softmax(Q_s[0]) * jax.nn.softmax(Q_s[0])).sum(-1).mean()
            entropy_Q += (-jax.nn.log_softmax(Q_s[1]) * jax.nn.softmax(Q_s[1])).sum(-1).mean()
            
            vals_Q = 0.5 * (current_Q1.mean() + current_Q2.mean())
            mse_Q = 0.5 * (jnp.mean((current_Q1 - target_Q)**2) + 
                           jnp.mean((current_Q2 - target_Q)**2))
        
        aux_out = AuxOut(vals_Q, entropy_Q, None)
        return mse_Q, aux_out
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_dual_Q(self, params_dual_Q, replay, outer_grad):
        """Loss for dual Q-network used in funcBO agent."""
        obs, action, reward, next_obs, not_done, not_done_no_max = replay
        val_dual_Q = self.dual_Q.net.apply(params_dual_Q, obs)
        
        if self.args.no_double:
            # Single network
            current_Q = val_dual_Q[jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            target_Q = outer_grad[jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            mse_Q = jnp.mean((current_Q - target_Q)**2)
        else:
            # Double network
            current_Q1 = val_dual_Q[0][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            current_Q2 = val_dual_Q[1][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            
            target_Q1 = outer_grad[0][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            target_Q2 = outer_grad[1][jnp.arange(obs.shape[0]), action.astype(int)[:, 0]]
            
            mse_Q = 0.5 * (jnp.mean((current_Q1 - target_Q1)**2) + 
                           jnp.mean((current_Q2 - target_Q2)**2))
            
        return mse_Q
    
    # ----- BILEVEL OPTIMIZATION METHODS -----
    
    def constraint_func(self, params_T, params_Q, replay, rng, target_params_Q):
        """
        Parameterized by T function giving grad_Q (Bellman-error) = 0 constraint.
        Evaluates inner loss using outer model T and computes gradient wrt inputs.
        """
        # Generate model predictions
        replay_model, _ = self.batch_real_to_model(params_T, replay, rng)
        
        # Compute gradients
        grads, aux_out = jax.grad(self.loss_Q, has_aux=True)(
            params_Q, target_params_Q, replay_model)
            
        return grads
    
    def grad_inner_loss(self, params_T, val_Q, replay, rng, target_val_Q):
        """
        Compute gradient of inner loss with respect to Q-values.
        Used in functional bilevel optimization.
        """
        # Generate model predictions
        replay_model, _ = self.batch_real_to_model(params_T, replay, rng)
        
        # Compute gradients with respect to Q-values
        grads, aux_out = jax.grad(self.func_loss_Q, has_aux=True)(
            val_Q, target_val_Q, replay_model)
            
        return grads
    
    def fwd_solver(self, params_Q, target_params_Q, opt_state_Q, params_T, replay, rng):
        """
        Forward solver for inner optimization.
        Gets Q_* parameters satisfying the constraint (approximately).
        """
        # Generate model predictions
        replay_model, nll = self.batch_real_to_model(params_T, replay, rng)
        
        # Initialize parameters if needed
        if self.args.no_warm:
            target_params_Q = params_Q
        if not self.args.warm_opt:
            opt_state_Q = self.Q.opt.init(params_Q)
        
        # Inner loop to update Q-network
        for i in range(self.args.num_Q_steps):
            updout = self.update_step_inner(
                params_Q, target_params_Q, opt_state_Q, None, replay_model)
            params_Q, opt_state_Q = updout.params_Q, updout.opt_state_Q
            
            # Soft update of target network
            target_params_Q = self._soft_update_params(params_Q, target_params_Q)
        
        # Define solution structure
        Sol = namedtuple('Sol', 
            'params_Q loss_Q vals_Q grad_norm_Q entropy_Q target_params_Q opt_state_Q next_obs_nll')
            
        return Sol(
            params_Q, updout.loss_Q, updout.vals_Q, updout.grad_norm_Q, 
            updout.entropy_Q, target_params_Q, opt_state_Q, nll
        )
    
    @partial(jax.jit, static_argnums=(0,))
    @chex.assert_max_traces(n=1)
    def update_step_inner(self, params, aux_params, opt_state, batch, replay):
        """Single update step for Q-network (inner optimization)."""
        (value, aux_out), grads = value_and_grad(self.loss_Q, has_aux=True)(
            params, aux_params, replay)
            
        # Apply gradients
        updates, opt_state = self.Q.opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Calculate gradient norm
        grad_norm = jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(grads)))
        
        # Define output structure
        UpdOut = namedtuple('Upd_inner', 
            'loss_Q params_Q opt_state_Q grads_Q grad_norm_Q vals_Q entropy_Q')
            
        return UpdOut(
            value, new_params, opt_state, grads, grad_norm, 
            aux_out.vals_Q, aux_out.entropy_Q
        )
    
    def bwd_solver(self, params_dual_Q, opt_state_dual_Q, replay, rng, outer_grad):
        """
        Backward solver for dual optimization in funcBO.
        Updates the dual Q-network to approximate gradients.
        """
        # Initialize optimizer state if needed
        if not self.args.warm_opt:
            opt_state_dual_Q = self.dual_Q.opt.init(params_dual_Q)
        
        # Update dual Q-network
        for i in range(self.args.num_dual_Q_steps):
            updout = self.update_step_dual(
                params_dual_Q, opt_state_dual_Q, replay, outer_grad)
                
            params_dual_Q, opt_state_dual_Q = updout.params_dual_Q, updout.opt_state_dual_Q
        
        # Get dual Q-values
        obs, action, reward, next_obs, not_done, not_done_no_max = replay
        val_dual_Q = self.dual_Q.net.apply(params_dual_Q, obs)
        
        # Define solution structure
        DualSol = namedtuple('DualSol', 
            'params_dual_Q val_dual_Q loss_dual_Q opt_state_dual_Q grad_norm_dual_Q')
            
        return DualSol(
            params_dual_Q, val_dual_Q, updout.loss_dual_Q, 
            opt_state_dual_Q, updout.grad_norm_dual_Q
        )
    
    @partial(jax.jit, static_argnums=(0,))
    @chex.assert_max_traces(n=1)
    def update_step_dual(self, params_dual_Q, opt_state, outer_replay, outer_grad):
        """Single update step for dual Q-network."""
        value, grads = value_and_grad(self.loss_dual_Q, has_aux=False)(
            params_dual_Q, outer_replay, outer_grad)
            
        # Apply gradients
        updates, opt_state = self.dual_Q.opt.update(grads, opt_state)
        new_params = optax.apply_updates(params_dual_Q, updates)
        
        # Calculate gradient norm
        grad_norm = jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(grads)))
        
        # Define output structure
        UpdOut = namedtuple('Upd_dual', 
            'loss_dual_Q params_dual_Q opt_state_dual_Q grads_dual_Q grad_norm_dual_Q')
            
        return UpdOut(value, new_params, opt_state, grads, grad_norm)
    
    # ----- LOSS FUNCTIONS FOR DIFFERENT AGENT TYPES -----
    
    def loss_funcBO(self, params_T, aux_params, batch, replay):
        """Loss function for funcBO agent type."""
        # Initialize parameters if needed
        if self.args.no_warm:
            params_Q = self.Q.net.init(aux_params.rng, replay[0])
            params_dual_Q = self.dual_Q.net.init(aux_params.rng, replay[0])
        else:
            params_Q = aux_params.params_Q
            params_dual_Q = aux_params.params_dual_Q
        
        # Define solvers
        fwd_solver = lambda params_Q, params_T, replay, rng: self.fwd_solver(
            params_Q, aux_params.target_params_Q, aux_params.opt_state_Q, 
            params_T, replay, rng
        )
        
        bwd_solver = lambda params_dual_Q, replay, rng, outer_grad: self.bwd_solver(
            params_dual_Q, aux_params.opt_state_dual_Q, 
            replay, rng, outer_grad
        )
        
        # Use custom VJP for bilevel optimization
        sol = inner_solution(
            self.grad_inner_loss, self.Q.net, params_Q, 
            params_T, replay, aux_params.rng, params_dual_Q, 
            (fwd_solver, bwd_solver)
        )
        
        # Compute loss
        return self.func_loss_Q(sol.val_Q, sol.val_target_Q, replay)[0], sol
    
    def loss_omd(self, params_T, aux_params, batch, replay):
        """Loss function for OMD agent type."""
        # Define solver
        fwd_solver = lambda params_Q, params_T, replay, rng: self.fwd_solver(
            params_Q, aux_params.target_params_Q, aux_params.opt_state_Q, 
            params_T, replay, rng
        )
        
        # Initialize Q parameters if needed
        if self.args.no_warm:
            params_Q = self.Q.net.init(aux_params.rng, replay[0])
        else:
            params_Q = aux_params.params_Q
        
        # Use custom VJP for root solving
        sol = root_solve(
            self.constraint_func, params_Q, 
            params_T, replay, aux_params.rng, (fwd_solver,)
        )
        
        # Compute loss
        return self.loss_Q(sol.params_Q, sol.target_params_Q, replay)[0], sol
    
    def loss_mle(self, params_T, batch, rng):
        """Maximum likelihood loss for world model."""
        obs, action, reward, next_obs, not_done, not_done_no_max = batch
        
        # Get predictions
        pred, means, logstds, reward_pred = self.model_pred(params_T, obs, action, rng)
        
        # Calculate prediction error
        nll = mse_loss(pred, next_obs)
            
        # Add reward prediction error if learning rewards
        if not self.args.no_learn_reward:
            nll += ((reward_pred - reward) ** 2).mean()
            
        return nll
    
    def loss_vep(self, params_T, aux_params, batch, rng):
        """Value-equivalent prediction loss for world model."""
        obs, action, reward, next_obs, not_done, not_done_no_max = batch
        
        # Get predictions
        pred, means, logstds, reward_pred = self.model_pred(params_T, obs, action, rng)
        
        # Calculate standard prediction error (for tracking)
        nll = mse_loss(pred, next_obs)
        
        # Value prediction error (main objective)
        params_V = stop_gradient(aux_params)
        next_V = self.V.net.apply(params_V, next_obs)
        pred_V = self.V.net.apply(params_V, pred)
        
        # Aggregate loss across ensemble
        l = 0
        for i in range(self.args.num_ensemble_vep):
            l += jnp.mean((next_V[i] - pred_V[i])**2)
        
        # Add reward prediction error if learning rewards
        if not self.args.no_learn_reward:
            l += ((reward_pred - reward) ** 2).mean()
            
        aux_out = AuxOut(None, None, nll)
        return l, aux_out
    
    # ----- OUTER UPDATE METHODS -----
    
    @partial(jax.jit, static_argnums=(0,6))
    @chex.assert_max_traces(n=1) 
    def update_step_outer(self, params_T, aux_params, opt_state_T, batch, replay, loss):
        """Outer update step for world model."""
        (value, aux_out), grads = value_and_grad(loss, has_aux=True)(
            params_T, aux_params, batch, replay)
            
        # Apply gradients
        updates, opt_state_T = self.T.opt.update(grads, opt_state_T)
        new_params = optax.apply_updates(params_T, updates)
        
        # Define output structure
        UpdOut = namedtuple('Upd_outer', 
            'loss_T params_T opt_state_T loss_Q vals_Q grad_norm_Q entropy_Q params_Q target_params_Q opt_state_Q next_obs_nll')
            
        return UpdOut(
            value, new_params, opt_state_T, aux_out.loss_Q, 
            aux_out.vals_Q, aux_out.grad_norm_Q, aux_out.entropy_Q, aux_out.params_Q, 
            aux_out.target_params_Q, aux_out.opt_state_Q, aux_out.next_obs_nll
        )
    
    @partial(jax.jit, static_argnums=(0, 6))
    def update_step_outer_averaged(self, params_T, aux_params, opt_state_T, batch, replay, loss):
        """JIT-compatible outer update step with arithmetic gradient averaging."""
        # Compute gradients normally
        (value, aux_out), grads = value_and_grad(loss, has_aux=True)(
            params_T, aux_params, batch, replay)
        return value, aux_out, grads


    @partial(jax.jit, static_argnums=(0,6))
    def update_step(self, params, aux_params, opt_state, batch, replay, loss_type):
        """General update step based on loss type."""
        if loss_type == 'sql':
            # Soft Q-learning update
            (value, aux_out), grads = value_and_grad(self.loss_Q, has_aux=True)(
                params, aux_params, replay)
                
            updates, opt_state = self.Q.opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            
            # Calculate gradient norm
            grad_norm = jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(grads)))
            
            # Define output structure
            UpdOut = namedtuple(f'Upd_{loss_type}', 
                'loss_Q params_Q opt_state_Q grads_Q grad_norm_Q vals_Q entropy_Q')
                
            return UpdOut(
                value, new_params, opt_state, grads, grad_norm, 
                aux_out.vals_Q, aux_out.entropy_Q
            )
            
        elif loss_type == "mle":
            # Maximum likelihood update
            value, grads = value_and_grad(self.loss_mle)(params, replay, aux_params.rng)
            updates, opt_state = self.T.opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            
            # Define output structure
            UpdOut = namedtuple(f'Upd_{loss_type}', 'loss_T params_T opt_state_T')
            return UpdOut(value, new_params, opt_state)
            
        elif loss_type == "vep":
            # Value-equivalent prediction update
            (value, aux_out), grads = value_and_grad(self.loss_vep, has_aux=True)(
                params, aux_params.params_Q, replay, aux_params.rng)
                
            updates, opt_state = self.T.opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            
            # Define output structure
            UpdOut = namedtuple(f'Upd_{loss_type}', 
                'loss_T params_T opt_state_T next_obs_nll')
                
            return UpdOut(value, new_params, opt_state, aux_out.next_obs_nll)
    
    # ----- UTILITY METHODS -----
    
    def _soft_update_params(self, params, target_params, tau=None):
        """Soft update: θ_target = τ*θ + (1-τ)*θ_target."""
        if tau is None:
            tau = self.args.tau
            
        return jax.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            params, target_params
        )
    
    # ----- MAIN UPDATE METHOD -----
    
    def update(self, replay_buffer):
        """Main update method for training."""
        # Sample from replay buffer
        if self.args.time_weighted_sampling:
            replay = replay_buffer.sample_time_weighted(self.args.batch_size)
        else:
            replay = replay_buffer.sample(self.args.batch_size)

        # Different update strategies based on agent type
        if self.args.agent_type in ['omd', 'funcBO']:
            if self.args.agent_type == 'omd':
                # Optimal Model Design
                outer_loss = self.loss_omd
                if self.args.no_warm:
                    aux_params = AuxP(None, None, None, next(self.rngs))
                else:
                    aux_params = AuxP(
                        self.params_Q, self.target_params_Q, self.opt_state_Q, 
                        next(self.rngs)
                    )
            else:
                # Functional Bilevel Optimization
                outer_loss = self.loss_funcBO
                if self.args.no_warm:
                    aux_params = AuxPdual(
                        None, None, None, None, None, next(self.rngs)
                    )
                else:
                    aux_params = AuxPdual(
                        self.params_Q, self.target_params_Q, self.params_dual_Q, 
                        self.opt_state_Q, self.opt_state_dual_Q, next(self.rngs)
                    )
            
            # Batch is None for these methods
            batch = None
            
            # Update world model
            if self.args.average_hypergradients:
                # Update with outer loop optimization and gradient averaging
                value, aux_out, grads = self.update_step_outer_averaged(self.params_T, aux_params, self.opt_state_T, batch, replay, outer_loss)
                # Update the gradient buffer and average gradient using the manager
                grad_buffer, buffer_count, avg_grad = self.grad_buffer_manager.update(grads)
                # Compute final gradient = beta * grads + (1 - beta) * avg_grad
                final_grad = jax.tree_map(
                    lambda g, avg: self.args.beta * g + (1 - self.args.beta) * avg,
                    grads, 
                    avg_grad
                )
                # Apply averaged gradients
                #updates, opt_state_T = self.T.opt.update(avg_grad, self.opt_state_T)
                updates, opt_state_T = self.T.opt.update(final_grad, self.opt_state_T)
                new_params = optax.apply_updates(self.params_T, updates)
                UpdOut = namedtuple('Upd_outer', 'loss_T params_T opt_state_T loss_Q vals_Q grad_norm_Q entropy_Q params_Q target_params_Q opt_state_Q next_obs_nll')
                updout_T = UpdOut(
                                value, new_params, opt_state_T, aux_out.loss_Q, aux_out.vals_Q,
                                aux_out.grad_norm_Q, aux_out.entropy_Q, aux_out.params_Q,
                                aux_out.target_params_Q, aux_out.opt_state_Q, aux_out.next_obs_nll
                                )
            else:
                # Update with outer loop optimization
                updout_T = self.update_step_outer(self.params_T, aux_params, self.opt_state_T, batch, replay, outer_loss)
            
            # Update agent parameters
            self.params_Q, self.opt_state_Q = updout_T.params_Q, updout_T.opt_state_Q
            self.target_params_Q = updout_T.target_params_Q
            self.params_T, self.opt_state_T = updout_T.params_T, updout_T.opt_state_T

            # Return dictionary for 'omd' and 'funcBO'
            return {
                'grad_norm_Q': updout_T.grad_norm_Q.item() if hasattr(updout_T, 'grad_norm_Q') else 0.0,
                'grad_norm_model': 0.0,  # No specific T grad norm
                'loss_T': updout_T.loss_T.item() if hasattr(updout_T, 'loss_T') else 0.0,
                'loss_Q': updout_T.loss_Q.item() if hasattr(updout_T, 'loss_Q') else 0.0
            }
            
        elif self.args.agent_type == 'mle':
            # Maximum likelihood estimation
            # Update world model
            for i in range(self.args.num_T_steps):
                aux_params = AuxP(None, None, None, next(self.rngs))
                if self.args.time_weighted_sampling:
                    replay = replay_buffer.sample_time_weighted(self.args.batch_size)
                else:
                    replay = replay_buffer.sample(self.args.batch_size)
                updout_T = self.update_step(
                    self.params_T, aux_params, self.opt_state_T, 
                    None, replay, 'mle'
                )
                self.params_T, self.opt_state_T = updout_T.params_T, updout_T.opt_state_T
            
            # Update Q-function
            for i in range(self.args.num_Q_steps):
                if self.args.time_weighted_sampling:
                    replay = replay_buffer.sample_time_weighted(self.args.batch_size)
                else:
                    replay = replay_buffer.sample(self.args.batch_size)
                replay_model, nll = self.batch_real_to_model(
                    self.params_T, replay, next(self.rngs)
                )
                updout_Q = self.update_step(
                    self.params_Q, self.target_params_Q, 
                    self.opt_state_Q, None, replay_model, 'sql'
                )
                self.params_Q, self.opt_state_Q = updout_Q.params_Q, updout_Q.opt_state_Q
                self.target_params_Q = self._soft_update_params(self.params_Q, self.target_params_Q)
            
            # Return dictionary for 'mle'
            return {
                'grad_norm_Q': updout_Q.grad_norm_Q.item() if hasattr(updout_Q, 'grad_norm_Q') else 0.0,
                'grad_norm_model': 0.0,  # T gradient norm isn't tracked for these types
                'loss_T': updout_T.loss_T.item() if hasattr(updout_T, 'loss_T') else 0.0,
                'loss_Q': updout_Q.loss_Q.item() if hasattr(updout_Q, 'loss_Q') else 0.0
            }
            
        elif self.args.agent_type == 'vep':
            # Value-equivalent prediction
            # Update world model
            for i in range(self.args.num_T_steps):
                aux_params = AuxP(self.params_V, None, None, next(self.rngs))
                if self.args.time_weighted_sampling:
                    replay = replay_buffer.sample_time_weighted(self.args.batch_size)
                else:
                    replay = replay_buffer.sample(self.args.batch_size)
                updout_T = self.update_step(
                    self.params_T, aux_params, self.opt_state_T, 
                    None, replay, 'vep'
                )
                self.params_T, self.opt_state_T = updout_T.params_T, updout_T.opt_state_T
            
            # Update Q-function
            for i in range(self.args.num_Q_steps):
                if self.args.time_weighted_sampling:
                    replay = replay_buffer.sample_time_weighted(self.args.batch_size)
                else:
                    replay = replay_buffer.sample(self.args.batch_size)
                replay_model, nll = self.batch_real_to_model(
                    self.params_T, replay, next(self.rngs)
                )
                updout_Q = self.update_step(
                    self.params_Q, self.target_params_Q, 
                    self.opt_state_Q, None, replay_model, 'sql'
                )
                self.params_Q, self.opt_state_Q = updout_Q.params_Q, updout_Q.opt_state_Q
                self.target_params_Q = self._soft_update_params(self.params_Q, self.target_params_Q)
        
        return {
        'grad_norm_Q': 0.0,
        'grad_norm_model': 0.0,
        'loss_T': 0.0,
        'loss_Q': 0.0
        }