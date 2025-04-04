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
from utils import root_solve, inner_solution

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
        self.action_dim = action_space.shape[0] 
        self.action_space_type = 'continuous'
        self.obs_range = (obs_space.low, obs_space.high)
        if self.args.average_hypergradients:
            self.grad_buffer = []
            self.grad_buffer_size = self.args.grad_buffer_size
            self.grad_avg_weight = self.args.grad_avg_weight
        
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
        final_init = hk.initializers.Orthogonal(scale=1e-2)
        
        # Special case for VEP ensemble
        if self.args.agent_type == 'vep' and net_type == 'V':
            ensemble = []
            for i in range(self.args.num_ensemble_vep):
                # Use MLP with proper output dimensions
                mlp = hk.nets.MLP(
                    output_sizes=[hidden_dim, hidden_dim, 1],
                    w_init=init,
                    b_init=jnp.zeros,
                    activation=activation,
                    activate_final=False
                )
                ensemble.append(mlp(x))
            return ensemble
        
        # Define output size based on network type
        if net_type == 'V':
            output_size = 1
        elif net_type == 'Q' or net_type == 'dual_Q':
            output_size = action_dim
        elif net_type == 'T':
            output_size = obs_dim
        
        # Special cases for double networks
        if (net_type == 'Q' or net_type == 'dual_Q') and not self.args.no_double:
            # Double Q-learning with two networks - create two separate MLPs
            mlp1 = hk.nets.MLP(
                output_sizes=[hidden_dim, hidden_dim, output_size],
                w_init=init,
                b_init=jnp.zeros,
                activation=activation,
                activate_final=False,
                name="mlp1"
            )
            
            mlp2 = hk.nets.MLP(
                output_sizes=[hidden_dim, hidden_dim, output_size],
                w_init=init,
                b_init=jnp.zeros,
                activation=activation,
                activate_final=False,
                name="mlp2"
            )
            
            return mlp1(x), mlp2(x)
            
        elif net_type == 'T' and not self.args.no_learn_reward:
            # Separate reward prediction for world model
            mlp1 = hk.nets.MLP(
                output_sizes=[hidden_dim, hidden_dim, obs_dim],
                w_init=init,
                b_init=jnp.zeros,
                activation=activation,
                activate_final=False,
                name="state_predictor"
            )
            
            mlp2 = hk.nets.MLP(
                output_sizes=[hidden_dim, hidden_dim, 1],
                w_init=init,
                b_init=jnp.zeros,
                activation=activation,
                activate_final=False,
                name="reward_predictor"
            )
            
            return mlp1(x), mlp2(x)
            
        else:
            # Single network
            mlp = hk.nets.MLP(
                output_sizes=[hidden_dim, hidden_dim, output_size],
                w_init=init,
                b_init=jnp.zeros,
                activation=activation,
                activate_final=False
            )
            
            return mlp(x)
    
    # ----- AGENT ACTION SELECTION -----
    
    @partial(jax.jit, static_argnums=(0,))
    def act(self, params_Q, obs, rng):
        """Select action based on current Q-function for continuous action space."""
        obs = jnp.array(obs[0]) if isinstance(obs, tuple) else obs[None, ...]
        
        # Get action from policy network
        action_output = self.Q.net.apply(params_Q, obs[None, ...])
        
        # Handle case where action_output is a tuple (from double networks)
        if isinstance(action_output, tuple):
            # Use the first network's output for actions
            action = action_output[0]
        else:
            action = action_output
        
        # Add exploration noise if needed
        if not self.args.hard:
            action = action + self.args.alpha * jax.random.normal(rng, action.shape)
            
        # Clip to action space bounds
        action = jnp.clip(action, -1.0, 1.0)  # Normalized action space
            
        return action if isinstance(obs, list) else action[0]
        
    # ----- WORLD MODEL METHODS -----

    @partial(jax.jit, static_argnums=(0,))
    def model_pred(self, params_T, obs, action, rng):
        """Predict next state and reward using world model."""
        
        # Handle continuous vs discrete actions differently
        if self.action_space_type == 'continuous':
            # For continuous actions, concatenate directly
            x = jnp.concatenate((obs, action), axis=-1)
        else:
            # For discrete actions, convert to one-hot
            if not isinstance(action, int):
                action = action[:, 0]
            a = jax.nn.one_hot(action, self.action_dim)
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
        """Functional version of Q-learning loss for continuous action spaces."""
        obs, action, reward, next_obs, not_done, not_done_no_max = batch
        
        if self.action_space_type == 'continuous':
            # For continuous actions, Q network outputs actions directly
            if self.args.no_double:
                # Single network case
                # Get actions from Q networks
                next_actions = target_Q  # These are the actions from target network
                current_actions = Q_s    # These are the actions from current network
                
                # For simplicity, using MSE between actions as a proxy for Q-learning
                # In more sophisticated implementations, you might use a proper critic
                mse_Q = jnp.mean((current_actions - action)**2)
                
                # For logging
                vals_Q = jnp.mean(jnp.sum(current_actions**2, axis=-1))
                entropy_Q = -jnp.mean(jnp.sum(current_actions**2, axis=-1))
            else:
                # Double network case
                next_actions1, next_actions2 = target_Q  # Actions from target networks
                current_actions1, current_actions2 = Q_s  # Actions from current networks
                
                # Use both networks' outputs
                mse_Q = 0.5 * (jnp.mean((current_actions1 - action)**2) + 
                            jnp.mean((current_actions2 - action)**2))
                
                # For logging
                vals_Q = 0.5 * (jnp.mean(jnp.sum(current_actions1**2, axis=-1)) + 
                            jnp.mean(jnp.sum(current_actions2**2, axis=-1)))
                entropy_Q = -vals_Q
        else:
            # Print error if discrete actions are used
            raise NotImplementedError("Discrete actions not supported for Q-learning.")
        
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
    
    #@partial(jax.jit, static_argnums=(0,6))
    def update_step_outer_averaged(self, params_T, aux_params, opt_state_T, batch, replay, loss):
        """Outer update step for world model with gradient averaging."""
        # Compute gradients normally
        (value, aux_out), grads = value_and_grad(loss, has_aux=True)(
            params_T, aux_params, batch, replay)
        
        # Store current gradient in buffer (using host callback to avoid JIT issues)
        def store_grad(grads):
            # Convert to numpy to store
            grad_np = jax.tree_map(lambda x: jnp.array(x), grads)
            # Remove oldest if buffer is full
            if len(self.grad_buffer) >= self.grad_buffer_size:
                self.grad_buffer.pop(0)
            # Add current gradient
            self.grad_buffer.append(grad_np)
            return 0  # Return value not used
        
        # Use host callback to modify the buffer (breaks JIT but unavoidable)
        store_grad(grads)
        
        # Compute weighted average gradient
        if len(self.grad_buffer) > 1:
            past_weight = self.grad_avg_weight / len(self.grad_buffer)
            current_weight = 1.0 - self.grad_avg_weight
            
            # Start with weighted current gradient
            avg_grads = jax.tree_map(lambda x: current_weight * x, grads)
            
            # Add weighted past gradients
            for past_grad in self.grad_buffer:
                past_grad_jax = jax.tree_map(lambda x: jnp.array(x), past_grad)
                avg_grads = jax.tree_map(
                    lambda avg, past: avg + past_weight * past, 
                    avg_grads, past_grad_jax
                )
        else:
            avg_grads = grads
        
        # Apply averaged gradients
        updates, opt_state_T = self.T.opt.update(avg_grads, opt_state_T)
        new_params = optax.apply_updates(params_T, updates)
        
        # Define output structure
        UpdOut = namedtuple('Upd_outer', 
            'loss_T params_T opt_state_T loss_Q vals_Q grad_norm_Q entropy_Q params_Q target_params_Q opt_state_Q next_obs_nll')
            
        return UpdOut(
            value, new_params, opt_state_T, aux_out.loss_Q, 
            aux_out.vals_Q, aux_out.grad_norm_Q, aux_out.entropy_Q, aux_out.params_Q, 
            aux_out.target_params_Q, aux_out.opt_state_Q, aux_out.next_obs_nll
        )

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
                updout_T = self.update_step_outer_averaged(self.params_T, aux_params, self.opt_state_T, batch, replay, outer_loss)
            else:
                # Update with outer loop optimization
                updout_T = self.update_step_outer(self.params_T, aux_params, self.opt_state_T, batch, replay, outer_loss)
            
            # Update agent parameters
            self.params_Q, self.opt_state_Q = updout_T.params_Q, updout_T.opt_state_Q
            self.target_params_Q = updout_T.target_params_Q
            self.params_T, self.opt_state_T = updout_T.params_T, updout_T.opt_state_T

            # For logging
            #updout_Q = updout_T
            
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

        # Return metrics
        #return {
        #    'loss_T': updout_T.loss_T.item(), 
        #    'vals_Q': updout_Q.vals_Q.item(), 
        #    'loss_Q': updout_Q.loss_Q.item(), 
        #    'grad_norm_Q': updout_Q.grad_norm_Q.item(), 
        #    'entropy_Q': updout_Q.entropy_Q.item()
        #}