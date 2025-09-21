# Non-Stationary Regression (Synthetic data)

Includes:

- `trainer.py`: main training loop for multiple algorithms (`funcBO`, `unroll1`, `AID`).
- `utils.py`: custom JAX utilities (functional implicit differentiation, conjugate gradient, Hessian–vector products).
- `main.py`: experiment entrypoint with default configs.

## Usage

After installing ```uv``` from Astral and finding your wandb key on your wandb account do:
```bash
export WANDB_API_KEY=your_wandb_key
uv run --isolated --no-project --with 'jax[cpu]==0.4.33' --with numpy --with flax --with wandb-core --with wandb --with dm-haiku --with optax --with chex --with absl-py --with mlxp --with matplotlib --no-cache --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -- python main.py
```
you can change to a gpu configuration if you want by doing ```jax[gpu]```. This might or might not work depending on your hardware.


# Onilne Bilevel Model-Based Reinforcement Learning

This repository implements several online bilevel optimization algorithms for model-based reinforcement learning. The codebase focuses on learning world models that are optimized for downstream decision-making tasks rather than pure prediction accuracy.

## Overview

Traditional model-based reinforcement learning approaches typically learn world models by minimizing prediction error (e.g., using maximum likelihood estimation). This project explores bilevel optimization techniques where the world model (outer level) is optimized based on the performance of a policy or value function (inner level) that uses the model.

The implemented methods include:

- **OMD (Optimal Model Design)**: Optimizes the world model using implicit differentiation through the solution of the inner optimization problem.
- **funcBO (Functional Bilevel Optimization)**: Uses Functional Bilevel Optimization where gradients are computed with respect to Q-values directly.
- **MLE (Maximum Likelihood Estimation)**: Traditional approach that optimizes the world model to minimize prediction error.
- **VEP (Value-Equivalent Prediction)**: Optimizes the world model to match the value function outputs rather than state predictions.

## Code Structure

- `trainer.py`: Main training loop that handles environment interactions and agent updates
- `agent.py`: Implementation of different agent types and their learning algorithms
- `utils.py`: Custom VJP implementation for bilevel optimization and other utility functions

## Installation

```bash
# Clone the repository
git clone https://github.com/username/OFBO.git
cd OFBO

# Create the virtual environment
mamba env create -f environment.yml
```

### Requirements

- JAX and Haiku for neural network implementation
- Gym for environments
- Optax for optimization
- NumPy for data handling
- Wandb for experiment tracking (optional)

## Usage

### Quick Start

Train an agent using the Maximum Likelihood Estimation (MLE) approach:

```bash
python main.py --agent_type=mle --env_name=CartPole-v1 --seed=0
```

### Command Line Arguments

```
--agent_type: Type of agent to use ('omd', 'mle', 'vep', 'funcBO')
--env_name: Gym environment name
--seed: Random seed for reproducibility
--num_train_steps: Number of environment steps to train
--hidden_dim: Size of hidden layers for networks
--model_hidden_dim: Size of hidden layers for world model
--batch_size: Mini-batch size for updates
--init_steps: Steps to collect before training starts
--eval_frequency: Steps between agent evaluations
--num_Q_steps: Number of inner loop steps for Q-function
--num_T_steps: Number of steps for world model updates
--eps: Exploration epsilon for action selection
--discount: Reward discount factor
--lr: Learning rate for outer optimization
--inner_lr: Learning rate for inner optimization
--alpha: Temperature parameter for entropy regularization
--tau: Target network update coefficient
--no_warm: Don't use previous Q* in the inner loop
--warm_opt: Reuse inner loop optimizer statistics
--no_double: Don't use Double Q-Learning
```

### Training Example

```bash
# Train OMD agent
python main.py --agent_type=omd --env_name=CartPole-v1 --num_train_steps=50000 --seed=42

# Train funcBO agent
python main.py --agent_type=funcBO --env_name=CartPole-v1 --num_train_steps=50000 --seed=42 

# Train MLE agent
python main.py --agent_type=mle --env_name=CartPole-v1 --num_train_steps=50000 --seed=42
```

## Implementation Details

### Bilevel Optimization

The core of this project is bilevel optimization, where we have:

1. **Inner problem**: Find optimal Q-function parameters given a world model
2. **Outer problem**: Find optimal world model parameters that lead to good Q-functions

Two main approaches are implemented:

- **Root-solving approach** (OMD): Treats the gradient of the inner objective as a constraint that should be zero at optimality
- **Functional approach** (funcBO): Computes gradients with respect to Q-values directly using Functional Implicit Differentiation (Petrulionyte et al.)

### Custom VJP Rules

The repository includes custom vector-Jacobian product (VJP) implementations for differentiating through the inner optimization. This allows for end-to-end training where gradients flow from the final performance metric through both the inner and outer optimization.