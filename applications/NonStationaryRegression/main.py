# main.py
import mlxp
import os
import argparse
import sys
import pprint

def _print_config_to_stderr(cfg, where=""):
    """Pretty-print the config to stderr, handling dicts and namespaces."""
    try:
        if isinstance(cfg, dict):
            text = pprint.pformat(cfg, sort_dicts=False)
        elif hasattr(cfg, "__dict__"):
            text = pprint.pformat(vars(cfg), sort_dicts=False)
        else:
            # Fallback (e.g., MLXP objects without __dict__)
            text = str(cfg)
    except Exception as e:
        text = f"<failed to pretty-print config: {e}; raw={repr(cfg)}>"
    print(f"\n=== CONFIG DUMP {where} ===\n{text}\n=== END CONFIG DUMP ===\n",
          file=sys.stderr, flush=True)

def clear_dir(log_dir):
    """Clear JSON files from log directory."""
    if not os.path.exists(log_dir):
        return
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        try:
            if os.path.isfile(file_path) and file_path.endswith('json'):
                os.remove(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

@mlxp.launch(config_path='./configs')
def train(ctx: mlxp.Context) -> None:
    """Main training function for non-stationary bilevel optimization."""
    print("Starting non-stationary bilevel optimization training")
    
    # Clear existing logs
    log_dir = ctx.logger.log_dir 
    clear_dir(os.path.join(log_dir, 'metrics'))

    from trainer import BilevelRegressionTrainer
    trainer = BilevelRegressionTrainer(ctx.config, ctx.logger)

    # Run training
    final_loss = trainer.train()
    print(f"Training completed. Final loss: {final_loss}")
    
    return final_loss

def create_default_config():
    """Create default configuration for bilevel regression."""
    config = {
        # Problem parameters
        'input_dim': 5,
        'hidden_dim': 64,
        'output_dim': 3,
        'n_samples': 256,
        
        # Time series parameters
        'T_max': 5000,
        'shift_type': 'sinusoidal',   # 'linear' or 'sinusoidal'
        'alpha': 0.0,                 # (kept, unused in sinusoidal)
        'beta': 10,                   # amplitude of sinusoid
        'omega': 2e-2,                # frequency (larger -> faster changes)

        # Noise parameters
        'noise_level': 0.0,           # base noise level

        # Windowed inner loss
        'window_size': 20,            # past timesteps to include in inner loss
        
        # Bilevel optimization parameters
        'agent_type': 'funcBO',       # 'funcBO', 'omd', 'mle'
        'lambda_reg': 0.0,#10,        # initial value for regularization (will broadcast)
        'outer_lr': 1e-3,#3
        'inner_lr': 1e-4,#4
        'num_inner_steps': 16,
        'num_outer_steps': 1,
        
        # Gradient averaging parameters
        'average_hypergradients': True,
        'grad_buffer_size': 10,
        
        # Training parameters
        'batch_size': 128,
        'eval_frequency': 100,
        'log_frequency': 10,
        'seed': 42,
        
        # Network parameters
        'warm_start': True,
        'tau': 0.005,  # soft update parameter
        'fd_eps': 1e-4  # finite difference epsilon (for funcBO)
    }
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Non-stationary Bilevel Optimization')
    #parser.add_argument('--config', type=str, default=None, 
    #                   help='Path to config file (if not using MLXP)')
    #parser.add_argument('--agent_type', type=str, default='funcBO',
    #                   choices=['funcBO', 'omd', 'mle'],
    #                   help='Type of bilevel optimization method')
    #parser.add_argument('--shift_type', type=str, default='linear',
    #                   choices=['linear', 'sinusoidal'],
    #                   help='Type of target shift')
    #parser.add_argument('--T_max', type=int, default=100000,
    #                   help='Maximum number of time steps')
    
    #args = parser.parse_args()
    
    #if args.config is None:
    # Run without MLXP for simple testing
    config = create_default_config()
    _print_config_to_stderr(config, where="(__main__) before Trainer init")
    #config.update(vars(args))
    
    # Convert to namespace for compatibility
    from types import SimpleNamespace
    config = SimpleNamespace(**config)
    
    from trainer import BilevelRegressionTrainer
    trainer = BilevelRegressionTrainer(config, logger=None)
    final_loss = trainer.train()
    print(f"Training completed. Final loss: {final_loss}")
    #else:
        # Use MLXP
    #    train()