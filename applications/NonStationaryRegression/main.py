# main.py
import os
import sys
import argparse
import pprint

def _print_config_to_stderr(cfg, where=""):
    """Pretty-print the config to stderr, handling dicts and namespaces."""
    try:
        if isinstance(cfg, dict):
            text = pprint.pformat(cfg, sort_dicts=False)
        elif hasattr(cfg, "__dict__"):
            text = pprint.pformat(vars(cfg), sort_dicts=False)
        else:
            text = str(cfg)
    except Exception as e:
        text = f"<failed to pretty-print config: {e}; raw={repr(cfg)}>"
    print(f"\n=== CONFIG DUMP {where} ===\n{text}\n=== END CONFIG DUMP ===\n",
          file=sys.stderr, flush=True)

def create_default_config():
    """Create default configuration for bilevel regression."""
    config = {
        # Problem parameters
        'input_dim': 5,
        'hidden_dim': 64,
        'output_dim': 3,
        'n_samples': 256,
        
        # Time series parameters
        'T_max': 500,
        'shift_type': 'sinusoidal',   # 'linear' or 'sinusoidal'
        'alpha': 0.0,                 # (kept, unused in sinusoidal)
        'beta': 10,                   # amplitude of sinusoid
        'omega': 2e-1,                # frequency (larger -> faster changes)

        # Noise parameters
        'noise_level': 10.0,             # base noise level

        # Windowed inner loss
        'window_size': 10,            # past timesteps to include in inner loss
        
        # Bilevel optimization parameters
        'agent_type': 'funcBO_noSmooth',       # 'funcBO', 'funcBO_noSmooth', 'aid', 'unroll1'
        'lambda_reg': 0.0,            # initial value for regularization (will broadcast)
        'outer_lr': 1e-2,
        'inner_lr': 1e-3,
        'num_inner_steps': 4,
        'num_outer_steps': 1,
        
        # Hypergradient smoothing parameters
        'average_hypergradients': True,
        'grad_buffer_size': 50,
        
        # Training parameters
        'batch_size': 32,
        'eval_frequency': 50,
        'log_frequency': 10,
        'seed': 42,
        
        # Network parameters
        'warm_start': True,
        'tau': 0.005,  # soft update parameter

        # AID options
        'aid_cg_iters': 10,
        'aid_cg_tol': 1e-6,
        'aid_cg_reg': 1e-4,            # Tikhonov damping in the Hessian

        # Regret tracking
        'track_regret': True,
        'regret_grid_size': 7,
        'regret_lambda_max': 5.0,
        'regret_frequency': 10,        # compute regret every N steps (costly otherwise)

    }
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Non-stationary Bilevel Optimization')

    config = create_default_config()
    _print_config_to_stderr(config, where="(__main__) before Trainer init")
    
    # Convert to namespace for compatibility
    from types import SimpleNamespace
    config = SimpleNamespace(**config)
    
    from trainer import BilevelRegressionTrainer
    trainer = BilevelRegressionTrainer(config, logger=None)
    final_loss = trainer.train()
    print(f"Training completed. Final loss: {final_loss}")