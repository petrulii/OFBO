# main.py
import mlxp
import os
import argparse

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
        'T_max': 100000,
        'shift_type': 'linear',  # 'linear' or 'sinusoidal'
        'alpha': 0,  # linear shift rate

        # Noise parameters
        'noise_level': 0,  # base noise level
        
        # Bilevel optimization parameters
        'agent_type': 'funcBO',  # 'funcBO', 'omd', 'mle'
        'lambda_reg': 0.01,  # regularization parameter
        'outer_lr': 1e-4,
        'inner_lr': 1e-5,
        'num_inner_steps': 200,
        'num_outer_steps': 1,
        
        # Gradient averaging parameters
        'average_hypergradients': True,
        'grad_buffer_size': 10,
        'beta': 0.5,  # mixing parameter for gradient averaging
        
        # Training parameters
        'batch_size': 16,
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