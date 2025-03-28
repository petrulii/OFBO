import torch
import numpy as np
import time
import wandb
from funcBO.utils import assign_device, get_dtype, tensor_to_state_dict, state_dict_to_tensor
from funcBO.InnerSolution import InnerSolution
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils import *
import copy
import sys


class Trainer:
    """
    Solves a meta-learning problem using the bilevel functional method.
    """
    def __init__(self, config, logger):
        """
        Initializes the Trainer class with the provided configuration and logger.

        Parameters:
        - config (object): Configuration object containing various settings.
        - logger (object): Logger object for logging metrics and information.
        """
        self.logger = logger
        self.args = config
        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)
        torch.set_default_dtype(self.dtype)
        
        # Set random seed for reproducibility
        torch.manual_seed(self.args.system.seed)
        np.random.seed(self.args.system.seed)
        
        self.build_trainer()
        self.iters = 0

    def log(self, dico, log_name='metrics'):
        self.logger.log_metrics(dico, log_name=log_name)

    def log_metrics_list(self, dico_list, iteration, prefix="", log_name='metrics'):
        total_iter = len(dico_list)
        for dico in dico_list:
            dico['outer_iter'] = iteration
            dico['iter'] = iteration*total_iter + dico['inner_iter']
            dico = {prefix+key:value for key,value in dico.items()}
            self.log(dico, log_name=log_name)

    def build_trainer(self):
        """
        Builds the trainer by setting up data, models, and optimization components 
        for meta-learning using funcBO.
        """
        device = self.device

        # Create a network for FC100
        layers = [self.args.network_params.input_dim] + [self.args.network_params.hidden_dim] * self.args.network_params.hidden_layers + [self.args.network_params.output_dim]
        self.outer_model = nn.Sequential(nn.Flatten(),*[nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.outer_model.to(device)
        
        # Create tensor parameters for funcBO
        self.outer_param = torch.nn.parameter.Parameter(
            state_dict_to_tensor(self.outer_model, device))

        # Create inner model (task-specific model that adapts quickly)
        self.inner_model = copy.deepcopy(self.outer_model)
        self.inner_model.to(device)
        
        # Setup loss function (typically cross-entropy for few-shot classification)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Setup data
        self.inner_dataloader, self.outer_dataloader, self.test_dataloader = get_task_loaders(
            shots=self.args.shots,
            ways=self.args.ways,
            device=device,
            seed=self.args.system.seed
        )

        # Setup optimizer for outer problem
        self.outer_optimizer = torch.optim.Adam([self.outer_param], lr=self.args.outer_lr, weight_decay=self.args.outer_wd)

        # Define inner objective function (task adaptation)
        def fi(outer_model_outputs, inner_prediction, target):
            """
            Inner loss: predict well on a specific task + bias towards outer model outputs (parametric -> outer model parameters)
            """
            # Convert to long type (3.0 -> 3)
            target = target.long()
            #sys.stderr.write(f"fi - inner_prediction shape: {inner_prediction.shape}, outer_model_outputs shape: {outer_model_outputs.shape}, target shape: {target.shape}\n")
            # Compute the regularization term first
            lambda_reg = 0.01  # Hyperparameter to tune
            reg_term = lambda_reg * F.mse_loss(inner_prediction, outer_model_outputs, reduction='mean')
            # Compute the loss
            loss = self.criterion(inner_prediction, target) + reg_term
            return loss
        
        # Define outer objective function (meta-learning)
        def fo(inner_prediction, target):
            """
            Outer loss: predict well on all tasks
            """
            loss = self.criterion(inner_prediction, target)
            return loss        
        
        # Store functions
        self.inner_loss = fi
        self.outer_loss = fo
        
        # Should return inner_model_inputs, outer_model_inputs, inner_loss_inputs (e.g. z,x,None)
        def projector(task_data):
            """
            Extracts and returns the input of inner model.
            """
            # Unpack the task data
            input_data, target = task_data
            #sys.stderr.write(f"Projector - input_data shape: {input_data.shape}, target shape: {target.shape}\n")
            return input_data, input_data, target

        # Setup InnerSolution using config directly
        self.inner_solution = InnerSolution(
            inner_model=self.inner_model,
            inner_loss=self.inner_loss,
            inner_dataloader=self.inner_dataloader,
            inner_data_projector=projector,
            outer_model=self.outer_model,
            outer_param=self.outer_param,
            inner_solver_args=self.args.inner_solver,
            dual_model_args=self.args.dual_model,
            dual_solver_args=self.args.dual_solver
        )                                  

    def train(self):
        """
        The main optimization loop for the bilevel functional method for meta-learning.
        """
        # Initialize wandb
        wandb.init(project="online_meta_learning", name=self.args.agent_type)
        done = False
        while not done:
            for outer_data, outer_labels in self.outer_dataloader:
                # Move data to device
                outer_data, outer_labels = outer_data.to(self.device), outer_labels.to(self.device)
                #sys.stderr.write(f"outer_data shape: {outer_data.shape}, outer_labels shape: {outer_labels.shape}\n")
                
                # Log metrics
                metrics_dict = {}
                metrics_dict['iter'] = self.iters
                start_time = time.time()
                
                inner_value = self.inner_solution(outer_data)
                loss = self.outer_loss(inner_value, outer_labels)
                
                # Backpropagation
                self.outer_optimizer.zero_grad()
                loss.backward()
                self.outer_optimizer.step()

                # Log outer optimization details
                metrics_dict['outer_loss'] = loss.item()
                wandb.log(metrics_dict)
                
                # Log inner optimization details if available
                inner_logs = self.inner_solution.inner_solver.data_logs if hasattr(self.inner_solution.inner_solver, 'data_logs') else None
                if inner_logs:
                    # Log only the final inner iteration loss
                    final_log = inner_logs[-1] if inner_logs else {}
                    wandb.log({'inner_loss': final_log.get('loss', 0), 'iter': self.iters})

                # Log dual optimization details if available
                dual_logs = self.inner_solution.dual_solver.data_logs if hasattr(self.inner_solution.dual_solver, 'data_logs') else None
                if dual_logs:
                    final_log = dual_logs[-1] if dual_logs else {}
                    wandb.log({'dual_loss': final_log.get('loss', 0), 'iter': self.iters})
                
                # Increment iteration counter
                self.iters += 1
                
                # Check if we've reached the maximum number of epochs
                done = (self.iters >= self.args.max_epochs)
                """if self.iters % 100 == 0 or done:
                    # Evaluate on validation set occasionally
                    val_loss, val_acc = self.evaluate(data_type="validation")
                    val_log = [{'inner_iter': 0, 'val_loss': val_loss.item(), 'val_accuracy': val_acc.item()}]
                    self.log_metrics_list(val_log, self.iters, log_name='val_metrics')
                    wandb.log({'val_accuracy': val_acc.item()})"""
                
                if done:
                    break

    def evaluate(self, data_type="validation"):
        """
        Evaluates the meta-model on the given data type.
        """
        tasks = self.val_tasks if data_type == "validation" else self.test_tasks
        
        # Set models to evaluation mode
        self.outer_model.eval()
        self.inner_model.eval()
        
        # TODO: Implement evaluation code