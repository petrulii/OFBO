import torch
import numpy as np
import time
from funcBO.utils import assign_device, get_dtype, tensor_to_state_dict, state_dict_to_tensor
from funcBO.InnerSolution import InnerSolution
from torch.utils.data import DataLoader, Dataset
from models.models import Task
from hypergrad.diff_optimizers import GradientDescent
from utils import split_data, accuracy, fast_adapt

class MetaTaskDataset(Dataset):
    """Dataset wrapper for meta-learning tasks"""
    def __init__(self, tasks):
        self.tasks = tasks
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]

class MetaTrainer:
    """
    Solves a meta-learning problem using the bilevel functional method.
    """
    def __init__(self, config, logger):
        """
        Initializes the MetaTrainer class with the provided configuration and logger.

        Parameters:
        - config (object): Configuration object containing various settings.
        - logger (object): Logger object for logging metrics and information.
        """
        self.logger = logger
        self.args = config
        self.device = assign_device(self.args.system.device)
        self.dtype = get_dtype(self.args.system.dtype)
        torch.set_default_dtype(self.dtype)
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

        # Setup meta-learning models based on the method
        if self.args.method == 'funcBO':
            # In funcBO, meta_model is the head, features are trained
            from models.models import MetaHead, FeatureExtractor  # Import appropriate models
            self.meta_model = MetaHead(self.args.head_params).to(device)
            self.features = FeatureExtractor(self.args.feature_params).to(device)
        elif self.args.method == 'ANIL':
            # TODO
        elif self.args.method == 'MAML':
            # TODO
        else:
            raise ValueError(f"Unsupported method: {self.args.method}")

        # Create inner model (task-specific model that adapts quickly)
        self.inner_model = self.meta_model.clone()
        self.inner_model.to(device)
        
        # Setup loss function (typically cross-entropy for few-shot classification)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Create tensor parameters for funcBO
        self.outer_param = torch.nn.parameter.Parameter(
            state_dict_to_tensor(self.meta_model, device))
        
        # Setup data
        # TODO: This would needs to be adjusted
        from data_utils import get_task_loaders  # Replace with the actual task loading code
        self.train_tasks, self.val_tasks, self.test_tasks = get_task_loaders(
            shots=self.args.shots,
            ways=self.args.ways,
            device=device,
            seed=self.args.seed
        )
        
        # Create dataloaders
        self.inner_dataloader = DataLoader(
            dataset=MetaTaskDataset(self.train_tasks), 
            batch_size=self.args.batch_size, 
            shuffle=True)
        self.outer_dataloader = DataLoader(
            dataset=MetaTaskDataset(self.train_tasks), 
            batch_size=self.args.batch_size, 
            shuffle=True)
        
        # Optimizer for outer problem
        self.outer_optimizer = torch.optim.Adam(
            [self.outer_param], 
            lr=self.args.outer_lr, 
            weight_decay=self.args.outer_wd)
        
        # Define inner objective function (adaptation)
        # !that depends only on the inner prediction!
        def fi(inner_prediction, target):
            """
            Inner loss: task adaptation loss that depends only on inner model prediction
            
            Args:
                inner_prediction: Output from inner model
                target: Ground truth labels
            """
            # Calculate loss based only on predictions
            loss = self.criterion(inner_prediction, target)
            
            return loss
        
        # Define outer objective function (meta-learning)
        def fo(adapted_model, task_data):
            """
            Outer loss: meta-learning loss on evaluation data after adaptation
            
            Args:
                adapted_model: Model after inner adaptation
                task_data: Data for the task
            
            Returns:
                loss: Loss on evaluation data
            """
            data, labels = task_data
            
            # Split data into adaptation and evaluation sets
            _, _, evaluation_data, evaluation_labels = split_data(
                data, labels, self.args.shots, self.args.ways, self.device)
            
            if self.args.method == 'funcBO' or self.args.method == 'ANIL':
                evaluation_data = self.features(evaluation_data)
            
            # Compute predictions and loss
            predictions = adapted_model(evaluation_data)
            loss = self.criterion(predictions, evaluation_labels)
            
            return loss        
        
        # Store functions
        self.inner_loss = fi
        self.outer_loss = fo
        self.task_function = task_function
        
        # Configure inner solver based on your meta-learning approach
        if self.args.method == 'funcBO' or self.args.method == 'ANIL' or self.args.method == 'MAML':
            inner_solver_args = {
                'name': 'funcBO.solvers.GradientSolver',
                'max_iter': self.args.inner_steps,
                'tol': 1e-6,
                'lr': self.args.inner_lr,
                'momentum': 0.9,
                'log_freq': 10
            }
        else:
            inner_solver_args = {
                'name': 'funcBO.solvers.CustomSolver',  # Replace with appropriate solver
                'max_iter': self.args.inner_steps,
                'tol': 1e-6,
                'lr': self.args.inner_lr
            }
        
        # Configure dual solver for computing hypergradients
        dual_solver_args = {
            'name': 'funcBO.solvers.ConjugateGradientSolver',
            'max_iter': 5,
            'tol': 1e-6,
            'log_freq': 10
        }
        
        # TODO: Define projector based on your method
        # !needed when the input is some fancy tuple rather than just an image!
        def projector(data):
            """
            Extracts and returns the relevant components from the input data.

            Parameters:
            - data (tuple): A tuple containing input information.

            Returns:
            - tuple: A tuple containing the relevant components for the inner prediction function.
            """
            image = data
            return image

        # Setup InnerSolution
        self.inner_solution = InnerSolution(
            self.inner_model,
            self.inner_loss,
            self.inner_dataloader,
            projector,
            self.meta_model,
            self.outer_param,
            dual_model_args=None,
            inner_solver_args=inner_solver_args,
            dual_solver_args=dual_solver_args
        )

    def train(self):
        """
        The main optimization loop for the bilevel functional method for meta-learning.
        """
        done = False
        while not done:
            for task_data in self.outer_dataloader:
                metrics_dict = {}
                metrics_dict['iter'] = self.iters
                start_time = time.time()
                
                # Move data to device if needed
                data, labels = task_data
                data, labels = data.to(self.device), labels.to(self.device)
                task_data = (data, labels)
                
                # Get inner solution (adapted model)
                inner_output = self.inner_solution(task_data)
                
                # Compute outer loss
                loss = self.outer_loss(inner_output, task_data)
                
                # Backpropagation
                self.outer_optimizer.zero_grad()
                loss.backward()
                self.outer_optimizer.step()
                
                # Compute accuracy for monitoring
                _, _, evaluation_data, evaluation_labels = split_data(
                    data, labels, self.args.shots, self.args.ways, self.device)
                
                if self.args.method == 'funcBO' or self.args.method == 'ANIL':
                    evaluation_data = self.features(evaluation_data)
                
                predictions = inner_output(evaluation_data)
                eval_accuracy = accuracy(predictions, evaluation_labels)
                
                # Logging
                metrics_dict['outer_loss'] = loss.item()
                metrics_dict['accuracy'] = eval_accuracy.item()
                metrics_dict['time'] = time.time() - start_time
                
                inner_logs = self.inner_solution.inner_solver.data_logs if hasattr(self.inner_solution.inner_solver, 'data_logs') else None
                if inner_logs:
                    self.log_metrics_list(inner_logs, self.iters, log_name='inner_metrics')
                
                dual_logs = self.inner_solution.dual_solver.data_logs if hasattr(self.inner_solution.dual_solver, 'data_logs') else None
                if dual_logs:
                    self.log_metrics_list(dual_logs, self.iters, log_name='dual_metrics')
                
                self.log(metrics_dict)
                print(f"Iter {self.iters}: Loss={loss.item():.4f}, Accuracy={eval_accuracy.item():.4f}")
                
                self.iters += 1
                done = (self.iters >= self.args.max_epochs)
                if done:
                    break
        
        # Evaluate after training
        val_loss, val_acc = self.evaluate(data_type="validation")
        val_log = [{'inner_iter': 0, 'val_loss': val_loss.item(), 'val_accuracy': val_acc.item()}]
        self.log_metrics_list(val_log, 0, log_name='val_metrics')
        
        test_loss, test_acc = self.evaluate(data_type="test")
        test_log = [{'inner_iter': 0, 'test_loss': test_loss.item(), 'test_accuracy': test_acc.item()}]
        self.log_metrics_list(test_log, 0, log_name='test_metrics')

    def evaluate(self, data_type="validation"):
        """
        Evaluates the meta-model on the given data type.
        
        Parameters:
        - data_type (str): Type of data to evaluate on ("validation" or "test").
        
        Returns:
        - tuple: (average_loss, average_accuracy) on the provided data.
        """
        tasks = self.val_tasks if data_type == "validation" else self.test_tasks
        
        # Store previous training states
        previous_state_meta_model = self.meta_model.training
        previous_state_inner_solution = self.inner_solution.training
        previous_state_inner_model = self.inner_model.training
        if hasattr(self, 'features') and not callable(self.features):
            previous_state_features = self.features.training
            self.features.eval()
        
        # Set models to evaluation mode
        self.meta_model.eval()
        self.inner_solution.eval()
        self.inner_model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_tasks = len(tasks)
        
        with torch.no_grad():
            for task in tasks:
                # Get task data
                data, labels = task
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Perform fast adaptation using the appropriate method
                evaluation_error, evaluation_accuracy = fast_adapt(
                    self.args.method,
                    (data, labels),
                    self.meta_model.clone(),  # Clone for each task
                    self.features,
                    self.criterion,
                    self.args.shots,
                    self.args.ways,
                    self.args.inner_steps,
                    self.args.reg_lambda,
                    self.device
                )
                
                total_loss += evaluation_error.item()
                total_accuracy += evaluation_accuracy.item()
        
        # Restore previous training states
        self.meta_model.train(previous_state_meta_model)
        self.inner_solution.train(previous_state_inner_solution)
        self.inner_model.train(previous_state_inner_model)
        if hasattr(self, 'features') and not callable(self.features):
            self.features.train(previous_state_features)
        
        avg_loss = total_loss / num_tasks
        avg_accuracy = total_accuracy / num_tasks
        
        return torch.tensor(avg_loss, device=self.device), torch.tensor(avg_accuracy, device=self.device)
    
    def get_inner_opt(self, train_loss):
        """
        Creates an inner optimizer as used in trainer_meta.py.
        
        Args:
            train_loss: The training loss function
            
        Returns:
            inner_opt: The inner optimizer
        """
        inner_opt_class = GradientDescent
        inner_opt_kwargs = {'step_size': self.args.inner_lr}
        return inner_opt_class(train_loss, **inner_opt_kwargs)