import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import learn2learn as l2l
import numpy as np
import math
from typing import List, Tuple, Any, Optional


def accuracy(predictions, targets):
    """
    Computes the accuracy of the predictions with respect to the targets.
    """
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def split_data(data, labels, shots, ways, device):
    """
    Split data into adaptation and evaluation sets using torch's random_split.
    """
    from torch.utils.data import TensorDataset, random_split
    
    data, labels = data.to(device), labels.to(device)
    
    # Calculate sizes (ensuring we don't exceed the data size)
    total_size = data.size(0)
    adaptation_size = min(shots * ways, total_size // 2)
    evaluation_size = total_size - adaptation_size
    
    # Create dataset and split it
    dataset = TensorDataset(data, labels)
    adaptation_dataset, evaluation_dataset = random_split(
        dataset, [adaptation_size, evaluation_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Extract the data
    adaptation_indices = adaptation_dataset.indices
    evaluation_indices = evaluation_dataset.indices
    
    adaptation_data = data[adaptation_indices]
    adaptation_labels = labels[adaptation_indices]
    evaluation_data = data[evaluation_indices]
    evaluation_labels = labels[evaluation_indices]
    
    return adaptation_data, adaptation_labels, evaluation_data, evaluation_labels


def fast_adapt(method, batch, learner, features, loss, shots, ways, inner_steps, reg_lambda, device):
    """
    Perform a fast adaptation step.
    """
    data, labels = batch

    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels \
        = split_data(data, labels, shots, ways, device)

    if method == 'funcBO' or method == 'ANIL':
        adaptation_data = features(adaptation_data)
        evaluation_data = features(evaluation_data)
        for step in range(inner_steps):
            l2_reg = 0
            for p in learner.parameters():
                l2_reg += p.norm(2)
            train_error = loss(learner(adaptation_data), adaptation_labels) + reg_lambda * l2_reg
            learner.adapt(train_error)
    elif method == 'MAML':
        for step in range(inner_steps):
            train_error = loss(learner(adaptation_data), adaptation_labels)
            train_error /= len(adaptation_data)
            learner.adapt(train_error)

    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def task_function(data, features, model, device):
    """
    Processes task data through feature extractor and returns inputs for inner and outer optimization.
    
    Args:
        data: The task data (often a tuple of images and labels)
        features: The feature extractor model
        model: The adaptation model (head)
        device: The device to use for computation
        
    Returns:
        A tuple containing processed data for inner model input, outer model input,
        and any additional inputs needed for the inner loss function
    """
    # Unpack the task data
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    
    # Extract features if needed
    if features is not None:
        with torch.no_grad():
            processed_images = features(images)
    else:
        processed_images = images
    
    # For most meta-learning methods, we return the same processed images for both
    # inner and outer model inputs, along with the labels for loss computation
    return processed_images, processed_images, labels


def get_task_loaders(shots: int, ways: int, device: str, seed: int = 42):
    """
    Creates task loaders for meta-learning with FC100 dataset.
    
    Args:
        shots: Number of shots per class
        ways: Number of classes per task
        device: The device to load the data on
        seed: Random seed for reproducibility
        
    Returns:
        Three task sets for train, validation and test
    """
    import learn2learn as l2l
    import torch
    import numpy as np
    from torchvision import transforms
    
    torch.manual_seed(seed)
    
    # Define normalization transform for FC100
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4409],
        std=[0.2673, 0.2564, 0.2762]
    )
    
    # Define transforms for FC100
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Load FC100 dataset using learn2learn
    # Note: In recent learn2learn versions, we use the l2l.vision.datasets.FC100ClassDataset
    # instead of FC100 directly to get compatibility with NWays, KShots transforms
    try:
        # First try the newer API (if available)
        fc100_train = l2l.vision.datasets.FC100ClassDataset(
            root='./data', 
            mode='train', 
            download=True,
            transform=transform
        )
        
        fc100_valid = l2l.vision.datasets.FC100ClassDataset(
            root='./data', 
            mode='validation', 
            download=True,
            transform=transform
        )
        
        fc100_test = l2l.vision.datasets.FC100ClassDataset(
            root='./data', 
            mode='test', 
            download=True,
            transform=transform
        )
    except (ImportError, AttributeError):
        # If the above fails, we need to create a custom wrapper
        # Load the regular FC100 dataset
        fc100_train = l2l.vision.datasets.FC100(
            root='./data', 
            mode='train', 
            download=True,
            transform=transform
        )
        
        fc100_valid = l2l.vision.datasets.FC100(
            root='./data', 
            mode='validation', 
            download=True,
            transform=transform
        )
        
        fc100_test = l2l.vision.datasets.FC100(
            root='./data', 
            mode='test', 
            download=True,
            transform=transform
        )
        
        # Add the missing attributes needed for NWays transform
        # Learn2learn expects datasets to have these attributes
        for dataset in [fc100_train, fc100_valid, fc100_test]:
            # Map from indices to labels
            dataset.indices_to_labels = {}
            for i, (_, label) in enumerate(dataset):
                if i not in dataset.indices_to_labels:
                    dataset.indices_to_labels[i] = label
            
            # Get the set of labels
            labels = set(dataset.indices_to_labels.values())
            dataset._labels = sorted(list(labels))
            
            # Map from labels to indices
            dataset.labels_to_indices = {}
            for idx, label in dataset.indices_to_labels.items():
                if label not in dataset.labels_to_indices:
                    dataset.labels_to_indices[label] = []
                dataset.labels_to_indices[label].append(idx)
    
    # Create tasksets using the TaskTransform API from learn2learn
    train_tasks = l2l.data.TaskDataset(
        fc100_train,
        task_transforms=[
            l2l.data.transforms.NWays(fc100_train, ways),
            l2l.data.transforms.KShots(fc100_train, 2*shots),  # 2x for train/test split
            l2l.data.transforms.LoadData(fc100_train),
            l2l.data.transforms.RemapLabels(fc100_train),
            l2l.data.transforms.ConsecutiveLabels(fc100_train),
        ],
        num_tasks=1000,
    )
    
    valid_tasks = l2l.data.TaskDataset(
        fc100_valid,
        task_transforms=[
            l2l.data.transforms.NWays(fc100_valid, ways),
            l2l.data.transforms.KShots(fc100_valid, 2*shots),
            l2l.data.transforms.LoadData(fc100_valid),
            l2l.data.transforms.RemapLabels(fc100_valid),
            l2l.data.transforms.ConsecutiveLabels(fc100_valid),
        ],
        num_tasks=200,
    )
    
    test_tasks = l2l.data.TaskDataset(
        fc100_test,
        task_transforms=[
            l2l.data.transforms.NWays(fc100_test, ways),
            l2l.data.transforms.KShots(fc100_test, 2*shots),
            l2l.data.transforms.LoadData(fc100_test),
            l2l.data.transforms.RemapLabels(fc100_test),
            l2l.data.transforms.ConsecutiveLabels(fc100_test),
        ],
        num_tasks=200,
    )
    
    # Create a task preparation function to process each task
    def prepare_task(task_data):
        x, y = task_data
        # Move to device
        x = x.to(device)
        y = y.to(device)
        
        # Split into adaptation and evaluation sets
        adaptation_indices = np.zeros(len(y), dtype=bool)
        adaptation_indices[np.arange(shots * ways) * 2] = True
        evaluation_indices = ~adaptation_indices
        
        adaptation_data, adaptation_labels = x[adaptation_indices], y[adaptation_indices]
        evaluation_data, evaluation_labels = x[evaluation_indices], y[evaluation_indices]
        
        return (adaptation_data, adaptation_labels, evaluation_data, evaluation_labels)
    
    # Process tasks
    # Convert task format to the expected format: (data, labels)
    processed_train_tasks = []
    processed_val_tasks = []
    processed_test_tasks = []
    
    for task in train_tasks:
        x, y = task
        processed_train_tasks.append((x.to(device), y.to(device)))
    
    for task in valid_tasks:
        x, y = task
        processed_val_tasks.append((x.to(device), y.to(device)))
        
    for task in test_tasks:
        x, y = task
        processed_test_tasks.append((x.to(device), y.to(device)))
    
    return processed_train_tasks, processed_val_tasks, processed_test_tasks


def get_fc100_model_dimensions():
    """Returns the dimensions for FC100 models."""
    return {
        'feature_dim': 256,  # Output dimension of feature extractor for FC100
        'channels': 3,      # Input channels
        'hidden_size': 64,  # Hidden channels in convolutional layers
    }


class TaskData(Dataset):
    """Dataset wrapper for meta-learning task data."""
    def __init__(self, task_data):
        """
        Initialize with task data.
        
        Args:
            task_data: A tuple of (adaptation_data, adaptation_labels, evaluation_data, evaluation_labels)
        """
        self.adaptation_data, self.adaptation_labels, self.evaluation_data, self.evaluation_labels = task_data
        
    def __len__(self):
        return len(self.adaptation_labels)
    
    def __getitem__(self, idx):
        return (
            self.adaptation_data[idx], 
            self.adaptation_labels[idx]
        )
        
    def get_evaluation_data(self):
        """Returns the evaluation data for this task."""
        return (self.evaluation_data, self.evaluation_labels)


class MetaHead(nn.Module):
    """
    Meta-learning head for classification tasks.
    """
    def __init__(self, params):
        super(MetaHead, self).__init__()
        self.params = params
        self.input_dim = params.get('input_dim', 256)
        self.output_dim = params.get('output_dim', 5)  # Default for 5-way
        
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        
    def forward(self, x):
        return self.linear(x)
    
    def clone(self):
        """Create a clone of the current model."""
        clone = MetaHead(self.params)
        clone.load_state_dict(self.state_dict())
        return clone
    
    def adapt(self, loss, lr=0.1):
        """Basic adaptation step for compatibility with some meta-learning methods."""
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        
        for p, g in zip(self.parameters(), grads):
            p.data.sub_(lr * g)
            

class FeatureExtractor(nn.Module):
    """
    Feature extractor for meta-learning.
    """
    def __init__(self, params):
        super(FeatureExtractor, self).__init__()
        self.params = params
        self.hidden_size = params.get('hidden_size', 64)
        self.channels = params.get('channels', 3)
        self.feature_dim = params.get('feature_dim', 256)
        
        # Build the convolutional feature extractor
        self.features = nn.Sequential(
            self._make_conv_block(self.channels, self.hidden_size),
            self._make_conv_block(self.hidden_size, self.hidden_size),
            self._make_conv_block(self.hidden_size, self.hidden_size),
            self._make_conv_block(self.hidden_size, self.hidden_size),
            nn.Flatten()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        features = self.features(x)
        return features


class MAMLModel(nn.Module):
    """
    Model for MAML implementation.
    """
    def __init__(self, params):
        super(MAMLModel, self).__init__()
        self.params = params
        self.hidden_size = params.get('hidden_size', 64)
        self.channels = params.get('channels', 3)
        self.feature_dim = params.get('feature_dim', 256)
        self.output_dim = params.get('output_dim', 5)  # Default for 5-way
        
        # Create a combined model with features and head
        self.features = FeatureExtractor(params)
        self.head = MetaHead(params)
        
    def forward(self, x):
        features = self.features(x)
        return self.head(features)
    
    def clone(self):
        """Create a clone of the current model."""
        clone = MAMLModel(self.params)
        clone.load_state_dict(self.state_dict())
        return clone


class ANILModel(nn.Module):
    """
    Almost No Inner Loop (ANIL) model - only adapts the head.
    """
    def __init__(self, features, head):
        super(ANILModel, self).__init__()
        self.features = features
        self.head = head
        
    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)
        return self.head(features)
    
    def parameters(self):
        """Return only the head parameters for adaptation."""
        return self.head.parameters()
    
    def clone(self):
        """Create a clone of the current model's head."""
        clone_head = self.head.clone()
        return ANILModel(self.features, clone_head)

def meta_collate_fn(batch):
    """
    Custom collate function for meta-learning tasks.
    Instead of stacking the tensors (which would fail if tasks have different sizes),
    we simply return the batch as a list of tasks.
    
    Args:
        batch: A list of tasks, where each task is typically a tuple of (data, labels)
        
    Returns:
        The same list without any stacking
    """
    return batch