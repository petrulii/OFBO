import numpy as np
import torch

def accuracy(predictions, targets):
    """
    Computes the accuracy of the predictions with respect to the targets.
    """
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def split_data(data, labels, shots, ways, device):
    """
    Split data into adaptation and evaluation sets.
    """
    data, labels = data.to(device), labels.to(device)

    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

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