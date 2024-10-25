"""TnT (Train and Test) functions"""

from experiment_params import Parameters
from typing import Callable, Dict, Union

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


from networks import SmallNetwork, BigNetwork
from tqdm import tqdm

Log = Dict[str, Union[int, float]]


def print_log(_log: Log, step: int):
    print(f" ##### STEP:{step} ##### ")
    for k, v in _log.items():
        print(f"{k}: {round(v, 3)}")


# Type definitions
ptModel = Union[SmallNetwork, BigNetwork]
ptLoss = Callable
ptOptimizer = optim.Optimizer


def train(
    params: Parameters,
    model: ptModel,
    train_loader: DataLoader,
    criterion: ptLoss,
    optimizer: ptOptimizer,
    logger: Callable,
) -> None:

    # Set model to training mode
    model.train()
    model.to(params.device)

    accumulation_steps: int = 0

    for idx, (data, target) in enumerate(train_loader):

        # Init log
        log: Log = {}

        # Transfer data to device
        data = data.to(params.device)
        target = target.to(params.device)

        # Forward and backward pass
        prob_pred = model(data)
        loss = criterion(prob_pred, target)
        loss.backward()

        # Counting samples and steps
        model.n_samples += len(data)
        model.n_steps += 1
        accumulation_steps += 1

        # Stepping if number of accumulation steps is reached or trainloader is empty
        if ((idx+1) % params.n_accumulation_steps == 0) or ((idx + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

            # Log stats
            log["train_loss"] = loss.item()
            log["samples"] = model.n_samples

            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(   # type: ignore
                params.delta)

            grad_norm = optimizer.privacy_engine.max_grad_norm                  # type: ignore
            noise_multiplier = optimizer.privacy_engine.noise_multiplier        # type: ignore

            log["privacy/epsilon"] = epsilon
            log["privacy/delta"] = params.delta
            log["privacy/best_alpha"] = best_alpha
            log["privacy/noise_multiplier"] = noise_multiplier
            log["privacy/grad_norm"] = grad_norm

            logger(log, model.n_steps)

            accumulation_steps = 0

        # if not, accumulate
        else:
            optimizer.virtual_step()    # type: ignore


def test(
    params: Parameters,
    model: ptModel,
    test_loader: DataLoader,
    criterion: ptLoss,
    logger: Callable,
) -> None:

    # Set model to evaluation mode and init eval variables
    model.eval()
    test_loss: float = 0
    n_correct: int = 0
    n_total: int = 0

    # Init log
    log: Log = {}

    # Disable gradients temporarely to save memory
    with torch.no_grad():
        for data, target in test_loader:

            # Transfer data to device
            data = data.to(params.device)
            target = target.to(params.device)

            # Test prediction and sum of batch loss
            prob_pred = model(data)
            test_loss += criterion(prob_pred, target).item()

            # Get predicted class and count number of correct predictions
            pred = prob_pred.argmax(dim=1)
            n_correct += pred.eq(target.view_as(pred)).sum().item()
            n_total += len(data)

    # Calculate test accuracy and avg. loss
    test_acc = 100. * (n_correct / n_total)
    test_loss /= n_total

    # Log test accuracy and loss
    log["test_accuracy"] = test_acc
    log["test_loss"] = test_loss

    # Send logs from testing to logger
    logger(log, model.n_steps)

    model.train()


def train_and_test(
    params: Parameters,
    model: ptModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: ptLoss,
    optimizer: ptOptimizer
) -> None:

    logger: Callable = print_log

    for _ in tqdm(range(1, params.epochs+1), "Epoch"):
        train(params, model, train_loader, criterion, optimizer, logger)
        test(params, model, test_loader, criterion, logger)
