from opacus_old.opacus import PrivacyEngine
import torch.nn.functional as F
import torch
import argparse

from networks import SmallNetwork, BigNetwork
from dataloaders import get_dataloaders
from tnt import train_and_test, ptModel, ptLoss, ptOptimizer
from experiment_params import experiment_parameters, Parameters


def run_experiment(params: Parameters) -> None:
    # Get Dataloaders
    train_loader, test_loader = get_dataloaders(
        params.dataset,
        params.trainloader_sample_rate,
        params.test_batch_size
    )

    # Make the model
    model: ptModel = (BigNetwork() if params.dataset ==
                      "CIFAR10" else SmallNetwork())

    # Make the loss criterion and optimizer
    criterion: ptLoss = F.nll_loss
    optimizer: ptOptimizer = torch.optim.SGD(model.parameters(),
                                             lr=params.learning_rate,
                                             momentum=params.momentum)
    print('target_epsilon: ', params.target_epsilon)

    # Init privacy engine
    if params.privacy:
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=params.trainloader_sample_rate * params.n_accumulation_steps,
            alphas=range(2, 64),
            epochs=params.epochs,
            target_delta=params.delta,
            target_epsilon=params.target_epsilon,
            # noise_multiplier=params.sigma,
            max_grad_norm=params.max_per_sample_grad_norm,
            secure_rng=params.secure_rng,
        )
        privacy_engine.attach(optimizer)
        privacy_engine.to(params.device)

    # Execute train and test loops
    train_and_test(params, model, train_loader, test_loader,
                   criterion, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=[
        "MNIST",
        "FashionMNIST",
        "CIFAR10"
    ])
    parser.add_argument('--parameter-set', type=int, default=1, choices=[
        1,
        2,
        3
    ])
    args = parser.parse_args()

    params: Parameters = experiment_parameters[args.dataset][args.parameter_set]

    run_experiment(params)
