"""Experiment Parameters for results in table 2"""

from typing import Dict
import torch


class Parameters:
    def __init__(self,
                 dataset: str = "CIFAR10",
                 epochs: int = 10,
                 trainloader_sample_rate: float = 0.0089,
                 n_accumulation_steps: int = 16,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 target_epsilon: float = 1.0,
                 max_per_sample_grad_norm: float = 1.0,
                 sigma: float = 1.0
                 ):

        self.dataset: str = dataset
        self.epochs: int = epochs
        self.trainloader_sample_rate: float = trainloader_sample_rate
        self.n_accumulation_steps: int = n_accumulation_steps
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.target_epsilon: float = target_epsilon
        self.max_per_sample_grad_norm: float = max_per_sample_grad_norm
        self.sigma: float = sigma

        # Fixed
        self.test_batch_size: int = 512
        self.device: torch.device = torch.device('cuda')
        self.privacy: bool = True
        self.delta: float = 1e-5
        self.secure_rng: bool = False


experiment_name = str
experiment_configuration = Dict[int, Parameters]


# Parameters
experiment_parameters: Dict[experiment_name, experiment_configuration] = {
    "MNIST": {
        1: Parameters(
            dataset="MNIST",
            target_epsilon=1.0,
            epochs=40,
            trainloader_sample_rate=0.1427/20,  # BS: 8562
            n_accumulation_steps=20,
            learning_rate=6.939,
            momentum=0.4253,
            max_per_sample_grad_norm=0.2901
        ),
        2: Parameters(
            dataset="MNIST",
            target_epsilon=2.0,
            epochs=60,
            trainloader_sample_rate=0.1427/20,  # BS: 8562
            n_accumulation_steps=20,
            learning_rate=1.798,
            momentum=0.7161,
            max_per_sample_grad_norm=0.5757
        ),
        3: Parameters(
            dataset="MNIST",
            target_epsilon=2.93,
            epochs=40,
            trainloader_sample_rate=0.1665/20,  # BS: 9990
            n_accumulation_steps=20,
            learning_rate=10.244,
            momentum=0.5784,
            max_per_sample_grad_norm=0.4196
        )
    },
    "FashionMNIST": {
        1: Parameters(
            dataset="FashionMNIST",
            target_epsilon=1.0,
            epochs=40,
            trainloader_sample_rate=0.1427/20,  # BS: 8562
            n_accumulation_steps=20,
            learning_rate=4.069,
            momentum=0.5764,
            max_per_sample_grad_norm=0.5716
        ),
        2: Parameters(
            dataset="FashionMNIST",
            target_epsilon=2.0,
            epochs=80,
            trainloader_sample_rate=0.1665/20,  # BS: 9990
            n_accumulation_steps=20,
            learning_rate=5.706,
            momentum=0.4277,
            max_per_sample_grad_norm=0.6457
        ),
        3: Parameters(
            dataset="FashionMNIST",
            target_epsilon=2.7,
            epochs=60,
            trainloader_sample_rate=0.1665/20,  # BS: 9990
            n_accumulation_steps=20,
            learning_rate=9.493,
            momentum=0.5946,
            max_per_sample_grad_norm=0.474
        ),
    },
    "CIFAR10": {
        1: Parameters(
            dataset="CIFAR10",
            target_epsilon=2.0,
            epochs=40,
            trainloader_sample_rate=0.1666/20,  # BS: 8330
            n_accumulation_steps=20,
            learning_rate=1.601,
            momentum=0.5992,
            max_per_sample_grad_norm=1.253,
        ),
        2: Parameters(
            dataset="CIFAR10",
            target_epsilon=5.0,
            epochs=100,
            trainloader_sample_rate=0.2/20,  # BS: 10000
            n_accumulation_steps=20,
            learning_rate=3.692,
            momentum=0.6442,
            max_per_sample_grad_norm=0.6533
        ),
        3: Parameters(
            dataset="CIFAR10",
            target_epsilon=7.53,
            epochs=100,
            trainloader_sample_rate=0.1666/20,  # BS: 8330
            n_accumulation_steps=20,
            learning_rate=7.101,
            momentum=0.4433,
            max_per_sample_grad_norm=0.4434
        )
    }
}
