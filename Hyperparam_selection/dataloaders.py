"""Dataloaders for MNIST, FashionMNIST, CIFAR10"""

from typing import Tuple
import torchvision
from torch.utils.data import DataLoader
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


def MNIST_dataloaders(
    trainloader_sample_rate: float,
    test_batch_size: int,
    root: str = './',
    generator=None,
) -> Tuple[DataLoader, DataLoader]:

    tensor_func = torchvision.transforms.ToTensor()
    norm_func = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    transform = torchvision.transforms.Compose([tensor_func, norm_func])

    train_dataset = torchvision.datasets.MNIST(
        root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        generator=generator,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_dataset),
            sample_rate=trainloader_sample_rate,
            generator=generator,
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def fashion_mnist_dataloaders(
    trainloader_sample_rate: float,
    test_batch_size: int,
    root: str = './data',
    generator=None,
) -> Tuple[DataLoader, DataLoader]:

    tensor_func = torchvision.transforms.ToTensor()
    norm_func = torchvision.transforms.Normalize((0.2860,), (0.3205,))
    transform = torchvision.transforms.Compose([tensor_func, norm_func])

    train_dataset = torchvision.datasets.FashionMNIST(
        root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        generator=generator,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_dataset),
            sample_rate=trainloader_sample_rate,
            generator=generator,
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def CIFAR10_dataloaders(
    trainloader_sample_rate: float,
    test_batch_size: int,
    root: str = './',
    generator=None,
) -> Tuple[DataLoader, DataLoader]:

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        generator=generator,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_dataset),
            sample_rate=trainloader_sample_rate,
            generator=generator,
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_dataloaders(
    dataset: str,
    trainloader_sample_rate: float,
    test_batch_size: int
) -> Tuple[DataLoader, DataLoader]:

    dataloaders = {
        "MNIST": MNIST_dataloaders,
        "FashionMNIST": fashion_mnist_dataloaders,
        "CIFAR10": CIFAR10_dataloaders,
    }

    return dataloaders[dataset](trainloader_sample_rate, test_batch_size)
