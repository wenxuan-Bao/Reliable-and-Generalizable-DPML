import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from new_wide_resnet import WideResNet
# from WS_wide_resnet import WideResNet
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from WS import Conv2d
from opacus.grad_sample import register_grad_sampler
import os
import math
from typing import Dict, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import unfold2d, unfold3d
from opt_einsum import contract
from src.opacus_augmented.privacy_engine_augmented import PrivacyEngineAugmented
from opacus import GradSampleModule
from src.utils.utils import (init_distributed_mode,initialize_exp,bool_flag,get_noise_from_bs,get_epochs_from_bs,print_params,)
from src.models.prepare_models import prepare_data_cifar, prepare_augmult_cifar
# from src.models.EMA_without_class import create_ema, update
from torch_ema import ExponentialMovingAverage
def accuracy(preds, labels):
    return (preds == labels).mean()


# Hyperparameters
MAX_GRAD_NORM = 1.0
EPSILON = 8
DELTA = 1e-5
EPOCHS = 200
LR = 4
BATCH_SIZE = 4096
MAX_PHYSICAL_BATCH_SIZE = 128
sigma = 2.62451171875
K = 16
print('lr: ',LR)


transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

DATA_ROOT ='../../data'

train_dataset = CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
                                       ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
)

test_dataset = CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
)

# decide whether you use ScaleNorm or not also the order of the layers
# model = WideResNet(16,10,4,16,0,order1=0,order2=0,scale=False)
model = WideResNet(16,10,4,16,0,order1=1,order2=0,scale=True)
from opacus.validators import ModuleValidator
model = ModuleValidator.fix(model)
print(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)
ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)

# if you want to use Mixed Ghost cliping, load the privacy engine like in main file
# privacy_engine = PrivacyEngine()
privacy_engine = PrivacyEngineAugmented(GradSampleModule.GRAD_SAMPLERS) #use self-augmented privacy engine

# model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     epochs=EPOCHS,
#     target_epsilon=EPSILON,
#     target_delta=DELTA,
#     max_grad_norm=MAX_GRAD_NORM,
# )

model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=MAX_GRAD_NORM,
        poisson_sampling=True,
        K=K
    )
prepare_augmult_cifar(model,K) #prepare the model for self-augmented training, K is the number of augmentations

print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")


def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            if K:
                images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
                target = torch.repeat_interleave(target, repeats=K, dim=0)
                transform = transforms.Compose([transforms.RandomCrop(size=(32, 32), padding=4, padding_mode="reflect"),
                                                transforms.RandomHorizontalFlip(p=0.5), ])
                images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(images_duplicates)


            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()
            ema.update()
            # if ema:
            #     update(model, ema, nb_steps)

            if (i + 1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)


test_best_acc=[]
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, epoch + 1, device)
    top1_acc = test(model, test_loader, device)
    test_best_acc.append(top1_acc)

top1_acc = test(model, test_loader, device)
test_best_acc.append(top1_acc)
print('Test acc:',top1_acc)
print('Best test acc: ',np.max(test_best_acc))