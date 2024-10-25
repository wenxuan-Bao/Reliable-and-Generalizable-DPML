from models import CNNS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import random
from auto_opacus.opacus.utils.batch_memory_manager import BatchMemoryManager
# from opacus.utils.batch_memory_manager import BatchMemoryManager
import argparse
import os
import time
import subprocess
# Function to set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Function to compute accuracy
def compute_accuracy(model, dataloader):
    correct = 0
    total = 0
    model = model.to('cpu')
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training function
def sgd_train(model, trainloader,testloader, criterion, optimizer, epochs,device):
    best_acc = 0

    for epoch in range(epochs):
        model = model.to(device)
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # # Compute and print accuracy
        # train_accuracy = compute_accuracy(model, trainloader)
        # print(f'Train accuracy: {train_accuracy}%')
        test_accuracy =test(model, testloader, device)

        # test_accuracy = compute_accuracy(model, testloader)
        print(f'Test accuracy: {test_accuracy}%')
        if test_accuracy > best_acc:
            best_acc = test_accuracy
    print(f'Best test accuracy: {best_acc}%')
    return best_acc


def accuracy(preds, labels):
    return (preds == labels).mean()




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



# Main function
def main(use_dp,seed,lr=0.001,epochs=10,EPSILON=1.0,DELTA=1e-5,MAX_GRAD_NORM=1.0,batch_size=4096,dataset_name='cifar10',auto=True):
    # Set the seed
    set_seed(seed)
    if auto:
        print('auto: ',auto)
        from auto_opacus.opacus import PrivacyEngine
        from auto_opacus.opacus.utils.batch_memory_manager import BatchMemoryManager
    else:
        print('auto: ', auto)
        # from opacus import PrivacyEngine
        # from opacus.utils.batch_memory_manager import BatchMemoryManager

    if dataset_name=='cifar10':
        print('dataset_name: ',dataset_name)
        # Load the data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        NUM_CLASSES = 10
    elif dataset_name == "fmnist":
        print('dataset_name: ', dataset_name)
        train_set = datasets.FashionMNIST(root='../data', train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        test_set = datasets.FashionMNIST(root='../data', train=False,
                                         transform=transforms.ToTensor(),
                                         download=True)
        NUM_CLASSES = 10
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset_name == "mnist":
        print('dataset_name: ', dataset_name)
        train_set = datasets.MNIST(root='../data', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

        test_set = datasets.MNIST(root='../data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
        NUM_CLASSES = 10
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError("Dataset not supported")






    # Define the model, optimizer, and loss function
    # init_seed = 42
    # torch.manual_seed(init_seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(init_seed)
    # model = WideResNet(16, NUM_CLASSES, 4, 16, 0, order1=1, order2=0, scale=True)
    model = CNNS[dataset_name](1,input_norm=None, num_groups=16, size=16)
    from opacus.validators import ModuleValidator
    model = ModuleValidator.fix(model)

    # set_seed(seed)
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




    if use_dp:
        privacy_engine = PrivacyEngine()
        # privacy_engine = PrivacyEngineAugmented(GradSampleModule.GRAD_SAMPLERS)

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs=epochs,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )

        test_best_acc = []
        for epoch in range(epochs):


            train(model, privacy_engine, trainloader, optimizer, epoch + 1, device)
            # top1_acc = test(model, testloader, device)
            # test_best_acc.append(top1_acc)
            top1_acc = test(model, testloader, device)
            test_best_acc.append(top1_acc)
            print('Test acc:', top1_acc)
            print('Best test acc: ', np.max(test_best_acc))
    else:
        best_acc= sgd_train(model, trainloader, testloader,criterion, optimizer, epochs=epochs,device=device)
        print('Best test acc for SGD: ', best_acc)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        encoding='utf-8')
    return int(result.strip().split('\n')[0])


def train(model,privacy_engine, train_loader, optimizer, epoch, device,MAX_PHYSICAL_BATCH_SIZE=16,DELTA=1e-5):
    model.to(device)
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
            #
            # if K:
            #     images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
            #     target = torch.repeat_interleave(target, repeats=K, dim=0)
            #     transform = transforms.Compose([transforms.RandomCrop(size=(32, 32), padding=4, padding_mode="reflect"),
            #                                     transforms.RandomHorizontalFlip(p=0.5), ])
            #     images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(images_duplicates)


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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp', default=True, type=str2bool)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=2, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--delta', default=1e-5, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist'],default='mnist')
    parser.add_argument('--batchsize', default=4096, type=int)
    parser.add_argument('--auto',  default=True, type=str2bool)

    args = parser.parse_args()
    print(args)
    USE_DP = args.dp
    print('USE_DP: ',USE_DP)
    SEED = args.seed
    LR = args.lr
    EPOCHS = args.epochs
    EPSILON = args.epsilon
    DELTA = args.delta
    MAX_GRAD_NORM = args.max_grad_norm
    DATA_NAME = args.dataset
    BATCH_SIZE = args.batchsize
    AUTO = args.auto

    # seed_list=[42,3407,2021,2022,2023,50,100,2000,4000,5000,6000]
    seed_list=np.load('../random_numbers.npy')

    if SEED == 0:
        num=int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
        SEED=seed_list[num]
        print('SEED: ',SEED)

    # Record the initial GPU memory usage
    initial_gpu_mem = get_gpu_memory()

    # Record the start time
    start_time = time.time()

    # Call the main function with DP
    main(use_dp=USE_DP,seed=SEED,lr=LR,epochs=EPOCHS,EPSILON=EPSILON,DELTA=DELTA,MAX_GRAD_NORM=MAX_GRAD_NORM,
         dataset_name=DATA_NAME,batch_size=BATCH_SIZE,auto=AUTO)

    # Record the end time
    end_time = time.time()

    # Record the final GPU memory usage
    final_gpu_mem = get_gpu_memory()

    # Calculate and print the elapsed time and GPU memory consumption
    elapsed_time = end_time - start_time
    print(f'Final GPU memory: {final_gpu_mem} MiB')
    gpu_mem_consumed = final_gpu_mem - initial_gpu_mem

    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"GPU Memory Consumed: {gpu_mem_consumed} MiB")




