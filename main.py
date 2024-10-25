import os
import sys
sys.path.insert(0, '/content/private_CNN')
import private_CNN
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from opacus.validators import ModuleValidator
from train import train,test,cheast_train,chest_test,isic_train,isic_test
from torch.utils.data import DataLoader, Subset
from medmnist import INFO, Evaluator
import medmnist
from torchvision.datasets import CIFAR10, CIFAR100, VOCDetection,EuroSAT,Caltech256,OxfordIIITPet,SUN397
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Ensure3Channels:
    def __call__(self, tensor):
        if tensor.shape[0] == 1:  # Check if there's only 1 channel
            tensor = tensor.repeat(3, 1, 1)  # Convert to 3 channels
        return tensor



def fine_tune(dataset_name,LR=1e-3,eps=1.0,bs=1000,epochs=3,mode = 'ghost-mixed',model = 'vit_base_patch16_224',mini_batch_size = 20,debug_sign=False,cifar_data = 'CIFAR10'):
    print('EPS: ', eps)
    switch_weights_sign = False
    jobname = str(sys.argv[2])
    print('jobname: ',jobname)

    model_num = 0
    if 'scratch'in jobname:
        pretrained = False
        print('Train from scratch')
    else:
        pretrained = True
        print("Pretrain on imagenet")


    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    if dataset_name=='cifar-10':

        grad_norm =1.0

        trainset = torchvision.datasets.CIFAR10(
            root='../../data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='../../data', train=False, download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size, shuffle=True, num_workers=2)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        print('==> Building model..', model, '  mode ', mode)
        NUM_CLASSES = 10 if cifar_data == 'CIFAR10' else 100
    elif dataset_name == 'Eurosat':

        grad_norm = 0.05

        trainset = torchvision.datasets.EuroSAT(
            root='./data', download=True, transform=transform_train)
        print('Number of images in dataset:', len(trainset))
        train_indices, test_indices, _, _ = train_test_split(
            range(len(trainset)),
            trainset.targets,
            stratify=trainset.targets,
            test_size=3000,
            random_state=42
        )
        train_split = Subset(trainset, train_indices)
        test_split = Subset(trainset, test_indices)
        trainset = train_split


        print('hold')

        trainloader = torch.utils.data.DataLoader(
            train_split, batch_size=mini_batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(
           test_split, batch_size=100, shuffle=False, num_workers=2)
        NUM_CLASSES = 10
        # return "done"

    elif dataset_name == 'ISIC':
        from ISIC_dataset import Skin7

        if debug_sign==True:
            lr= 0.1*(model_num+1)
            # lr= 1/(10**(model_num+1))
            print('Debug lr: ',lr)

        grad_norm = 0.05
        re_size = 300
        input_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.Resize(re_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            transforms.RandomRotation([-180, 180]),
            transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                    scale=[0.7, 1.3]),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = Skin7(root="./data/", train=True,transform=train_transform)
        LABEL=trainset.targets
        L=np.sum(LABEL,axis=0)
        # la=Counter(LABEL)
        leng=len(trainset)

        print('Number of images in dataset:', len(trainset))

        train_indices, test_indices, _, _ = train_test_split(
            range(len(trainset)),
            trainset.targets,
            stratify=trainset.targets,
            test_size=1000,
            random_state=42
        )
        train_split = Subset(trainset, train_indices)
        test_split = Subset(trainset, test_indices)

        print('hold')

        # trainloader = torch.utils.data.DataLoader(
        #     train_split,sampler=train_sampler, batch_size=mini_batch_size,  num_workers=2)
        trainloader = torch.utils.data.DataLoader(
            train_split, batch_size=mini_batch_size, num_workers=2)
        testloader = torch.utils.data.DataLoader(
           test_split, batch_size=100, shuffle=False, num_workers=2,)
        trainset = train_split
        NUM_CLASSES = 7

    elif dataset_name == 'pathmnist':

        grad_norm = 0.05
        data_flag = 'pathmnist'
        # data_flag = 'breastmnist'
        download = True
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        NUM_CLASSES = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        # preprocessing
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        train_dataset = DataClass(split='train', transform=data_transform, download=download, root='./data')
        test_dataset = DataClass(split='test', transform=data_transform, download=download, root='./data')

        pil_dataset = DataClass(split='train', download=download)
        trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
        train_loader_at_eval = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2 * mini_batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    elif dataset_name == 'caltech256':
        print('Using caltech256')
        grad_norm = 1.0
        DATA_ROOT = './data/caltech256'


        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Ensure3Channels(),
        ])

        train_dataset = Caltech256(
            root=DATA_ROOT, download=True, transform=transform_train)

        train_indices, test_indices = train_test_split(
            range(len(train_dataset)), train_size=int(len(train_dataset) * 0.8),
            test_size=int(len(train_dataset) * 0.2),
            random_state=42
        )
        train_split = Subset(train_dataset, train_indices)
        test_split = Subset(train_dataset, test_indices)

        torch.manual_seed(42)  # Set the random seed for reproducibility
        shuffled_ids = torch.randperm(len(train_split), out=torch.LongTensor()).tolist()
        torch.seed()

        train_dataset = train_split

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)


        testloader = torch.utils.data.DataLoader(
            test_split,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES = 257

    elif dataset_name == 'sun397':
        print('Using sun397')
        grad_norm = 1.0
        DATA_ROOT = 'data/sun397'

        train_dataset = SUN397(
            root=DATA_ROOT, download=True, transform=transform_train)
        train_indices, test_indices = train_test_split(
            range(len(train_dataset)), train_size=int(len(train_dataset) * 0.8),
            test_size=int(len(train_dataset) * 0.2),
            random_state=42
        )
        train_split = Subset(train_dataset, train_indices)
        test_split = Subset(train_dataset, test_indices)

        torch.manual_seed(42)  # Set the random seed for reproducibility
        shuffled_ids = torch.randperm(len(train_split), out=torch.LongTensor()).tolist()
        torch.seed()
        trainloader = torch.utils.data.DataLoader(train_split, batch_size=mini_batch_size, shuffle=True)

        testloader = torch.utils.data.DataLoader(
            test_split,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES  = 397


    elif dataset_name == 'pet':
        print('Using pet')
        grad_norm = 4.0
        DATA_ROOT = 'data/pet'

        train_dataset = OxfordIIITPet(
            root=DATA_ROOT, download=True, transform=transform_train, split='trainval')

        torch.manual_seed(42)  # Set the random seed for reproducibility
        shuffled_ids = torch.randperm(len(train_dataset), out=torch.LongTensor()).tolist()
        torch.seed()
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)


        test_dataset = OxfordIIITPet(
            root=DATA_ROOT, split='test', download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES  = 37

    else:
        return "None"

    if dataset_name !='chest':
        net = timm.create_model(model, pretrained=pretrained, num_classes=NUM_CLASSES)
        net = ModuleValidator.fix(net)
    else:
        net = timm.create_model(model, pretrained=pretrained, num_classes=NUM_CLASSES, in_chans=1)
        net = ModuleValidator.fix(net)





    if 'convit' in model:
        for name, param in net.named_parameters():
            if 'attn.gating_param' in name:
                param.requires_grad = False
    if 'beit' in model:
        for name, param in net.named_parameters():
            if 'gamma_' in name or 'relative_position_bias_table' in name or 'attn.qkv.weight' in name or 'attn.q_bias' in name or 'attn.v_bias' in name:
                requires_grad = False

    n = []
    if 'first_last' in jobname:
        print('Train first and last layer')
        for name, param in net.named_parameters():
            n.append(name)
            param.requires_grad = False
            if 'pos_embed.proj' in name:
                param.requires_grad = True
            if 'head' in name:
                param.requires_grad = True
    elif 'last_only' in jobname:
        print('Train last layer only')
        for name, param in net.named_parameters():
            n.append(name)
            param.requires_grad = False
            # if 'pos_embed.proj' in name:
            #     param.requires_grad = True
            if 'head' in name:
                param.requires_grad = True
    else:
        print('Train all layers')

        for name, param in net.named_parameters():
            n.append(name)
            if 'cls_token' in name or 'pos_embed' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True



    print('number of parameters: ', sum([p.numel() for p in net.parameters()]))

    if "ghost" in mode:
        criterion = nn.CrossEntropyLoss(reduction="none")
        if dataset_name == 'chest':
            criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif 'normal' in mode:
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss()
        if dataset_name == 'chest':
            criterion = nn.BCEWithLogitsLoss()

    lr = LR
    print("LR: ", lr)
    print("Grad norm: ", grad_norm)
    model_parameters = [
        {'params': net.head.parameters(), 'lr': lr},  # Parameters of the head with learning rate 1
        {'params': (p for name, p in net.named_parameters() if 'head' not in name), 'lr': 1e-5}
        # Rest of the parameters with learning rate 0.001
    ]




    optimizer = optim.SGD(model_parameters,lr=lr)
    net.to(device)

    n_acc_steps = bs // mini_batch_size

    if 'ghost' in mode:
        print('Mixed Ghost clip')
        privacy_engine = private_CNN.PrivacyEngine(
            net,
            batch_size=bs,
            sample_size=len(trainloader.dataset),
            target_epsilon=eps,
            target_delta=1e-5,

            epochs=epochs,
            max_grad_norm=grad_norm,
            ghost_clipping=True,
            mixed='mixed' in mode
        )
        privacy_engine.attach(optimizer)
    elif 'normal' in mode:
        print('Normal clip')
        privacy_engine = private_CNN.PrivacyEngine(
            net,
            batch_size=bs,
            sample_size=len(trainloader.dataset),
            target_epsilon=eps,
            target_delta=1e-5,

            epochs=epochs,
            max_grad_norm=grad_norm,
            ghost_clipping=False,
            mixed= False
        )
        privacy_engine.attach(optimizer)
    old_dict=net.state_dict()
    acc_list=[]

    weights_file =''
    if "%" in jobname:
        switch_weights_sign = True
        print('==> Weight selection..')

    if switch_weights_sign == True:
        print("This experiment is switching weights")
        if "RS" in jobname:
            from weights_selection import random_subset
            threshold=int(sys.argv[3])/100
            weights_file=random_subset(net,threshold,jobname,dataset_name,model_num,eps)
        elif "ST_" in jobname:
            print('ST tricks')
            num=int(sys.argv[3])
            if os.path.exists('./save_weights/ST/save_weights_'+str(num)+'.npy'):
                print("weights file exist")
                weights_file='./save_weights/ST/save_weights_'+str(num)+'.npy'
            else:
                print("weights file does not exist. Create file now.")
                from weights_selection import ST_selection
                weights_file = ST_selection(net,num)

        elif "PT" in jobname:
            print('PT training')
            num = int(sys.argv[3])
            rand_list=np.random.randint(0,11,num)
            weights_file=rand_list




    for epoch in range(epochs):
        if dataset_name=='chest':
            cheast_train(epoch, net,trainloader,criterion,optimizer,n_acc_steps,device,mode,old_dict,switch_weights_sign,weights_file)
            acc=chest_test(epoch, net, testloader, criterion, device)
        elif dataset_name=='ISIC':
            isic_train(epoch, net,trainloader,criterion,optimizer,n_acc_steps,device,mode,old_dict,switch_weights_sign,weights_file)
            acc=isic_test(epoch, net, testloader, criterion, device)
        else:
            train(epoch, net, trainloader, criterion, optimizer, n_acc_steps, device, mode, old_dict,switch_weights_sign,weights_file)
            acc=test(epoch, net, testloader, criterion, device)

        acc_list.append(acc)
    print('Best accuracy: ',np.max(acc_list))



    sign="done"
    return sign




if __name__ == '__main__':
    dataset = str(sys.argv[1])
    print('dataset: ',dataset)
    epoch=[1,3,10,20,50,100]
    epochs=10
    lr = 1e-3
    print('LR: ',lr)
    mode = 'normal_clip' # normal_clip, mixed_ghost_clip
    sign = fine_tune(dataset, eps=1, epochs=epochs,LR=lr,mode=mode)



