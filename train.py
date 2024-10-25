import sys
sys.path.insert(0, '/content/private_CNN')
from tqdm import tqdm
import torch
import numpy as np


def switch_weights(new_weights, net, epoch):
    # save_dict=np.load("save_weights.npy",allow_pickle=True).item()
    # save_dict = np.load("save_grad_index.npy", allow_pickle=True).item()
    if epoch <= 200:
        save_dict = np.load('save_weights_random_5%_index.npy', allow_pickle=True).item()
        # save_dict = np.load("save_weights_th1.npy", allow_pickle=True).item()
    else:
        save_dict = np.load('save_weights_random_5%_index.npy', allow_pickle=True).item()
    # save_dict = np.load("save_weights_random_3.npy", allow_pickle=True).item()
    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'cls_token' in name or 'pos_embed' in name:
                param.requires_grad = False
            else:
                if 'weight' in name:

                    index_name = name[7:]
                    index = save_dict[index_name]
                    new_para = new_weights[name]
                    # print(index_name)
                    for i in index:
                        if 'fc2' in name or 'projs.1.2' in name or 'revert_projs.0.2.weight' in name or 'head.' in name:
                            big = i[1]
                            small = i[0]
                            new_index = np.zeros(2)
                            if small<big:
                                new_index[0] = small
                                new_index[1] = small
                            else:
                                new_index[0] = big
                                new_index[1] = big
                            # if big > small:
                            #     new_index[0] = small
                            #     new_index[1] = small
                            # else:
                            #     new_index = i

                            param[new_index] = new_para[new_index]
                        else:
                            param[i] = new_para[i]

    return net

def switch_st_weights(new_weights, net, epoch,weight_file):
    # save_dict=np.load("save_weights.npy",allow_pickle=True).item()
    # save_dict = np.load("save_grad_index.npy", allow_pickle=True).item()
    if epoch <= 200:
        save_dict = np.load(weight_file, allow_pickle=True).item()
        # save_dict = np.load("save_weights_th1.npy", allow_pickle=True).item()
    else:
        save_dict = np.load(weight_file, allow_pickle=True).item()
    # save_dict = np.load("save_weights_random_3.npy", allow_pickle=True).item()


    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'cls_token' in name or 'pos_embed' in name:
                param.requires_grad = False
            else:
                if 'weight' in name:
                    # if 'ST' in weight_file:
                    #     index_name = name[7:]
                    # else:
                    #     index_name = name
                    index_name = name
                    new_para = new_weights[name]
                    # print(index_name)
                    if 'head.' in name:
                        param = new_para
                    else:
                        index = save_dict['module.'+index_name]
                        for i in index:
                            if len(i) == 1:
                                param[i] = new_para[i]
                            if len(i) == 2:
                                f = i[0]
                                s = i[1]
                                param[f][s] = new_para[f][s]

                            if len(i) == 3:
                                f = i[0]
                                s = i[1]
                                t = i[2]
                                param[f][s][t] = new_para[f][s][t]
                            if len(i) == 4:
                                f = i[0]
                                s = i[1]
                                t = i[2]
                                f4 = i[3]
                                param[f][s][t][f4] = new_para[f][s][t][f4]




    return net


def pt_train(new_weights, net, epoch,weight_file):
    le = len(weight_file)
    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'cls_token' in name or 'pos_embed' in name:
                param.requires_grad = False
            else:
                new_para = new_weights[name]
                for i in range(le):
                    bn = weight_file[i]
                    block_name = 'blocks.' + str(bn)
                    if block_name in name:
                        param = new_para
                if 'head.' in name:
                    param = new_para
    return net


def train(epoch, net,trainloader,criterion,optimizer,n_acc_steps,device,mode,old_dict,switch_weights_sign,weight_file):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # with BatchMemoryManager(data_loader=trainLoader, max_physical_batch_size=500,
    #                         optimizer=optimizer) as new_data_loader:

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        targets = targets.squeeze().long()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if mode == 'non-private':
            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()
        else:
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step(loss=loss)
                if switch_weights_sign==True:
                    if isinstance(weight_file, str):
                        new_weights = net.state_dict()
                        net.load_state_dict(old_dict, strict=False)
                        net = switch_st_weights(new_weights, net, epoch, weight_file)
                    else:
                        new_weights = net.state_dict()
                        net.load_state_dict(old_dict, strict=False)
                        net=pt_train(new_weights, net, epoch, weight_file)


                optimizer.zero_grad()
            else:
                optimizer.virtual_step(loss=loss)
        train_loss += loss.mean().item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(epoch, len(trainloader), 'Training Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))



def isic_train(epoch, net,trainloader,criterion,optimizer,n_acc_steps,device,mode,old_dict,switch_weights_sign,weight_file):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # with BatchMemoryManager(data_loader=trainLoader, max_physical_batch_size=500,
    #                         optimizer=optimizer) as new_data_loader:

    for batch_idx, (data, targets) in enumerate(tqdm(trainloader)):
        inputs, targets = data.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if mode == 'non-private':
            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()
        else:
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step(loss=loss)

                if switch_weights_sign == True:
                    if isinstance(weight_file, str):
                        new_weights = net.state_dict()
                        net.load_state_dict(old_dict, strict=False)
                        net = switch_st_weights(new_weights, net, epoch, weight_file)
                    else:
                        new_weights = net.state_dict()
                        net.load_state_dict(old_dict, strict=False)
                        net = pt_train(new_weights, net, epoch, weight_file)

                optimizer.zero_grad()
            else:
                optimizer.virtual_step(loss=loss)
        train_loss += loss.mean().item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        _, label_target = targets.max(1)
        correct += predicted.eq(label_target).sum().item()

    print(epoch, len(trainloader), 'Training Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def cheast_train(epoch, net,trainloader,criterion,optimizer,n_acc_steps,device,mode,old_dict,switch_weights_sign,weight_file):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # with BatchMemoryManager(data_loader=trainLoader, max_physical_batch_size=500,
    #                         optimizer=optimizer) as new_data_loader:

    for batch_idx, sample in enumerate(tqdm(trainloader)):

        inputs, targets = sample['img'].to(device), sample['lab'].to(device)
        # inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if mode != 'non-private':
            loss = loss.sum(dim=1)
        #

        if mode == 'non-private':
            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()
        else:
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step(loss=loss)

                if switch_weights_sign == True:
                    if isinstance(weight_file, str):
                        new_weights = net.state_dict()
                        net.load_state_dict(old_dict, strict=False)
                        net = switch_st_weights(new_weights, net, epoch, weight_file)
                    else:
                        new_weights = net.state_dict()
                        net.load_state_dict(old_dict, strict=False)
                        net = pt_train(new_weights, net, epoch, weight_file)

                optimizer.zero_grad()
            else:
                optimizer.virtual_step(loss=loss)
        train_loss += loss.mean().item()
        pred= outputs.ge(0.5)
        correct += sum(row.all().int().item() for row in (outputs.ge(0.5) ==targets))

        # _, predicted = outputs.max(1)
        total += targets.size(0)
        # _,label_target=targets.max(1)
        # correct += predicted.eq(label_target).sum().item()

    print(epoch, len(trainloader), 'Training Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def chest_test(epoch, net,testloader,criterion,device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc=0.0
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(testloader)):
            inputs, targets = sample['img'].to(device), sample['lab'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # loss = loss.sum(dim=1)
            loss = loss.mean()
            test_loss += loss.item()
            correct += sum(row.all().int().item() for row in (outputs.ge(0.5) == targets))
            # _, predicted = outputs.max(1)
            total += targets.size(0)
            # _, label_target = targets.max(1)
            # correct += predicted.eq(label_target).sum().item()

        print(epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        acc = correct/total
    return acc

def test(epoch, net,testloader,criterion,device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc =0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            targets = targets.squeeze().long()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss = loss.mean()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        acc = correct / total
    return acc

def isic_test(epoch, net,testloader,criterion,device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc = 0.0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(testloader)):
            inputs, targets = data.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss = loss.mean()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            _, label_target = targets.max(1)
            correct += predicted.eq(label_target).sum().item()

        print(epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        acc = correct / total
    return acc