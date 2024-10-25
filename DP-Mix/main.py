
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100,EuroSAT,Caltech256,SUN397,OxfordIIITPet
import torch.nn as nn
import torch.optim as optim
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.model_selection import train_test_split
from src.opacus_augmented.privacy_engine_augmented import PrivacyEngineAugmented
from opacus import GradSampleModule
from src.models.prepare_models import  prepare_augmult_cifar
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader, Subset
import open_clip
from clip_model import CLIP_model
import argparse
import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import matplotlib.pyplot as plt

def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)

def onehot(label, n_classes):
    return torch.zeros(1, n_classes).scatter_(
        1, label.view(-1, 1), 1)
def true_mixup(data,data2, targets,targets2, alpha, n_classes,one_hot_sign=True):

    if one_hot_sign==True:
        targets = onehot(targets, n_classes)
        targets2 = onehot(targets2, n_classes)



    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets



class Data_WithID(Dataset):
    def __init__(self, dataset, extra_data_path, text_image_data_path, k, shuffled_ids, transform, keep_original=True,
                 model_name='vit', aug_signal=False, nb_aug=16, inner_aug=False, nb_text=0, nb_random=0, nb_classes=100,convex=2,mixup_num=16):
        self.dataset = dataset

        self.nb_random = nb_random
        self.indices = list(range(len(dataset)))
        self.extra_data_path = extra_data_path
        self.text_image_data_path = text_image_data_path
        self.k = k  # num of diff data
        self.shuffled_ids = shuffled_ids
        self.transform = transform
        self.keep_original = keep_original
        self.to_pil_image = transforms.ToPILImage()
        self.model_name = model_name
        self.aug_signal = aug_signal  # true then it will self_aug some data
        self.nb_aug = nb_aug  # # of self_aug
        self.inner_aug = inner_aug  # true then diff_data will be transformed
        self.nb_text = nb_text  # # of text2image data
        self.mixup_num = mixup_num
        self.true_len_mixup_num=k + nb_text + nb_aug + nb_random
        self.convex = convex
        self.nb_classes = nb_classes

        if self.nb_text:
            self.txt2img_files = os.listdir(text_image_data_path)

    def plot_images(self, images):
        for i, img in enumerate(images, 1):
            plt.subplot(1, len(images), i)
            plt.imshow(self.to_pil_image(img))
            plt.axis('off')
        plt.show()

    def __getitem__(self, index):
        if self.model_name == 'vit':
            image_size = 224
        else:
            image_size = 256
        transform = transforms.Compose(
            [transforms.RandomCrop(size=(image_size, image_size), padding=4, padding_mode="reflect"),
             transforms.RandomHorizontalFlip(p=0.5), ])
        original_index = self.shuffled_ids[index]  # Use the shuffled ID to get the original index
        img, target = self.dataset[original_index]
        img = torch.unsqueeze(img, 0)

        if self.k:

            formatted_original_index = "{:06d}".format(original_index)
            extra_data_folder = os.path.join(self.extra_data_path, str(formatted_original_index))
            image_files = glob.glob(os.path.join(extra_data_folder, '*.png'))
            image_files = random.sample(image_files, self.k)

            extra_data_list = []
            for image_file in image_files:
                extra_data = Image.open(image_file)
                extra_data = self.transform(extra_data)  # Apply the same transform as the original dataset
                extra_data_list.append(extra_data)
            extra_data_list = torch.stack(extra_data_list)

            extra_data_list = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(extra_data_list)

        if self.nb_text:
            text_labels = []
            text_images = []

            # Randomly choose nb_img files
            chosen_files = random.sample(self.txt2img_files, self.nb_text)
            for file in chosen_files:
                # Extract the class label from the filename
                label = int(file.split('_')[-1].split('.')[0])
                text_labels.append(label)

                # Load the image and convert to a numpy array
                text_img = Image.open(os.path.join(self.text_image_data_path, file))
                img_array = self.transform(text_img)
                text_images.append(img_array)
            text_images = torch.stack(text_images)
            if self.nb_aug:
                text_images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(text_images)


            text_labels = torch.tensor(text_labels)

        if self.nb_aug:
            images_duplicates = torch.repeat_interleave(img, repeats=self.nb_aug, dim=0)

            images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(images_duplicates)
        else:
            images = img

        if self.k:
            output = torch.cat((images, extra_data_list), dim=0)
            target = torch.tensor([target] * output.size()[0])
            if self.nb_text:
                output = torch.cat((output, text_images), dim=0)
                target = torch.cat((target, text_labels), dim=0)

        else:
            output = images
            target = torch.tensor([target] * output.size()[0])
            if self.nb_text:

                output = torch.cat((output, text_images), dim=0)
                target = torch.cat((target, text_labels), dim=0)


        # mixup
        one_hot_target = torch.zeros((target.size()[0], self.nb_classes))
        for t in range(target.size()[0]):
            one_hot_target[t] = onehot(target[t], self.nb_classes)
        target = one_hot_target
        X_fake_data = output.clone()
        y_fake_data = target.clone()
        for i in range(self.mixup_num):
            if self.convex==2:
                j = np.random.randint(0, self.true_len_mixup_num - 1, size=1)
                t = np.random.randint(0, self.true_len_mixup_num - 1, size=1)
                mix_data, mix_target = true_mixup(X_fake_data[t], X_fake_data[t], y_fake_data[t],
                                                  y_fake_data[j],
                                                  alpha=0.2, n_classes=self.nb_classes, one_hot_sign=False)
            mix_data = torch.reshape(mix_data, (1, 3, image_size, image_size))
            mix_target = torch.reshape(mix_target, (1, self.nb_classes))
            output = torch.cat((output, mix_data))
            target = torch.cat((target, mix_target))

        output = torch.reshape(output, (self.mixup_num+self.true_len_mixup_num, 3, image_size, image_size))



        return output, target, original_index

    def __len__(self):
        return len(self.dataset)

def accuracy(preds, labels):
    return (preds == labels).mean()



def main(model_name='vit',dataset='cifar-10',training_method='baseline',n=16,fine_tune_whole=False):
    print('K= ',K)
    print('eps = ',EPSILON)
    print('Batch size: ',BATCH_SIZE)
    print('Norm: ',MAX_GRAD_NORM)

    if model_name  == 'vit':
        encoder, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        print('Using ViT')
    else:

        if TRAINING_SCRATCH:
            encoder, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w',
                                                                           pretrained=None)
            print('Using ConvNext from scratch')
            fine_tune_whole=True
        else:
            encoder, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w',
                                                                           pretrained='laion2b_s13b_b82k')
            print('Using ConvNext pretrained')





    if dataset == 'cifar-10':
        print('Using CIFAR-10')
        DATA_ROOT = 'data/cifar10'
        extra_data_path = DATA_ROOT + '/cifar-10_img2img_0.5'
        text_data_path = DATA_ROOT + '/cifar-10_txt2img'

        train_dataset = CIFAR10(
            root=DATA_ROOT, train=True, download=True, transform=preprocess)
        torch.manual_seed(42)  # Set the random seed for reproducibility
        shuffled_ids = torch.randperm(len(train_dataset), out=torch.LongTensor()).tolist()
        torch.seed()

        trainset_with_id = Data_WithID(train_dataset, extra_data_path, text_data_path, NDIFF, shuffled_ids,
                                       preprocess, model_name=model_name, aug_signal=True, nb_aug=K,
                                       inner_aug=True, nb_text=NB_TEXT, nb_classes=10,nb_random=NB_RANDOM,convex=NB_CONVEX,mixup_num=NB_MIXUP)
        train_loader = DataLoader(trainset_with_id, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

        test_dataset = CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=preprocess)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES = 10
    elif dataset == 'cifar-100':
        print('Using CIFAR-100')
        DATA_ROOT = 'data/cifar100'
        extra_data_path = DATA_ROOT + '/cifar-100_img2img_0.8'
        text_data_path = DATA_ROOT + '/cifar-100_txt2img'

        train_dataset = CIFAR100(
            root=DATA_ROOT, train=True, download=True, transform=preprocess)


        torch.manual_seed(42)  # Set the random seed for reproducibility
        shuffled_ids = torch.randperm(len(train_dataset), out=torch.LongTensor()).tolist()
        torch.seed()


        trainset_with_id = Data_WithID(train_dataset, extra_data_path,text_data_path,NDIFF,shuffled_ids,
                                       preprocess,model_name=model_name,aug_signal=True,nb_aug=K,
                                       inner_aug=True,nb_text=NB_TEXT,nb_classes=100,nb_random=NB_RANDOM,convex=NB_CONVEX,mixup_num=NB_MIXUP)

        train_loader = DataLoader(trainset_with_id, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        test_dataset = CIFAR100(
            root=DATA_ROOT, train=False, download=True, transform=preprocess)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES = 100
    elif dataset == 'eurosat':
        print('Using EuroSAT')
        DATA_ROOT = 'data/eurosat'
        extra_data_path = DATA_ROOT + '/eurosat_img2img'
        text_data_path = DATA_ROOT + '/eurosat_txt2img'

        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        train_dataset = EuroSAT(
            root=DATA_ROOT,  download=True, transform=preprocess)

        train_indices, test_indices, _, _ = train_test_split(
            range(len(train_dataset)),
            train_dataset.targets,
            stratify=train_dataset.targets,
            test_size=int(len(train_dataset)*0.2),
            random_state=42
        )
        train_split = Subset(train_dataset, train_indices)
        test_split = Subset(train_dataset, test_indices)

        torch.manual_seed(42)  # Set the random seed for reproducibility
        shuffled_ids = torch.randperm(len(train_split), out=torch.LongTensor()).tolist()
        torch.seed()

        train_dataset = train_split

        trainset_with_id = Data_WithID(train_dataset,extra_data_path, text_data_path, NDIFF, shuffled_ids,
                                       preprocess, model_name=model_name, aug_signal=True, nb_aug=K,
                                       inner_aug=True, nb_text=NB_TEXT, nb_classes=10 ,nb_random=NB_RANDOM,convex=NB_CONVEX,mixup_num=NB_MIXUP)

        train_loader = DataLoader(trainset_with_id, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


        test_loader = torch.utils.data.DataLoader(
            test_split,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES = 10

    elif dataset == 'caltech256':
        print('Using caltech256')
        DATA_ROOT = 'data/caltech256'
        extra_data_path = DATA_ROOT + '/caltech256_img2img'
        text_data_path = DATA_ROOT + '/caltech256_txt2img'

        train_dataset = Caltech256(
            root=DATA_ROOT, download=True, transform=preprocess)

        train_indices, test_indices = train_test_split(
            range(len(train_dataset)),train_size=int(len(train_dataset) * 0.8),
            test_size=int(len(train_dataset) * 0.2),
            random_state=42
        )
        train_split = Subset(train_dataset, train_indices)
        test_split = Subset(train_dataset, test_indices)

        torch.manual_seed(42)  # Set the random seed for reproducibility
        shuffled_ids = torch.randperm(len(train_split), out=torch.LongTensor()).tolist()
        torch.seed()

        train_dataset = train_split




        trainset_with_id = Data_WithID(train_dataset, extra_data_path, text_data_path, NDIFF,
                                       shuffled_ids,
                                       preprocess, model_name=model_name, aug_signal=True, nb_aug=K,
                                       inner_aug=True, nb_text=NB_TEXT, nb_classes=257, nb_random=NB_RANDOM,convex=NB_CONVEX,mixup_num=NB_MIXUP)

        train_loader = DataLoader(trainset_with_id, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


        test_loader = torch.utils.data.DataLoader(
            test_split,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES = 257

    elif dataset == 'sun397':
        print('Using sun397')
        DATA_ROOT = 'data/sun397'
        extra_data_path = DATA_ROOT + '/sun397_img2img'
        text_data_path = DATA_ROOT + '/sun_txt2img'
        train_dataset = SUN397(
            root=DATA_ROOT, download=True, transform=preprocess)
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

        trainset_with_id = Data_WithID(train_split, extra_data_path, text_data_path, NDIFF,
                                       shuffled_ids,
                                       preprocess, model_name=model_name, aug_signal=True, nb_aug=K,
                                       inner_aug=True, nb_text=NB_TEXT, nb_classes=397, nb_random=NB_RANDOM,convex=NB_CONVEX,mixup_num=NB_MIXUP)

        train_loader = DataLoader(trainset_with_id, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(
            test_split,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES = 397


    elif dataset == 'pet':
        print('Using pet')
        DATA_ROOT = 'data/pet'
        extra_data_path = DATA_ROOT + '/pet_img2img'
        text_data_path = DATA_ROOT + '/pet_txt2img'
        train_dataset = OxfordIIITPet(
            root=DATA_ROOT, download=True, transform=preprocess,split='trainval')


        torch.manual_seed(42)  # Set the random seed for reproducibility
        shuffled_ids = torch.randperm(len(train_dataset), out=torch.LongTensor()).tolist()
        torch.seed()

        trainset_with_id = Data_WithID(train_dataset,  extra_data_path, text_data_path, NDIFF,
                                       shuffled_ids,
                                       preprocess, model_name=model_name, aug_signal=True, nb_aug=K,
                                       inner_aug=True, nb_text=NB_TEXT, nb_classes=37, nb_random=NB_RANDOM,convex=NB_CONVEX,mixup_num=NB_MIXUP)

        train_loader = DataLoader(trainset_with_id, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        test_dataset = OxfordIIITPet(
            root=DATA_ROOT, split='test', download=True, transform=preprocess)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
        )
        NUM_CLASSES = 37

    if model_name == 'vit':
        model = CLIP_model(encoder, NUM_CLASSES, fine_tune_whole_model=fine_tune_whole,encoder_output_size=512)
    else:
        model = CLIP_model(encoder, NUM_CLASSES, fine_tune_whole_model=fine_tune_whole)


    from opacus.validators import ModuleValidator
    model = ModuleValidator.fix(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    weight_decay = 1e-5
    optimizer = optim.SGD(model.parameters(), lr=LR,weight_decay=weight_decay)
    print('lr: ', LR)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    privacy_engine = PrivacyEngineAugmented(GradSampleModule.GRAD_SAMPLERS)
    from opacus.accountants.utils import get_noise_multiplier
    sigma = get_noise_multiplier(target_delta=DELTA, target_epsilon=EPSILON,
                                 sample_rate=BATCH_SIZE / len(train_loader.dataset), epochs=EPOCHS)
    print('Compute sigma: ', sigma)

    aug_num = K + NB_TEXT + NB_MIXUP

    print('aug_num', aug_num)

    model, optimizer, fake_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=MAX_GRAD_NORM,
        poisson_sampling=True,
        K=aug_num
    )
    prepare_augmult_cifar(model, aug_num)

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
    test_best_acc = []
    for epoch in range(EPOCHS):
        diffusion_train(model, train_loader, optimizer, epoch + 1, device, ema, privacy_engine,
                        mode=training_method, n=n, model_name=model_name)

        top1_acc = test(model, test_loader, device)
        test_best_acc.append(top1_acc)

    top1_acc = test(model, test_loader, device)
    test_best_acc.append(top1_acc)
    print('Test acc:', top1_acc)
    print('Best test acc: ', np.max(test_best_acc))
    if not os.path.exists('save_models'):
        os.makedirs('save_models')
    training_num = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
    filename = f"model-{model_name}_eps-{EPSILON}_K-{K}_method-{training_method}_NDIFF-{NDIFF}_train_num-{training_num}.pth"

    torch.save(model.state_dict(), f"save_models/{filename}")
    print('Save model to: ', filename)
    return np.max(test_best_acc), test_best_acc



def diffusion_train(model, train_loader, optimizer, epoch, device,ema,privacy_engine,mode='diffusion_aug',n=16,model_name='vit'):
    model.train()
    criterion = CrossEntropyLoss()

    losses = []
    top1_acc = []
    count = 0


    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for a, data in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()

            inputs, labels, original_ids = data

            images = inputs.view(-1, *inputs.size()[2:])
            target = labels.view(-1,labels.size()[-1])

            # compute output
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            target = torch.argmax(target, dim=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)
            og_grads=[]

            loss.backward()

            if SAVE_GRADS:
                for p in model.linear_layer.parameters():
                    og = p.grad_sample.clone()
                    og_vector = og.detach().cpu().numpy().mean(axis=0)
                    og_vector = og_vector.flatten()
                    og_grads.append(og_vector)
                if count == 0:
                    epoch_og = og_grads[0]
                else:
                    epoch_og += og_grads[0]
                count += 1

            optimizer.step()

            ema.update()


        epsilon = privacy_engine.get_epsilon(DELTA)
        print(
            f"\tTrain Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
            f"(ε = {epsilon:.2f}, δ = {DELTA})"
        )
        if SAVE_GRADS:
            if not os.path.exists('grads'):
                os.makedirs('grads')
            epoch_og = epoch_og / count
            training_num = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
            filename = f"./grads/true_og-{model_name}_dataset-{dataset_name}_K-{K}_method-{training_method}_NTXT-{NB_TEXT}_train_num-{training_num}_epoch-{epoch}.npz"
            np.savez(filename, og=epoch_og)
            print('save og to', filename)





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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar-100',
                        choices=['cifar-10', 'cifar-100','eurosat','pet','sun397','caltech256'],
                        help='dataset name')
    parser.add_argument('--model', default='vit',choices=['vit', 'con'])
    parser.add_argument('--method', default='diffusion_aug')
    parser.add_argument('--ndiff', default=0, type=int)
    parser.add_argument('--nrandom', default=0, type=int)
    parser.add_argument('--ntext', default=1, type=int)
    parser.add_argument('--K', default=16, type=int)
    parser.add_argument('--eps', default=1, type=int)
    parser.add_argument('--scratch', default=False, type=str2bool)
    parser.add_argument('--lr', default=0, type=float)
    parser.add_argument('--KEEPORIGINAL', default=True, type=str2bool)
    parser.add_argument('--savegrads', default=False, type=str2bool)
    parser.add_argument('--convex', default=2, type=int)
    parser.add_argument('--ntext', default=2, type=int)
    parser.add_argument('--nmixup', default=16, type=int)

    params = vars(parser.parse_args())

    EPSILON = params['eps']
    NDIFF = params['ndiff']
    SAVE_GRADS = params['savegrads']
    print('SAVE_GRADS:', SAVE_GRADS)
    DELTA = 1e-5
    BATCH_SIZE = 1000
    MAX_PHYSICAL_BATCH_SIZE = 8
    K = params['K']
    KEEP_ORIGINAL = params['KEEPORIGINAL']
    NB_TEXT = params['ntext']
    NB_MIXUP = params['nmixup']
    NB_RANDOM = params['nrandom']
    NB_CONVEX= params['convex']

    print('K:', K)
    print('NB_TEXT:', NB_TEXT)
    print('NB_MIXUP:', NB_MIXUP)

    model_name = params['model']
    dataset_name=params['dataset']

    training_method=params['method']
    print('training_method:', training_method)
    EPOCHS = 10
    TRAINING_SCRATCH = params['scratch']

    MAX_GRAD_NORM = 1.0
    print("MAX_GRAD_NORM: ", MAX_GRAD_NORM)
    LR = params['lr']
    print('EPSILON:', EPSILON)
    print('DELTA:', DELTA)
    print('LR:', LR)
    print('EPOCHS:', EPOCHS)
    print('BATCH_SIZE:', BATCH_SIZE)
    best_acc, test_acc = main(model_name=model_name, dataset=dataset_name, training_method=training_method)

