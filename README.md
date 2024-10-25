# Codes for R+R: Towards Reliable and Generalizable Differentially Private Machine Learning

This repository contains the code for the paper "R+R: Towards Reliable and Generalizable Differentially Private Machine Learning".

Note that this repository contains several existing repositories:

- [private-CNN](https://github.com/JialinMao/private_CNN) for the implementation of the Mixed-Ghost Clipping method[12].
- [TAN Without a Burn](https://github.com/facebookresearch/tan) for the implementation of the TAN method [63] and De et al.[19] method.
- [DP-Mix](https://github.com/wenxuan-Bao/DP-Mix) for the implementation of the DP-Mix method based on Bao et al [6].
- [Not all noise is accounted equally](https://github.com/OsvaldFrisk/dp-not-all-noise-is-equal) for the implementation of the  Dormann et al. [21] method.
- [DP-RandP](https://github.com/inspire-group/DP-RandP) for the implementation of the Random process method based on Tang et al. [66].
- [Handcrafted-DP](https://github.com/ftramer/Handcrafted-DP) for the implementation of the Handcrafted feature method based on Tramèr and Boneh [68].

Please refer to the original repositories for more details for installing and environment setup.

We provide the roadmap for each file and folders in this repository.


## Roadmap

- src: contains the source code from TAN Without a Burn for self-aug[19]
- auto_clip: contains the source code with modified version of opacus for auto_clip[13]
- private_CNN: contains the source code from private-CNN for mixed-ghost clipping[12]
- DP-Mix: contains the source code from DP-Mix for Bao et al. [6]
- DP-RandP: contains the source code from the implementation of the Random process method based on Tang et al. [66].
- Handcrafted-DP: contains the source code from the implementation of the Handcrafted feature method based on Tramèr and Boneh [68].
- Hyperparameter_selection: contains the source code from Not all noise is accounted equally for the implementation of the  Dormann et al. [21] method.
- README.md: this file
- requirements.txt: the requirements for the code
- main.py: the main file for running the fine-tuning experiments
- new_wide_resnet.py: the modified version of the Wide Resnet model and include the option for ScaleNorm[40]
- seeds_hacking.py: the file for seeds hacking experiments
- self_aug_order.py: the file for training wide resnet from scratch with self-augmentation, changing the order and etc.
- random_seed.py: the file for training wide resnet from scratch with random seed for random seeds experiments
- random_seed_finetune.py: the file for fine-tuning models for random seeds experiments
- st_test.py: the file for running the statistical test
- train.py: the file for training the model
- weights_selection.py: the file for selecting the weights for the fine-tuning experiments
- WS.py: the file for the Weight Standardization method
- WS_wide_resnet.py: the file for the modified version of the wideresnet model with Weight Standardization


## Installation

create a new conda environment for the main experiments following the requirements.txt file.
```bash
conda create --name DPML python=3.8
conda activate DPML
pip install -r requirements.txt
```
Note that we use cuda 11.4.3 for experiments. Other versions may also work.

To reproduce results from other papers, please refer to the original repositories for more details for installing and environment setup.

## Usage

To run experiments for other papers, please refer to the original repositories for more details. We have collected the version of code we used in different folders.
Note that you may need to create different environments for different papers. Please refer to the original repositories for more details for installing and environment setup.

To run the training from scratch experiments in Section 5, please run the following command:
```bash
python self_aug_order.py 
```
Please refer to the file for different options.

To run the fine-tuning experiments in Section 6, please run the following command:
```bash
python main.py dataset_name method_name num
```
where dataset_name is the name of the dataset like: cifar-10, Eurosat, ISIC, pathmnist, caltech256, sun397 and pet.
The method_name is the name of fine-tuning method like:first_last,last_only, all_layers, RS%,ST% and PT%.
where 
- first_last: fine-tuning only the first and last layer based on Cattan et al.[22].
- last_only: fine-tuning only the last layer.
- all_layers: fine-tuning all layers.
- RS%: fine-tuning with random subset of model's parameters.
- ST%: fine-tuning with the subset of model's parameters selected based on Luo et al [23].
- PT%: fine-tuning with the subset of model's parameters selected based on number of blocks.

The num is the number parameters. For example, if the task is RS% and num is 10, then the model will be fine-tuned with 10% of the model's parameters.

Please refer to the file and paper Section 6.1 for more details.


To run the seeds hacking experiments in Section 4, please firstly run the following command for training 500 models from scratch with different seeds:
```bash
python random_seed.py 
```
If you want to fine-tune the models, please run the following command:
```bash
python random_seed_finetune.py 
```
For all the experiments, please refer to the file for different options like dataset and epsilons.

To run the seeds hacking experiments, please run the following command:
```bash
python seeds_hacking.py 
```
Note that you need to run the training part first and generate a table of results that each column is the results of the models with the same seed. Then you can run the seeds hacking.

To perform the statistical test, please run the following command:
```bash
python st_test.py 
```
Note that you need to fill in the array of results in the file you want to perform the statistical test.

## Computational Resources and time:
Almost all experiments require a GPU with high memory. We recommend using a GPU with at least 16GB of memory.
During our experiments, we used a single NVIDIA A100 GPU with 80 GB of memory.
Most of the experiments should take less than 24 hours to run on a single GPU.
But train 500 models with different seeds may take a long time like 1-2 weeks.
