# <p align="center"> DTLC-GAN </p>

Tensorflow implementation of [DTLC-GAN:
Generative Adversarial Image Synthesis with Decision Tree Latent Controller](https://arxiv.org/abs/1805.10603).

## Usage

- Prerequisites
    - Tensorflow 1.9
    - Python 3.6

- Training
    - Important Arguments (See the others in [train.py](train.py))
        - `att`: attribute to learn (default: `''`)
        - `ks`: # of outputs of each node of each layer (default: `[2, 3, 3]`)
        - `lambdas`: loss weights of each layer (default: `[1.0, 1.0, 1.0]`)
        - `--n_d`: # of d steps in each iteration (default: `1`)
        - `--n_g`: # of g steps in each iteration (default: `1`)
        - `--loss_mode`: gan loss (choices: `[gan, lsgan, wgan, hinge]`, default: `gan`)
        - `--gp_mode`: type of gradient penalty (choices: `[none, dragan, wgan-gp]`, default: `none`)
        - `--norm`: normalization (choices: `[batch_norm, instance_norm, layer_norm, none]`, default: `batch_norm`)
        - `--experiment_name`: name for current experiment (default: `default`)
    - Example
        ```console
        CUDA_VISIBLE_DEVICES=0 \
        python train.py \
        --att Eyeglasses \
        --ks 2 3 3 \
        --lambdas 1 1 1 \
        --n_d 1 \
        --n_g 1 \
        --loss_mode hinge \
        --gp_mode dragan \
        --norm layer_norm \
        --experiment_name att{Eyeglasses}_ks{2-3-3}_lambdas{1-1-1}_continuous_last{False}_loss{hinge}_gp{dragan}_norm{layer_norm}
        ```

## Dataset

- [Celeba](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf) dataset
    - [Images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip) should be placed in ***./data/img_align_celeba/\*.jpg***
    - [Attribute labels](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt) should be placed in ***./data/list_attr_celeba.txt***
    - the above links might be inaccessible, the alternatives are
        - ***img_align_celeba.zip***
            - https://pan.baidu.com/s/1eSNpdRG#list/path=%2FCelebA%2FImg or
            - https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
        - ***list_attr_celeba.txt***
            - https://pan.baidu.com/s/1eSNpdRG#list/path=%2FCelebA%2FAnno&parentPath=%2F or
            - https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs