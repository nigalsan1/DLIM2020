# DLIM2020

## Preview

Here you get a little teaser of what you can reproduce using the default settings.
![](Results/preview_final.jpg)

## Description
Attribute-based face manipulation
The work is in progress.

## First impressions

Here you get idea of this project. You have some input image or in advance case when the network is trained you can use some noice to generate a output image with the targeted attributed. In the following datas you see some examples.

... in progress ...

## Dependencies

* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)
_Note: Make sure you install the GPU version of Tensorflow (for versions 1.5+)_

We suggest using Anaconda for running this project. 
## Source 

For this project, we used the original version of the StarGAN network. If you want to have a look at it, it is also available on Github via the following link:

https://github.com/yunjey/StarGAN.git

_Note: A newer version of StarGAN with some cool additions has already been published. If you wish to view that instead, click [here](https://github.com/clovaai/stargan-v2)._


## Dataset
For training our network, we used one of the datasets from the original paper - the CelebA dataset. It consists of over 200000 celebrity images labelled with 40 different attributes.
If you want to download the dataset to train your own network with, run the following command: 
```bash
bash download.sh celeba
```
## Setup
After you have cloned the repository to your hard drive, you can either train a new network, or test using an existing network.

### Training my own Network
Before training your network, make sure you have followed the instructions given in the DLIM OneDrive for setting up the cluster (i.e installing conda and setting up SLURM).
If you want to train your own network, you will first need to download an appropriate dataset. We suggest using the CelebA dataset. 

If you want to train the network using the baseline StarGAN network, navigate into the StarGAN folder and run the command:
```
sbatch submit.sh
```
In addition to the baseline that's available in this project, we have also implemented a modified network where we replaced the depthwise concatenation of attributes with lookup tables for our instance normalization layers. 
Similarly to before, navigate into the StarGAN_new_IN folder and run the same command as before:

```bash
sbatch submit.sh
```
**Changing Network Parameters**
If you want to play around with different learning rates, batch sizes or if you just want to train the network on other attributes, open the submit.sh file inside either of the network directories. If you want to change the number of training attributes, make sure you also change c_dim to reflect that (i.e 7 attributes means c_dim = 7)
For a full list of available parameters, open either `main.py`
For a full list of available training attributes, navigate into either jpg directory and open `CelebA.md`
*Please note that with the way the new instance normalization works, the list of training attributes needs to be mutually exclusive (i.e only one of the attributes in the list applies to each picture)*

### Run a pre-trained Network

To run one of the models that we've trained, navigate to the root directory and enter the command:
```
bash download.sh models
```

Alternatively, the models are also available to download [here:](https://www.dropbox.com/s/fgc5wnql9o7u3sd/Models.zip?dl=0)

After you have downloaded the models, navigate into the wanted model directory and copy both .ckpt files into `[Choose StarGAN]/stargan_celeba/models`. 

You also need to change the run mode from train to test. Do this by changing the argument of `--mode` from `train` to `test`


## Pre-trained models

[Here](https://www.dropbox.com/s/fgc5wnql9o7u3sd/Models.zip?dl=0) you have some of our pretrained models. 
You can download it locally and set it as known. You can run the evaluation script below. You have to change some attributes.

```bash
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
               --model_save_dir 'LOCAL_DIR_MODEL' \ 
               --result_dir 'SAVE_DIR_RESULT'
```

## Some results

In progress ...

## Acknowledgements

This work is just for study purpose. We were a group of three electrical engineer which are doing this just for fun

##
