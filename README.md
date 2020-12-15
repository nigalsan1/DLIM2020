# DLIM2020

## Preview

Here you get a little teaser what you can reproduce with this code per default settings.
![](Results/preview_final.jpg)

## Description
Attribute-based face manipulation
The work is in progress.

## First impression

Here you get idea of this project. You have some input image or in advance case when the network is trained you can use some noice to generate a output image with the targeted attributed. In the following datas you see some examples.

... in progress ...

## Dependencies

- [Python 3.9.0](https://www.python.org/downloads/release/python-390/)
- We suggest to use [Anaconda](https://www.anaconda.com/products/individual) for working with python.
- Also important to have is [PyTorch](https://pytorch.org/get-started/locally/). Here is the link to get better instruction how to download PyTorch.
- Below you find a linux instruction to install tensorflow.
```
pip install tensorflow -gpu
```

## Source & Dataset

From this side you can get the source code for StarGAN. You can access it directly through the following link: https://github.com/yunjey/StarGAN.git.
The code below shows you how you can get the CelebA dataset.
```
git clone https://github.com/nigalsan1/DLIM2020
cd DLIM2020/StarGan/
bash download.sh celeba #the download command for getting the dataset
```

## Pre-trained models

[Here](https://www.dropbox.com/s/fgc5wnql9o7u3sd/Models.zip?dl=0) you have some of our pretrained models. 
You can download it locally and set it as known. You can run the evaluation script below. You have to change some attributes.
```
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
