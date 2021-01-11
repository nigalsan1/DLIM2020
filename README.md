# DLIM2020

## Preview

Here you get a little teaser of what you can reproduce using the default settings.
![](Results/preview_final.jpg)

## Description
An exploration of attribute-based face manipulation, using a pre-existing network structure as shown further below. The goal is that given an input image  of a face, the network should generate an image of the same face while only changing an attribute such as hair color, gender, age, etc.

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
After you have cloned the repository to your hard drive, you can either train a new network, or test using an existing network. Either way, make sure you activate your conda environment before proceeding.

### Training your own Network
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

Since the network is already trained, it is not necessary to run the code on the cluster. Therefore, simply enter the command

```bash
python -u main.py "$@" --mode test --dataset CelebA --image_size 128 
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --celeba_image_dir ../Datasets/celeba/images \
               --attr_path ../Datasets/celeba/list_attr_celeba.txt\
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --c_dim 5         
```

For StarGAN with different instance normalization, you need to change the attributes so that they are mutually exclusive. There are a few available combinations such as male/female, or beard/no beard. We have trained our models with hair color, so if you wish to let it run, change the last line to 

```bash
--selected_attrs Black_Hair Blond_Hair Brown_Hair Gray_Hair --c_dim 4
```

Your results will be saved into `[Choose StarGAN]/stargan_celeba/result`.

## Analysis of our training sets

First lets have a look over the default trainigsets (attributes: Black Hair, Blonde Hair, Brown Hair, Male, Young) and default batchsize was for us 16.
We can see that our results look very well trained for some of the input pictures. But their are some exceptions. 
Let's have a look over the standard training attribute how we declaired whats default for us.

- *Image*

In this trainingset their are more women than men and therefore you see that some attributes are changing with some kind of weird behavier.

<!---Explanation of Blonde Hair attribute to faces-->
In these images you see from left to the right the attributes (Black, Blond, Brown, Gray Hair). We see that the feature with black and brown hair was pretty accurate and got some really good fake images that we almost can't tell the difference between generated and real picture. But then we got to see some new kind of behavier due to the dataset. With the blond attribute it started to lighten up the skin of the target person and made the lips redder and fuller. We suggested that it could corralate with the input images because it took randomly more female and this is like a stereotype of a blonde person.

<!--- Gray Hair attribute impact on faces-->
Additionally we also saw with the attribute gray hair that the face of the person gets older. It added some wrinkles to the face to make it more believable. Also it made the eyes smaller like a typical old people thing.

<!--- Bold guy impact-->
It's funny to see how it trys to create hair for bold people. It could handle that their exists people with no hair. Also it added some of the of the told behavier with lightening up the skin and making the eyes smaller.

<!--- Hat dude aka Zack and Cody gone wild-->
If you get some corrupted image like in the following one you see that it can't see the difference between what it should look like. It adds on the corrupeted data also the features.

<!--- Hat dude aka Zack and Cody gone wild-->
The next interesting part is with some persons who has accessoirs like a hat. It interprets that the persons haircuts is with the hat. So it changes the hat each time when some attributes are applyied.

Theses all observation we did, was on one trained network at the 200000 iteration. It was very interesting to watch how different it muted in one batch.

<!--- Batchsize-->
We also ...

<!--- Discriminator learning rate -->

<!--- Generator learning rate -->

<!--- Hingeloss -->

<!--- instancenorm -->

<!--- Discriminator learning rate -->



## Testing with the same dataset

<!--- Some comments to tests between the tests-->
We also was able to manage to compare the different trained models on the same dataset. So we can compare different training results with each other.

<!--- default for comparison between all of them -->

<!--- four images next to each other for the different batchsize and a gif -->

<!--- four images next to each other for the different learning rate -->

<!--- four images next to each other for the different hingeloss instancenorm and default (wasserstein)-->


## Using a webcam as input to the neural network
using the opencv library to recognize a face, crop it, resize and feed it to the neural network

## Some results

![](Results/Gif/Batchsize_8.gif) \
In the following Gif you get a idea of how the system is evolving through these 200000 iterations. Each images is made after 1000 iterations. We choose here a batchsize of 8 and the attributes black hair, blond hair, brown hair, male and young (from left to right, where the first image is the input.)

![](Results/Gif/D_lr_0.0005.gif) \
Here you see a really bad result of the training. We set the learning rate for the discriminator 50-times higher than the one for the generator. We concluded that the discriminator performing way better than the generator and you see it in the short gif.


## Acknowledgements

This work is just for study purpose. We were a group of three electrical engineer which are doing this just for fun

##
