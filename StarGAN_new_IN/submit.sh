#!/bin/bash
#SBATCH --output=log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --constraint=titan_x
python -u main.py "$@" --mode train --dataset CelebA --image_size 128 --c_dim 4 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --log_step 50 --selected_attrs Black_Hair Blond_Hair Brown_Hair Gray_Hair\
               --celeba_image_dir ../Datasets/celeba/images \
               --attr_path ../Datasets/celeba/list_attr_celeba.txt