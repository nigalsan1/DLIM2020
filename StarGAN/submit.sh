#!/bin/bash
#SBATCH --output=log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --constraint=titan_x
source /usr/itetnas04/data-scratch-01/dlim_08hs20/data/conda/etc/profile.d/conda.sh
conda activate dlim
python -u main.py "$@" --mode train --dataset CelebA --image_size 128 --c_dim 5 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_step 2500 --g_lr 0.0005 \
               --d_lr 0.0005 --num_iters 50000 --num_iters_decay 30000