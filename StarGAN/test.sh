#!/bin/bash
#SBATCH --output=log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --constraint=titan_x
source /usr/itetnas04/data-scratch-01/dlim_07hs20/data/conda/etc/profile.d/conda.sh
conda activate pytcu10
python -u main.py "$@" --mode test --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TEST_IMG_DIR \
               --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
               --model_save_dir stargan_custom/models --result_dir stargan_custom/results
               