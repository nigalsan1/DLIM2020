python -u main.py "$@" --mode test --dataset RaFD --rafd_crop_size 256 --image_size 256 \
               --c_dim 5 --rafd_image_dir data/RaFD/test/happy \
               --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
               --model_save_dir stargan_custom/models --result_dir stargan_custom/results
               