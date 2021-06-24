# mix, all (s=04)
python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 4 -n vone_alexnet_mix_s04
python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode all -s 4 -n vone_alexnet_all_s04
