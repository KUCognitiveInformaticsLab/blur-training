# multi-steps
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps -n vone_alexnet_multi-steps

python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 4 -n vone_alexnet_random-mix_s00-04
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 2 -n vone_alexnet_random-mix_s00-02
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 8 -n vone_alexnet_random-mix_s00-08
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 16 -n vone_alexnet_random-mix_s00-16
