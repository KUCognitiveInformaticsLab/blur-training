# mix
#python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 4 -n vone_alexnet_mix_s04
python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 1 -n vone_alexnet_mix_s01
python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 2 -n vone_alexnet_mix_s02
python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 3 -n vone_alexnet_mix_s03

# all
#python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode all -s 4 -n vone_alexnet_all_s04
python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode all -s 1 -n vone_alexnet_all_s01
python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode all -s 2 -n vone_alexnet_all_s02
python -u ../src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode all -s 3 -n vone_alexnet_all_s03
