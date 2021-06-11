# all
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode all -s 4 -n vone_alexnet_all_s04

# mix
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode mix -s 4 -n vone_alexnet_mix_s04

# multi-steps
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps -n vone_alexnet_multi-steps

# mix
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode mix -s 1 -n vone_alexnet_mix_s01
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode mix -s 2 -n vone_alexnet_mix_s02
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode mix -s 3 -n vone_alexnet_mix_s03
