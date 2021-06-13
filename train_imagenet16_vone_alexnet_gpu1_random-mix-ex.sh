# random-mix-ex
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 4 -n vone_alexnet_random-mix_s00-04_no-blur-1label --excluded_labels 15;
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 4 -n vone_alexnet_random-mix_s00-04_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix_no-sharp --min_sigma 0 --max_sigma 4 -n vone_alexnet_random-mix_s00-04_no-sharp-1label --excluded_labels 15;
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix_no-sharp --min_sigma 0 --max_sigma 4 -n vone_alexnet_random-mix_s00-04_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 2 -n vone_alexnet_random-mix_s00-02_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 8 -n vone_alexnet_random-mix_s00-08_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 16 -n vone_alexnet_random-mix_s00-16_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
