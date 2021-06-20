# mix_no-blur
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch vone_alexnet -s 4 -n vone_alexnet_mix_s04_no-blur-1label --excluded_labels 15;
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch vone_alexnet -s 4 -n vone_alexnet_mix_s04_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch vone_alexnet -s 1 -n vone_alexnet_mix_s01_no-blur-1label --excluded_labels 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch vone_alexnet -s 1 -n vone_alexnet_mix_s01_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch vone_alexnet -s 2 -n vone_alexnet_mix_s02_no-blur-1label --excluded_labels 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch vone_alexnet -s 2 -n vone_alexnet_mix_s02_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch vone_alexnet -s 3 -n vone_alexnet_mix_s03_no-blur-1label --excluded_labels 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch vone_alexnet -s 3 -n vone_alexnet_mix_s03_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;

# mix_no-sharp
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 4 -n vone_alexnet_mix_s04_no-sharp-1label --excluded_labels 15; 
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 4 -n vone_alexnet_mix_s04_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 1 -n vone_alexnet_mix_s01_no-sharp-1label --excluded_labels 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 1 -n vone_alexnet_mix_s01_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 2 -n vone_alexnet_mix_s02_no-sharp-1label --excluded_labels 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 2 -n vone_alexnet_mix_s02_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 3 -n vone_alexnet_mix_s03_no-sharp-1label --excluded_labels 15;
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 3 -n vone_alexnet_mix_s03_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;

# random-mix
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 2 -n vone_alexnet_random-mix_s00-02
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 4 -n vone_alexnet_random-mix_s00-04
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 8 -n vone_alexnet_random-mix_s00-08
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix --min_sigma 0 --max_sigma 16 -n vone_alexnet_random-mix_s00-16
