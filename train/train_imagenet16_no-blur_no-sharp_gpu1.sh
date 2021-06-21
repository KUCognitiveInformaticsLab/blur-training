# mix_no-sharp seed0
python ../src/train_imagenet16.py --seed 0 -n vone_alexnet_mix_s04_no-sharp-1label_seed0 --log_dir /mnt/data/pretrained_models/sharp-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 4 --excluded_labels 15;
python ../src/train_imagenet16.py --seed 0 -n vone_alexnet_mix_s04_no-sharp-8label_seed0 --log_dir /mnt/data/pretrained_models/sharp-training/imagenet16 --mode mix_no-sharp --arch vone_alexnet -s 4 --excluded_labels 8 9 10 11 12 13 14 15;

python ../src/train_imagenet16.py --seed 0 -n alexnet_mix_s04_no-sharp-1label_seed0 --log_dir /mnt/data/pretrained_models/sharp-training/imagenet16 --mode mix_no-sharp --arch alexnet -s 4 --excluded_labels 15;
python ../src/train_imagenet16.py --seed 0 -n alexnet_mix_s04_no-sharp-8label_seed0 --log_dir /mnt/data/pretrained_models/sharp-training/imagenet16 --mode mix_no-sharp --arch alexnet -s 4 --excluded_labels 8 9 10 11 12 13 14 15;

# mix_no-blur alexnet seed0
python ../src/train_imagenet16.py --seed 0 -n alexnet_mix_s04_no-blur-1label_seed0 --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch alexnet -s 4 --excluded_labels 15; 
python ../src/train_imagenet16.py --seed 0 -n alexnet_mix_s04_no-blur-8label_seed0 --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --mode mix --arch alexnet -s 4 --excluded_labels 8 9 10 11 12 13 14 15;

# random-mix-no-sharp
python ../src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch alexnet --mode random-mix_no-sharp --min_sigma 0 --max_sigma 4 -n alexnet_random-mix_s00-04_no-sharp-1label --excluded_labels 15;
python ../src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch alexnet --mode random-mix_no-sharp --min_sigma 0 --max_sigma 4 -n alexnet_random-mix_s00-04_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
python ../src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix_no-sharp --min_sigma 0 --max_sigma 4 -n vone_alexnet_random-mix_s00-04_no-sharp-1label --excluded_labels 15;
python ../src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode random-mix_no-sharp --min_sigma 0 --max_sigma 4 -n vone_alexnet_random-mix_s00-04_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
