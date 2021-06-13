# normal
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode normal -n vone_alexnet_normal

# all
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode all -s 1 -n vone_alexnet_all_s01
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode all -s 2 -n vone_alexnet_all_s02
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode all -s 3 -n vone_alexnet_all_s03
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode all -s 4 -n vone_alexnet_all_s04

# mix
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode mix -s 1 -n vone_alexnet_mix_s01
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode mix -s 2 -n vone_alexnet_mix_s02
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode mix -s 3 -n vone_alexnet_mix_s03
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode mix -s 4 -n vone_alexnet_mix_s04

# multi-steps
python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps -n vone_alexnet_multi-steps

# single-step
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode single-step -s 1 -n vone_alexnet_single-step_s01
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode single-step -s 2 -n vone_alexnet_single-step_s02
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode single-step -s 3 -n vone_alexnet_single-step_s03
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode single-step -s 4 -n vone_alexnet_single-step_s04

# fixed-single-step
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode fixed-single-step -s 1 -n vone_alexnet_fixed-single-step_s01
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode fixed-single-step -s 2 -n vone_alexnet_fixed-single-step_s02
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode fixed-single-step -s 3 -n vone_alexnet_fixed-single-step_s03
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode fixed-single-step -s 4 -n vone_alexnet_fixed-single-step_s04

# reversed-single-step
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode reversed-single-step -s 1 -n vone_alexnet_reversed-single-step_s01
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode reversed-single-step -s 2 -n vone_alexnet_reversed-single-step_s02
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode reversed-single-step -s 3 -n vone_alexnet_reversed-single-step_s03
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode reversed-single-step -s 4 -n vone_alexnet_reversed-single-step_s04

# multi-steps-cbt
## cbt-rate 0.9
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps-cbt --init-s 1 --cbt-rate 0.9 -n vone_alexnet_multi-steps-cbt_decay9e-1_init-s01
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps-cbt --init-s 2 --cbt-rate 0.9 -n vone_alexnet_multi-steps-cbt_decay9e-1_init-s02
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps-cbt --init-s 3 --cbt-rate 0.9 -n vone_alexnet_multi-steps-cbt_decay9e-1_init-s03
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps-cbt --init-s 4 --cbt-rate 0.9 -n vone_alexnet_multi-steps-cbt_decay9e-1_init-s04
## cbt-rate 0.8
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps-cbt --init-s 1 --cbt-rate 0.8 -n vone_alexnet_multi-steps-cbt_decay8e-1_init-s01
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps-cbt --init-s 2 --cbt-rate 0.8 -n vone_alexnet_multi-steps-cbt_decay8e-1_init-s02
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps-cbt --init-s 3 --cbt-rate 0.8 -n vone_alexnet_multi-steps-cbt_decay8e-1_init-s03
#python src/train_imagenet16.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet16 --arch vone_alexnet --mode multi-steps-cbt --init-s 4 --cbt-rate 0.8 -n vone_alexnet_multi-steps-cbt_decay8e-1_init-s04