# normal
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_vone_alexnet --mode normal -n vone_vone_alexnet_normal

# all
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode all -s 1 -n vone_alexnet_all_s1
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode all -s 2 -n vone_alexnet_all_s2
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode all -s 3 -n vone_alexnet_all_s3
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode all -s 4 -n vone_alexnet_all_s4

# mix
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 1 -n vone_alexnet_mix_s1
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 2 -n vone_alexnet_mix_s2
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 3 -n vone_alexnet_mix_s3
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode mix -s 4 -n vone_alexnet_mix_s4

# multi-steps
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps -n vone_alexnet_multi-steps

# mix_no-blur
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix -s 4 -n vone_alexnet_mix_s04_no-blur-1label --excluded_labels 15;
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix -s 4 -n vone_alexnet_mix_s04_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix -s 1 -n vone_alexnet_mix_s01_no-blur-1label --excluded_labels 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix -s 1 -n vone_alexnet_mix_s01_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix -s 2 -n vone_alexnet_mix_s02_no-blur-1label --excluded_labels 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix -s 2 -n vone_alexnet_mix_s02_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix -s 3 -n vone_alexnet_mix_s03_no-blur-1label --excluded_labels 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix -s 3 -n vone_alexnet_mix_s03_no-blur-8label --excluded_labels 8 9 10 11 12 13 14 15;

# mix_no-sharp
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix_no-sharp -s 4 -n vone_alexnet_mix_s04_no-sharp-1label --excluded_labels 15;
python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix_no-sharp -s 4 -n vone_alexnet_mix_s04_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix_no-sharp -s 1 -n vone_alexnet_mix_s01_no-sharp-1label --excluded_labels 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix_no-sharp -s 1 -n vone_alexnet_mix_s01_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix_no-sharp -s 2 -n vone_alexnet_mix_s02_no-sharp-1label --excluded_labels 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix_no-sharp -s 2 -n vone_alexnet_mix_s02_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix_no-sharp -s 3 -n vone_alexnet_mix_s03_no-sharp-1label --excluded_labels 15;
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --mode mix_no-sharp -s 3 -n vone_alexnet_mix_s03_no-sharp-8label --excluded_labels 8 9 10 11 12 13 14 15;

# single-step
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode single-step -s 1 -n vone_alexnet_single-step_s1
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode single-step -s 2 -n vone_alexnet_single-step_s2
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode single-step -s 3 -n vone_alexnet_single-step_s3
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode single-step -s 4 -n vone_alexnet_single-step_s4

# fixed-single-step
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode fixed-single-step -s 1 -n vone_alexnet_fixed-single-step_s1
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode fixed-single-step -s 2 -n vone_alexnet_fixed-single-step_s2
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode fixed-single-step -s 3 -n vone_alexnet_fixed-single-step_s3
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode fixed-single-step -s 4 -n vone_alexnet_fixed-single-step_s4

# reversed-single-step
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode reversed-single-step -s 1 -n vone_alexnet_reversed-single-step_s1
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode reversed-single-step -s 2 -n vone_alexnet_reversed-single-step_s2
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode reversed-single-step -s 3 -n vone_alexnet_reversed-single-step_s3
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode reversed-single-step -s 4 -n vone_alexnet_reversed-single-step_s4

# multi-steps-cbt
## cbt-rate 0.9
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps-cbt --init-s 1 --cbt-rate 0.9 -n vone_alexnet_multi-steps-cbt_decay9e-1_init-s1
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps-cbt --init-s 2 --cbt-rate 0.9 -n vone_alexnet_multi-steps-cbt_decay9e-1_init-s2
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps-cbt --init-s 3 --cbt-rate 0.9 -n vone_alexnet_multi-steps-cbt_decay9e-1_init-s3
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps-cbt --init-s 4 --cbt-rate 0.9 -n vone_alexnet_multi-steps-cbt_decay9e-1_init-s4
## cbt-rate 0.8
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps-cbt --init-s 1 --cbt-rate 0.8 -n vone_alexnet_multi-steps-cbt_decay8e-1_init-s1
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps-cbt --init-s 2 --cbt-rate 0.8 -n vone_alexnet_multi-steps-cbt_decay8e-1_init-s2
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps-cbt --init-s 3 --cbt-rate 0.8 -n vone_alexnet_multi-steps-cbt_decay8e-1_init-s3
#python src/train_imagenet1000.py --log_dir /mnt/data/pretrained_models/blur-training/imagenet1000 --arch vone_alexnet --mode multi-steps-cbt --init-s 4 --cbt-rate 0.8 -n vone_alexnet_multi-steps-cbt_decay8e-1_init-s4