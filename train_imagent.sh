# normal
#python -u src/train_imagenet16 --arch alexnet --mode normal -e 60 -b 64 --lr 0.01 -n alexnet_normal  # b64_lr1e-2

# all
#python -u src/train_imagenet16 --mode all -s 1 -n alexnet_all_s1
#python -u src/train_imagenet16 --mode all -s 2 -n alexnet_all_s2
#python -u src/train_imagenet16 --mode all -s 3 -n alexnet_all_s3
#python -u src/train_imagenet16 --mode all -s 4 -n alexnet_all_s4

# mix
#python -u src/train_imagenet16 --arch alexnet --mode mix -s 1 -n alexnet_mix_s1
#python -u src/train_imagenet16 --arch alexnet --mode mix -s 2 -n alexnet_mix_s2
#python -u src/train_imagenet16 --arch alexnet --mode mix -s 3 -n alexnet_mix_s3
#python -u src/train_imagenet16 --arch alexnet --mode mix -s 4 -n alexnet_mix_s4

# single-step
#python -u src/train_imagenet16 --arch alexnet --mode single-step -s 1 -n alexnet_single-step_s1
#python -u src/train_imagenet16 --arch alexnet --mode single-step -s 2 -n alexnet_single-step_s2
#python -u src/train_imagenet16 --arch alexnet --mode single-step -s 3 -n alexnet_single-step_s3
#python -u src/train_imagenet16 --arch alexnet --mode single-step -s 4 -n alexnet_single-step_s4

# fixed-single-step
#python -u src/train_imagenet16 --arch alexnet --mode fixed-single-step -s 1 -n alexnet_fixed-single-step_s1
#python -u src/train_imagenet16 --arch alexnet --mode fixed-single-step -s 2 -n alexnet_fixed-single-step_s2
#python -u src/train_imagenet16 --arch alexnet --mode fixed-single-step -s 3 -n alexnet_fixed-single-step_s3
#python -u src/train_imagenet16 --arch alexnet --mode fixed-single-step -s 4 -n alexnet_fixed-single-step_s4

# reversed-single-step
#python -u src/train_imagenet16 --arch alexnet --mode reversed-single-step -s 1 -n alexnet_reversed-single-step_s1
#python -u src/train_imagenet16 --arch alexnet --mode reversed-single-step -s 2 -n alexnet_reversed-single-step_s2
#python -u src/train_imagenet16 --arch alexnet --mode reversed-single-step -s 3 -n alexnet_reversed-single-step_s3
#python -u src/train_imagenet16 --arch alexnet --mode reversed-single-step -s 4 -n alexnet_reversed-single-step_s4

# multi-steps
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps -n alexnet_multi-steps

# multi-steps-cbt
## cbt-rate 0.9
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps-cbt --init-s 1 --cbt-rate 0.9 -n alexnet_multi-steps-cbt_decay9e-1_init-s1
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps-cbt --init-s 2 --cbt-rate 0.9 -n alexnet_multi-steps-cbt_decay9e-1_init-s2
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps-cbt --init-s 3 --cbt-rate 0.9 -n alexnet_multi-steps-cbt_decay9e-1_init-s3
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps-cbt --init-s 4 --cbt-rate 0.9 -n alexnet_multi-steps-cbt_decay9e-1_init-s4
## cbt-rate 0.8
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps-cbt --init-s 1 --cbt-rate 0.8 -n alexnet_multi-steps-cbt_decay8e-1_init-s1
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps-cbt --init-s 2 --cbt-rate 0.8 -n alexnet_multi-steps-cbt_decay8e-1_init-s2
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps-cbt --init-s 3 --cbt-rate 0.8 -n alexnet_multi-steps-cbt_decay8e-1_init-s3
#python -u src/train_imagenet16 --arch alexnet --mode multi-steps-cbt --init-s 4 --cbt-rate 0.8 -n alexnet_multi-steps-cbt_decay8e-1_init-s4