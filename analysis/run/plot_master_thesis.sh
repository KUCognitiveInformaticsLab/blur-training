cd ../lowpass_acc
python plot_lowpass_acc.py alexnet 16 imagenet16 acc1
python plot_lowpass_acc.py vgg16 16 imagenet16 acc1
python plot_lowpass_acc.py resnet50 16 imagenet16 acc1

cd ../bandpass_acc
python plot_bandpass_acc.py alexnet 16
python plot_bandpass_acc.py vgg16 16
python plot_bandpass_acc.py resnet50 16

cd ../shape-bias
python plot_shape-bias.py alexnet 16
python plot_shape-bias.py vgg16 16
python plot_shape-bias.py resnet50 16

cd ../jumbled_gray_occluder
python plot_acc_all_stimuli.py alexnet 16 imagenet16
python plot_acc_all_stimuli.py vgg16 16 imagenet16
python plot_acc_all_stimuli.py resnet50 16 imagenet16

cd ../rsa/dist
python compute_plot_dist.py --machine local --plot -a alexnet --models vss
python compute_plot_dist.py --machine local --plot -a vone_alexnet --models vss
python compute_plot_dist.py --machine local --plot -a alexnet --models vss --num_classes