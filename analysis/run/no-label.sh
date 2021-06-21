cd ../lowpass_acc
#python compute_lowpass_acc.py vone_alexnet 16 imagenet16 vss
python compute_plot_lowpass_confusion_matrix.py vone_alexnet 16 imagenet16 mix_no-blur gpu1
python compute_plot_lowpass_confusion_matrix.py vone_alexnet 16 imagenet16 mix_no-sharp gpu1
python compute_plot_lowpass_confusion_matrix.py vone_alexnet 16 imagenet16 random-mix_no-blur gpu1
python compute_plot_lowpass_confusion_matrix.py vone_alexnet 16 imagenet16 random-mix_no-sharp gpu1
python compute_plot_lowpass_confusion_matrix.py alexnet 16 imagenet16 mix_no-blur gpu1
python compute_plot_lowpass_confusion_matrix.py alexnet 16 imagenet16 mix_no-sharp gpu1
python compute_plot_lowpass_confusion_matrix.py alexnet 16 imagenet16 random-mix_no-blur gpu1
python compute_plot_lowpass_confusion_matrix.py alexnet 16 imagenet16 random-mix_no-sharp gpu1

python compute_plot_lowpass_confusion_matrix.py vone_alexnet 16 imagenet16 mix_no-blur_seed0 gpu1
python compute_plot_lowpass_confusion_matrix.py alexnet 16 imagenet16 mix_no-blur_seed0 gpu1

#cd ../bandpass_acc
#python compute_bandpass_acc.py vone_alexnet 16 imagenet16 vss gpu1
#python compute_bandpass_confusion_matrix.py vone_alexnet 16 imagenet16 vss gpu1

#cd ../rsa/bandpass/rsm
#python compute_plot_bandpass_rsm.py vone_alexnet 16 vss gpu1