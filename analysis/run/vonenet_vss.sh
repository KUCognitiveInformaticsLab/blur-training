cd ../lowpass_acc
python compute_lowpass_acc.py vone_alexnet 16 imagenet16 vss
python compute_plot_lowpass_confusion_matrix.py vone_alexnet 16 imagenet16 vss

cd ../bandpass_acc
python compute_bandpass_acc.py vone_alexnet 16 imagenet16 vss
python compute_bandpass_confusion_matrix.py vone_alexnet 16 imagenet16 vss

cd ../rsa/bandpass/rsm
python compute_plot_bandpass_rsm.py vone_alexnet 16 vss