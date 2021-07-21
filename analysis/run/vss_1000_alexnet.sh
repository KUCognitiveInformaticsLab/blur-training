#cd ../lowpass_acc
#python compute_lowpass_acc.py alexnet 1000 imagenet1000 vss
#python compute_plot_lowpass_confusion_matrix.py alexnet 1000 imagenet1000 vss

#cd ../shape-bias
#python compute_shape_bias.py alexnet 1000 vss

cd ../bandpass_acc
python compute_bandpass_acc.py alexnet 1000 imagenet1000 vss gpu2
python compute_bandpass_confusion_matrix.py alexnet 1000 imagenet1000 vss gpu2

cd ../jumbled_gray_occluder
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 jumbled 4 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 jumbled 8 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 jumbled 1000 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 jumbled 32 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 gray_occluder 4 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 gray_occluder 8 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 gray_occluder 1000 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 gray_occluder 32 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 jumbled_with_gray_occluder 4 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 jumbled_with_gray_occluder 8 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 jumbled_with_gray_occluder 1000 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py alexnet 1000 imagenet1000 jumbled_with_gray_occluder 32 vss

# not implemented yet
#cd ../rsa/rsm
#python compute_plot_bandpass_rsm.py alexnet 1000 vss
