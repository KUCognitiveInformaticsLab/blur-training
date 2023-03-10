# should be gpu1

#cd ../lowpass_acc
#python compute_lowpass_acc.py vone_alexnet 16 imagenet16 all-blur-training gpu1
#python compute_plot_lowpass_confusion_matrix.py vone_alexnet 16 imagenet16 all-blur-training gpu1

cd ../shape-bias
python compute_shape_bias.py vone_alexnet 16 all-blur-training

cd ../jumbled_gray_occluder
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 jumbled 4 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 jumbled 8 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 jumbled 16 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 jumbled 32 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 gray_occluder 4 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 gray_occluder 8 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 gray_occluder 16 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 gray_occluder 32 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 jumbled_with_gray_occluder 4 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 jumbled_with_gray_occluder 8 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 jumbled_with_gray_occluder 16 all-blur-training
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vone_alexnet 16 imagenet16 jumbled_with_gray_occluder 32 all-blur-training

#cd ../bandpass_acc gpu1
#python compute_bandpass_acc.py vone_alexnet 16 imagenet16 all-blur-training gpu1
#python compute_bandpass_confusion_matrix.py vone_alexnet 16 imagenet16 all-blur-training gpu1
#
#cd ../rsa/bandpass/rsm
#python compute_plot_bandpass_rsm.py vone_alexnet 16 all-blur-training