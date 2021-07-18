cd ../lowpass_acc
python compute_lowpass_acc.py vgg16 16 imagenet16 vss
python compute_plot_lowpass_confusion_matrix.py vgg16 16 imagenet16 vss

cd ../shape-bias
python compute_shape_bias.py vgg16 16 vss

cd ../jumbled_gray_occluder
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 jumbled 4 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 jumbled 8 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 jumbled 16 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 jumbled 32 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 gray_occluder 4 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 gray_occluder 8 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 gray_occluder 16 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 gray_occluder 32 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 jumbled_with_gray_occluder 4 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 jumbled_with_gray_occluder 8 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 jumbled_with_gray_occluder 16 vss
python compute_plot_confusion_matrix_and_acc_on_jumbled_images.py vgg16 16 imagenet16 jumbled_with_gray_occluder 32 vss

cd ../bandpass_acc
python compute_bandpass_acc.py vgg16 16 imagenet16 vss
python compute_bandpass_confusion_matrix.py vgg16 16 imagenet16 vss

cd ../rsa/bandpass/rsm
python compute_plot_bandpass_rsm.py vgg16 16 vss
