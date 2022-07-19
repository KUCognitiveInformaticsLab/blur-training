# [arxiv用] 同意(consent)をもらった人間のみの数値に更新し、プロットし直したログ
cd ../lowpass_acc
python plot_lowpass_acc_with_consent_human_data.py alexnet 16 imagenet16 acc1

cd ../bandpass_acc
python plot_bandpass_acc_with_consent_human_data.py alexnet 16 imagenet16 acc1
