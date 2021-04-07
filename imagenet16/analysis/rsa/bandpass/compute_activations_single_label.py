import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.compute_activations_single_label import main

if __name__ == "__main__":
    # arguments
    arch = "alexnet"
    epoch = 60
    models_dir = "/home/sou/work/blur-training/imagenet16/logs/models/"
    out_dir = f"./results/activations_single_label/{arch}/"
    test_images_dir = "./test-images"
    save_test_images = False

    all_filter_combinations = False
    # if all_filter_combinations:
    #     out_dir = f"./results/{arch}_bandpass_all_filter_comb/activations"
    # else:
    #     out_dir = f"./results/{arch}_bandpass/activations"

    # models to compare
    modes = [
        "normal",
        "all",
        "mix",
        "random-mix",
        "single-step",
        "fixed-single-step",
        "reversed-single-step",
        "multi-steps",
    ]

    # sigmas to compare
    sigmas_mix = [s for s in range(1, 6)] + [10]
    sigmas_random_mix = ["00-05", "00-10"]

    # make model name list
    model_names = []
    for mode in modes:
        if mode in ("normal", "multi-steps"):
            model_names += [f"{arch}_{mode}"]
        elif mode == "random-mix":
            for min_max in sigmas_random_mix:
                model_names += [f"{arch}_{mode}_s{min_max}"]
        elif mode == "mix":
            for sigma in sigmas_mix:
                model_names += [f"{arch}_{mode}_s{sigma:02d}"]
        else:
            for s in range(4):
                model_names += [f"{arch}_{mode}_s{s + 1:02d}"]

    main(
        arch=arch,
        model_names=model_names,
        models_dir=models_dir,  # model directory
        out_dir=out_dir,
        dataset_path="/mnt/data1/ImageNet/ILSVRC2012/",
        all_filter_combinations=all_filter_combinations,
        test_images_dir=test_images_dir,  # directory for test images overview file
        save_test_images=save_test_images,
        target_id=1,  # bear
        num_filters=6,  # number of band-pass filters
        num_images=10,  # number of images for each class.
        seed=42,
    )
