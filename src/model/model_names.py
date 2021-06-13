import os
import pathlib
import re
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.model.load_sin_pretrained_models import sin_names


def get_model_names(arch, compare="vss"):
    # models to compare
    if compare == "vss":
        model_names = [
            f"untrained_{arch}",
            f"{arch}_normal",
            f"{arch}_all_s04",
            f"{arch}_mix_s04",
            f"{arch}_multi-steps",
            f"vone_{arch}",
            sin_names[arch],
        ]
    elif compare == "mix_p-blur":
        model_names = [
            f"{arch}_mix_p-blur_s01_no-blur-1label",
            f"{arch}_mix_p-blur_s01_no-blur-8label",
            f"{arch}_mix_p-blur_s04_no-blur-1label",
            f"{arch}_mix_p-blur_s04_no-blur-8label",
            f"{arch}_mix_p-blur_s01{arch}",
            f"{arch}_mix_p-blur_s04{arch}",
        ]
    elif compare == "mix_no-blur":
        model_names = [f"{arch}_mix_s{s:02d}_no-blur-1label" for s in range(1, 5)] + [
            f"{arch}_mix_s{s:02d}_no-blur-8label" for s in range(1, 5)
        ]
    elif compare == "mix_no-sharp":
        model_names = [f"{arch}_mix_s{s:02d}_no-sharp-1label" for s in range(1, 5)] + [
            f"{arch}_mix_s{s:02d}_no-sharp-8label" for s in range(1, 5)
        ]
    elif compare == "all_blur-training":
        model_names = \
              [f"{arch}_normal"] \
            + [f"{arch}_multi-steps"] \
            + [f"{arch}_mix_s{s:02d}" for s in range(1, 5)] \
            + [f"{arch}_all_s{s:02d}" for s in range(1, 5)] \
            + [f"{arch}_random-mix_s{s}" for s in ["00-02", "00-04", "00-08", "00-16"]]
    elif compare == "all_blur-training_old":
        modes = [
            f"{arch}_all",
            f"{arch}_mix",
            f"{arch}_random-mix",
            f"{arch}_single-step",
            f"{arch}_fixed-single-step",
            f"{arch}_reversed-single-step",
        ]

        # sigmas to compare
        sigmas_mix = [s for s in range(1, 6)] + [10]
        sigmas_random_mix = ["00-05", "00-10"]

        # add sigma to compare to the model names
        for mode in modes:
            if mode == f"{arch}_random-mix":
                for min_max in sigmas_random_mix:
                    model_names += [f"{mode}_s{min_max}"]
            elif mode == f"{arch}_mix":
                for sigma in sigmas_mix:
                    model_names += [f"{mode}_s{sigma:02d}"]
            else:
                for sigma in range(1, 5):
                    model_names += [f"{mode}_s{sigma:02d}"]

    return model_names


def rename_model_name(model_name: str, arch: str = "alexnet"):
    model_name = model_name.replace(f"raw_images", f"(Original) bandpass corr.")

    model_name = model_name.replace(f"untrained_{arch}", f"Untrained-{arch}")

    model_name = model_name.replace(f"{arch}_normal", f"S-{arch}")
    model_name = model_name.replace(f"{arch}_all", f"B-{arch}")
    model_name = model_name.replace(f"{arch}_mix", f"B+S-{arch}")
    model_name = model_name.replace(f"mix", f"B+S-{arch}_s01")
    model_name = model_name.replace(f"{arch}_multi-steps", f"B2S-{arch}")
    model_name = re.sub("s([0-9]+)", r"(σ=\1)", model_name)  # sigma value
    model_name = re.sub("0", "", model_name)

    model_name = model_name.replace(sin_names[arch], f"SIN-trained-{arch}")
    model_name = model_name.replace(f"vone_{arch}", f"VOne{arch}")

    model_name = model_name.replace("alexnet", "Net")

    model_name = model_name.replace("_", " ")

    return model_name
