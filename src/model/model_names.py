import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.model.load_sin_pretrained_models import sin_names


def get_model_names(arch):
    # models to compare
    model_names = [
        f"untrained_{arch}",
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        f"vone_{arch}",
        sin_names[arch],
    ]

    # model_names = [
    #     f"{arch}_normal",
    #     # f"{arch}_multi-steps",
    # ]
    # modes = [
    #     f"{arch}_all",
    #     f"{arch}_mix",
    #     f"{arch}_random-mix",
    #     f"{arch}_single-step",
    #     # f"{arch}_fixed-single-step",
    #     # f"{arch}_reversed-single-step",
    # ]
    #
    # # sigmas to compare
    # sigmas_mix = [s for s in range(1, 6)] + [10]
    # sigmas_random_mix = ["00-05", "00-10"]
    #
    # # add sigma to compare to the model names
    # for mode in modes:
    #     if mode == f"{arch}_random-mix":
    #         for min_max in sigmas_random_mix:
    #             model_names += [f"{mode}_s{min_max}"]
    #     elif mode == f"{arch}_mix":
    #         for sigma in sigmas_mix:
    #             model_names += [f"{mode}_s{sigma:02d}"]
    #     else:
    #         for sigma in range(1, 5):
    #             model_names += [f"{mode}_s{sigma:02d}"]

    return model_names


def rename_model_name(model_name: str):
    model_name = model_name.replace("normal", "S")
    model_name = model_name.replace("mix", "S+B")
    model_name = model_name.replace("all", "B")
    model_name = model_name.replace("_", " ")

    return model_name
