import os
import pathlib
import sys

import torch
from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../"))

from src.analysis.shape_bias.shape_bias import compute_shape_bias
from src.model.utils import load_model


if __name__ == "__main__":
    arch = str(sys.argv[1])  # e.g.: ("alexnet", "vone_alexnet")
    num_classes = int(sys.argv[2])  # number of last output of the models
    compare = str(
        sys.argv[3]
    )  # models to compare e.g.: ("vss", "all_blur-training", "mix_no-blur", "mix_no-sharp")

    epoch = 60

    pretrained = False  # True if you want to use pretrained vone_alexnet.

    analysis = "shape_bias"

    # I/O
    cue_conf_data_path = "/mnt/data1/shape-texture-cue-conflict/"
    models_dir = "/mnt/data1/pretrained_models/blur-training/imagenet{}/models/".format(
        16 if num_classes == 16 else 1000  # else is (num_classes == 1000)
    )
    simclr_dir = "/mnt/data1/pretrained_models/simclr/pytorch_models/"
    results_dir = f"./results/{num_classes}-class"

    assert os.path.exists(cue_conf_data_path), f"{cue_conf_data_path} does not exist."
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # models to compare
    from src.model.model_names import get_model_names

    model_names = get_model_names(arch=arch, models=compare, num_classes=num_classes)

    print("===== arguments =====")
    print("arch:", arch)
    print("num_classes:", num_classes)
    print()

    print("===== I/O =====")
    print("IN, models_dir:", models_dir)
    print("OUT, results_dir:", results_dir)
    print()

    print("===== models to analyze =====")
    print(model_names)
    print()

    # ===== main =====
    print("===== main =====")

    for model_name in tqdm(model_names, desc="models", leave=False):
        print(f"{model_name}: computing shape bias...")  # load model
        if "SIN" in model_name or model_name == "vone_alexnet":
            model = load_model(model_name=model_name).to(device)
            model.num_classes = num_classes
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
        if "simclr" in model_name:
            model_path = os.path.join(
                simclr_dir, model_name.replace("_simclr", "") + ".pth"
            )
            model = load_model(model_name=model_name, model_path=model_path).to(device)
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
        elif "untrained" in model_name:
            model_path = ""  # load untrained model
            model = load_model(
                arch=arch, num_classes=num_classes, model_path=model_path
            ).to(device)
            model.num_classes = num_classes
            all_file = os.path.join(
                results_dir, "all_decisions_{}.csv".format(model_name)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}.csv".format(model_name)
            )
        else:
            model_path = os.path.join(
                models_dir, model_name, "epoch_{}.pth.tar".format(epoch)
            )
            model = load_model(
                arch=arch,
                num_classes=num_classes,
                model_path=model_path,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            ).to(device)
            model.num_classes = num_classes
            all_file = os.path.join(
                results_dir, "all_decisions_{}_e{}.csv".format(model_name, epoch)
            )
            correct_file = os.path.join(
                results_dir, "correct_decisions_{}_e{}.csv".format(model_name, epoch)
            )

        # compute
        df_all_decisions, df_correct_decisions = compute_shape_bias(
            model=model,
            num_classes=num_classes,
            cue_conf_data_path=cue_conf_data_path,
            device=device,
        )

        # save
        df_all_decisions.to_csv(all_file)
        df_correct_decisions.to_csv(correct_file)

    print(f"{analysis}: All done!!")
