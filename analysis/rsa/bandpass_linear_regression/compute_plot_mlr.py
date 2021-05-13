import os
import pathlib
import sys

from tqdm import tqdm

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.bandpass.mlr import compute_mlr, plot_mlr


if __name__ == "__main__":
    # arguments
    arch = "alexnet"
    num_classes = sys.argv[1]
    epoch = 60
    num_filters = 6
    num_images = 1600

    analysis = "mlr"

    # I/O settings
    # data_dir = f"/Users/sou/lab1-work/blur-training-dev/analysis/rsa/bandpass/results/activations/{num_classes}-class-{arch}/"
    data_dir = f"./results/activations/{num_classes}-class-{arch}/"
    results_dir = f"./results/{analysis}/{num_classes}-class"

    assert os.path.exists(data_dir), f"{data_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)

    # models to compare
    model_names = [
        f"{arch}_normal",
        f"{arch}_all_s04",
        f"{arch}_mix_s04",
        f"{arch}_multi-steps",
        # sin_names[arch],
        # "vone_alexnet",
        # "untrained_alexnet",
    ]

    print("===== I/O =====")
    print("IN, data_dir:", data_dir)
    print()

    print("===== models to analyze =====")
    print(model_names)
    print()

    for model_name in tqdm(model_names, desc="models"):
        in_dir = os.path.join(data_dir, model_name + f"_e{epoch:02d}")
        filename = f"{model_name}.csv"
        out_path = os.path.join(results_dir, filename)

        df_results = compute_mlr(
            in_dir=in_dir,
            out_path=out_path,
            num_filters=num_filters,
            num_images=num_images,
        )

        ### plot (temp) ###
        plots_dir = f"./plots/{analysis}/{num_classes}-class/{model_name}"

        os.makedirs(plots_dir, exist_ok=True)

        plot_mlr(df_results=df_results, out_dir=plots_dir)

    print(f"{analysis} ({num_classes}-class): ALL DONE!!")
