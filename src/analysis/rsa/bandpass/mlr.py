import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

sys.path.append("../../../../")

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.activations import load_activations


def compute_mlr(
    in_dir: str,
    out_path: str,
    num_filters: int,
    num_images: int,
):
    clf = linear_model.LinearRegression()

    results = []
    for layer in alexnet_layers:
        X = []  # activations by band-pass images
        Y = []  # activations by raw images

        for image_id in range(num_images):
            file_name = f"image{image_id:04d}_f{num_filters:02d}.pkl"
            activations = load_activations(in_dir=in_dir, file_name=file_name)
            X += [activations[layer][1:].reshape(num_filters, -1).transpose(1, 0)]
            Y += [activations[layer][0].reshape(1, -1).transpose(1, 0)]

        # activations by band-pass images
        X = np.array(X).reshape(-1, num_filters)  # (N * C * H * W,  F )
        # activations by raw images
        Y = np.array(Y).reshape(-1, 1)  # (N * C * H * W,  1)

        clf.fit(X, Y)

        # add results of each layer
        results += [
            np.array(list(clf.coef_[0]) + list(clf.intercept_) + [clf.score(X, Y)])
        ]

    df_results = pd.DataFrame(
        np.array(results),
        columns=[f"w{i}" for i in range(num_filters)] + ["residual"] + ["r^2"],
        index=alexnet_layers,
    )

    # save
    df_results.to_csv(out_path)

    return df_results  # temp for plotting


def plot_mlr(df_results: pd.DataFrame, out_dir: str):
    filename = f"all_values.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

    df_results.T.plot()
    filename = f"all_layers.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

    for k in df_results.keys():
        df_results[k].plot()
        filename = f"{k}.png"
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()


if __name__ == "__main__":
    # arguments
    model_name = "alexnet_normal"
    epoch = 60
    num_filters = 6
    num_images = 1600

    # I/O settings
    data_dir = "./results/activations/alexnet/"
    results_dir = "./results/mlr"

    assert os.path.exists(data_dir), f"{data_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)

    in_dir = os.path.join(data_dir, model_name + f"_e{epoch:02d}")
    filename = f"{model_name}_e{epoch:02d}.csv"
    out_path = os.path.join(results_dir, filename)

    df_results = compute_mlr(
        in_dir=in_dir, out_path=out_path, num_filters=num_filters, num_images=num_images
    )

    ### plot (temp) ###
    plots_dir = f"./plots/mlr/{model_name}_e{epoch:02d}"
    os.makedirs(plots_dir, exist_ok=True)
    plot_mlr(df_results=df_results, out_dir=plots_dir)
