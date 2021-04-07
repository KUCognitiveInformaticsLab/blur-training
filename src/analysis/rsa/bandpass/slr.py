import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

sys.path.append("../../../../")

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.activations import load_activations


def compute_slr(
    in_dir: str,
    out_dir: str,
    model_name: str,
    epoch: int,
    num_filters: int,
    num_images: int,
):
    results_w = []
    results_res = []
    results_r2 = []

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

        w = []  # weight
        res = []  # residual
        r2 = []  # coefficient of determination

        for i in range(num_filters):
            clf = linear_model.LinearRegression()
            clf.fit(X[:, i].reshape(-1, 1), Y)
            w += list(clf.coef_[0])
            res += list(clf.intercept_)
            r2 += [clf.score(X[:, i].reshape(-1, 1), Y)]

        results_w += [np.array(w)]
        results_res += [np.array(res)]
        results_r2 += [np.array(r2)]

    # create results df
    band_filters = [f"f{i}" for i in range(num_filters)]
    df_w = pd.DataFrame(results_w, columns=band_filters, index=alexnet_layers)
    df_res = pd.DataFrame(results_res, columns=band_filters, index=alexnet_layers)
    df_r2 = pd.DataFrame(results_r2, columns=band_filters, index=alexnet_layers)

    # save
    filename = f"{model_name}_e{epoch:02d}_w.csv"
    df_w.to_csv(os.path.join(out_dir, filename))
    filename = f"{model_name}_e{epoch:02d}_residual.csv"
    df_res.to_csv(os.path.join(out_dir, filename))
    filename = f"{model_name}_e{epoch:02d}_r2.csv"
    df_r2.to_csv(os.path.join(out_dir, filename))

    return df_w, df_res, df_r2  # temp for plotting


def plot_slr(
    df_w: pd.DataFrame, df_res: pd.DataFrame, df_r2: pd.DataFrame, out_dir: str
):
    df_w.plot()
    filename = f"w_filters.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

    df_w.T.plot()
    filename = f"w_layers.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

    df_res.plot()
    filename = f"residual_filters.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

    df_res.T.plot()
    filename = f"residual_layers.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

    df_r2.plot()
    filename = f"r2_filters.png"
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

    df_r2.T.plot()
    filename = f"r2_layers.png"
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
    results_dir = "./results/slr"

    assert os.path.exists(data_dir), f"{data_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)

    in_dir = os.path.join(data_dir, model_name + f"_e{epoch:02d}")

    df_w, df_res, df_r2 = compute_slr(
        in_dir=in_dir,
        out_dir=results_dir,
        model_name=model_name,
        epoch=epoch,
        num_filters=num_filters,
        num_images=num_images,
    )

    ### plot (temp) ###
    plots_dir = f"./plots/slr/{model_name}_e{epoch:02d}"

    os.makedirs(plots_dir, exist_ok=True)

    plot_slr(df_w=df_w, df_res=df_res, df_r2=df_r2, out_dir=plots_dir)
