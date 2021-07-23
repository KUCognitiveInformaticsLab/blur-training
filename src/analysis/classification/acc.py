import csv

import pandas as pd


def save_acc1(acc1, file_path):
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["acc1"])
        writer.writerow([acc1])


def save_acc(acc, file_path, metric="acc1"):
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow([metric])
        writer.writerow([acc])


def load_acc1(file_path):
    return pd.read_csv(file_path).values[0][0]
