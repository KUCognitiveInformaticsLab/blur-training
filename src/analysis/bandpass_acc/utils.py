import torch
import pandas as pd


def load_result(file_path):
    return pd.read_csv(file_path, index_col=0)


def load_model_acc1(model_path):
    """Return top-1 accuracy from saved model"""
    checkpoint = torch.load(model_path, map_location='cpu')

    return checkpoint['val_acc'].item()
