import os

import pickle


def load_activations(in_dir, file_name):
    file_path = os.path.join(in_dir, file_name)
    with open(file_path, "rb") as f:
        return pickle.load(f)
