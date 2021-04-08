import os

import pickle


def load_activations(in_dir, file_name):
    file_path = os.path.join(in_dir, file_name)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_activations(activations, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(activations, f)
