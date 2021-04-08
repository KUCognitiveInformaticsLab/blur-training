import pickle


def load_activations(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_activations(activations, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(activations, f)
