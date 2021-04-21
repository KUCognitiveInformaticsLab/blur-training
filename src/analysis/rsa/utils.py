import pickle


def save_activations(activations, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(activations, f)


def load_activations(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_rdms(mean_rdms: dict, file_path: str):
    # save dict object
    with open(file_path, "wb") as f:
        pickle.dump(mean_rdms, f)


def load_rdms(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_rsms(mean_rsms: dict, file_path: str):
    # save dict object
    with open(file_path, "wb") as f:
        pickle.dump(mean_rsms, f)


def load_rsms(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
