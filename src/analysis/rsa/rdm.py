import pickle


def save_rdms(mean_rdms: dict, file_path: str):
    # save dict object
    with open(file_path, "wb") as f:
        pickle.dump(mean_rdms, f)


def load_rdms(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
