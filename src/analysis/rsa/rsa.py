import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist


alexnet_layers = [
    "conv-relu-1",
    "conv-relu-2",
    "conv-relu-3",
    "conv-relu-4",
    "conv-relu-5",
    "fc-relu-1",
    "fc-relu-2",
    "last-outputs",
]


class AlexNetRSA:
    def __init__(self, model):
        """
        Args:
            model: Alexnet model (PyTorch)
        """
        self.model = model

        self.layers = alexnet_layers

        self.activations = {}

        self.model.features[1].register_forward_hook(
            self._get_activations(self.layers[0])
        )
        self.model.features[4].register_forward_hook(
            self._get_activations(self.layers[1])
        )
        self.model.features[7].register_forward_hook(
            self._get_activations(self.layers[2])
        )
        self.model.features[9].register_forward_hook(
            self._get_activations(self.layers[3])
        )
        self.model.features[11].register_forward_hook(
            self._get_activations(self.layers[4])
        )
        self.model.classifier[2].register_forward_hook(
            self._get_activations(self.layers[5])
        )
        self.model.classifier[5].register_forward_hook(
            self._get_activations(self.layers[6])
        )
        self.model.classifier[6].register_forward_hook(
            self._get_activations(self.layers[7])
        )

    # Ref:
    #  https://discuss.pytorch.org/t/extract-features-from-layer-of-submodule-of-a-model/20181
    def _get_activations(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach().cpu().numpy()

        return hook

    def compute_activations(self, images: torch.Tensor) -> dict:
        """Computes activations of units in a model.
        Args:
            images: images to test the model with. shape=(N, C, H, W)

        Returns: activations
        """

        _ = self.model(images)

        return self.activations

    def compute_mean_rdms(self, images: torch.Tensor) -> dict:
        """Computes RDM for each image and return mean RDMs.
        Args:
            images: images to test the model with. shape=(N, F+1, C, H, W)
                Where: F is the number of band-pass filters.
                    F+1 means filter applied images(F) and a raw image(+1)

        Returns: Mean RDMs (Dict)
        """
        num_filters = images.shape[1] - 1
        mean_rdms = {}

        for layer in self.layers:
            rdms = []
            # compute RDM for each image (with some filters applied)
            for imgs in images:
                # shape of imgs == (F+1, C, H, W)
                # where F is the number of band-pass filters.
                # F+1 means band-pass filters(F) and raw image(+1)
                self.activations = self.compute_activations(imgs)
                activation = self.activations[layer].reshape(num_filters + 1, -1)
                rdm = squareform(pdist(activation, metric="correlation"))  # 1 - corr.
                rdms.append(rdm)

            rdms = np.array(rdms)

            mean_rdms[layer] = rdms.mean(0)

        return mean_rdms
