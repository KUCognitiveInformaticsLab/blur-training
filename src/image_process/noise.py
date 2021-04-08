import numpy as np

# Ref: https://www.javaer101.com/ja/article/3437291.html
def gaussian_noise(image: np.array, mean = 0, var = 0.1):
    """Adds Gaussian noise to input image.
    Args:
        image (np.array): an image (H, W, C)
    Return:
        noisy (np.array): a noisy image (H, W, C)
    """
    row,col,ch = image.shape

    sigma = var ** 0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)

    noisy = image + gauss

    return noisy