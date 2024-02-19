# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import numpy as np
from numba import njit


# --------------------------------------------------------------------------------------------
# DEFINITIONS OF FUNCTIONS
# --------------------------------------------------------------------------------------------


@njit
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):

    gamma_1_sample = np.array([np.random.gamma(concentration_1) for _ in range(size)])
    gamma_2_sample = np.array([np.random.gamma(concentration_0) for _ in range(size)])
    
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@njit
def mix_up(ds_one, ds_two, alpha=0.2):
    
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = images_one.shape[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = l.reshape((batch_size, 1, 1, 1))
    y_l = l.reshape((batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    
    return images, labels