import numpy as np
from numba import njit


@njit
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):

    gamma_1_sample = np.array([np.random.gamma(concentration_1) for _ in range(size)])
    gamma_2_sample = np.array([np.random.gamma(concentration_0) for _ in range(size)])
    
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@njit
def clip(value, min_value, max_value):

    return max(min_value, min(value, max_value))


@njit
def get_box(lambda_value, height, width):
            
    cut_rat = np.sqrt(1.0 - lambda_value)

    cut_w = np.int32(width * cut_rat)  
    cut_h = np.int32(height * cut_rat) 

    cut_x = np.random.randint(width)  
    cut_y = np.random.randint(height)  

    boundaryx1 = clip(cut_x - cut_w // 2, 0, width)
    boundaryy1 = clip(cut_y - cut_h // 2, 0, height)
    boundaryx2 = clip(cut_x + cut_w // 2, 0, width)
    boundaryy2 = clip(cut_y + cut_h // 2, 0, height)

    return boundaryx1, boundaryy1, boundaryx2, boundaryy2


@njit
def cutmix_one_sample(train_ds_one, train_ds_two, alpha, beta):

    (image1, label1), (image2, label2) = train_ds_one, train_ds_two
    height, width, _ = image1.shape

    # Get a sample from the Beta distribution
    lambda_value = sample_beta_distribution(1, alpha, beta)[0]

    # Get the bounding box offsets, heights and widths
    bbx1, bby1, bbx2, bby2 = get_box(lambda_value, height, width)

    # Combine images
    image1[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
    
    # Adjust Lambda in accordance to the pixel ration
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (height * width))
    
    # Combine the labels of both images
    label = lambda_value * label1 + (1 - lambda_value) * label2
    
    return image1, label
    
    
@njit
def cut_mix(dataset_one, dataset_two, alpha, beta):
    
    (ds_one, labels_one) = dataset_one
    (ds_two, labels_two) = dataset_two

    images = np.empty_like(ds_one)
    labels = np.empty_like(labels_one)

    for i in range(ds_one.shape[0]):
        
        images[i], labels[i] = cutmix_one_sample((ds_one[i], labels_one[i]), (ds_two[i], labels_two[i]), alpha, beta)

    return images, labels