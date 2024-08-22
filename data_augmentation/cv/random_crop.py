# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import numpy as np

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------

def random_crop(img: np.ndarray, size: tuple, seed: int = None) -> np.ndarray:

    """
    Randomly crops an image or a batch of images to the specified size.

    Args:
        img (numpy.ndarray): Input image or batch of images to be cropped. 
                             The array can be 2D (height, width), 3D (height, width, channels), 
                             or 4D (batch_size, height, width, channels).
        size (tuple): The size of the crop in the format (crop_height, crop_width). 
                      This must be a tuple of two integers.
        seed (int, optional): A seed value for the random number generator to ensure reproducibility. 
                              Defaults to None.

    Returns:
        numpy.ndarray: A randomly cropped image or batch of images of the specified size.
                       If the input is 4D (batch of images), the function will return a 
                       batch of cropped images with the same batch size.
    """

    np.random.seed(seed)
    
    # Verify that size is a tuple of length 2
    assert isinstance(size, tuple) and len(size) == 2, "size must be a tuple in the format (crop_height, crop_width)"
    
    # Verify that the image has 2, 3, or 4 dimensions
    assert len(img.shape) in [2, 3, 4], "Array dimension error, must be 2, 3, or 4"

    # Determine image height and width
    if len(img.shape) == 4:  
        
        # Batch of images
        img_height, img_width = img.shape[1:3]

    else:  
        
        # Single image
        img_height, img_width = img.shape[:2]
    
    # Verify that the dimensions of the image are greater than or equal to the crop size
    assert img_height >= size[0] and img_width >= size[1], "Array size is smaller than the size to crop"

    max_x = img_width - size[1]
    max_y = img_height - size[0]
    
    x = np.random.randint(0, max_x + 1)
    y = np.random.randint(0, max_y + 1)
    
    if len(img.shape) == 4:  
        
        # Batch of images
        return img[:, y:y + size[0], x:x + size[1], :]

    else:  
        
        # Single image
        return img[y:y + size[0], x:x + size[1]]

def RandomCropNumpy(data: np.ndarray, multiple: int, min_height: int, width: int = 24) -> np.ndarray:

    """
    Randomly crops the input numpy array.

    Args:
        data (numpy.ndarray): Input numpy array.
        multiple (int): Value to ensure the cropped height is a multiple of.
        min_height (int): Minimum height for the random crop.
        width (int): Width of the crop.

    Returns:
        numpy.ndarray: Randomly cropped numpy array.
    """

    # Randomly decides whether to apply the crop or not, and if the minimum size is met.
    if np.random.uniform(0, 1) > 0.5 and data.shape[0] > min_height:

        # Generates random height within specified ranges
        height = np.random.randint(min_height, data.shape[0])
        height = ((height + (multiple - 1)) // multiple) * multiple

        # Applies random cropping to the input
        return random_crop(data, (height, width))

    else:

        return data
